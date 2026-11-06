/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * hyper_connection_test: what source/hyper_connection.c and the host reference of the hyper-connection kernels do.
 * No card: a prepare function is arithmetic on a description, the arena a number, the kernel handles made up.
 *
 *   - hyper_kernels_prepare against a table that records what it is asked: the four functions once each, in
 *     the order of the members, from kernel_hyper_connection with the size of their record; the first failure ends
 *     it with that status, names the function and leaves the handles untouched; a missing argument or a table
 *     without kernel_prepare is refused. Every module of kernel_hyper_connection the program carries, one for each
 *     architecture, holds the four names as symbols. On a card of the host-memory backend of each class the four are
 *     prepared, and launch_run takes every launch prepared from an accepted description.
 *   - The three prepare functions, a site in both forms: an accepted description, its launches checked by hand
 *     (kernel, grid, block, the record's every offset); every rule of hyper_connection.h broken once, each range of a
 *     description shrunk, moved off the alignment, out of the arena and onto another: refused with one error line
 *     that holds the rule's words, the launches untouched; a kernel it does not launch may be missing. Of a chunk's
 *     rows (MODEL_WIDE_MOST_ROWS, a wide step's) a site and the embedding are accepted, every launch over the
 *     rows; one more row is refused. The final collapse takes a step's rows (LINEAR_MAXIMUM_ROWS), not one
 *     more.
 *   - The reference on cases worked by hand: a site of zero mixes (pre 1/2 + eps, post 1, comb all equal, rows and
 *     columns adding to one), a random site (doubly stochastic within 2^-16), a collapse with pre (1, 0, 0, 0), posts
 *     whose sums are exact, the combine, a norm of known size, and the distance in float32 places; the draft's tap,
 *     the collapse with pre 1/4 each, of streams 1 to 4 is 2.5, and of four equal streams that stream.
 *   - The draft's inputs (dspark_v1/DESIGN.md C1) and the final's kept collapse: hyper_kernels_prepare leaves
 *     draft_inputs and final_kept unprepared; accepted descriptions of both modes and every tap at 1, 8 and 4,096
 *     rows, their launch checked by hand; every rule of the draft inputs broken once (rows, mode, tap, a weight for
 *     the mean, a reserved field, the kernel, each range shrunk, moved off the alignment, out of the arena and onto
 *     another), refused with one error line, the launch untouched; the final with a collapse range: the draft's final
 *     (final_kept) with its record's comb_offset, refused without that kernel, which the final without one does not
 *     need; the collapse too short, on the head's input and at byte 0 refused.
 * A failure is one line on the standard output.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "log_expectation.h"
#include "anemone/hyper_connection.h"
#include "anemone/hyper_connection_records.h"
#include "anemone/model_layout.h"
#include "anemone/status.h"
#include "anemone/step_records.h"
#include "backend_cuda_kernel_images.h"
#include "host_memory_backend.h"
#include "hyper_connection_reference.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define HIDDEN MODEL_HIDDEN
#define STREAMS MODEL_STREAMS
#define COMB_VALUES (STREAMS * STREAMS)
#define KERNEL_COUNT 4u
#define ALL_HANDLES (KERNEL_COUNT + 2u)             /* the four and the draft's two, which draft.h prepares */
#define WIDE_ROWS MODEL_WIDE_MOST_ROWS
#define DRAFT_INPUT_RANGES 3u
#define FIRST_RECORDED_HANDLE 40u                   /* what the recording table gives the first kernel it is asked */
#define FIRST_MADE_UP_HANDLE 11u
#define CARD_MEMORY_BYTES 4096u                     /* of the stand-in card the kernels are asked of */
#define UNTOUCHED_BYTE 0xA5
#define CAPTURE_CAPACITY 2048u
#define ROWS ((uint64_t)3u)
#define TOO_MANY_ROWS (MODEL_WIDE_MOST_ROWS + 1u)
#define VOCABULARY ((uint64_t)7u)
#define ARENA_BYTES (1ull << 40u)                   /* more than any case lays out */
#define ALIGNMENT LINEAR_OFFSET_ALIGNMENT_BYTES
#define MISALIGNMENT (ALIGNMENT / 2u)
#define SITE_RANGES 11u
#define EMBED_RANGES 4u
#define FINAL_RANGES 4u
#define SCALE_BYTES (HYPER_SCALES * sizeof(float))
#define SUMS_ROW_BYTES (HIDDEN * sizeof(float))
#define RANDOM_STREAM 0x5E1Eu
#define MIX_SPREAD 2.0                              /* random mixes, scales and bases from -1 to 1 */
#define SCALE_POSITIONS \
	(2u * MODEL_MIXES) /* of the hash: behind the mixes and the bases */
#define HASH_UNIT 0x1p-64                           /* a hash of 64 bits as a fraction of one */
#define STOCHASTIC_TOLERANCE 0x1p-16
#define HALF 0.5f
#define QUARTER 0.25f
#define KNOWN_WEIGHT 2.0f
#define KNOWN_ROUTED 1.5f
#define KNOWN_SUM 2.5f                              /* the four streams 1 to 4 a quarter each */
#define NORM_TOLERANCE 0x1p-40
#define FLOAT32_ONE_BITS 0x3F800000u
#define ACROSS_ZERO_PLACES (2u * FLOAT32_ONE_BITS)   /* from -1 to 1 */
#define LAUNCHED_KERNELS 2u                         /* the most kernels one description launches: a site's */

/* Words of the one error line of each rule. */
#define RULE_MISSING "a missing argument"
#define RULE_NOT_PREPARED "is not prepared"
#define RULE_RESERVED "a reserved field"
#define RULE_ROWS "a launch takes 1 to"
#define RULE_VOCABULARY "an embedding of no tokens"
#define RULE_ALIGNED "an offset is a multiple"
#define RULE_ARENA "leaves the arena"
#define RULE_TOUCHED "the launches touch"
#define RULE_EXACT "its format stores"
#define RULE_SHARED "share a byte"
#define RULE_DRAFT_COUNTS "rows; mode"
#define RULE_COLLAPSE_AT_ZERO "where no state array lies"

enum prepared { ATTENTION_SITE, FFN_SITE, EMBED, FINAL, PREPARED_COUNT };
enum dimension { GRID_COLUMNS, GRID_ROWS, GRID_DEPTH };

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_hyper_connection_test_XXXXXX";
static int capture_descriptor = -1;
static int error_stream_descriptor = -1;
static struct hyper_kernels_record made_up_kernels;

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

/* ---- the kernels of a card ---- */

struct kernel_asked {
    const char *kernel_file_name;
    const char *function_name;
    uint32_t argument_byte_count;
};
static const struct kernel_asked kernels_expected[KERNEL_COUNT] = {
	{ "kernel_hyper_connection", "hyper_mix_finish",
	  sizeof(struct hyper_finish_arguments) },
	{ "kernel_hyper_connection", "hyper_final",
	  sizeof(struct hyper_finish_arguments) },
	{ "kernel_hyper_connection", "hyper_post_entry",
	  sizeof(struct hyper_post_arguments) },
	{ "kernel_hyper_connection", "hyper_embed",
	  sizeof(struct hyper_embed_arguments) },
};
static struct kernel_asked kernels_asked[KERNEL_COUNT];
static uint32_t asked_count = 0;
static uint32_t failing_call_number = 0;            /* counted from 1; 0: no call fails */

_Static_assert(sizeof(struct hyper_kernels_record) ==
		       ALL_HANDLES * sizeof(uint32_t),
	       "four handles, the draft's two");

MUST_CHECK static result_status
recording_kernel_prepare(struct device_card *card, const char *kernel_file_name,
			 const char *function_name,
			 uint32_t argument_byte_count, uint32_t *kernel_out)
{
    if (card != NULL && asked_count < KERNEL_COUNT) {
        kernels_asked[asked_count] = (struct kernel_asked){kernel_file_name, function_name, argument_byte_count};
    }
    asked_count++;
    if (asked_count == failing_call_number) {
	    return STATUS_DEVICE_ERROR;
    }
    *kernel_out = FIRST_RECORDED_HANDLE + asked_count - 1;
    return STATUS_OK;
}

static void test_kernels_prepare(void)
{
	static const struct backend_operations recording_backend = {
		.name = "recording", .kernel_prepare = recording_kernel_prepare
	};
	static const struct backend_operations backend_without_kernels = {
		.name = "no kernels"
	};
	struct hyper_kernels_record kernels;
	struct hyper_kernels_record untouched;
	uint32_t handles[KERNEL_COUNT];
	struct device_card *card = NULL;
	char logged[CAPTURE_CAPACITY];
	context_text = "the kernels of a card";
	memset(&kernels, UNTOUCHED_BYTE, sizeof(kernels));
	memset(&untouched, UNTOUCHED_BYTE, sizeof(untouched));
	test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
	expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES,
						  &card) == STATUS_OK,
	       "a card opens");
	capture_begin();
	expect(hyper_kernels_prepare(NULL, card, &kernels) ==
		       STATUS_INVALID_ARGUMENT,
	       "no table");
	expect(hyper_kernels_prepare(&backend_without_kernels, card,
				     &kernels) == STATUS_INVALID_ARGUMENT,
	       "a table without kernel_prepare");
	expect(hyper_kernels_prepare(&recording_backend, NULL, &kernels) ==
		       STATUS_INVALID_ARGUMENT,
	       "no card");
	expect(hyper_kernels_prepare(&recording_backend, card, NULL) ==
		       STATUS_INVALID_ARGUMENT,
	       "nowhere to put the handles");
	expect(hyper_kernels_prepare(&test_host_memory_backend, card,
				     &kernels) == STATUS_UNSUPPORTED,
	       "a card of class 0 has no build of the kernels: its status comes back");
	capture_end(logged, sizeof(logged));
	for (uint32_t call_number = 1; call_number <= KERNEL_COUNT;
	     call_number++) {
		expect(asked_count == 0,
		       "a table is asked nothing more once a call was refused or has failed");
		failing_call_number = call_number;
		capture_begin();
		expect(hyper_kernels_prepare(&recording_backend, card,
					     &kernels) == STATUS_DEVICE_ERROR,
		       "the status of the kernel_prepare that fails");
		capture_end(logged, sizeof(logged));
		expect_one_error_line_with(
			logged, kernels_expected[call_number - 1].function_name,
			"it is named");
		expect(asked_count == call_number,
		       "the first failure ends the preparation");
		asked_count = 0;
	}
    failing_call_number = 0;
    expect(memcmp(&kernels, &untouched, sizeof(kernels)) == 0, "a failure leaves the handles untouched");
    expect(hyper_kernels_prepare(&recording_backend, card, &kernels) ==
		   STATUS_OK,
	   "every kernel is prepared");
    memcpy(handles, &kernels, sizeof(handles));
    for (uint32_t kernel_index = 0; kernel_index < KERNEL_COUNT; kernel_index++) {
        const struct kernel_asked *const asked = &kernels_asked[kernel_index];
        const struct kernel_asked *const expected = &kernels_expected[kernel_index];
        expect(asked_count == KERNEL_COUNT && asked->kernel_file_name != NULL && asked->function_name != NULL &&
                   strcmp(asked->kernel_file_name, expected->kernel_file_name) == 0 &&
                   strcmp(asked->function_name, expected->function_name) == 0 &&
                   asked->argument_byte_count == expected->argument_byte_count &&
                   handles[kernel_index] == FIRST_RECORDED_HANDLE + kernel_index,
               "each function is asked once, in the order of the members, by its file, its name and the size of its "
               "record, and its member holds its handle");
    }
    expect(kernels.draft_inputs == BACKEND_NO_KERNEL &&
		   kernels.final_kept == BACKEND_NO_KERNEL,
	   "the draft's kernels are left unprepared: a card without the draft keeps its handles");
    test_host_memory_backend.card_close(card, 0);
}

/* ---- descriptions laid out one range behind another ---- */

static struct linear_range next_range(uint64_t *cursor, uint64_t byte_count)
{
	const struct linear_range range = { *cursor, byte_count };
	*cursor += (byte_count + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT +
		   ALIGNMENT;
	return range;
}

static struct hyper_site_description site_of(bool is_combined, uint64_t rows)
{
    uint64_t cursor = ALIGNMENT;
    struct hyper_site_description site = { .row_count = (uint32_t)rows };
    site.streams = next_range(&cursor, rows * MODEL_STREAMS_ROW_BYTES);
    site.carried_pre = next_range(&cursor, rows * MODEL_SITE_ROW_BYTES);
    site.mixes = next_range(&cursor, rows * MODEL_MIXES_ROW_BYTES);
    site.scale = next_range(&cursor, SCALE_BYTES);
    site.base = next_range(&cursor, MODEL_MIXES_ROW_BYTES);
    site.norm_weight = next_range(&cursor, MODEL_HIDDEN_ROW_BYTES);
    site.post = next_range(&cursor, rows * MODEL_SITE_ROW_BYTES);
    site.comb = next_range(&cursor, rows * MODEL_COMB_ROW_BYTES);
    site.hidden = next_range(&cursor, rows * MODEL_HIDDEN_ROW_BYTES);
    site.output = next_range(&cursor, rows * MODEL_HIDDEN_ROW_BYTES);
    if (is_combined) {
        site.routed_sums = next_range(&cursor, rows * SUMS_ROW_BYTES);
    }
    return site;
}

static struct hyper_embed_description embed_of(uint64_t rows)
{
    uint64_t cursor = ALIGNMENT;
    struct hyper_embed_description embed = { .row_count = (uint32_t)rows,
					     .vocabulary_size = VOCABULARY };
    embed.step = next_range(&cursor, STEP_BLOCK_BYTES_VALUE +
					     rows * STEP_ROW_BYTES_VALUE);
    embed.embedding = next_range(&cursor, VOCABULARY * MODEL_HIDDEN_ROW_BYTES);
    embed.streams = next_range(&cursor, rows * MODEL_STREAMS_ROW_BYTES);
    embed.carried_pre = next_range(&cursor, rows * MODEL_SITE_ROW_BYTES);
    return embed;
}

static struct hyper_final_description final_of(uint64_t rows)
{
    uint64_t cursor = ALIGNMENT;
    struct hyper_final_description final = { .row_count = (uint32_t)rows };
    final.streams = next_range(&cursor, rows * MODEL_STREAMS_ROW_BYTES);
    final.carried_pre = next_range(&cursor, rows * MODEL_SITE_ROW_BYTES);
    final.norm_weight = next_range(&cursor, MODEL_HIDDEN_ROW_BYTES);
    final.hidden = next_range(&cursor, rows * MODEL_HIDDEN_ROW_BYTES);
    return final;
}

/* One of the three descriptions, as the prepare functions take it. */
struct described {
    enum prepared kind;
    struct hyper_site_description site;
    struct hyper_embed_description embed;
    struct hyper_final_description final;
};

/* The ranges of a description in the order of its members, and where each is exact. */
static uint32_t ranges_of(struct described *described,
			  struct linear_range **ranges_out, bool *is_exact_out)
{
	struct hyper_site_description *const site = &described->site;
	struct linear_range *const site_ranges[SITE_RANGES] = {
		&site->streams, &site->carried_pre, &site->mixes,
		&site->scale,	&site->base,	    &site->norm_weight,
		&site->post,	&site->comb,	    &site->hidden,
		&site->output,	&site->routed_sums
	};
	struct linear_range *const embed_ranges[EMBED_RANGES] = {
		&described->embed.step, &described->embed.embedding,
		&described->embed.streams, &described->embed.carried_pre
	};
	struct linear_range *const final_ranges[FINAL_RANGES] = {
		&described->final.streams, &described->final.carried_pre,
		&described->final.norm_weight, &described->final.hidden
	};
	static const bool site_exact[SITE_RANGES] = { false, false, false,
						      true,  true,  true,
						      false, false, false,
						      false, false };
	static const bool embed_exact[EMBED_RANGES] = { false, true, false,
							false };
	static const bool final_exact[FINAL_RANGES] = { false, false, true,
							false };
	/* the attention site's: all but the routed sums, the last */
	static const uint32_t range_counts[PREPARED_COUNT] = {
		SITE_RANGES - 1u, SITE_RANGES, EMBED_RANGES, FINAL_RANGES
	};
	const bool is_site = described->kind == ATTENTION_SITE ||
			     described->kind == FFN_SITE;
	const uint32_t count = range_counts[described->kind];
	for (uint32_t index = 0; index < count; index++) {
		ranges_out[index] = is_site ? site_ranges[index] :
					      (described->kind == EMBED ?
						       embed_ranges[index] :
						       final_ranges[index]);
		is_exact_out[index] = is_site ? site_exact[index] :
						(described->kind == EMBED ?
							 embed_exact[index] :
							 final_exact[index]);
	}
    return count;
}

MUST_CHECK static result_status
prepare(const struct described *described, uint64_t arena_byte_count,
	const struct hyper_kernels_record *kernels,
	struct kernel_launch *launches)
{
    switch (described->kind) {
    case ATTENTION_SITE:
    case FFN_SITE:
	    return hyper_site_prepare(&described->site, arena_byte_count,
				      kernels, launches);
    case EMBED:
	    return hyper_embed_prepare(&described->embed, arena_byte_count,
				       kernels, launches);
    case FINAL:
    case PREPARED_COUNT:
        break;
    }
    return hyper_final_prepare(&described->final, arena_byte_count, kernels,
			       launches);
}

/* A description that must be refused: one error line with the words, every launch byte untouched. */
static void expect_refused(const struct described *described,
			   uint64_t arena_byte_count,
			   const struct hyper_kernels_record *kernels,
			   const char *words, const char *what)
{
	struct kernel_launch launches[HYPER_SITE_LAUNCH_COUNT];
	struct kernel_launch untouched[HYPER_SITE_LAUNCH_COUNT];
	char logged[CAPTURE_CAPACITY];
	memset(launches, UNTOUCHED_BYTE, sizeof(launches));
	memset(untouched, UNTOUCHED_BYTE, sizeof(untouched));
	capture_begin();
	const result_status status =
		prepare(described, arena_byte_count, kernels, launches);
	capture_end(logged, sizeof(logged));
	expect(status == STATUS_INVALID_ARGUMENT &&
		       memcmp(launches, untouched, sizeof(launches)) == 0,
	       what);
	expect_one_error_line_with(logged, words, what);
}

static void expect_launch(const struct kernel_launch *launch, uint32_t kernel,
			  uint32_t grid_columns, uint32_t block_threads,
			  const void *record, uint32_t record_byte_count,
			  const char *what)
{
	unsigned char behind[LAUNCH_ARGUMENT_BYTES];
	memset(behind, 0, sizeof(behind));
	expect(launch->kernel == kernel &&
		       launch->grid[GRID_COLUMNS] == grid_columns &&
		       launch->grid[GRID_ROWS] == ROWS &&
		       launch->grid[GRID_DEPTH] == 1 &&
		       launch->block[GRID_COLUMNS] == block_threads &&
		       launch->block[GRID_ROWS] == 1 &&
		       launch->block[GRID_DEPTH] == 1 &&
		       launch->dynamic_shared_byte_count == 0 &&
		       launch->branch == LAUNCH_CHAIN &&
		       launch->argument_byte_count == record_byte_count &&
		       memcmp(launch->argument, record, record_byte_count) ==
			       0 &&
		       memcmp(launch->argument + record_byte_count, behind,
			      LAUNCH_ARGUMENT_BYTES - record_byte_count) == 0,
	       what);
}

static void test_accepted(void)
{
	struct kernel_launch launches[HYPER_SITE_LAUNCH_COUNT];
	for (uint32_t combined = 0; combined <= 1; combined++) {
		const struct hyper_site_description site =
			site_of(combined != 0, ROWS);
		context_text = combined != 0 ? "an FFN site" :
					       "an attention site";
		expect(hyper_site_prepare(&site, ARENA_BYTES, &made_up_kernels,
					  launches) == STATUS_OK,
		       "accepted");
		const struct hyper_finish_arguments finish = {
			site.streams.byte_offset, site.carried_pre.byte_offset,
			site.mixes.byte_offset,	  site.scale.byte_offset,
			site.base.byte_offset,	  site.post.byte_offset,
			site.comb.byte_offset,	  site.norm_weight.byte_offset,
			site.hidden.byte_offset
		};
		const struct hyper_post_arguments post = {
			site.streams.byte_offset,
			site.post.byte_offset,
			site.comb.byte_offset,
			site.output.byte_offset,
			site.routed_sums.byte_offset,
			combined,
			0
		};
		expect_launch(&launches[HYPER_SITE_MIX_FINISH],
			      made_up_kernels.mix_finish, 1,
			      HYPER_FINISH_THREADS, &finish, sizeof(finish),
			      "the mix finish");
		expect_launch(&launches[HYPER_SITE_POST], made_up_kernels.post,
			      HYPER_PIECES, HYPER_PIECE_THREADS, &post,
			      sizeof(post),
			      "the post, with the combine on the FFN site");
	}
    context_text = "the embedding";
    const struct hyper_embed_description embed = embed_of(ROWS);
    expect(hyper_embed_prepare(&embed, ARENA_BYTES, &made_up_kernels,
			       launches) == STATUS_OK,
	   "accepted");
    const struct hyper_embed_arguments embed_record = {
	    embed.step.byte_offset,
	    embed.embedding.byte_offset,
	    embed.streams.byte_offset,
	    embed.carried_pre.byte_offset,
	    VOCABULARY,
	    0
    };
    expect_launch(launches, made_up_kernels.embed, HYPER_PIECES,
		  HYPER_PIECE_THREADS, &embed_record, sizeof(embed_record),
		  "its launch");
    context_text = "the final collapse";
    const struct hyper_final_description final = final_of(ROWS);
    expect(hyper_final_prepare(&final, ARENA_BYTES, &made_up_kernels,
			       launches) == STATUS_OK,
	   "accepted");
    const struct hyper_finish_arguments final_record = {
	    .streams_offset = final.streams.byte_offset,
	    .pre_offset = final.carried_pre.byte_offset,
	    .weight_offset = final.norm_weight.byte_offset,
	    .hidden_offset = final.hidden.byte_offset
    };
    expect_launch(launches, made_up_kernels.final_collapse, 1,
		  HYPER_ROW_THREADS, &final_record, sizeof(final_record),
		  "its launch");
}

/* A chunk's rows (a wide step's), a site of each kind and the embedding: accepted, every launch over the rows on
 * grid[1]. The final collapse takes a step's rows alone: 8 accepted, 9 refused. */
static void test_chunk_rows(void)
{
	struct kernel_launch launches[HYPER_SITE_LAUNCH_COUNT];
	const uint64_t rows = MODEL_WIDE_MOST_ROWS;
	for (uint32_t combined = 0; combined <= 1; combined++) {
		const struct hyper_site_description site =
			site_of(combined != 0, rows);
		context_text = combined != 0 ?
				       "an FFN site of a chunk's rows" :
				       "an attention site of a chunk's rows";
		expect(hyper_site_prepare(&site, ARENA_BYTES, &made_up_kernels,
					  launches) == STATUS_OK &&
			       launches[HYPER_SITE_MIX_FINISH].grid[GRID_ROWS] ==
				       rows &&
			       launches[HYPER_SITE_POST].grid[GRID_ROWS] ==
				       rows,
		       "accepted, both launches over the rows");
	}
    context_text = "the embedding of a chunk's rows";
    const struct hyper_embed_description embed = embed_of(rows);
    expect(hyper_embed_prepare(&embed, ARENA_BYTES, &made_up_kernels,
			       &launches[0]) == STATUS_OK &&
		   launches[0].grid[GRID_ROWS] == rows,
	   "accepted, its launch over the rows");
    context_text = "the final collapse of a step's rows";
    const struct hyper_final_description final = final_of(LINEAR_MAXIMUM_ROWS);
    const struct described more = {
	    .kind = FINAL, .final = final_of(LINEAR_MAXIMUM_ROWS + 1u)
    };
    expect(hyper_final_prepare(&final, ARENA_BYTES, &made_up_kernels,
			       &launches[0]) == STATUS_OK &&
		   launches[0].grid[GRID_ROWS] == LINEAR_MAXIMUM_ROWS,
	   "accepted, its launch over the rows");
    expect_refused(&more, ARENA_BYTES, &made_up_kernels, RULE_ROWS, "one row more than a step's refused");
}

/* Every rule broken once, on each kind of description (the FFN site's ranges include the routed sums). */
static void test_refused(void)
{
    static const char *const kind_names[PREPARED_COUNT] = {"an attention site", "an FFN site", "the embedding",
                                                           "the final collapse"};
    for (uint32_t kind = 0; kind < PREPARED_COUNT; kind++) {
        struct described valid = {.kind = (enum prepared)kind, .site = site_of(kind == FFN_SITE, ROWS),
                                  .embed = embed_of(ROWS), .final = final_of(ROWS)};
	struct linear_range *ranges[SITE_RANGES];
	bool is_exact[SITE_RANGES];
	struct kernel_launch launches[HYPER_SITE_LAUNCH_COUNT];
	const uint32_t range_count = ranges_of(&valid, ranges, is_exact);
        const uint32_t launched[PREPARED_COUNT][LAUNCHED_KERNELS] = {
            {made_up_kernels.mix_finish, made_up_kernels.post},
            {made_up_kernels.mix_finish, made_up_kernels.post},
            {made_up_kernels.embed, made_up_kernels.embed},
            {made_up_kernels.final_collapse, made_up_kernels.final_collapse}};
        context_text = kind_names[kind];
	expect_refused(&valid, LINEAR_OFFSET_ALIGNMENT_BYTES, &made_up_kernels,
		       RULE_ARENA, "an arena too small");
	for (uint32_t kernel = 0; kernel < KERNEL_COUNT; kernel++) {
		struct hyper_kernels_record missing = made_up_kernels;
		uint32_t handles[ALL_HANDLES];
		memcpy(handles, &missing, sizeof(handles));
		const bool is_launched = handles[kernel] == launched[kind][0] ||
					 handles[kernel] == launched[kind][1];
		handles[kernel] = BACKEND_NO_KERNEL;
		memcpy(&missing, handles, sizeof(missing));
		if (is_launched) {
			expect_refused(&valid, ARENA_BYTES, &missing,
				       RULE_NOT_PREPARED,
				       "a kernel it launches not prepared");
		} else {
			expect(prepare(&valid, ARENA_BYTES, &missing,
				       launches) == STATUS_OK,
			       "a kernel it does not launch may be missing");
		}
        }
        for (uint32_t row_count = 0; row_count <= TOO_MANY_ROWS; row_count += TOO_MANY_ROWS) {
            struct described broken = valid;
            broken.site.row_count = broken.embed.row_count = broken.final.row_count = row_count;
            expect_refused(&broken, ARENA_BYTES, &made_up_kernels, RULE_ROWS, "no rows, or too many");
        }
        struct described broken = valid;
        broken.site.reserved_zero = broken.final.reserved_zero = 1;
        broken.embed.vocabulary_size = 0;
        expect_refused(&broken, ARENA_BYTES, &made_up_kernels, kind == EMBED ? RULE_VOCABULARY : RULE_RESERVED,
                       "a reserved field not zero, or no vocabulary");
        for (uint32_t index = 0; index < range_count; index++) {
		struct linear_range *broken_ranges[SITE_RANGES];
		bool unused[SITE_RANGES];
		broken = valid;
		(void)ranges_of(&broken, broken_ranges, unused);
		broken_ranges[index]->byte_count -= sizeof(float);
		expect_refused(&broken, ARENA_BYTES, &made_up_kernels,
			       is_exact[index] ? RULE_EXACT : RULE_TOUCHED,
			       "a range shorter than its bytes");
		broken = valid;
		(void)ranges_of(&broken, broken_ranges, unused);
		broken_ranges[index]->byte_offset += MISALIGNMENT;
		expect_refused(&broken, ARENA_BYTES, &made_up_kernels,
			       RULE_ALIGNED, "a range off the alignment");
		broken = valid;
		(void)ranges_of(&broken, broken_ranges, unused);
		broken_ranges[index]->byte_offset = ARENA_BYTES;
		expect_refused(&broken, ARENA_BYTES, &made_up_kernels,
			       RULE_ARENA, "a range out of the arena");
		broken = valid;
		(void)ranges_of(&broken, broken_ranges, unused);
		broken_ranges[index]->byte_offset =
			broken_ranges[(index + 1u) % range_count]->byte_offset;
		expect_refused(&broken, ARENA_BYTES, &made_up_kernels,
			       RULE_SHARED, "a range on another");
        }
        expect_refused(&valid, ARENA_BYTES, NULL, RULE_MISSING, "no kernels");
    }
    context_text = "an FFN site";
    struct described empty_sums = {.kind = FFN_SITE, .site = site_of(true, ROWS)};
    empty_sums.site.routed_sums.byte_count = 0;
    expect_refused(&empty_sums, ARENA_BYTES, &made_up_kernels, RULE_TOUCHED, "routed sums of no bytes");
}

/* ---- the program's modules hold the kernels; the host-memory backend takes every launch ---- */

/* Whether a module holds name as a symbol: its bytes and the NUL that ends them, behind no byte of a name. */
static bool module_holds(const struct cuda_kernel_image *image,
			 const char *name)
{
    const size_t byte_count = strlen(name) + 1u;
    const unsigned char *const end = image->bytes + image->byte_count;
    for (const unsigned char *place = image->bytes; place < end; place++) {
        place = memmem(place, (size_t)(end - place), name, byte_count);
        if (place == NULL) {
            return false;
        }
        const unsigned char before = place == image->bytes ? (unsigned char)'\0' : place[-1];
        if (!(before == '_' || (before >= '0' && before <= '9') || (before >= 'A' && before <= 'Z') ||
              (before >= 'a' && before <= 'z'))) {
            return true;
        }
    }
    return false;
}

static void test_kernel_names(void)
{
    uint32_t architecture_count = 0;
    uint32_t module_count = 0;
    context_text = "the modules of kernel_hyper_connection in the program";
    for (uint32_t index = 0; index < cuda_kernel_image_count; index++) {
	    const struct cuda_kernel_image *const image =
		    &cuda_kernel_images[index];
	    bool is_new_architecture = true;
	    for (uint32_t before = 0; before < index; before++) {
		    is_new_architecture =
			    is_new_architecture &&
			    strcmp(cuda_kernel_images[before].architecture,
				   image->architecture) != 0;
	    }
	    architecture_count += is_new_architecture ? 1u : 0u;
	    if (strcmp(image->kernel_name, KERNEL_FILE_HYPER_CONNECTION) != 0) {
		    continue;
	    }
	    module_count++;
	    for (uint32_t kernel_index = 0; kernel_index < KERNEL_COUNT;
		 kernel_index++) {
		    expect(image->bytes != NULL &&
				   module_holds(image,
						kernels_expected[kernel_index]
							.function_name),
			   "the module holds each function the kernels are prepared by as a symbol");
	    }
    }
    expect(module_count != 0 && module_count == architecture_count, "a module for each architecture");
}

static void test_launches_run(void)
{
	static const uint32_t classes[] = { ARITHMETIC_CLASS_SM_80,
					    ARITHMETIC_CLASS_SM_120 };
	for (uint32_t class_index = 0; class_index < COUNT_OF(classes);
	     class_index++) {
		struct device_card *card = NULL;
		struct hyper_kernels_record kernels;
		struct kernel_launch launches[HYPER_SITE_LAUNCH_COUNT];
		context_text = class_index == 0 ?
				       "a card of class 80 in host memory" :
				       "a card of class 120 in host memory";
		test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
		test_host_memory_backend_set_arithmetic_class(
			0, classes[class_index]);
		expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES,
							  &card) == STATUS_OK &&
			       hyper_kernels_prepare(&test_host_memory_backend,
						     card,
						     &kernels) == STATUS_OK,
		       "the four kernels are prepared on a card of the class");
		for (uint32_t combined = 0; combined <= 1; combined++) {
			const struct hyper_site_description site =
				site_of(combined != 0, ROWS);
			expect(hyper_site_prepare(&site, ARENA_BYTES, &kernels,
						  launches) == STATUS_OK &&
				       test_host_memory_backend.launch_run(
					       card, launches,
					       HYPER_SITE_LAUNCH_COUNT) ==
					       STATUS_OK,
			       "a card takes the launches of a site");
		}
		const struct hyper_embed_description embed = embed_of(ROWS);
		const struct hyper_final_description final = final_of(ROWS);
		expect(hyper_embed_prepare(&embed, ARENA_BYTES, &kernels,
					   &launches[0]) == STATUS_OK &&
			       hyper_final_prepare(&final, ARENA_BYTES,
						   &kernels,
						   &launches[1]) == STATUS_OK &&
			       test_host_memory_backend.launch_run(
				       card, launches,
				       HYPER_SITE_LAUNCH_COUNT) == STATUS_OK,
		       "a card takes the launches of the embedding and the final collapse");
		test_host_memory_backend.card_close(card, 0);
	}
}

/* ---- the reference ---- */

static double random_fraction(uint64_t position)
{
    return (double)test_linear_hash(RANDOM_STREAM, position) * HASH_UNIT - (double)HALF;
}

/* Whether every row and every column of comb adds up to one within tolerance. */
static bool is_doubly_stochastic(const float *comb, double tolerance)
{
    bool holds = true;
    for (uint32_t line = 0; line < STREAMS; line++) {
        double row_sum = 0.0;
        double column_sum = 0.0;
        for (uint32_t place = 0; place < STREAMS; place++) {
            row_sum += (double)comb[line * STREAMS + place];
            column_sum += (double)comb[place * STREAMS + line];
        }
        holds = holds && row_sum > 1.0 - tolerance && row_sum < 1.0 + tolerance && column_sum > 1.0 - tolerance &&
                column_sum < 1.0 + tolerance;
    }
    return holds;
}

static void test_reference(void)
{
	static float mixes[MODEL_MIXES];
	static float scale[HYPER_SCALES];
	static float base[MODEL_MIXES];
	static uint16_t streams[STREAMS * HIDDEN];
	static uint16_t collapsed[HIDDEN];
	static uint16_t output[HIDDEN];
	static uint16_t weight[HIDDEN];
	static uint16_t updated[STREAMS * HIDDEN];
	static float sums[HIDDEN];
	static double normed[HIDDEN];
	float pre[STREAMS];
	float post[STREAMS];
	float comb[COMB_VALUES];
	bool holds = true;
	context_text = "the reference's Sinkhorn";
	test_hyper_sinkhorn(mixes, scale, base, pre, post, comb);
	for (uint32_t element = 0; element < COMB_VALUES; element++) {
		holds = holds && comb[element] == comb[0] &&
			(element >= STREAMS ||
			 (pre[element] == HALF + MODEL_HYPER_EPSILON &&
			  post[element] == 1.0f));
	}
    expect(holds && is_doubly_stochastic(comb, STOCHASTIC_TOLERANCE),
           "of zero mixes: pre 1/2 + eps, post 1, comb all equal, rows and columns adding to one");
    for (uint32_t mix = 0; mix < MODEL_MIXES; mix++) {
	    mixes[mix] = (float)(random_fraction(mix) * MIX_SPREAD);
	    base[mix] =
		    (float)(random_fraction(MODEL_MIXES + mix) * MIX_SPREAD);
    }
    for (uint32_t scale_index = 0; scale_index < HYPER_SCALES; scale_index++) {
	    scale[scale_index] =
		    (float)(random_fraction(SCALE_POSITIONS + scale_index) *
			    MIX_SPREAD);
    }
    test_hyper_sinkhorn(mixes, scale, base, pre, post, comb);
    holds = true;
    for (uint32_t stream = 0; stream < STREAMS; stream++) {
	    holds = holds && pre[stream] > 0.0f &&
		    pre[stream] <= 1.0f + MODEL_HYPER_EPSILON &&
		    post[stream] >= 0.0f && post[stream] <= KNOWN_WEIGHT;
    }
    expect(holds && is_doubly_stochastic(comb, STOCHASTIC_TOLERANCE),
           "of random mixes: pre and post in their ranges, comb doubly stochastic");

    context_text = "the reference's collapse, norm and post";
    test_linear_values(streams, STREAMS * HIDDEN, RANDOM_STREAM, TEST_LINEAR_VALUES_PLAIN);
    const float first_only[STREAMS] = {1.0f, 0.0f, 0.0f, 0.0f};
    test_hyper_collapse(streams, first_only, collapsed);
    expect(memcmp(collapsed, streams, sizeof(collapsed)) == 0, "a collapse with pre (1, 0, 0, 0) is stream 0");
    for (uint32_t dimension = 0; dimension < HIDDEN; dimension++) {
        collapsed[dimension] = test_linear_bfloat16_nearest(1.0);
        weight[dimension] = test_linear_bfloat16_nearest((double)KNOWN_WEIGHT);
        output[dimension] = test_linear_bfloat16_nearest((double)QUARTER);
        sums[dimension] = KNOWN_ROUTED;
        for (uint32_t stream = 0; stream < STREAMS; stream++) {
            streams[stream * HIDDEN + dimension] = test_linear_bfloat16_nearest((double)(stream + 1u));
        }
    }
    test_hyper_norm(collapsed, weight, normed);
    holds = true;
    for (uint32_t dimension = 0; dimension < HIDDEN; dimension++) {
        holds = holds && normed[dimension] > (double)KNOWN_WEIGHT - NORM_TOLERANCE &&
                normed[dimension] < (double)KNOWN_WEIGHT + NORM_TOLERANCE;
    }
    expect(holds, "the norm of ones, weighted by 2, is 2");
    const float no_post[STREAMS] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float unit_post[STREAMS] = {1.0f, 1.0f, 1.0f, 1.0f};
    float quarters[COMB_VALUES];
    float identity[COMB_VALUES];
    for (uint32_t element = 0; element < COMB_VALUES; element++) {
        quarters[element] = QUARTER;
        identity[element] = element / STREAMS == element % STREAMS ? 1.0f : 0.0f;
    }
    test_hyper_post(streams, no_post, quarters, output, NULL, updated);
    holds = true;
    for (uint32_t value = 0; value < STREAMS * HIDDEN; value++) {
        holds = holds && updated[value] == test_linear_bfloat16_nearest((double)KNOWN_SUM);
    }
    expect(holds, "a post of comb all a quarter and post 0 makes every stream their mean");
    test_hyper_post(streams, unit_post, identity, output, sums, updated);
    holds = true;
    for (uint32_t value = 0; value < STREAMS * HIDDEN; value++) {
        const uint32_t stream_number = value / HIDDEN + 1u;
        const double expected = (double)stream_number + (double)KNOWN_ROUTED + (double)QUARTER;
        holds = holds && updated[value] == test_linear_bfloat16_nearest(expected);
    }
    expect(holds, "the combine adds the routed sums to the shared expert's output, and an identity comb keeps each "
                  "stream");
    float one = 1.0f;
    float next = 0.0f;
    const uint32_t next_bits = FLOAT32_ONE_BITS + 1u;
    memcpy(&next, &next_bits, sizeof(next));
    expect(test_hyper_float_places(one, next) == 1 && test_hyper_float_places(-0.0f, 0.0f) == 0 &&
               test_hyper_float_places(one, -one) == ACROSS_ZERO_PLACES,
           "float32 places: one apart for neighbours, none between the zeros, across zero both ways");
}

/* ---- the draft's inputs and the final's kept collapse ---- */

static struct hyper_draft_inputs_description
draft_inputs_of(uint32_t mode, uint32_t tap, uint32_t row_count)
{
	const bool is_mean = mode == HYPER_STREAM_MEAN;
	uint64_t cursor = ALIGNMENT;
	struct hyper_draft_inputs_description inputs = { .row_count = row_count,
							 .mode = mode,
							 .tap = tap };
	inputs.input =
		next_range(&cursor, (uint64_t)row_count *
					    (is_mean ? MODEL_STREAMS_ROW_BYTES :
						       MODEL_HIDDEN_ROW_BYTES));
	inputs.output = next_range(
		&cursor,
		(uint64_t)row_count * (is_mean ? MODEL_DRAFT_TAP_ROW_BYTES :
						 MODEL_HIDDEN_ROW_BYTES));
	if (!is_mean) {
		inputs.weight = next_range(&cursor, MODEL_HIDDEN_ROW_BYTES);
	}
    return inputs;
}

static void
expect_inputs_refused(const struct hyper_draft_inputs_description *inputs,
		      const struct hyper_kernels_record *kernels,
		      const char *words, const char *what)
{
	struct kernel_launch launch;
	struct kernel_launch untouched;
	char logged[CAPTURE_CAPACITY];
	memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
	memset(&untouched, UNTOUCHED_BYTE, sizeof(untouched));
	capture_begin();
	const result_status status = hyper_draft_inputs_prepare(
		inputs, ARENA_BYTES, kernels, &launch);
	capture_end(logged, sizeof(logged));
	expect(status == STATUS_INVALID_ARGUMENT &&
		       memcmp(&launch, &untouched, sizeof(launch)) == 0,
	       what);
	expect_one_error_line_with(logged, words, what);
}

static void test_draft_inputs(void)
{
	static const uint32_t row_counts[] = { 1u, LINEAR_MAXIMUM_ROWS,
					       WIDE_ROWS };
	unsigned char behind[LAUNCH_ARGUMENT_BYTES];
	memset(behind, 0, sizeof(behind));
	context_text = "the draft's inputs, accepted";
	for (uint32_t mode = 0; mode <= HYPER_ROW_NORM; mode++) {
		for (uint32_t tap = 0;
		     tap < (mode == HYPER_STREAM_MEAN ? MODEL_DRAFT_TAPS : 1u);
		     tap++) {
			for (size_t index = 0; index < COUNT_OF(row_counts);
			     index++) {
				const struct hyper_draft_inputs_description
					inputs = draft_inputs_of(
						mode, tap, row_counts[index]);
				struct kernel_launch launch;
				const struct hyper_draft_inputs_arguments
					record = { inputs.input.byte_offset,
						   inputs.weight.byte_offset,
						   inputs.output.byte_offset,
						   mode == HYPER_STREAM_MEAN ?
							   MODEL_DRAFT_FEATURES :
							   MODEL_HIDDEN,
						   tap * MODEL_HIDDEN,
						   mode,
						   0 };
				expect(hyper_draft_inputs_prepare(
					       &inputs, ARENA_BYTES,
					       &made_up_kernels,
					       &launch) == STATUS_OK &&
					       launch.kernel ==
						       made_up_kernels
							       .draft_inputs &&
					       launch.grid[GRID_COLUMNS] == 1 &&
					       launch.grid[GRID_ROWS] ==
						       row_counts[index] &&
					       launch.grid[GRID_DEPTH] == 1 &&
					       launch.block[GRID_COLUMNS] ==
						       HYPER_ROW_THREADS &&
					       launch.block[GRID_ROWS] == 1 &&
					       launch.block[GRID_DEPTH] == 1 &&
					       launch.dynamic_shared_byte_count ==
						       0 &&
					       launch.argument_byte_count ==
						       sizeof(record) &&
					       memcmp(launch.argument, &record,
						      sizeof(record)) == 0 &&
					       memcmp(launch.argument +
							      sizeof(record),
						      behind,
						      LAUNCH_ARGUMENT_BYTES -
							      sizeof(record)) ==
						       0,
				       "a launch of grid (1, rows) of a row's threads, the record of the mode and the tap");
			}
		}
	}
    context_text = "the draft's inputs, refused";
    for (uint32_t mode = 0; mode <= HYPER_ROW_NORM; mode++) {
	    const struct hyper_draft_inputs_description valid = draft_inputs_of(
		    mode,
		    mode == HYPER_STREAM_MEAN ? MODEL_DRAFT_TAPS - 1u : 0u,
		    ROWS);
	    struct hyper_kernels_record missing = made_up_kernels;
	    missing.draft_inputs = BACKEND_NO_KERNEL;
	    expect_inputs_refused(&valid, &missing, RULE_NOT_PREPARED,
				  "the kernel not prepared");
	    expect_inputs_refused(&valid, NULL, RULE_MISSING, "no kernels");
	    struct hyper_draft_inputs_description broken = valid;
	    broken.reserved_zero = 1;
	    expect_inputs_refused(&broken, &made_up_kernels, RULE_RESERVED,
				  "a reserved field not zero");
	    for (uint32_t row_count = 0; row_count <= WIDE_ROWS + 1u;
		 row_count += WIDE_ROWS + 1u) {
		    broken = valid;
		    broken.row_count = row_count;
		    expect_inputs_refused(&broken, &made_up_kernels,
					  RULE_DRAFT_COUNTS,
					  "no rows, or more than a chunk's");
	    }
	    broken = valid;
	    broken.mode = HYPER_ROW_NORM + 1u;
	    expect_inputs_refused(&broken, &made_up_kernels, RULE_DRAFT_COUNTS,
				  "a mode outside the enumeration");
	    broken = valid;
	    broken.tap = mode == HYPER_STREAM_MEAN ? MODEL_DRAFT_TAPS : 1u;
	    expect_inputs_refused(&broken, &made_up_kernels, RULE_DRAFT_COUNTS,
				  "a tap past the taps, or the norm's not 0");
	    struct linear_range *ranges[DRAFT_INPUT_RANGES];
	    const uint32_t range_count = mode == HYPER_STREAM_MEAN ?
						 DRAFT_INPUT_RANGES - 1u :
						 DRAFT_INPUT_RANGES;
	    for (uint32_t index = 0; index < range_count; index++) {
		    const bool is_weight = index == DRAFT_INPUT_RANGES - 1u;
		    broken = valid;
		    ranges[0] = &broken.input;
		    ranges[1] = &broken.output;
		    ranges[DRAFT_INPUT_RANGES - 1u] = &broken.weight;
		    ranges[index]->byte_count -= sizeof(float);
		    expect_inputs_refused(&broken, &made_up_kernels,
					  is_weight ? RULE_EXACT : RULE_TOUCHED,
					  "a range shorter than its bytes");
		    broken = valid;
		    ranges[index]->byte_offset += MISALIGNMENT;
		    expect_inputs_refused(&broken, &made_up_kernels,
					  RULE_ALIGNED,
					  "a range off the alignment");
		    broken = valid;
		    ranges[index]->byte_offset = ARENA_BYTES;
		    expect_inputs_refused(&broken, &made_up_kernels, RULE_ARENA,
					  "a range out of the arena");
		    broken = valid;
		    ranges[index]->byte_offset =
			    ranges[(index + 1u) % range_count]->byte_offset;
		    expect_inputs_refused(&broken, &made_up_kernels,
					  RULE_SHARED, "a range on another");
	    }
	    if (mode == HYPER_STREAM_MEAN) {
		    broken = valid;
		    broken.weight =
			    draft_inputs_of(HYPER_ROW_NORM, 0, ROWS).weight;
		    expect_inputs_refused(&broken, &made_up_kernels,
					  RULE_DRAFT_COUNTS,
					  "a weight given for the mean");
	    }
    }
    context_text = "the final's kept collapse";
    struct described kept = {.kind = FINAL, .final = final_of(ROWS)};
    uint64_t cursor = kept.final.hidden.byte_offset + kept.final.hidden.byte_count + ALIGNMENT;
    kept.final.collapse = next_range(&cursor, ROWS * MODEL_HIDDEN_ROW_BYTES);
    struct kernel_launch launch;
    const struct hyper_finish_arguments kept_record = {
	    .streams_offset = kept.final.streams.byte_offset,
	    .pre_offset = kept.final.carried_pre.byte_offset,
	    .comb_offset = kept.final.collapse.byte_offset,
	    .weight_offset = kept.final.norm_weight.byte_offset,
	    .hidden_offset = kept.final.hidden.byte_offset
    };
    expect(hyper_final_prepare(&kept.final, ARENA_BYTES, &made_up_kernels,
			       &launch) == STATUS_OK,
	   "accepted");
    expect_launch(
	    &launch, made_up_kernels.final_kept, 1, HYPER_ROW_THREADS,
	    &kept_record, sizeof(kept_record),
	    "the draft's final, its record writes the collapse at comb_offset");
    struct hyper_kernels_record missing = made_up_kernels;
    missing.final_kept = BACKEND_NO_KERNEL;
    expect_refused(&kept, ARENA_BYTES, &missing, RULE_NOT_PREPARED, "the draft's final not prepared");
    const struct described plain = {.kind = FINAL, .final = final_of(ROWS)};
    expect(hyper_final_prepare(&plain.final, ARENA_BYTES, &missing, &launch) ==
			   STATUS_OK &&
		   launch.kernel == made_up_kernels.final_collapse,
	   "without a collapse, the backbone's final, the draft's not needed");
    struct described broken_kept = kept;
    broken_kept.final.collapse.byte_count -= sizeof(float);
    expect_refused(&broken_kept, ARENA_BYTES, &made_up_kernels, RULE_TOUCHED, "a collapse too short");
    broken_kept = kept;
    broken_kept.final.collapse.byte_offset = kept.final.hidden.byte_offset;
    expect_refused(&broken_kept, ARENA_BYTES, &made_up_kernels, RULE_SHARED, "a collapse on the head's input");
    broken_kept = (struct described){.kind = FINAL, .final = final_of(ROWS)};
    broken_kept.final.streams.byte_offset += kept.final.collapse.byte_count + ALIGNMENT;
    broken_kept.final.carried_pre.byte_offset += kept.final.collapse.byte_count + ALIGNMENT;
    broken_kept.final.norm_weight.byte_offset += kept.final.collapse.byte_count + ALIGNMENT;
    broken_kept.final.hidden.byte_offset += kept.final.collapse.byte_count + ALIGNMENT;
    broken_kept.final.collapse =
	    (struct linear_range){ 0, kept.final.collapse.byte_count };
    expect_refused(&broken_kept, ARENA_BYTES, &made_up_kernels, RULE_COLLAPSE_AT_ZERO, "a collapse at byte 0");
}

static void test_tap_reference(void)
{
    static uint16_t streams[STREAMS * HIDDEN];
    static uint16_t tap[HIDDEN];
    const float quarters[STREAMS] = { HYPER_STREAM_MEAN_PRE,
				      HYPER_STREAM_MEAN_PRE,
				      HYPER_STREAM_MEAN_PRE,
				      HYPER_STREAM_MEAN_PRE };
    bool holds = true;
    context_text = "the reference of a tap";
    for (uint32_t dimension = 0; dimension < HIDDEN; dimension++) {
        for (uint32_t stream = 0; stream < STREAMS; stream++) {
            streams[stream * HIDDEN + dimension] = test_linear_bfloat16_nearest((double)(stream + 1u));
        }
    }
    test_hyper_collapse(streams, quarters, tap);
    for (uint32_t dimension = 0; dimension < HIDDEN; dimension++) {
        holds = holds && tap[dimension] == test_linear_bfloat16_nearest((double)KNOWN_SUM);
    }
    expect(holds, "the mean of streams 1 to 4 is 2.5");
    test_linear_values(streams, HIDDEN, RANDOM_STREAM, TEST_LINEAR_VALUES_PLAIN);
    for (uint32_t stream = 1; stream < STREAMS; stream++) {
        memcpy(&streams[(size_t)stream * HIDDEN], streams, HIDDEN * sizeof(uint16_t));
    }
    test_hyper_collapse(streams, quarters, tap);
    expect(memcmp(tap, streams, sizeof(tap)) == 0, "the mean of four equal streams is that stream: the products by "
                                                    "1/4 and their sums are exact");
}

int main(void)
{
    uint32_t handles[ALL_HANDLES];
    error_stream_descriptor = dup(STDERR_FILENO);
    capture_descriptor = mkstemp(capture_path);
    if (error_stream_descriptor < 0 || capture_descriptor < 0) {
        (void)printf("FAILED: the temporary file that stands in for the error stream could not be made\n");
        return EXIT_FAILURE;
    }
    for (uint32_t kernel_index = 0; kernel_index < ALL_HANDLES; kernel_index++) {
        handles[kernel_index] = FIRST_MADE_UP_HANDLE + kernel_index;
    }
    memcpy(&made_up_kernels, handles, sizeof(made_up_kernels));
    test_kernels_prepare();
    test_kernel_names();
    test_accepted();
    test_chunk_rows();
    test_refused();
    test_launches_run();
    test_reference();
    test_draft_inputs();
    test_tap_reference();
    context_text = "the end of the test";
    expect(close(capture_descriptor) == 0 && unlink(capture_path) == 0 && close(error_stream_descriptor) == 0 &&
               test_host_memory_backend_open_card_count() == 0,
           "the temporary file is removed and no card is left open");
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("hyper_connection_test passed\n");
    return EXIT_SUCCESS;
}
