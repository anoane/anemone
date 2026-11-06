/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * index_scan_test: what source/index_scan.c prepares, and the host reference of its kernels (index_scan_reference.h).
 * No card: a prepare function is arithmetic on a description, the arena is a number, the handles are made up.
 *
 *   - index_scan_kernels_prepare against a table that records what it is asked: all kernels, in the
 *     order of the members, of kernel_index_scan with their record sizes; a missing argument, a table without
 *     kernel_prepare and a kernel_prepare that fails, each with its status, its error line and the handles untouched.
 *   - index_scan_prepare at each ratio, at 1 and 8 rows, at the smallest context, 16,384 tokens and the
 *     most, and as a candidate source and reader: its launches (the candidates' for a source alone), every byte, as
 *     index_scan_records.h states them, worked out here by hand, and their count.
 *   - Each refusal of index_scan.h and prepare_common.h: INVALID_ARGUMENT, one error line that names the rule, the
 *     launches untouched. Every range once off the alignment, a byte short, leaving the arena; two ranges that share
 *     a byte; an arena a byte short of the last range.
 *   - The reference: every FP16 pattern back from its value; ties to even, the largest numbers and the subnormals by
 *     hand; order keys against the order of the values; the lane orders place every value once, and a few values
 *     at the bytes model_layout.h's formulas give them, worked out by hand; a score in each sum, a float32 tree that
 *     loses the 1 of 2^24 + 1 where double and the heads in order keep it, and two selections by hand, ties to the
 *     lowest position (and to the highest with the break).
 * A failure is one line on the standard output: the case, and what was expected of it.
 */
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "log_expectation.h"
#include "anemone/index_scan.h"
#include "anemone/index_scan_records.h"
#include "anemone/model_layout.h"
#include "anemone/status.h"
#include "anemone/step_records.h"
#include "host_memory_backend.h"
#include "index_scan_reference.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define FIRST_RECORDED_HANDLE 40u
#define CLASS_80 ARITHMETIC_CLASS_SM_80 /* the class of the tables' card */
#define SCAN_HANDLE 7u
#define SELECT_HANDLE 9u
#define CANDIDATES_HANDLE 11u
#define SPLIT_FIRST_HANDLE 20u
#define SPLIT_CUT_HIGH_HANDLE 21u
#define SPLIT_HISTOGRAM_LOW_HANDLE 22u
#define SPLIT_CUT_LOW_HANDLE 23u
#define SPLIT_OUTPUT_HANDLE 24u
#define DEEP_SPLIT_MIDDLE_ROWS 16u
#define WIDE_LEGACY_ROWS 64u
#define CARD_MEMORY_BYTES 4096u
#define UNTOUCHED_BYTE 0xA5
#define CAPTURE_CAPACITY 2048u
#define ALIGNMENT LINEAR_OFFSET_ALIGNMENT_BYTES
#define SMALLEST_CONTEXT MODEL_CONTEXT_GRANULE
#define LARGEST_CONTEXT MODEL_CONTEXT_WITHOUT_CANDIDATES
#define SESSIONS 3u
#define SELECTED MODEL_SELECTED
#define HEADS MODEL_INDEX_HEADS
#define DIMS MODEL_INDEX_HEAD_DIMS
#define HALF_PATTERNS 65536u
#define HALF_NOT_A_NUMBER 0x7E00u
#define HALF_INFINITY 0x7C00u
#define HALF_LARGEST 0x7BFFu                    /* 65,504 */
#define HALF_ONE 0x3C00u
#define HALF_ONE_AND_TWO_PLACES 0x3C02u
#define HALF_TWO_SMALLEST 0x0002u               /* 2^-23 */
#define HALF_SIGN 0x8000u
#define HALF_PLACE_AT_ONE (-10)
#define HALF_SUBNORMAL_PLACE (-24)
#define OVERFLOW_AT 65520.0
#define BELOW_OVERFLOW 65519.0
#define THREE_HALF_PLACES 3.0                   /* 1.5 places: a tie to the even of two neighbours */
#define ORDER_STRIDE_FIRST 31u                  /* patterns compared with each other: one in 31 by one in 29 */
#define ORDER_STRIDE_SECOND 29u
#define HAND_KEYS 600u                          /* of a hand selection: above 512, so the cut is made */
#define HAND_TIES_FROM 300u                     /* the rising case: positions from here share the largest score */
/* The hand score: one key dim of 3; heads 0 .. 3 meet it at 1, 2, 1 and -1 with weights 2, 2, 0.5 and -1:
 * 2 * 3 + 2 * 6 + 0.5 * 3 + (-1) * 0 (the last dot is below 0) = 19.5. */
#define HAND_KEY_VALUE 3.0
#define HAND_DOUBLE 2.0
#define HAND_SCORE 19.5
#define BFLOAT16_TWO 0x4000u
#define BFLOAT16_HALF 0x3F00u
#define BFLOAT16_ONE 0x3F80u
#define BFLOAT16_MINUS_ONE 0xBF80u
#define BFLOAT16_TWO_TO_24 0x4B80u              /* 2^24: in float32, 2^24 + 1 rounds back to it (a tie to even) */
#define BFLOAT16_MINUS_TWO_TO_24 0xCB80u
enum hand_head { HAND_PLAIN, HAND_DOUBLED, HAND_HALVED, HAND_BELOW_ZERO };
/* The tree's hand case: every dot 1; head 0 weighs 2^24 and meets head 8's 1 in v_0, head 4's -2^24 meets them last. */
enum tree_head { TREE_LARGE = 0, TREE_LARGE_NEGATIVE = 4, TREE_ONE = 8 };
/* Lane orders by hand from model_layout.h: (head or key, dim) and the byte of its FP16 value. */
struct placed_by_hand {
    uint32_t vector;
    uint32_t dim;
    uint32_t byte_offset;
};
static const struct placed_by_hand queries_by_hand[] = {
    {9, 10, 92},        /* tile 0, step 0, g 1, register 3 (head 9 = 1 + 8), t 1 (dim 10 = 2 + 8): lane 5 */
    {30, 127, 8126},    /* tile 1, step 7, g 6, register 3, t 3, the second half: 15 x 512 + 27 x 16 + 12 + 2 */
};
static const struct placed_by_hand keys_by_hand[] = {
    {3, 0, 768}, {3, 8, 772}, {3, 17, 778},     /* key 3 at 768: run 0, step 0 slots 0 and 2, step 1 slot 1 */
    {5, 42, 1364},                              /* key 5 at 1,280: run 1 at 64, step 2 at 16, slot 2 */
    {7, 127, 2046},                             /* the last half of the tile */
};

/* Words of the error line of each rule. */
#define RULE_COUNTS "1 to 4096 rows; ratio 1 or 2"
#define RULE_ALIGNED "an offset is a multiple of"
#define RULE_ARENA "leaves the arena"
#define RULE_TOUCHED "the launches touch"
#define RULE_SHARED "share a byte"
#define RULE_KERNEL "is not prepared"
#define RULE_MISSING "a missing argument"

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_index_scan_test_XXXXXX";
static int capture_descriptor = -1;             /* the temporary file that stands in for the error stream */
static int error_stream_descriptor = -1;        /* a duplicate of descriptor 2, kept to put it back */

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
static struct kernel_asked kernels_asked[INDEX_SCAN_KERNEL_COUNT];
static uint32_t asked_count = 0;
static uint32_t failing_call_number = 0;        /* counted from 1; 0: no call fails */

/* kernel_prepare of a table that records what it is asked and gives the handles FIRST_RECORDED_HANDLE and on. */
MUST_CHECK static result_status
recording_kernel_prepare(struct device_card *card, const char *kernel_file_name,
			 const char *function_name,
			 uint32_t argument_byte_count, uint32_t *kernel_out)
{
	if (card != NULL && asked_count < INDEX_SCAN_KERNEL_COUNT) {
		kernels_asked[asked_count] =
			(struct kernel_asked){ kernel_file_name, function_name,
					       argument_byte_count };
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
	static const struct backend_operations recording = {
		.name = "recording", .kernel_prepare = recording_kernel_prepare
	};
	static const struct backend_operations without_kernels = {
		.name = "no kernels"
	};
	static const char *const names[INDEX_SCAN_KERNEL_COUNT] = {
		KERNEL_INDEX_SCAN,	     KERNEL_INDEX_SELECT,
		KERNEL_INDEX_CANDIDATES,     KERNEL_INDEX_SPLIT_HISTOGRAM_HIGH,
		KERNEL_INDEX_SPLIT_CUT_HIGH, KERNEL_INDEX_SPLIT_HISTOGRAM_LOW,
		KERNEL_INDEX_SPLIT_CUT_LOW,  KERNEL_INDEX_SPLIT_OUTPUT
	};
	static const uint32_t sizes[INDEX_SCAN_KERNEL_COUNT] = {
		sizeof(struct index_scan_arguments_record),
		sizeof(struct index_select_arguments_record),
		sizeof(struct index_candidates_arguments),
		sizeof(struct index_split_arguments),
		sizeof(struct index_split_arguments),
		sizeof(struct index_split_arguments),
		sizeof(struct index_split_arguments),
		sizeof(struct index_split_arguments)
	};
	struct device_card *card = NULL;
	struct index_scan_kernels kernels;
	struct index_scan_kernels untouched;
	char logged[CAPTURE_CAPACITY];
	context_text = "the kernels of a card";
	memset(&untouched, UNTOUCHED_BYTE, sizeof(untouched));
	kernels = untouched;
	test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
	expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES,
						  &card) == STATUS_OK,
	       "a card opens");
	capture_begin();
	expect(index_scan_kernels_prepare(NULL, card, CLASS_80, &kernels) ==
			       STATUS_INVALID_ARGUMENT &&
		       index_scan_kernels_prepare(&without_kernels, card,
						  CLASS_80, &kernels) ==
			       STATUS_INVALID_ARGUMENT &&
		       index_scan_kernels_prepare(&recording, NULL, CLASS_80,
						  &kernels) ==
			       STATUS_INVALID_ARGUMENT &&
		       index_scan_kernels_prepare(&recording, card, CLASS_80,
						  NULL) ==
			       STATUS_INVALID_ARGUMENT,
	       "a missing argument or kernel_prepare is refused");
	capture_end(logged, sizeof(logged));
	expect(test_log_is_expected(logged, RULE_MISSING, false) &&
		       asked_count == 0,
	       "a refusal asks no kernel and says why");
	for (uint32_t kernel = 0; kernel < INDEX_SCAN_KERNEL_COUNT; kernel++) {
		asked_count = 0;
		failing_call_number = kernel + 1u;
		capture_begin();
		expect(index_scan_kernels_prepare(&recording, card, CLASS_80,
						  &kernels) ==
			       STATUS_DEVICE_ERROR,
		       "the status of a kernel_prepare that fails is the answer");
		capture_end(logged, sizeof(logged));
		expect(memcmp(&kernels, &untouched, sizeof(kernels)) == 0 &&
			       asked_count == failing_call_number,
		       "a failure leaves all handles untouched");
		expect_one_error_line_with(logged, names[kernel],
					   "the failure names its kernel");
	}
    failing_call_number = 0;
    asked_count = 0;
    expect(index_scan_kernels_prepare(&recording, card, CLASS_80, &kernels) ==
			   STATUS_OK &&
		   kernels.scan ==
			   FIRST_RECORDED_HANDLE + INDEX_SCAN_LAUNCH_SCAN &&
		   kernels.select ==
			   FIRST_RECORDED_HANDLE + INDEX_SCAN_LAUNCH_SELECT &&
		   kernels.candidates ==
			   FIRST_RECORDED_HANDLE + INDEX_SCAN_LAUNCH_CANDIDATES,
	   "the handles are given in the order of the members");
    for (uint32_t kernel = 0; kernel < INDEX_SCAN_KERNEL_COUNT; kernel++) {
	    expect(strcmp(kernels_asked[kernel].kernel_file_name,
			  KERNEL_FILE_INDEX_SCAN) == 0 &&
			   strcmp(kernels_asked[kernel].function_name,
				  names[kernel]) == 0 &&
			   kernels_asked[kernel].argument_byte_count ==
				   sizes[kernel],
		   "each kernel is asked of its file with the size of its record");
    }
    for (uint32_t pass = 0; pass < INDEX_SPLIT_LAUNCH_COUNT; pass++) {
	    expect(kernels.split[pass] ==
			   FIRST_RECORDED_HANDLE +
				   INDEX_SCAN_LEGACY_LAUNCH_COUNT + pass,
		   "the split handles follow the legacy handles in pass order");
    }
    asked_count = 0;
    expect(index_scan_kernels_prepare(&recording, card,
				      ARITHMETIC_CLASS_SM_120_NATIVE,
				      &kernels) == STATUS_OK &&
		   asked_count == INDEX_SCAN_KERNEL_COUNT,
	   "native preparation gets every kernel");
    for (uint32_t kernel = 0; kernel < INDEX_SCAN_KERNEL_COUNT; kernel++) {
	    expect(strcmp(kernels_asked[kernel].kernel_file_name,
			  KERNEL_FILE_INDEX_SCAN_NATIVE) == 0 &&
			   strcmp(kernels_asked[kernel].function_name,
				  names[kernel]) == 0 &&
			   kernels_asked[kernel].argument_byte_count ==
				   sizes[kernel],
		   "native preparation gets the same ABI from the native module");
    }
    test_host_memory_backend.card_close(card, 0);
}

/* ---- the launches ---- */

static const struct index_scan_kernels made_up_kernels = {
	.scan = SCAN_HANDLE,
	.select = SELECT_HANDLE,
	.candidates = CANDIDATES_HANDLE,
	.split = { SPLIT_FIRST_HANDLE, SPLIT_CUT_HIGH_HANDLE,
		   SPLIT_HISTOGRAM_LOW_HANDLE, SPLIT_CUT_LOW_HANDLE,
		   SPLIT_OUTPUT_HANDLE }
};

/* The next range of byte_count bytes from *end, which moves on to the next multiple of the alignment. */
static struct linear_range next_range(uint64_t *end, uint64_t byte_count)
{
	const struct linear_range range = { *end, byte_count };
	*end += (byte_count + ALIGNMENT - 1u) / ALIGNMENT * ALIGNMENT;
	return range;
}

/* A description every rule accepts in a candidate role, its ranges one behind another from 0; *arena_out: where
 * the last one ends. A role holds the candidates' range, a reader scores 16,384 entries a row. */
static struct index_scan_description
valid_description_of_role(uint32_t rows, uint32_t ratio, uint32_t context,
			  uint32_t role, uint64_t *arena_out)
{
    const uint32_t entries = context / ratio;
    const uint32_t row_scores = role == INDEX_CANDIDATES_READER ?
					MODEL_CONTEXT_WITHOUT_CANDIDATES :
					entries;
    const uint64_t stride =
	    (uint64_t)entries / MODEL_KEYS_PER_TILE * MODEL_TILE_BYTES;
    uint64_t end = 0;
    struct index_scan_description description = { .context_token_limit =
							  context,
						  .session_capacity = SESSIONS,
						  .ratio = ratio,
						  .row_count = rows,
						  .candidate_role = role };
    description.step =
	    next_range(&end, STEP_BLOCK_BYTES_VALUE +
				     (uint64_t)rows * STEP_ROW_BYTES_VALUE);
    description.keys = next_range(&end, SESSIONS * stride);
    description.queries =
	    next_range(&end, (uint64_t)rows * MODEL_INDEX_QUERIES_ROW_BYTES);
    description.weights =
	    next_range(&end, (uint64_t)rows * MODEL_INDEX_WEIGHTS_ROW_BYTES);
    description.scores =
	    next_range(&end, (uint64_t)rows * row_scores * INDEX_SCORE_BYTES);
    if ((rows <= INDEX_SPLIT_MAX_ROWS ||
	 (rows <= INDEX_SPLIT_DEEP_MAX_ROWS &&
	  row_scores >= INDEX_SPLIT_DEEP_MIN_ENTRIES)) &&
	row_scores >= INDEX_SPLIT_MIN_ENTRIES) {
	    (void)next_range(&end, (uint64_t)rows * INDEX_SPLIT_ROW_BYTES);
    }
    description.selection =
	    next_range(&end, (uint64_t)rows * MODEL_SELECTION_ROW_BYTES);
    *arena_out = description.selection.byte_offset + description.selection.byte_count;
    if (role != INDEX_CANDIDATES_NONE) {
	    description.candidates = next_range(
		    &end, (uint64_t)rows * MODEL_CANDIDATES_ROW_BYTES);
	    *arena_out = description.candidates.byte_offset +
			 description.candidates.byte_count;
    }
    return description;
}

/* The same of no candidate role. */
static struct index_scan_description valid_description(uint32_t rows,
						       uint32_t ratio,
						       uint32_t context,
						       uint64_t *arena_out)
{
	return valid_description_of_role(rows, ratio, context,
					 INDEX_CANDIDATES_NONE, arena_out);
}

/* The split ABI, independently spelled out: the cuts reduce one row, the other passes partition its positions. */
static void
expected_split_launches(const struct index_scan_description *description,
			bool is_block, struct kernel_launch *launches)
{
	const struct index_split_arguments record = {
		.step_offset = description->step.byte_offset,
		.scores_offset = description->scores.byte_offset,
		.output_offset = is_block ?
					 description->candidates.byte_offset :
					 description->selection.byte_offset,
		.candidates_offset = INDEX_NO_CANDIDATES,
		.scratch_offset = description->scores.byte_offset +
				  description->scores.byte_count,
		.context_token_limit = description->context_token_limit,
		.ratio = description->ratio,
		.segments = INDEX_SPLIT_SEGMENTS,
		.is_block = is_block ? 1u : 0u
	};
	for (uint32_t pass = 0; pass < INDEX_SPLIT_LAUNCH_COUNT; pass++) {
		struct kernel_launch *const launch = &launches[pass];
		memset(launch, 0, sizeof(*launch));
		launch->kernel = made_up_kernels.split[pass];
		launch->grid[0] = description->row_count;
		launch->grid[1] = launch->kernel == SPLIT_CUT_HIGH_HANDLE ||
						  launch->kernel ==
							  SPLIT_CUT_LOW_HANDLE ?
					  1u :
					  INDEX_SPLIT_SEGMENTS;
		launch->grid[LAUNCH_DIMENSIONS - 1u] = 1;
		launch->block[0] = INDEX_SPLIT_THREADS;
		launch->block[1] = 1;
		launch->block[LAUNCH_DIMENSIONS - 1u] = 1;
		launch->argument_byte_count = sizeof(record);
		memcpy(launch->argument, &record, sizeof(record));
	}
}

/* The launches of a description, by hand: the records in the order of their members, the key tiles of a session
 * one behind another (model_layout.h); the candidates' launch for a source alone. Their count. */
static uint32_t
expected_launches(const struct index_scan_description *description,
		  struct kernel_launch launches[INDEX_SCAN_LAUNCH_COUNT])
{
	const bool is_reader = description->candidate_role ==
			       INDEX_CANDIDATES_READER;
	const bool is_source = description->candidate_role ==
			       INDEX_CANDIDATES_SOURCE;
	const uint32_t entries =
		description->context_token_limit / description->ratio;
	const uint32_t row_scores =
		is_reader ? MODEL_CONTEXT_WITHOUT_CANDIDATES : entries;
	const uint32_t segments =
		(row_scores / MODEL_KEYS_PER_TILE + INDEX_SEGMENT_TILES - 1u) /
		INDEX_SEGMENT_TILES;
	const uint64_t stride =
		(uint64_t)entries / MODEL_KEYS_PER_TILE * MODEL_TILE_BYTES;
	const uint64_t candidates =
		is_reader ? description->candidates.byte_offset :
			    INDEX_NO_CANDIDATES;
	const struct index_scan_arguments_record scan = {
		description->step.byte_offset,
		description->keys.byte_offset,
		stride,
		description->queries.byte_offset,
		description->weights.byte_offset,
		description->scores.byte_offset,
		candidates,
		row_scores,
		description->ratio,
		description->context_token_limit,
		description->session_capacity,
		0,
		0
	};
	const struct index_select_arguments_record select = {
		description->step.byte_offset,
		description->scores.byte_offset,
		description->selection.byte_offset,
		candidates,
		description->context_token_limit,
		description->ratio,
		0,
		0
	};
	const struct index_candidates_arguments kept = {
		description->step.byte_offset,
		description->scores.byte_offset,
		description->candidates.byte_offset,
		entries,
		description->context_token_limit,
		0,
		0
	};
	const void *const records[INDEX_SCAN_LAUNCH_COUNT] = { &scan, &select,
							       &kept };
	const uint32_t record_sizes[INDEX_SCAN_LAUNCH_COUNT] = {
		(uint32_t)sizeof(scan), (uint32_t)sizeof(select),
		(uint32_t)sizeof(kept)
	};
	const uint32_t handles[INDEX_SCAN_LAUNCH_COUNT] = { SCAN_HANDLE,
							    SELECT_HANDLE,
							    CANDIDATES_HANDLE };
	const uint32_t count = is_source ? INDEX_SCAN_LEGACY_LAUNCH_COUNT :
					   INDEX_SCAN_LAUNCH_CANDIDATES;
	memset(launches, UNTOUCHED_BYTE,
	       INDEX_SCAN_LAUNCH_COUNT * sizeof(*launches));
	memset(launches, 0, count * sizeof(*launches));
	for (uint32_t launch = 0; launch < count; launch++) {
		const bool is_scan = launch == INDEX_SCAN_LAUNCH_SCAN;
		struct kernel_launch *const made = &launches[launch];
		made->kernel = handles[launch];
		made->grid[0] = description->row_count;
		made->grid[1] = is_scan ? segments : 1u;
		made->grid[LAUNCH_DIMENSIONS - 1u] = 1;
		made->block[0] = is_scan ? INDEX_SCAN_THREADS :
					   INDEX_SELECT_THREADS_VALUE;
		made->block[1] = 1;
		made->block[LAUNCH_DIMENSIONS - 1u] = 1;
		made->dynamic_shared_byte_count =
			is_scan ? MODEL_INDEX_QUERIES_ROW_BYTES : 0u;
		made->argument_byte_count = record_sizes[launch];
		memcpy(made->argument, records[launch],
		       made->argument_byte_count);
	}
	if ((description->row_count <= INDEX_SPLIT_MAX_ROWS ||
	     (description->row_count <= INDEX_SPLIT_DEEP_MAX_ROWS &&
	      row_scores >= INDEX_SPLIT_DEEP_MIN_ENTRIES)) &&
	    row_scores >= INDEX_SPLIT_MIN_ENTRIES) {
		const struct kernel_launch legacy_candidates =
			launches[INDEX_SCAN_LAUNCH_CANDIDATES];
		memset(&launches[1], UNTOUCHED_BYTE,
		       (INDEX_SCAN_LAUNCH_COUNT - 1u) * sizeof(*launches));
		expected_split_launches(description, false, &launches[1]);
		uint32_t split_count = 1u + INDEX_SPLIT_LAUNCH_COUNT;
		if (is_source) {
			if (entries >= INDEX_SPLIT_MIN_CANDIDATE_ENTRIES) {
				expected_split_launches(description, true,
							&launches[split_count]);
				split_count += INDEX_SPLIT_LAUNCH_COUNT;
			} else {
				launches[split_count++] = legacy_candidates;
			}
		}
		return split_count;
	}
    return count;
}

/* One accepted description: its launches every byte, those it does not have untouched, and their count. */
static void expect_accepted(const struct index_scan_description *description,
			    uint64_t arena)
{
	struct kernel_launch launches[INDEX_SCAN_LAUNCH_COUNT];
	struct kernel_launch expected[INDEX_SCAN_LAUNCH_COUNT];
	uint32_t count = 0;
	memset(launches, UNTOUCHED_BYTE, sizeof(launches));
	const uint32_t expected_count =
		expected_launches(description, expected);
	expect(index_scan_prepare(description, arena, &made_up_kernels,
				  launches, &count) == STATUS_OK &&
		       count == expected_count &&
		       memcmp(launches, expected, sizeof(launches)) == 0,
	       "the launches are the records' table, every byte, and no more (the arena ends at the last byte)");
}

static void test_accepted(void)
{
	static const uint32_t ratios[] = { MODEL_DIRECT_RATIO,
					   MODEL_POOLED_RATIO };
	static const uint32_t rows[] = { 1,
					 LINEAR_MAXIMUM_ROWS,
					 LINEAR_MAXIMUM_ROWS + 1u,
					 DEEP_SPLIT_MIDDLE_ROWS,
					 INDEX_SPLIT_DEEP_MAX_ROWS,
					 INDEX_SPLIT_DEEP_MAX_ROWS + 1u,
					 WIDE_LEGACY_ROWS,
					 MODEL_WIDE_MOST_ROWS };
	static const uint32_t contexts[] = {
		SMALLEST_CONTEXT,
		LARGEST_CONTEXT,
		INDEX_SPLIT_MIN_ENTRIES,
		INDEX_SPLIT_MIN_CANDIDATE_ENTRIES - MODEL_CONTEXT_GRANULE,
		INDEX_SPLIT_MIN_CANDIDATE_ENTRIES,
		INDEX_SPLIT_MIN_CANDIDATE_ENTRIES + MODEL_CONTEXT_GRANULE,
		INDEX_SPLIT_DEEP_MIN_ENTRIES - MODEL_CONTEXT_GRANULE,
		INDEX_SPLIT_DEEP_MIN_ENTRIES,
		INDEX_SPLIT_DEEP_MIN_ENTRIES + MODEL_CONTEXT_GRANULE,
		MODEL_CONTEXT_MOST
	};
	static const uint32_t roles[] = { INDEX_CANDIDATES_SOURCE,
					  INDEX_CANDIDATES_READER };
	context_text = "the launches of an accepted description";
	for (size_t row = 0; row < COUNT_OF(rows); row++) {
		for (size_t context = 0; context < COUNT_OF(contexts);
		     context++) {
			uint64_t arena = 0;
			for (size_t ratio = 0; ratio < COUNT_OF(ratios);
			     ratio++) {
				const struct index_scan_description description =
					valid_description(rows[row],
							  ratios[ratio],
							  contexts[context],
							  &arena);
				expect_accepted(&description, arena);
			}
			for (size_t role = 0; role < COUNT_OF(roles); role++) {
				const struct index_scan_description description =
					valid_description_of_role(
						rows[row], MODEL_DIRECT_RATIO,
						contexts[context], roles[role],
						&arena);
				expect_accepted(&description, arena);
			}
		}
    }
}

/* step_row_first and split_rows (index_span's launches over a span of a view's slabs): a span of 64 rows from row 8
 * of a split shape (8 rows' rule) takes the split selection with its annex behind its scores, every record's
 * row_first the first row, the step read up to its last row; a span past the most rows is refused. */
#define OFFSET_FIRST_ROW 8u
#define OFFSET_SPAN_ROWS 64u
static void test_offset_rows(void)
{
    context_text = "a span of a view's rows from a row on";
    const uint32_t rows = OFFSET_SPAN_ROWS, context = INDEX_SPLIT_MIN_ENTRIES;
    const uint64_t stride =
	    (uint64_t)context / MODEL_KEYS_PER_TILE * MODEL_TILE_BYTES;
    uint64_t end = 0;
    struct index_scan_description description = {
	    .context_token_limit = context,
	    .session_capacity = SESSIONS,
	    .ratio = MODEL_DIRECT_RATIO,
	    .row_count = rows,
	    .step_row_first = OFFSET_FIRST_ROW,
	    .split_rows = LINEAR_MAXIMUM_ROWS
    };
    description.step = next_range(
	    &end, STEP_BLOCK_BYTES_VALUE + (uint64_t)(OFFSET_FIRST_ROW + rows) *
						   STEP_ROW_BYTES_VALUE);
    description.keys = next_range(&end, SESSIONS * stride);
    description.queries =
	    next_range(&end, (uint64_t)rows * MODEL_INDEX_QUERIES_ROW_BYTES);
    description.weights =
	    next_range(&end, (uint64_t)rows * MODEL_INDEX_WEIGHTS_ROW_BYTES);
    description.scores =
	    next_range(&end, (uint64_t)rows * context * INDEX_SCORE_BYTES);
    (void)next_range(&end, (uint64_t)rows * INDEX_SPLIT_ROW_BYTES);
    description.selection =
	    next_range(&end, (uint64_t)rows * MODEL_SELECTION_ROW_BYTES);
    const uint64_t arena = description.selection.byte_offset + description.selection.byte_count;
    struct kernel_launch launches[INDEX_SCAN_LAUNCH_COUNT];
    uint32_t count = 0;
    expect(index_scan_prepare(&description, arena, &made_up_kernels, launches,
			      &count) == STATUS_OK &&
		   count == 1u + INDEX_SPLIT_LAUNCH_COUNT,
	   "a span of a split shape takes the scan and the split selection");
    struct index_scan_arguments_record scan;
    memcpy(&scan, launches[INDEX_SCAN_LAUNCH_SCAN].argument, sizeof(scan));
    expect(scan.row_first == OFFSET_FIRST_ROW &&
		   launches[INDEX_SCAN_LAUNCH_SCAN].grid[0] == rows,
	   "the scan's record takes the span's first row, its grid the span's rows");
    for (uint32_t launch = 1;
	 launch < count && launch < INDEX_SCAN_LAUNCH_COUNT; launch++) {
	    struct index_split_arguments split;
	    memcpy(&split, launches[launch].argument, sizeof(split));
	    expect(split.row_first == OFFSET_FIRST_ROW &&
			   launches[launch].grid[0] == rows &&
			   split.scratch_offset ==
				   description.scores.byte_offset +
					   description.scores.byte_count,
		   "each split launch takes the span's first row and the annex behind its scores");
    }
    description.step_row_first = MODEL_WIDE_MOST_ROWS - rows + 1u;
    count = 0;
    expect(index_scan_prepare(&description, UINT64_MAX, &made_up_kernels,
			      launches, &count) == STATUS_INVALID_ARGUMENT,
	   "a span past the most rows is refused");
}

/* A description that breaks a rule: refused with one error line holding the words, the launches untouched. */
static void expect_refused(const struct index_scan_description *description,
			   uint64_t arena,
			   const struct index_scan_kernels *kernels,
			   bool has_launches, const char *words,
			   const char *what)
{
	struct kernel_launch launches[INDEX_SCAN_LAUNCH_COUNT];
	struct kernel_launch untouched[INDEX_SCAN_LAUNCH_COUNT];
	char logged[CAPTURE_CAPACITY];
	memset(launches, UNTOUCHED_BYTE, sizeof(launches));
	memset(untouched, UNTOUCHED_BYTE, sizeof(untouched));
	capture_begin();
	uint32_t count = UNTOUCHED_BYTE;
	const result_status status =
		index_scan_prepare(description, arena, kernels,
				   has_launches ? launches : NULL, &count);
	capture_end(logged, sizeof(logged));
	expect(status == STATUS_INVALID_ARGUMENT && count == UNTOUCHED_BYTE &&
		       memcmp(launches, untouched, sizeof(launches)) == 0,
	       what);
	expect_one_error_line_with(logged, words, what);
}

static void test_refused(void)
{
	static const uint32_t bad_rows[] = { 0, MODEL_WIDE_MOST_ROWS + 1u };
	static const uint32_t bad_ratios[] = { 0, MODEL_POOLED_RATIO + 1u };
	static const uint32_t bad_contexts[] = {
		0, SMALLEST_CONTEXT + 1u, MODEL_CONTEXT_MOST + SMALLEST_CONTEXT
	};
	struct index_scan_kernels without_scan = made_up_kernels;
	struct index_scan_kernels without_select = made_up_kernels;
	struct index_scan_kernels without_candidates = made_up_kernels;
	without_scan.scan = BACKEND_NO_KERNEL;
	without_select.select = BACKEND_NO_KERNEL;
	without_candidates.candidates = BACKEND_NO_KERNEL;
	uint64_t arena = 0;
	const struct index_scan_description valid =
		valid_description(LINEAR_MAXIMUM_ROWS, MODEL_POOLED_RATIO,
				  LARGEST_CONTEXT, &arena);
	struct index_scan_description broken = valid;
	struct linear_range *const ranges[] = {
		&broken.step,	 &broken.keys,	 &broken.queries,
		&broken.weights, &broken.scores, &broken.selection
	};
	context_text = "a refused description";
	expect_refused(NULL, arena, &made_up_kernels, true, RULE_MISSING,
		       "no description");
	expect_refused(&valid, arena, NULL, true, RULE_MISSING, "no kernels");
	expect_refused(&valid, arena, &made_up_kernels, false, RULE_MISSING,
		       "no launches to fill");
	expect_refused(&valid, arena, &without_scan, true, RULE_KERNEL,
		       "the scan's kernel is not prepared");
	expect_refused(&valid, arena, &without_select, true, RULE_KERNEL,
		       "the selection's kernel is not prepared");
	expect_refused(&valid, arena, &without_candidates, true, RULE_KERNEL,
		       "the candidates' kernel is not prepared");
	for (size_t index = 0; index < COUNT_OF(bad_rows); index++) {
		broken = valid;
		broken.row_count = bad_rows[index];
		expect_refused(&broken, arena, &made_up_kernels, true,
			       RULE_COUNTS, "a row count out of range");
		broken = valid;
		broken.ratio = bad_ratios[index];
		expect_refused(&broken, arena, &made_up_kernels, true,
			       RULE_COUNTS, "a ratio of neither 1 nor 2");
	}
    for (size_t index = 0; index < COUNT_OF(bad_contexts); index++) {
        broken = valid;
        broken.context_token_limit = bad_contexts[index];
        expect_refused(&broken, arena, &made_up_kernels, true, RULE_COUNTS, "a context of none, part granules or more");
    }
    broken = valid;
    broken.session_capacity = 0;
    expect_refused(&broken, arena, &made_up_kernels, true, RULE_COUNTS, "no session");
    broken = valid;
    broken.session_capacity = SESSIONS + 1u;
    expect_refused(&broken, arena, &made_up_kernels, true, RULE_TOUCHED, "a session more than the key tiles hold");
    for (size_t range = 0; range < COUNT_OF(ranges); range++) {
        broken = valid;
        ranges[range]->byte_offset += 1u;
        expect_refused(&broken, arena, &made_up_kernels, true, RULE_ALIGNED, "a range off the alignment");
        broken = valid;
        ranges[range]->byte_count -= 1u;
        expect_refused(&broken, arena, &made_up_kernels, true, RULE_TOUCHED, "a range a byte short");
        broken = valid;
        ranges[range]->byte_offset = arena - ranges[range]->byte_count + ALIGNMENT;
        expect_refused(&broken, arena, &made_up_kernels, true, RULE_ARENA, "a range that leaves the arena");
    }
    broken = valid;
    broken.selection.byte_offset = broken.scores.byte_offset;
    expect_refused(&broken, arena, &made_up_kernels, true, RULE_SHARED, "the selection on the scores");
    expect_refused(&valid, arena - 1u, &made_up_kernels, true, RULE_ARENA, "an arena a byte short of the last range");
    broken = valid;
    broken.candidate_role = INDEX_CANDIDATES_READER + 1u;
    expect_refused(&broken, arena, &made_up_kernels, true, RULE_COUNTS, "a candidate role not known");
    uint64_t role_arena = 0;
    const struct index_scan_description reader = valid_description_of_role(
	    LINEAR_MAXIMUM_ROWS, MODEL_DIRECT_RATIO, LARGEST_CONTEXT,
	    INDEX_CANDIDATES_READER, &role_arena);
    broken = reader;
    broken.ratio = MODEL_POOLED_RATIO;
    expect_refused(&broken, role_arena, &made_up_kernels, true, RULE_COUNTS, "a candidate role at ratio 2");
    broken = reader;
    broken.candidates.byte_offset += 1u;
    expect_refused(&broken, role_arena, &made_up_kernels, true, RULE_ALIGNED, "the candidates off the alignment");
    broken = reader;
    broken.candidates.byte_count -= 1u;
    expect_refused(&broken, role_arena, &made_up_kernels, true, RULE_TOUCHED, "the candidates a byte short");
    broken = reader;
    broken.candidates.byte_offset = broken.selection.byte_offset;
    expect_refused(&broken, role_arena, &made_up_kernels, true, RULE_SHARED, "the candidates on the selection");
    struct kernel_launch launches[INDEX_SCAN_LAUNCH_COUNT];
    capture_begin();
    const result_status uncounted =
	    index_scan_prepare(&valid, arena, &made_up_kernels, launches, NULL);
    char logged[CAPTURE_CAPACITY];
    capture_end(logged, sizeof(logged));
    expect(uncounted == STATUS_INVALID_ARGUMENT,
	   "no place for the count of the launches");
}

static void test_split_refused(void)
{
    uint64_t arena = 0;
    const struct index_scan_description valid = valid_description_of_role(
	    INDEX_SPLIT_MAX_ROWS, MODEL_DIRECT_RATIO, MODEL_CONTEXT_MOST,
	    INDEX_CANDIDATES_SOURCE, &arena);
    struct index_scan_description broken = valid;
    struct index_scan_kernels kernels = made_up_kernels;
    context_text = "split preparation and its scratch annex";
    for (uint32_t pass = 0; pass < INDEX_SPLIT_LAUNCH_COUNT; pass++) {
	    kernels = made_up_kernels;
	    kernels.split[pass] = BACKEND_NO_KERNEL;
	    expect_refused(&valid, arena, &kernels, true, RULE_KERNEL,
			   "each split pass needs a prepared kernel");
    }
    broken.scores.byte_count -= 1u;
    expect_refused(&broken, arena, &made_up_kernels, true, RULE_TOUCHED,
                   "scratch bytes cannot hide a short score payload");
    broken = valid;
    broken.selection.byte_offset = valid.scores.byte_offset + valid.scores.byte_count;
    expect_refused(&broken, arena, &made_up_kernels, true, RULE_SHARED,
                   "the scratch annex may not overwrite selected positions");
    broken = valid;
    broken.candidates.byte_offset = valid.scores.byte_offset + valid.scores.byte_count;
    expect_refused(&broken, arena, &made_up_kernels, true, RULE_SHARED,
                   "the scratch annex may not overwrite candidate blocks");
    broken = valid;
    broken.scores.byte_offset = arena;
    const uint64_t scratch_bytes =
	    (uint64_t)valid.row_count * INDEX_SPLIT_ROW_BYTES;
    const uint64_t score_end = arena + valid.scores.byte_count;
    expect_refused(&broken, score_end, &made_up_kernels, true, RULE_ARENA,
                   "score payload fits but the scratch annex is absent");
    expect_refused(&broken, score_end + scratch_bytes - 1u, &made_up_kernels, true, RULE_ARENA,
                   "the scratch annex is a byte short");
    expect_accepted(&broken, score_end + scratch_bytes);
    broken.scores.byte_offset = UINT64_MAX - valid.scores.byte_count - (ALIGNMENT - 1u);
    expect_refused(&broken, UINT64_MAX, &made_up_kernels, true, RULE_ARENA,
                   "a score payload near UINT64_MAX leaves no space for scratch");
    broken = valid;
    broken.scores.byte_count += 1u;
    expect_refused(&broken, arena, &made_up_kernels, true, RULE_ALIGNED,
                   "the scratch annex must begin on an aligned offset");
    broken = valid;
    broken.scores.byte_count += ALIGNMENT;
    broken.selection.byte_offset += ALIGNMENT;
    broken.candidates.byte_offset += ALIGNMENT;
    expect_accepted(&broken, arena + ALIGNMENT);

    for (uint32_t pass = 0; pass < INDEX_SPLIT_LAUNCH_COUNT; pass++) {
	    kernels.split[pass] = BACKEND_NO_KERNEL;
    }
    static const struct {
        uint32_t rows, context;
    } fallbacks[] = { { 1, LARGEST_CONTEXT },
		      { INDEX_SPLIT_MAX_ROWS + 1u,
			INDEX_SPLIT_DEEP_MIN_ENTRIES - MODEL_CONTEXT_GRANULE },
		      { INDEX_SPLIT_DEEP_MAX_ROWS + 1u, MODEL_CONTEXT_MOST },
		      { WIDE_LEGACY_ROWS, MODEL_CONTEXT_MOST },
		      { MODEL_WIDE_MOST_ROWS, MODEL_CONTEXT_MOST } };
    for (size_t index = 0; index < COUNT_OF(fallbacks); index++) {
        uint64_t fallback_arena = 0;
	const struct index_scan_description fallback =
		valid_description(fallbacks[index].rows, MODEL_DIRECT_RATIO,
				  fallbacks[index].context, &fallback_arena);
	struct kernel_launch launches[INDEX_SCAN_LAUNCH_COUNT];
	uint32_t count = 0;
	expect(index_scan_prepare(&fallback, fallback_arena, &kernels, launches,
				  &count) == STATUS_OK &&
		       count == INDEX_SCAN_LAUNCH_CANDIDATES &&
		       launches[INDEX_SCAN_LAUNCH_SELECT].kernel ==
			       SELECT_HANDLE,
	       "small and wide fallback needs no scratch or split handles");
    }
}

/* ---- the reference ---- */

static void test_rounding(void)
{
    uint32_t back = 0, ordered = 0, compared = 0;
    context_text = "the host's FP16";
    for (uint32_t bits = 0; bits < HALF_PATTERNS; bits++) {
        const double value = test_index_half_value((uint16_t)bits);
        back += test_index_half_nearest(value) == (isnan(value) ? HALF_NOT_A_NUMBER : bits) ? 1u : 0u;
    }
    expect(back == HALF_PATTERNS, "every FP16 pattern comes back from its value (a not-a-number as the quiet one)");
    expect(test_index_half_nearest(1.0 + ldexp(1.0, HALF_PLACE_AT_ONE - 1)) == HALF_ONE &&
               test_index_half_nearest(1.0 + ldexp(THREE_HALF_PLACES, HALF_PLACE_AT_ONE - 1)) ==
                   HALF_ONE_AND_TWO_PLACES &&
               test_index_half_nearest(BELOW_OVERFLOW) == HALF_LARGEST &&
               test_index_half_nearest(OVERFLOW_AT) == HALF_INFINITY &&
               test_index_half_nearest(-OVERFLOW_AT) == (HALF_INFINITY | HALF_SIGN) &&
               test_index_half_nearest(ldexp(1.0, HALF_SUBNORMAL_PLACE - 1)) == 0 &&
               test_index_half_nearest(ldexp(THREE_HALF_PLACES, HALF_SUBNORMAL_PLACE - 1)) == HALF_TWO_SMALLEST &&
               test_index_half_nearest(-0.0) == HALF_SIGN,
           "ties to even, the largest numbers, subnormals and the sign of zero, by hand");
    for (uint32_t first = 0; first < HALF_PATTERNS; first += ORDER_STRIDE_FIRST) {
        for (uint32_t second = 0; second < HALF_PATTERNS; second += ORDER_STRIDE_SECOND) {
            const double a = test_index_half_value((uint16_t)first), b = test_index_half_value((uint16_t)second);
            if (!isnan(a) && !isnan(b) && (a != b || first == second)) {
                compared++;
                ordered += (a < b) == (test_index_order_key((uint16_t)first) < test_index_order_key((uint16_t)second))
                               ? 1u : 0u;
            }
        }
    }
    expect(compared != 0 && ordered == compared && test_index_order_key(HALF_SIGN) < test_index_order_key(0),
           "order keys follow the values, -0 below +0");
}

/* The FP16 value placed at a byte of a lane order is the natural one of (vector, dim): vector 128 + dim. */
static bool is_placed_by_hand(const unsigned char *placed, const struct placed_by_hand *cases, size_t case_count)
{
    bool is_hand = true;
    for (size_t index = 0; index < case_count; index++) {
        uint16_t value = 0;
        memcpy(&value, placed + cases[index].byte_offset, sizeof(value));
        is_hand = is_hand && (uint32_t)value == cases[index].vector * DIMS + cases[index].dim;
    }
    return is_hand;
}

static void test_layouts_and_hand_cases(void)
{
    static uint16_t query[HEADS * DIMS];
    static unsigned char placed[MODEL_INDEX_QUERIES_ROW_BYTES];
    static unsigned char tile[MODEL_TILE_BYTES];
    static uint16_t scores[HAND_KEYS];
    static uint32_t seen[HEADS * DIMS];
    uint16_t key[DIMS] = {0};
    uint16_t weights[HEADS] = {0};
    int32_t selected[SELECTED];
    double sums[TEST_INDEX_SUM_HEADS_IN_ORDER + 1u];
    bool is_hand = true;
    context_text = "the host's layouts, a score and selections by hand";
    for (uint32_t value = 0; value < HEADS * DIMS; value++) {
        query[value] = (uint16_t)value;
    }
    test_index_place_queries(query, placed);
    for (uint32_t key_place = 0; key_place < MODEL_KEYS_PER_TILE; key_place++) {
	    test_index_place_key(&query[(size_t)key_place * DIMS], tile,
				 key_place);
    }
    for (size_t half = 0; half < sizeof(placed) / sizeof(uint16_t); half++) {
        uint16_t value = 0;
        memcpy(&value, placed + half * sizeof(uint16_t), sizeof(value));
        seen[value]++;
        if (half < sizeof(tile) / sizeof(uint16_t)) {
            memcpy(&value, tile + half * sizeof(uint16_t), sizeof(value));
            seen[value]++;
        }
    }
    for (uint32_t value = 0; value < HEADS * DIMS; value++) {
	    is_hand =
		    is_hand &&
		    seen[value] ==
			    (value < MODEL_KEYS_PER_TILE * DIMS ? 1u + 1u : 1u);
    }
    expect(is_hand, "the lane orders place every value once");
    expect(is_placed_by_hand(placed, queries_by_hand, COUNT_OF(queries_by_hand)) &&
               is_placed_by_hand(tile, keys_by_hand, COUNT_OF(keys_by_hand)),
           "values lie at the bytes model_layout.h's formulas give them");
    memset(query, 0, sizeof(query));
    key[0] = test_index_half_nearest(HAND_KEY_VALUE);
    query[(size_t)HAND_PLAIN * DIMS] = HALF_ONE;
    query[(size_t)HAND_DOUBLED * DIMS] = test_index_half_nearest(HAND_DOUBLE);
    query[(size_t)HAND_HALVED * DIMS] = HALF_ONE;
    query[(size_t)HAND_BELOW_ZERO * DIMS] = HALF_ONE | HALF_SIGN;
    weights[HAND_PLAIN] = BFLOAT16_TWO;
    weights[HAND_DOUBLED] = BFLOAT16_TWO;
    weights[HAND_HALVED] = BFLOAT16_HALF;
    weights[HAND_BELOW_ZERO] = BFLOAT16_MINUS_ONE;
    for (uint32_t sum = TEST_INDEX_SUM_DOUBLE; sum <= TEST_INDEX_SUM_HEADS_IN_ORDER; sum++) {
        test_index_scores(query, weights, key, 1, (enum test_index_sum)sum, &sums[sum]);
    }
    expect(sums[TEST_INDEX_SUM_DOUBLE] == HAND_SCORE && sums[TEST_INDEX_SUM_TREE] == HAND_SCORE &&
               sums[TEST_INDEX_SUM_HEADS_IN_ORDER] == HAND_SCORE,
           "a score worked out by hand, the same in every sum");
    memset(query, 0, sizeof(query));
    memset(weights, 0, sizeof(weights));
    key[0] = HALF_ONE;
    for (uint32_t head = 0; head < HEADS; head++) {
        query[(size_t)head * DIMS] = HALF_ONE;
    }
    weights[TREE_LARGE] = BFLOAT16_TWO_TO_24;
    weights[TREE_LARGE_NEGATIVE] = BFLOAT16_MINUS_TWO_TO_24;
    weights[TREE_ONE] = BFLOAT16_ONE;
    for (uint32_t sum = TEST_INDEX_SUM_DOUBLE; sum <= TEST_INDEX_SUM_HEADS_IN_ORDER; sum++) {
        test_index_scores(query, weights, key, 1, (enum test_index_sum)sum, &sums[sum]);
    }
    expect(sums[TEST_INDEX_SUM_DOUBLE] == 1.0 && sums[TEST_INDEX_SUM_HEADS_IN_ORDER] == 1.0 &&
               sums[TEST_INDEX_SUM_TREE] == 0.0 && !signbit(sums[TEST_INDEX_SUM_TREE]),
           "the tree: (2^24 + 1) in v_0 loses the 1 in float32, then -2^24 leaves +0; double and in order keep 1");
    /* one score everywhere: the lowest 512 positions; with the break the highest */
    memset(scores, 0, sizeof(scores));
    test_index_select(scores, HAND_KEYS, false, selected);
    expect(selected[0] == 0 && selected[SELECTED - 1u] == (int32_t)(SELECTED - 1u), "all tied: the lowest 512");
    test_index_select(scores, HAND_KEYS, true, selected);
    expect(selected[0] == (int32_t)(HAND_KEYS - SELECTED) && selected[SELECTED - 1u] == (int32_t)(HAND_KEYS - 1u),
           "all tied, the break: the highest 512");
    /* rising scores (subnormal patterns) to HAND_TIES_FROM, then the largest number: those and the highest rising */
    for (uint32_t position = 0; position < HAND_KEYS; position++) {
        scores[position] = (uint16_t)(position < HAND_TIES_FROM ? position : HALF_LARGEST);
    }
    test_index_select(scores, HAND_KEYS, false, selected);
    is_hand = true;
    for (uint32_t output = 0; output < SELECTED; output++) {
        is_hand = is_hand && selected[output] == (int32_t)(HAND_KEYS - SELECTED + output);
    }
    expect(is_hand, "rising, then tied above them: positions 88 to 599");
    test_index_select(scores, HAND_TIES_FROM, false, selected);
    expect(selected[0] == 0 &&
		   selected[HAND_TIES_FROM - 1u] ==
			   (int32_t)(HAND_TIES_FROM - 1u) &&
		   selected[HAND_TIES_FROM] == INDEX_NO_POSITION &&
		   selected[SELECTED - 1u] == INDEX_NO_POSITION,
	   "512 entries or fewer: every one, then -1");
}

int main(void)
{
    capture_descriptor = mkstemp(capture_path);
    error_stream_descriptor = dup(STDERR_FILENO);
    if (capture_descriptor < 0 || error_stream_descriptor < 0 || unlink(capture_path) != 0) {
        (void)printf("FAILED: the temporary file that stands in for the error stream\n");
        return EXIT_FAILURE;
    }
    test_kernels_prepare();
    test_accepted();
    test_offset_rows();
    test_refused();
    test_split_refused();
    test_rounding();
    test_layouts_and_hand_cases();
    (void)close(capture_descriptor);
    (void)close(error_stream_descriptor);
    if (failures != 0) {
        (void)printf("index_scan_test: %d failures\n", failures);
        return EXIT_FAILURE;
    }
    (void)printf("index_scan_test passed\n");
    return EXIT_SUCCESS;
}
