/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * draft_test: source/draft.c against draft.h and draft_records.h, and the host reference of the draft's kernels
 * (draft_reference.h) on cases worked by hand (dspark_v1/DESIGN.md H3). No card: a prepare function is arithmetic on
 * a description, the arena a number, the handles made up or a recording table's.
 *
 *   - draft_kernels_prepare: a missing argument refused; against a recording table, the eight kernels asked
 *     once each in order, by their file (the core's native build for class 120 native), name and record size; the
 *     first failure ends it with that status, names the kernel and writes nothing; success fills the draft's member
 *     of each family and the draft's own handles. On a card of each class of the host-memory backend all are found.
 *   - The three prepare functions: accepted descriptions, their launch checked by hand (kernel, grid, block, every
 *     member of the record, zeros behind it); every rule of draft.h broken once, each range shrunk, moved off the
 *     alignment, out of the arena and onto another: refused with one error line holding the rule's words, the launch
 *     untouched.
 *   - The reference: the verify kernel's answer for every count of proposals k = 0 .. 7, every first mismatch and
 *     every place of the end token, against the emitted count worked out here, with a row of another session in the
 *     step; a record of fewer rows than its step; a step without a DRAFTS row; the Markov bias and the confidence of
 *     whole numbers whose sums are exact, and their float32 orders against the double sums.
 * A failure is one line on the standard output.
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/draft.h"
#include "anemone/status.h"
#include "draft_reference.h"
#include "host_memory_backend.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"
#include "log_expectation.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define INVALID STATUS_INVALID_ARGUMENT
#define UNTOUCHED_BYTE 0xA5
#define UNTOUCHED_WORD 0xA5A5A5A5u
#define CAPTURE_CAPACITY 4096u
#define CARD_MEMORY_BYTES 1048576u
#define ARENA_BYTES (1ull << 40u)
#define ALIGNMENT LINEAR_OFFSET_ALIGNMENT_BYTES
#define MISALIGNMENT (ALIGNMENT / 2u)
#define DRAFT_KERNEL_COUNT \
	8u /* the draft's members of four families, then kernel_draft's three */
#define FIRST_RECORDED_HANDLE 60u
#define MADE_UP_HANDLE 20u
#define POSITIONS MODEL_DRAFT_BLOCK
#define MOST_POSITIONS STEP_DRAFT_MOST
#define ROWS LINEAR_MAXIMUM_ROWS
#define VOCABULARY MODEL_VOCABULARY_VALUE
#define END_TOKEN 1u /* the end token of the tests: below the vocabulary */
#define SMALL_VOCABULARY \
	19u /* the Markov bias's reference case: entries of a launch's last block */
#define SEQUENCE 7u
#define FIRST_POSITION 100
#define OTHER_SESSION 3u
#define VERIFYING_SESSION 1u
#define TOKEN_BASE \
	1000 /* proposal i is TOKEN_BASE + i; the end token is END_TOKEN */
#define OTHER_TOKEN 77
#define WORDS_UNTOUCHED 0x5Au
#define ROUND_UNTOUCHED_WORD \
	0x5A5A5A5Au /* a word of the round the verify kernel leaves */
#define ANCHOR_AND_OTHER_ROWS \
	2u /* of a verify step: the other session's row and the anchor */
#define SHORT_RECORD_ROWS \
	5u /* a verify record of fewer rows than its step: 4 emitted at most */
#define CORE_BLOCK_ASKED 4u /* the fifth kernel asked: the block core */
#define STREAM_EMBEDDING 0x4D31u /* the hash streams of the reference's cases */
#define STREAM_HEAD 0x4D32u
#define STREAM_COLLAPSE 0x4331u
#define STREAM_WEIGHTS 0x4332u
#define RULE_MISSING "a missing argument"
#define RULE_NOT_PREPARED "is not prepared"
#define RULE_RESERVED "a reserved field"
#define RULE_POSITIONS "draft positions: 1 to"
#define RULE_VERIFY_COUNTS "rows, end token"
#define RULE_VERIFY_FLAGS "a round table with slab results"
#define RULE_MARKOV_COUNTS "a position below the draft's"
#define RULE_BATCHED_SLOT "a batched draft's slot"
#define RULE_ALIGNED "an offset is a multiple"
#define RULE_ARENA "leaves the arena"
#define RULE_TOUCHED "the launches touch"
#define RULE_EXACT "its format stores"
#define RULE_SHARED "share a byte"
#define MOST_RANGES 6u
/* the reference's whole numbers: Markov weights and embeddings of -2 .. 2, sums of at most 256 * 4 */
#define WHOLE_SPAN 5u
#define WHOLE_LOWEST 2
#define DOUBLE_TOLERANCE \
	1e-5 /* of the float32 orders against the double sums, relative */

enum { DIMENSION_X, DIMENSION_Y, DIMENSION_Z };

static int failures = 0;
static const char *context_text = "";
static char capture_path[] = "/tmp/anemone_draft_test_XXXXXX";
static int capture_descriptor = -1;
static int error_stream_descriptor = -1;
static const struct draft_kernels_record made_up_kernels = {
	MADE_UP_HANDLE, MADE_UP_HANDLE + 1u, MADE_UP_HANDLE + 2u
};

static void expect(bool condition, const char *what)
{
	if (!condition) {
		(void)printf("FAILED: %s: %s\n", context_text, what);
		failures++;
	}
}

static void capture_begin(void)
{
	expect(capture_descriptor >= 0 && fflush(stderr) == 0 &&
		       ftruncate(capture_descriptor, 0) == 0 &&
		       lseek(capture_descriptor, 0, SEEK_SET) == 0 &&
		       dup2(capture_descriptor, STDERR_FILENO) == STDERR_FILENO,
	       "the error stream is pointed at the temporary file");
}

static void capture_end(char *text, size_t text_capacity)
{
	const bool is_open = capture_descriptor >= 0 &&
			     error_stream_descriptor >= 0 &&
			     fflush(stderr) == 0;
	const ssize_t byte_count = is_open ? pread(capture_descriptor, text,
						   text_capacity - 1, 0) :
					     -1;
	expect(is_open &&
		       dup2(error_stream_descriptor, STDERR_FILENO) ==
			       STDERR_FILENO &&
		       byte_count >= 0,
	       "the temporary file is read back and the error stream restored");
	text[byte_count >= 0 ? (size_t)byte_count : 0] = '\0';
}

static void expect_one_error_line_with(const char *text, const char *words,
				       const char *what)
{
	if (!test_log_is_expected(text, words, true)) {
		(void)printf(
			"FAILED: %s: %s: no single error line with \"%s\": \"%s\"\n",
			context_text, what, words, text);
		failures++;
	}
}

/* ---- the kernels of a card ---- */

struct kernel_asked {
	const char *kernel_file_name;
	const char *function_name;
	uint32_t argument_byte_count;
};
static struct kernel_asked kernels_asked[DRAFT_KERNEL_COUNT];
static uint32_t asked_count = 0;
static uint32_t failing_call_number = 0; /* counted from 1; 0: no call fails */

MUST_CHECK static result_status
recording_kernel_prepare(struct device_card *card, const char *kernel_file_name,
			 const char *function_name,
			 uint32_t argument_byte_count, uint32_t *kernel_out)
{
	if (card != NULL && asked_count < DRAFT_KERNEL_COUNT) {
		kernels_asked[asked_count] =
			(struct kernel_asked){ kernel_file_name, function_name,
					       argument_byte_count };
	}
	asked_count++;
	if (asked_count == failing_call_number) {
		return STATUS_DEVICE_ERROR;
	}
	*kernel_out = FIRST_RECORDED_HANDLE + asked_count - 1u;
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
	const struct kernel_asked expected[DRAFT_KERNEL_COUNT] = {
		{ KERNEL_FILE_HYPER_CONNECTION, KERNEL_HYPER_DRAFT_INPUTS,
		  HYPER_DRAFT_INPUTS_ARGUMENT_BYTES },
		{ KERNEL_FILE_HYPER_CONNECTION, KERNEL_HYPER_FINAL_KEPT,
		  HYPER_FINISH_ARGUMENT_BYTES },
		{ KERNEL_FILE_ROUTER, KERNEL_ROUTER_SELECT_DRAFT,
		  ROUTER_SELECT_ARGUMENT_BYTES },
		{ KERNEL_FILE_ATTENTION_STATE, KERNEL_ATTENTION_WINDOW,
		  ATTENTION_PREPARE_ARGUMENT_BYTES },
		{ KERNEL_FILE_ATTENTION_CORE_NATIVE,
		  KERNEL_ATTENTION_CORE_BLOCK, ATTENTION_CORE_ARGUMENT_BYTES },
		{ KERNEL_FILE_DRAFT, KERNEL_VERIFY_MATCHES,
		  VERIFY_MATCHES_ARGUMENT_BYTES },
		{ KERNEL_FILE_DRAFT, KERNEL_DRAFT_MARKOV_BIAS,
		  DRAFT_MARKOV_BIAS_ARGUMENT_BYTES },
		{ KERNEL_FILE_DRAFT, KERNEL_DRAFT_CONFIDENCE,
		  DRAFT_CONFIDENCE_ARGUMENT_BYTES }
	};
	struct hyper_kernels_record hyper;
	struct router_kernels_record router;
	struct attention_state_kernels state;
	struct attention_core_kernels core;
	struct draft_kernels_record draft;
	struct device_card *card = NULL;
	char logged[CAPTURE_CAPACITY];
	context_text = "the draft's kernels";
	memset(&hyper, UNTOUCHED_BYTE, sizeof(hyper));
	memset(&router, UNTOUCHED_BYTE, sizeof(router));
	memset(&state, UNTOUCHED_BYTE, sizeof(state));
	memset(&core, UNTOUCHED_BYTE, sizeof(core));
	memset(&draft, UNTOUCHED_BYTE, sizeof(draft));
	test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
	expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES,
						  &card) == STATUS_OK,
	       "a card opens");
	capture_begin();
	expect(draft_kernels_prepare(NULL, card, ARITHMETIC_CLASS_SM_80, &hyper,
				     &router, &state, &core,
				     &draft) == INVALID &&
		       draft_kernels_prepare(&backend_without_kernels, card,
					     ARITHMETIC_CLASS_SM_80, &hyper,
					     &router, &state, &core,
					     &draft) == INVALID &&
		       draft_kernels_prepare(&recording_backend, NULL,
					     ARITHMETIC_CLASS_SM_80, &hyper,
					     &router, &state, &core,
					     &draft) == INVALID &&
		       draft_kernels_prepare(&recording_backend, card,
					     ARITHMETIC_CLASS_SM_80, NULL,
					     &router, &state, &core,
					     &draft) == INVALID &&
		       draft_kernels_prepare(
			       &recording_backend, card, ARITHMETIC_CLASS_SM_80,
			       &hyper, &router, &state, &core, NULL) == INVALID,
	       "a missing table, kernel_prepare, card or place for the handles");
	capture_end(logged, sizeof(logged));
	for (uint32_t call_number = 1; call_number <= DRAFT_KERNEL_COUNT;
	     call_number++) {
		asked_count = 0;
		failing_call_number = call_number;
		capture_begin();
		expect(draft_kernels_prepare(&recording_backend, card,
					     ARITHMETIC_CLASS_SM_120_NATIVE,
					     &hyper, &router, &state, &core,
					     &draft) == STATUS_DEVICE_ERROR &&
			       asked_count == call_number,
		       "the status of the kernel_prepare that fails; nothing asked after it");
		capture_end(logged, sizeof(logged));
		expect_one_error_line_with(
			logged, expected[call_number - 1u].function_name,
			"it is named");
	}
	expect(hyper.draft_inputs == UNTOUCHED_WORD &&
		       hyper.final_kept == UNTOUCHED_WORD &&
		       router.select_draft == UNTOUCHED_WORD &&
		       state.window == UNTOUCHED_WORD &&
		       core.attention_core_block == UNTOUCHED_WORD &&
		       draft.verify_matches == UNTOUCHED_WORD,
	       "a failure writes no handle");
	failing_call_number = 0;
	asked_count = 0;
	expect(draft_kernels_prepare(
		       &recording_backend, card, ARITHMETIC_CLASS_SM_120_NATIVE,
		       &hyper, &router, &state, &core, &draft) == STATUS_OK &&
		       asked_count == DRAFT_KERNEL_COUNT,
	       "every kernel is asked once");
	for (uint32_t index = 0; index < DRAFT_KERNEL_COUNT; index++) {
		expect(kernels_asked[index].kernel_file_name != NULL &&
			       strcmp(kernels_asked[index].kernel_file_name,
				      expected[index].kernel_file_name) == 0 &&
			       strcmp(kernels_asked[index].function_name,
				      expected[index].function_name) == 0 &&
			       kernels_asked[index].argument_byte_count ==
				       expected[index].argument_byte_count,
		       "in order, by its file (the core's native build), name and record size");
	}
	const uint32_t handles[DRAFT_KERNEL_COUNT] = {
		hyper.draft_inputs,	   hyper.final_kept,
		router.select_draft,	   state.window,
		core.attention_core_block, draft.verify_matches,
		draft.markov_bias,	   draft.confidence
	};
	for (uint32_t index = 0; index < DRAFT_KERNEL_COUNT; index++) {
		expect(handles[index] == FIRST_RECORDED_HANDLE + index,
		       "each handle in its family's member, in order");
	}
	asked_count = 0;
	expect(draft_kernels_prepare(&recording_backend, card,
				     ARITHMETIC_CLASS_SM_120, &hyper, &router,
				     &state, &core, &draft) == STATUS_OK &&
		       strcmp(kernels_asked[CORE_BLOCK_ASKED].kernel_file_name,
			      KERNEL_FILE_ATTENTION_CORE) == 0,
	       "the emulating class 120 takes the core's plain build");
	test_host_memory_backend.card_close(card, 0);
	static const uint32_t classes[] = { ARITHMETIC_CLASS_SM_80,
					    ARITHMETIC_CLASS_SM_120 };
	for (size_t index = 0; index < COUNT_OF(classes); index++) {
		test_host_memory_backend_reset(1, CARD_MEMORY_BYTES);
		test_host_memory_backend_set_arithmetic_class(0,
							      classes[index]);
		card = NULL;
		expect(test_host_memory_backend.card_open(0, CARD_MEMORY_BYTES,
							  &card) == STATUS_OK &&
			       draft_kernels_prepare(
				       &test_host_memory_backend, card,
				       classes[index], &hyper, &router, &state,
				       &core, &draft) == STATUS_OK,
		       "the program's modules hold every kernel of the draft, on a card of each class");
		test_host_memory_backend.card_close(card, 0);
	}
}

/* ---- descriptions ---- */

static struct linear_range next_range(uint64_t *cursor, uint64_t byte_count)
{
	const struct linear_range range = { *cursor, byte_count };
	*cursor += (byte_count + ALIGNMENT - 1u) / ALIGNMENT * ALIGNMENT +
		   ALIGNMENT;
	return range;
}

static void expect_launch(const struct kernel_launch *launch, uint32_t kernel,
			  uint32_t grid_columns, uint32_t grid_rows,
			  uint32_t threads, const void *record,
			  uint32_t record_byte_count, const char *what)
{
	unsigned char behind[LAUNCH_ARGUMENT_BYTES];
	memset(behind, 0, sizeof(behind));
	expect(launch->kernel == kernel &&
		       launch->grid[DIMENSION_X] == grid_columns &&
		       launch->grid[DIMENSION_Y] == grid_rows &&
		       launch->grid[DIMENSION_Z] == 1 &&
		       launch->block[DIMENSION_X] == threads &&
		       launch->block[DIMENSION_Y] == 1 &&
		       launch->block[DIMENSION_Z] == 1 &&
		       launch->dynamic_shared_byte_count == 0 &&
		       launch->branch == LAUNCH_CHAIN &&
		       launch->argument_byte_count == record_byte_count &&
		       memcmp(launch->argument, record, record_byte_count) ==
			       0 &&
		       memcmp(launch->argument + record_byte_count, behind,
			      LAUNCH_ARGUMENT_BYTES - record_byte_count) == 0,
	       what);
}

/* One kind of description and its prepare, so that one refusal loop serves the three. */
enum described_kind { VERIFY, MARKOV, CONFIDENCE, KIND_COUNT };
struct described {
	enum described_kind kind;
	struct verify_matches_description verify;
	struct draft_markov_bias_description markov;
	struct draft_confidence_description confidence;
};

static struct described described_of(enum described_kind kind)
{
	uint64_t cursor = ALIGNMENT;
	struct described described;
	memset(&described, 0, sizeof(described));
	described.kind = kind;
	struct verify_matches_description *const verify = &described.verify;
	verify->step = next_range(
		&cursor,
		STEP_BLOCK_BYTES_VALUE + (uint64_t)ROWS * STEP_ROW_BYTES_VALUE);
	verify->result = next_range(
		&cursor,
		STEP_RESULT_BYTES + (uint64_t)ROWS * STEP_ROW_RESULT_BYTES);
	verify->round = next_range(&cursor, STEP_ROUND_BYTES);
	verify->draft_step = next_range(
		&cursor, STEP_BLOCK_BYTES_VALUE +
				 (uint64_t)POSITIONS * STEP_ROW_BYTES_VALUE);
	verify->draft_views = next_range(
		&cursor, (uint64_t)POSITIONS * STEP_DRAFT_VIEW_BYTES);
	verify->row_count = ROWS;
	verify->draft_positions = POSITIONS;
	verify->end_token_id = END_TOKEN;
	verify->vocabulary_size = VOCABULARY;
	cursor = ALIGNMENT;
	struct draft_markov_bias_description *const markov = &described.markov;
	markov->draft_step = next_range(&cursor, STEP_BLOCK_BYTES_VALUE +
							 STEP_ROW_BYTES_VALUE);
	markov->results = next_range(&cursor, (uint64_t)POSITIONS *
						      STEP_DRAFT_RESULT_BYTES);
	markov->embedding = next_range(&cursor, (uint64_t)VOCABULARY *
							DRAFT_MARKOV_ROW_BYTES);
	markov->head = next_range(&cursor, (uint64_t)VOCABULARY *
						   DRAFT_MARKOV_ROW_BYTES);
	markov->logits = next_range(&cursor, (uint64_t)POSITIONS *
						     MODEL_LOGITS_ROW_BYTES);
	markov->markov = next_range(&cursor, (uint64_t)POSITIONS *
						     DRAFT_MARKOV_ROW_BYTES);
	markov->position = POSITIONS - 1u;
	markov->draft_positions = POSITIONS;
	markov->vocabulary_size = VOCABULARY;
	cursor = ALIGNMENT;
	struct draft_confidence_description *const confidence =
		&described.confidence;
	confidence->draft_step = next_range(&cursor, STEP_BLOCK_BYTES_VALUE);
	confidence->collapse = next_range(
		&cursor, (uint64_t)POSITIONS * DRAFT_COLLAPSE_ROW_BYTES);
	confidence->markov = next_range(
		&cursor, (uint64_t)POSITIONS * DRAFT_MARKOV_ROW_BYTES);
	confidence->weight = next_range(
		&cursor, MODEL_HALVES(MODEL_DRAFT_CONFIDENCE_INPUTS));
	confidence->results = next_range(
		&cursor, (uint64_t)POSITIONS * STEP_DRAFT_RESULT_BYTES);
	confidence->round = next_range(&cursor, STEP_ROUND_BYTES);
	confidence->draft_positions = POSITIONS;
	return described;
}

/* The ranges of a description in the order of its members, and whether each is a tensor of the pack (exact). */
static uint32_t ranges_of(struct described *described,
			  struct linear_range **ranges, bool *is_exact)
{
	struct verify_matches_description *const verify = &described->verify;
	struct draft_markov_bias_description *const markov = &described->markov;
	struct draft_confidence_description *const confidence =
		&described->confidence;
	struct linear_range *const all[KIND_COUNT][MOST_RANGES] = {
		{ &verify->step, &verify->result, &verify->round,
		  &verify->draft_step, &verify->draft_views, NULL },
		{ &markov->draft_step, &markov->results, &markov->embedding,
		  &markov->head, &markov->logits, &markov->markov },
		{ &confidence->draft_step, &confidence->collapse,
		  &confidence->markov, &confidence->weight,
		  &confidence->results, &confidence->round }
	};
	static const bool exact[KIND_COUNT][MOST_RANGES] = {
		{ false, false, false, false, false, false },
		{ false, false, true, true, false, false },
		{ false, false, false, true, false, false }
	};
	uint32_t count = 0;
	for (uint32_t index = 0;
	     index < MOST_RANGES && all[described->kind][index] != NULL;
	     index++, count++) {
		ranges[index] = all[described->kind][index];
		is_exact[index] = exact[described->kind][index];
	}
	return count;
}

MUST_CHECK static result_status
prepare(const struct described *described, uint64_t arena_byte_count,
	const struct draft_kernels_record *kernels,
	struct kernel_launch *launch)
{
	switch (described->kind) {
	case VERIFY:
		return verify_matches_prepare(
			&described->verify, arena_byte_count, kernels, launch);
	case MARKOV:
		return draft_markov_bias_prepare(
			&described->markov, arena_byte_count, kernels, launch);
	case CONFIDENCE:
	case KIND_COUNT:
		break;
	}
	return draft_confidence_prepare(&described->confidence,
					arena_byte_count, kernels, launch);
}

static void expect_refused(const struct described *described,
			   const struct draft_kernels_record *kernels,
			   const char *words, const char *what)
{
	struct kernel_launch launch, untouched;
	char logged[CAPTURE_CAPACITY];
	memset(&launch, UNTOUCHED_BYTE, sizeof(launch));
	memset(&untouched, UNTOUCHED_BYTE, sizeof(untouched));
	capture_begin();
	const result_status status =
		prepare(described, ARENA_BYTES, kernels, &launch);
	capture_end(logged, sizeof(logged));
	expect(status == INVALID &&
		       memcmp(&launch, &untouched, sizeof(launch)) == 0,
	       what);
	expect_one_error_line_with(logged, words, what);
}

static void test_accepted(void)
{
	struct kernel_launch launch;
	context_text = "the verify kernel, accepted";
	const struct described verify = described_of(VERIFY);
	const struct verify_matches_arguments verify_record = {
		verify.verify.step.byte_offset,
		verify.verify.result.byte_offset,
		verify.verify.round.byte_offset,
		verify.verify.draft_step.byte_offset,
		verify.verify.draft_views.byte_offset,
		ROWS,
		POSITIONS,
		END_TOKEN,
		MODEL_DRAFT_NOISE_TOKEN,
		0,
		0,
		0,
		0
	};
	expect(prepare(&verify, ARENA_BYTES, &made_up_kernels, &launch) ==
		       STATUS_OK,
	       "accepted");
	expect_launch(
		&launch, made_up_kernels.verify_matches, 1, 1,
		VERIFY_MATCHES_THREADS, &verify_record, sizeof(verify_record),
		"one warp, the record of every range, the rows, R, the end and noise tokens");
	struct described table = verify;
	table.verify.flags = DRAFT_ROUND_TABLE;
	table.verify.verifier = 3;
	table.verify.round_capacity = 1;
	struct verify_matches_arguments table_record = verify_record;
	table_record.flags = DRAFT_ROUND_TABLE;
	table_record.verifier = 3;
	table_record.round_capacity = 1;
	expect(prepare(&table, ARENA_BYTES, &made_up_kernels, &launch) ==
		       STATUS_OK,
	       "a BATCH verifier accepted");
	expect_launch(&launch, made_up_kernels.verify_matches, 1, 1,
		      VERIFY_MATCHES_THREADS, &table_record,
		      sizeof(table_record),
		      "a BATCH verifier's record: its index, flags and rounds");
	context_text = "the Markov bias, accepted";
	for (uint32_t position = 0; position < POSITIONS; position++) {
		struct described markov = described_of(MARKOV);
		markov.markov.position = position;
		const struct draft_markov_bias_arguments markov_record = {
			markov.markov.draft_step.byte_offset,
			markov.markov.results.byte_offset,
			markov.markov.embedding.byte_offset,
			markov.markov.head.byte_offset,
			markov.markov.logits.byte_offset +
				(uint64_t)position * MODEL_LOGITS_ROW_BYTES,
			markov.markov.markov.byte_offset +
				(uint64_t)position * DRAFT_MARKOV_ROW_BYTES,
			position,
			VOCABULARY,
			VOCABULARY,
			0
		};
		expect(prepare(&markov, ARENA_BYTES, &made_up_kernels,
			       &launch) == STATUS_OK,
		       "accepted");
		expect_launch(
			&launch, made_up_kernels.markov_bias,
			VOCABULARY / DRAFT_MARKOV_ENTRIES_PER_BLOCK, 1,
			DRAFT_MARKOV_THREADS, &markov_record,
			sizeof(markov_record),
			"a warp an entry; the logits and Markov rows of the position");
	}
	struct described markov = described_of(MARKOV);
	markov.markov.vocabulary_size = SMALL_VOCABULARY;
	markov.markov.embedding.byte_count = markov.markov.head.byte_count =
		(uint64_t)SMALL_VOCABULARY * DRAFT_MARKOV_ROW_BYTES;
	const uint32_t small_blocks =
		(SMALL_VOCABULARY + DRAFT_MARKOV_ENTRIES_PER_BLOCK - 1u) /
		DRAFT_MARKOV_ENTRIES_PER_BLOCK;
	expect(prepare(&markov, ARENA_BYTES, &made_up_kernels, &launch) ==
			       STATUS_OK &&
		       launch.grid[DIMENSION_X] == small_blocks,
	       "a vocabulary of no whole blocks: the last block's warps beyond it do nothing");
	context_text = "the confidence, accepted";
	const struct described confidence = described_of(CONFIDENCE);
	const struct draft_confidence_arguments confidence_record = {
		confidence.confidence.draft_step.byte_offset,
		confidence.confidence.collapse.byte_offset,
		confidence.confidence.markov.byte_offset,
		confidence.confidence.weight.byte_offset,
		confidence.confidence.results.byte_offset,
		confidence.confidence.round.byte_offset,
		POSITIONS,
		0,
		0,
		0
	};
	expect(prepare(&confidence, ARENA_BYTES, &made_up_kernels, &launch) ==
		       STATUS_OK,
	       "accepted");
	expect_launch(&launch, made_up_kernels.confidence, 1, MOST_POSITIONS,
		      DRAFT_CONFIDENCE_THREADS, &confidence_record,
		      sizeof(confidence_record),
		      "a block for each place of the round");
}

static void test_refused(void)
{
	static const char *const kind_names[KIND_COUNT] = { "the verify kernel",
							    "the Markov bias",
							    "the confidence" };
	for (uint32_t kind = 0; kind < KIND_COUNT; kind++) {
		const struct described valid =
			described_of((enum described_kind)kind);
		struct described broken = valid;
		struct linear_range *ranges[MOST_RANGES];
		bool is_exact[MOST_RANGES];
		context_text = kind_names[kind];
		expect_refused(&valid, NULL, RULE_MISSING, "no kernels");
		struct draft_kernels_record missing;
		memset(&missing, UNTOUCHED_BYTE, sizeof(missing));
		missing.verify_matches = missing.markov_bias =
			missing.confidence = BACKEND_NO_KERNEL;
		expect_refused(&valid, &missing, RULE_NOT_PREPARED,
			       "the kernel not prepared");
		const uint32_t range_count =
			ranges_of(&broken, ranges, is_exact);
		for (uint32_t index = 0; index < range_count; index++) {
			broken = valid;
			(void)ranges_of(&broken, ranges, is_exact);
			ranges[index]->byte_count -= 1u;
			expect_refused(&broken, &made_up_kernels,
				       is_exact[index] ? RULE_EXACT :
							 RULE_TOUCHED,
				       "a range a byte short");
			broken = valid;
			(void)ranges_of(&broken, ranges, is_exact);
			ranges[index]->byte_offset += MISALIGNMENT;
			expect_refused(&broken, &made_up_kernels, RULE_ALIGNED,
				       "a range off the alignment");
			broken = valid;
			(void)ranges_of(&broken, ranges, is_exact);
			ranges[index]->byte_offset = ARENA_BYTES;
			expect_refused(&broken, &made_up_kernels, RULE_ARENA,
				       "a range out of the arena");
			if (range_count > 1u) {
				broken = valid;
				(void)ranges_of(&broken, ranges, is_exact);
				ranges[index]->byte_offset =
					ranges[(index + 1u) % range_count]
						->byte_offset;
				expect_refused(&broken, &made_up_kernels,
					       RULE_SHARED,
					       "a range on another");
			}
		}
	}
	context_text = "the verify kernel, refused";
	struct described broken = described_of(VERIFY);
	for (uint32_t positions = 0; positions <= MOST_POSITIONS + 1u;
	     positions += MOST_POSITIONS + 1u) {
		broken = described_of(VERIFY);
		broken.verify.draft_positions = positions;
		expect_refused(&broken, &made_up_kernels, RULE_POSITIONS,
			       "no draft positions, or more than a round's");
	}
	for (uint32_t rows = 0; rows <= ROWS + 1u; rows += ROWS + 1u) {
		broken = described_of(VERIFY);
		broken.verify.row_count = rows;
		expect_refused(&broken, &made_up_kernels, RULE_VERIFY_COUNTS,
			       "no rows, or more than a thin step's");
	}
	broken = described_of(VERIFY);
	broken.verify.end_token_id = VOCABULARY;
	expect_refused(&broken, &made_up_kernels, RULE_VERIFY_COUNTS,
		       "an end token past the vocabulary");
	broken = described_of(VERIFY);
	broken.verify.flags = DRAFT_SLAB_RESULTS;
	expect_refused(&broken, &made_up_kernels, RULE_VERIFY_FLAGS,
		       "slab results without a round table");
	broken = described_of(VERIFY);
	broken.verify.verifier = 1;
	expect_refused(&broken, &made_up_kernels, RULE_VERIFY_FLAGS,
		       "a later verifier without a round table");
	broken = described_of(VERIFY);
	broken.verify.flags = DRAFT_ROUND_TABLE;
	expect_refused(&broken, &made_up_kernels, RULE_VERIFY_FLAGS,
		       "a round table of no rounds");
	broken.verify.round_capacity = 1;
	broken.verify.row_count = DRAFT_BATCH_MOST_ROWS + 1u;
	expect_refused(&broken, &made_up_kernels, RULE_VERIFY_COUNTS,
		       "more rows than a BATCH step's");
	broken = described_of(VERIFY);
	broken.verify.vocabulary_size = MODEL_DRAFT_NOISE_TOKEN;
	expect_refused(&broken, &made_up_kernels, RULE_VERIFY_COUNTS,
		       "a vocabulary that holds no noise token");
	context_text = "the Markov bias, refused";
	broken = described_of(MARKOV);
	broken.markov.position = POSITIONS;
	expect_refused(&broken, &made_up_kernels, RULE_MARKOV_COUNTS,
		       "a position past the draft's");
	broken = described_of(MARKOV);
	broken.markov.vocabulary_size = 0;
	expect_refused(&broken, &made_up_kernels, RULE_MARKOV_COUNTS,
		       "no vocabulary");
	broken = described_of(MARKOV);
	broken.markov.embedding_rows = broken.markov.vocabulary_size - 1u;
	expect_refused(&broken, &made_up_kernels, RULE_MARKOV_COUNTS,
		       "fewer embedding rows than the cut's");
	broken = described_of(MARKOV);
	broken.markov.flags = DRAFT_ROUND_TABLE;
	expect_refused(&broken, &made_up_kernels, RULE_RESERVED,
		       "a Markov flag other than BATCHED");
	broken = described_of(MARKOV);
	broken.markov.flags = DRAFT_BATCHED;
	expect_refused(&broken, &made_up_kernels, RULE_TOUCHED,
		       "BATCHED Markov ranges of one slot a position");
	broken = described_of(MARKOV);
	broken.markov.draft_positions = MOST_POSITIONS + 1u;
	expect_refused(&broken, &made_up_kernels, RULE_POSITIONS,
		       "more draft positions than a round's");
	context_text = "the confidence, refused";
	broken = described_of(CONFIDENCE);
	broken.confidence.flags = DRAFT_SLAB_RESULTS;
	expect_refused(&broken, &made_up_kernels, RULE_RESERVED,
		       "a flag other than the round table");
	broken = described_of(CONFIDENCE);
	broken.confidence.flags = DRAFT_BATCHED;
	expect_refused(&broken, &made_up_kernels, RULE_BATCHED_SLOT,
		       "a batched confidence without a round table");
	broken = described_of(CONFIDENCE);
	broken.confidence.verifier = 1;
	expect_refused(&broken, &made_up_kernels, RULE_BATCHED_SLOT,
		       "a verifier slot without BATCHED");
	broken = described_of(CONFIDENCE);
	broken.confidence.draft_positions = 0;
	expect_refused(&broken, &made_up_kernels, RULE_POSITIONS,
		       "no draft positions");
}

/* ---- the reference ---- */

/* A step of an anchor and k proposals of the verifying session, a row of another session first. The draws: the
 * proposal's own token up to mismatch (index of the first draw that differs, k + 1 for none), END_TOKEN at index
 * end_index (none where it is 0). */
struct verify_step {
	struct step_block block;
	struct step_row_record rows[ROWS];
	struct step_row_result results[ROWS];
};

static void verify_step_make(uint32_t proposals, uint32_t mismatch,
			     uint32_t end_index, struct verify_step *step)
{
	memset(step, 0, sizeof(*step));
	step->block = (struct step_block){ SEQUENCE,
					   proposals + ANCHOR_AND_OTHER_ROWS,
					   0 };
	step->rows[0] = (struct step_row_record){ FIRST_POSITION, OTHER_TOKEN,
						  OTHER_SESSION,
						  STEP_ROW_LOGITS };
	step->results[0] = (struct step_row_result){ OTHER_TOKEN, 0.0f };
	for (uint32_t index = 0; index <= proposals; index++) {
		struct step_row_record *const row = &step->rows[index + 1u];
		*row = (struct step_row_record){
			FIRST_POSITION + (int32_t)index,
			TOKEN_BASE + (int32_t)index, VERIFYING_SESSION,
			STEP_ROW_LOGITS | (index == 0 ? STEP_ROW_DRAFTS :
							STEP_ROW_PROPOSED)
		};
		/* the draw from this row is y_(index + 1): the next proposal's token, else another */
		const uint32_t drawn_index = index + 1u;
		const int32_t drawn =
			drawn_index == end_index ?
				(int32_t)END_TOKEN :
			drawn_index < mismatch ?
				TOKEN_BASE + (int32_t)drawn_index :
				OTHER_TOKEN + (int32_t)drawn_index;
		step->results[index + 1u] =
			(struct step_row_result){ drawn, 0.0f };
	}
}

static void test_verify_reference(void)
{
	context_text = "the verify kernel's reference";
	struct verify_step step;
	struct step_round round;
	struct step_block draft_block;
	struct step_row_record draft_rows[MOST_POSITIONS];
	unsigned char views[MOST_POSITIONS * STEP_DRAFT_VIEW_BYTES];
	uint32_t case_count = 0;
	for (uint32_t proposals = 0; proposals < ROWS - 1u; proposals++) {
		for (uint32_t mismatch = 1; mismatch <= proposals + 1u;
		     mismatch++) {
			for (uint32_t end_index = 0;
			     end_index <= proposals + 1u; end_index++) {
				verify_step_make(proposals, mismatch, end_index,
						 &step);
				/* by hand: y_i matches d_i for i < mismatch, unless y_i is
				 * the end token, which emits and stops */
				uint32_t emitted = 1;
				while (emitted <= proposals &&
				       emitted < mismatch &&
				       emitted != end_index) {
					emitted++;
				}
				memset(&round, WORDS_UNTOUCHED, sizeof(round));
				test_draft_verify(&step.block, step.rows,
						  step.results, ROWS, POSITIONS,
						  END_TOKEN,
						  MODEL_DRAFT_NOISE_TOKEN,
						  &round, &draft_block,
						  draft_rows, views);
				const int32_t anchor_token =
					step.results[emitted].token_id;
				bool holds =
					round.sequence == SEQUENCE &&
					round.session_slot ==
						VERIFYING_SESSION &&
					round.emitted_count == emitted &&
					round.anchor_position ==
						FIRST_POSITION +
							(int32_t)emitted &&
					round.anchor_token == anchor_token &&
					round.proposal_count == POSITIONS &&
					round.reserved_zero == 0 &&
					(uint32_t)round.proposals[0] ==
						ROUND_UNTOUCHED_WORD;
				holds = holds &&
					draft_block.sequence == SEQUENCE &&
					draft_block.row_count == POSITIONS;
				for (uint32_t position = 0;
				     position < POSITIONS; position++) {
					struct step_block view;
					struct step_row_record view_row;
					memcpy(&view,
					       views + (size_t)position *
							       STEP_DRAFT_VIEW_BYTES,
					       sizeof(view));
					memcpy(&view_row,
					       views +
						       (size_t)position *
							       STEP_DRAFT_VIEW_BYTES +
						       STEP_BLOCK_BYTES_VALUE,
					       sizeof(view_row));
					holds = holds &&
						draft_rows[position].position ==
							round.anchor_position +
								(int32_t)
									position &&
						draft_rows[position].token_id ==
							(position == 0 ?
								 anchor_token :
								 (int32_t)
									 MODEL_DRAFT_NOISE_TOKEN) &&
						draft_rows[position]
								.session_slot ==
							VERIFYING_SESSION &&
						draft_rows[position].flags ==
							STEP_ROW_LOGITS &&
						view.sequence == SEQUENCE &&
						view.row_count == 1 &&
						memcmp(&view_row,
						       &draft_rows[position],
						       sizeof(view_row)) == 0;
				}
				expect(holds,
				       "the emitted count of keyed match, the anchor, the draft's rows and views");
				case_count++;
			}
		}
	}
	/* by hand: of a step of 8 rows that accepts every proposal, the record's first 5 (the other session's, the
	 * anchor at P and the proposals at P + 1 .. P + 3) give y_1 .. y_4, the draw of the row at P + 3 the anchor */
	verify_step_make(ROWS - ANCHOR_AND_OTHER_ROWS, ROWS, 0, &step);
	test_draft_verify(&step.block, step.rows, step.results,
			  SHORT_RECORD_ROWS, POSITIONS, END_TOKEN,
			  MODEL_DRAFT_NOISE_TOKEN, &round, &draft_block,
			  draft_rows, views);
	expect(round.emitted_count == SHORT_RECORD_ROWS - 1u &&
		       round.anchor_token ==
			       step.results[SHORT_RECORD_ROWS - 1u].token_id &&
		       round.anchor_position ==
			       FIRST_POSITION + (int32_t)SHORT_RECORD_ROWS - 1,
	       "the record's rows alone are read");
	step.block.row_count = 1;
	step.rows[0].flags = STEP_ROW_LOGITS;
	test_draft_verify(&step.block, step.rows, step.results, ROWS, POSITIONS,
			  END_TOKEN, MODEL_DRAFT_NOISE_TOKEN, &round,
			  &draft_block, draft_rows, views);
	expect(round.session_slot == STEP_NO_SESSION &&
		       round.emitted_count == 0 && draft_block.row_count == 0,
	       "a step without a DRAFTS row: no session, no rows");
	expect(case_count != 0, "the cases ran");
}

static uint16_t whole(uint64_t stream, uint64_t position)
{
	return test_linear_bfloat16_nearest(
		(double)(int32_t)(test_linear_hash(stream, position) %
				  WHOLE_SPAN) -
		(double)WHOLE_LOWEST);
}

static void test_markov_and_confidence_reference(void)
{
	static uint16_t embedding[SMALL_VOCABULARY * MODEL_MARKOV_RANK];
	static uint16_t head[SMALL_VOCABULARY * MODEL_MARKOV_RANK];
	static float logits[SMALL_VOCABULARY];
	static uint16_t markov_row[MODEL_MARKOV_RANK];
	static uint16_t collapse_row[MODEL_HIDDEN];
	static uint16_t weights[MODEL_DRAFT_CONFIDENCE_INPUTS];
	context_text = "the Markov bias's reference";
	for (uint32_t index = 0; index < SMALL_VOCABULARY * MODEL_MARKOV_RANK;
	     index++) {
		embedding[index] = whole(STREAM_EMBEDDING, index);
		head[index] = whole(STREAM_HEAD, index);
	}
	bool holds = true;
	for (uint32_t token = 0; token < SMALL_VOCABULARY; token++) {
		for (uint32_t entry = 0; entry < SMALL_VOCABULARY; entry++) {
			logits[entry] = (float)entry;
		}
		test_draft_markov_bias(embedding, head, SMALL_VOCABULARY, token,
				       logits, markov_row);
		holds = holds &&
			memcmp(markov_row,
			       &embedding[(size_t)token * MODEL_MARKOV_RANK],
			       sizeof(markov_row)) == 0;
		for (uint32_t entry = 0; entry < SMALL_VOCABULARY; entry++) {
			holds = holds &&
				(double)logits[entry] ==
					(double)entry +
						test_draft_markov_bias_exact(
							embedding, head, token,
							entry);
		}
	}
	expect(holds,
	       "whole numbers: every logit plus the exact sum, the Markov row W1's row of the token");
	context_text = "the confidence's reference";
	for (uint32_t index = 0; index < MODEL_HIDDEN; index++) {
		collapse_row[index] = whole(STREAM_COLLAPSE, index);
	}
	for (uint32_t index = 0; index < MODEL_DRAFT_CONFIDENCE_INPUTS;
	     index++) {
		weights[index] = whole(STREAM_WEIGHTS, index);
	}
	expect((double)test_draft_confidence(collapse_row, markov_row,
					     weights) ==
		       test_draft_confidence_exact(collapse_row, markov_row,
						   weights),
	       "whole numbers: the exact sum");
	test_linear_values(collapse_row, MODEL_HIDDEN, STREAM_COLLAPSE,
			   TEST_LINEAR_VALUES_PLAIN);
	test_linear_values(weights, MODEL_DRAFT_CONFIDENCE_INPUTS,
			   STREAM_WEIGHTS, TEST_LINEAR_VALUES_PLAIN);
	const double exact =
		test_draft_confidence_exact(collapse_row, markov_row, weights);
	const double ordered = (double)test_draft_confidence(
		collapse_row, markov_row, weights);
	expect(fabs(ordered - exact) <=
		       DOUBLE_TOLERANCE * fabs(exact) + DOUBLE_TOLERANCE,
	       "plain values: the float32 order within 1e-5 of the double sum");
}

int main(void)
{
	error_stream_descriptor = dup(STDERR_FILENO);
	capture_descriptor = mkstemp(capture_path);
	if (error_stream_descriptor < 0 || capture_descriptor < 0) {
		(void)printf(
			"FAILED: the temporary file that stands in for the error stream could not be made\n");
		return EXIT_FAILURE;
	}
	test_kernels_prepare();
	test_accepted();
	test_refused();
	test_verify_reference();
	test_markov_and_confidence_reference();
	context_text = "the end of the test";
	expect(close(capture_descriptor) == 0 && unlink(capture_path) == 0 &&
		       close(error_stream_descriptor) == 0 &&
		       test_host_memory_backend_open_card_count() == 0,
	       "the temporary file is removed and no card is left open");
	if (failures != 0) {
		return EXIT_FAILURE;
	}
	(void)printf("draft_test passed\n");
	return EXIT_SUCCESS;
}
