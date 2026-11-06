/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * step_card_test: the step driver (source/step.c) on every card of this machine through the CUDA backend, the cards
 * a pipeline in the order they are enumerated:
 *     card 0 fills the logits of each row with 1.0 and writes 2.0 at token MARK + r of row r, and fills the streams;
 *     each card hands the logits (517,120 B a row) and the streams (40,976 B a row: four streams and the carried pre,
 *     the hop of every step in first_token_plan.md 2.8) to the next, which fills a word of its own; the last runs
 *     the sampler (U8), greedy. With --hop streams the hops carry the streams alone and the last card fills the
 *     logits itself: the hop a step of the model makes.
 * Steps of ROWS rows and of one row in turn, every row asking for logits: each answer must be MARK + r with the logit
 * 2.0. Then the median wall time of a step of each row count, with the hops and their bytes: what the eager driver
 * costs on this machine. One card is a pipeline of one stage with no hand-off.
 *
 *     step_card_test [--steps <count>] [--hop streams | --break hop]
 *
 * --break hop leaves out the hand-off of the logits into the last card: the test then passes only if there are two
 * cards or more and every answer was reported wrong. Without a card it says that it compared nothing, and fails if
 * it was to show a break. A failure is one line on the standard output.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "anemone/backend_records.h"
#include "anemone/model_layout.h"
#include "anemone/sampler.h"
#include "anemone/step.h"
#include "prepare_common.h"

#define ARENA_BYTES 67108864ull                          /* 64 MiB */
#define STEP_OFFSET 0ull
#define SESSION_OFFSET 4096ull
#define RESULT_OFFSET 8192ull
#define SCRATCH_OFFSET 16384ull
#define STREAMS_OFFSET 1048576ull
#define LOGITS_OFFSET 2097152ull
#define WEIGHTS_OFFSET 8388608ull
#define ROWS 8u
#define PROGRAMS 2u
#define STREAM_ROW_BYTES (MODEL_STREAMS_ROW_BYTES + MODEL_SITE_ROW_BYTES)
#define VOCABULARY MODEL_VOCABULARY_VALUE
#define MARK 1000u
#define ONE_BITS 0x3F800000u                             /* 1.0f */
#define TWO_BITS 0x40000000u                             /* 2.0f */
#define TWO 2.0f
#define MOST_LAUNCHES (ROWS + 3u)                        /* card 0: the logits, a mark a row, the streams; sampler */
#define HAND_OFFS_PER_HOP 2u
#define DEFAULT_STEPS 64u
#define MOST_STEPS 100000u
#define WAIT_LIMIT 10000000000ull
#define NANOSECONDS_PER_SECOND 1000000000ull
#define NANOSECONDS_PER_MICROSECOND 1e3
#define DECIMAL_BASE 10
#define ARGUMENT_PAIR 2
#define MEDIAN_DIVISOR 2

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };

struct card_program {
	struct kernel_launch launches[BACKEND_MAXIMUM_CARDS][MOST_LAUNCHES];
	struct program_card_record cards[BACKEND_MAXIMUM_CARDS];
	struct segment_record segments[BACKEND_MAXIMUM_CARDS];
	struct hand_off_record
		hand_offs[BACKEND_MAXIMUM_CARDS * HAND_OFFS_PER_HOP];
	struct execution_program program;
};

static struct device_card *cards[BACKEND_MAXIMUM_CARDS];
static uint32_t fill_kernels[BACKEND_MAXIMUM_CARDS];
static struct sampler_kernels_record sampler_kernels;
static struct card_program programs[PROGRAMS];
static uint64_t step_nanoseconds[PROGRAMS][MOST_STEPS];

static uint64_t clock_nanoseconds(void)
{
    struct timespec now = {0};
    return clock_gettime(CLOCK_MONOTONIC, &now) == 0
               ? (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND + (uint64_t)now.tv_nsec : 0;
}

/* A launch of the fill kernel: word_count words of value from the offset on, its grid as backend_records.h asks. */
static void fill_launch(struct kernel_launch *launch, uint32_t card,
			uint64_t offset, uint64_t word_count, uint32_t value)
{
	const struct fill_record record = { .arena_byte_offset = offset,
					    .word_count = word_count,
					    .value = value };
	const uint64_t blocks =
		(word_count + FILL_THREADS_PER_BLOCK_VALUE - 1u) /
		FILL_THREADS_PER_BLOCK_VALUE;
	prepare_launch_fill(launch, fill_kernels[card],
			    blocks == 0		      ? 1u :
			    blocks > FILL_MOST_BLOCKS ? FILL_MOST_BLOCKS :
							(uint32_t)blocks,
			    1, FILL_THREADS_PER_BLOCK_VALUE, 0, &record,
			    sizeof(record));
}

/* The pipeline of the header comment for steps of up to row_count rows, without the last logits hop if broken, and
 * with the logits filled on the last card and no logits hop if streams_only. */
MUST_CHECK static result_status program_build(struct card_program *made,
					      uint32_t card_count,
					      uint32_t row_count,
					      bool is_broken, bool streams_only)
{
    const uint64_t logits_bytes = (uint64_t)row_count * VOCABULARY * sizeof(float);
    const struct sampler_description sampler = {
	    { STEP_OFFSET, STEP_BLOCK_BYTES_VALUE +
				   (uint64_t)row_count * STEP_ROW_BYTES_VALUE },
	    { SESSION_OFFSET, STEP_SESSION_BYTES },
	    { LOGITS_OFFSET, logits_bytes },
	    { WEIGHTS_OFFSET, logits_bytes },
	    { RESULT_OFFSET,
	      STEP_RESULT_BYTES + (uint64_t)row_count * STEP_ROW_RESULT_BYTES },
	    VOCABULARY,
	    1,
	    row_count,
	    0
    };
    const uint32_t last = card_count - 1u;
    result_status status = STATUS_OK;
    memset(made, 0, sizeof(*made));
    for (uint32_t card = 0; card < card_count && status == STATUS_OK; card++) {
	    struct kernel_launch *const launches = made->launches[card];
	    uint32_t count = 0;
	    if (card == (streams_only ? last : 0)) {
		    fill_launch(&launches[count++], card, LOGITS_OFFSET,
				(uint64_t)row_count * VOCABULARY, ONE_BITS);
		    for (uint32_t row = 0; row < row_count; row++) {
			    fill_launch(&launches[count++], card,
					LOGITS_OFFSET +
						((uint64_t)row * VOCABULARY +
						 MARK + row) *
							sizeof(float),
					1, TWO_BITS);
		    }
	    }
	    if (card == 0) {
		    fill_launch(&launches[count++], card, STREAMS_OFFSET,
				(uint64_t)row_count * STREAM_ROW_BYTES /
					FILL_WORD_BYTES,
				card);
	    } else if (card != last) {
		    fill_launch(&launches[count++], card, SCRATCH_OFFSET, 1,
				card);
	    }
	    if (card == last) {
		    status = sampler_prepare_entry(&sampler, ARENA_BYTES,
						   &sampler_kernels,
						   &launches[count++]);
	    }
	    made->cards[card] = (struct program_card_record){
		    launches,
		    ARENA_BYTES,
		    STEP_OFFSET,
		    card == last ? SESSION_OFFSET : PROGRAM_NOWHERE,
		    card == last ? RESULT_OFFSET : PROGRAM_NOWHERE,
		    PROGRAM_NOWHERE,
		    STREAMS_OFFSET,
		    PROGRAM_NOWHERE,
		    PROGRAM_NOWHERE,
		    0,
		    count,
		    NULL,
		    0,
		    PROGRAM_NOWHERE
	    };
	    made->segments[card] =
		    (struct segment_record){ card, card,  SEGMENT_NO_LAYER,
					     0,	   count, 0 };
	    if (card != last && !streams_only &&
		!(is_broken && card + 1u == last)) {
		    made->hand_offs[made->program.hand_off_count++] =
			    (struct hand_off_record){
				    LOGITS_OFFSET,
				    LOGITS_OFFSET,
				    0,
				    VOCABULARY * (uint32_t)sizeof(float),
				    0,
				    card,
				    card + 1u,
				    card,
				    0
			    };
	    }
	    if (card != last) {
		    made->hand_offs[made->program.hand_off_count++] =
			    (struct hand_off_record){ STREAMS_OFFSET,
						      STREAMS_OFFSET,
						      0,
						      STREAM_ROW_BYTES,
						      0,
						      card,
						      card + 1u,
						      card,
						      0 };
	    }
    }
    made->program = (struct execution_program){
	    made->cards,
	    made->segments,
	    made->hand_offs,
	    { .context_token_limit = MODEL_CONTEXT_WITHOUT_CANDIDATES,
	      .row_capacity = ROWS,
	      .session_capacity = 1,
	      .state_width = STATE_WIDTH_16 },
	    card_count,
	    card_count,
	    made->program.hand_off_count,
	    row_count,
	    last,
	    NULL,
	    NULL,
	    0,
	    0,
	    PROGRAM_THIN,
	    0,
	    NULL,
	    0,
	    0,
	    0,
	    0,
	    NULL,
	    NULL,
	    card_count
    };
    return status;
}

static int compare_nanoseconds(const void *first, const void *second)
{
    const uint64_t first_value = *(const uint64_t *)first, second_value = *(const uint64_t *)second;
    return first_value < second_value ? -1 : first_value > second_value ? 1 : 0;
}

/* The cards open with the fill kernel, the last also with the sampler, the last card's logits zero. */
MUST_CHECK static result_status cards_open(uint32_t card_count)
{
	struct kernel_launch zeros;
	result_status status = STATUS_OK;
	for (uint32_t card = 0; card < card_count && status == STATUS_OK;
	     card++) {
		status = backend_cuda_value.card_open(card, ARENA_BYTES,
						      &cards[card]);
		status = status != STATUS_OK ?
				 status :
				 backend_cuda_value.kernel_prepare(
					 cards[card], KERNEL_FILE_FILL,
					 KERNEL_FILL_WORDS, FILL_RECORD_BYTES,
					 &fill_kernels[card]);
	}
	status = status != STATUS_OK ?
			 status :
			 sampler_kernels_prepare(&backend_cuda_value,
						 cards[card_count - 1u],
						 &sampler_kernels);
	if (status == STATUS_OK) {
		fill_launch(&zeros, card_count - 1u, LOGITS_OFFSET,
			    (uint64_t)ROWS * VOCABULARY, 0);
		status = backend_cuda_value.launch_run(cards[card_count - 1u],
						       &zeros, 1);
		status = status != STATUS_OK ?
				 status :
				 backend_cuda_value.launch_wait(
					 cards[card_count - 1u], WAIT_LIMIT);
	}
    return status;
}

int main(int argument_count, char **arguments)
{
	struct step_driver *driver = NULL;
	struct step_row_record rows[ROWS];
	struct step_row_result results[ROWS];
	uint32_t card_count = 0, step_count = DEFAULT_STEPS, wrong_count = 0,
		 answered_count = 0;
	uint32_t steps_of[PROGRAMS] = { 0 };
	bool is_broken = false, streams_only = false, is_usage = true;
	for (int index = 1; index + 1 < argument_count && is_usage;
	     index += ARGUMENT_PAIR) {
		const char *const name = arguments[index],
				  *const value = arguments[index + 1];
		char *end = NULL;
		const unsigned long long number =
			strtoull(value, &end, DECIMAL_BASE);
		const bool is_steps = strcmp(name, "--steps") == 0;
		is_usage = is_steps ? *end == '\0' && number != 0 &&
					      number <= MOST_STEPS :
			   strcmp(name, "--break") == 0 ?
				      strcmp(value, "hop") == 0 :
				      strcmp(name, "--hop") == 0 &&
					      strcmp(value, "streams") == 0;
		step_count = is_steps && is_usage ? (uint32_t)number :
						    step_count;
		is_broken = is_broken || strcmp(name, "--break") == 0;
		streams_only = streams_only || strcmp(name, "--hop") == 0;
	}
    if (!is_usage || argument_count % ARGUMENT_PAIR == 0 || (is_broken && streams_only)) {
        (void)printf("usage: step_card_test [--steps 1..%u] [--hop streams | --break hop]\n", MOST_STEPS);
        return TEST_WRONG_USAGE;
    }
    result_status status = backend_cuda_value.card_count(&card_count);
    if (status != STATUS_OK || card_count == 0) {
	    (void)printf("step_card_test: no card (%s): compared nothing%s\n",
			 status_text(status),
			 is_broken ? ", so no break was shown: FAILED" : "");
	    return status == STATUS_OK && !is_broken ? TEST_PASSED :
						       TEST_FAILED;
    }
    card_count = card_count > BACKEND_MAXIMUM_CARDS ? BACKEND_MAXIMUM_CARDS :
						      card_count;
    status = cards_open(card_count);
    for (uint32_t program = 0; program < PROGRAMS && status == STATUS_OK;
	 program++) {
	    status = program_build(&programs[program], card_count,
				   program == 0 ? 1u : ROWS, is_broken,
				   streams_only);
    }
    const struct execution_program *const pointers[PROGRAMS] = {
	    &programs[0].program, &programs[1].program
    };
    status = status != STATUS_OK ?
		     status :
		     step_driver_create(&backend_cuda_value, cards, pointers,
					PROGRAMS, NULL, WAIT_LIMIT, &driver);
    for (uint32_t step = 0, position = 0;
	 step < step_count && status == STATUS_OK; step++) {
	    const uint32_t program = step % PROGRAMS,
			   row_count = program == 0 ? 1u : ROWS;
	    position = position + row_count > MODEL_CONTEXT_WITHOUT_CANDIDATES ?
			       0 :
			       position; /* begun again */
	    for (uint32_t row = 0; row < row_count; row++) {
		    rows[row] =
			    (struct step_row_record){ (int32_t)(position + row),
						      (int32_t)(MARK + row), 0,
						      STEP_ROW_LOGITS };
	    }
	    const uint64_t started = clock_nanoseconds();
	    status = step_run_entry(driver, rows, row_count, results);
	    step_nanoseconds[program][steps_of[program]++] =
		    clock_nanoseconds() - started;
	    for (uint32_t row = 0; row < row_count && status == STATUS_OK;
		 row++) {
		    const bool is_right = results[row].token_id ==
						  (int32_t)(MARK + row) &&
					  results[row].logit == TWO;
		    wrong_count += is_right ? 0 : 1u;
		    answered_count++;
		    if (!is_right && !is_broken) {
			    (void)printf("FAILED: step %" PRIu32 " row %" PRIu32
					 ": token %" PRId32
					 " logit %.9g, not %" PRIu32 " and 2\n",
					 step, row, results[row].token_id,
					 (double)results[row].logit,
					 MARK + row);
		    }
	    }
	    position += row_count;
    }
    for (uint32_t program = 0;
	 program < PROGRAMS && status == STATUS_OK && steps_of[program] != 0;
	 program++) {
	    qsort(step_nanoseconds[program], steps_of[program],
		  sizeof(uint64_t), compare_nanoseconds);
	    const uint64_t median =
		    step_nanoseconds[program]
				    [steps_of[program] / MEDIAN_DIVISOR];
	    (void)printf(
		    "STEP_TIME rows=%" PRIu32 " cards=%" PRIu32
		    " steps=%" PRIu32 " hop_bytes_per_row=%" PRIu64
		    " median_us=%.1f fastest_us=%.1f\n",
		    program == 0 ? 1u : ROWS, card_count, steps_of[program],
		    (streams_only ? 0 : (uint64_t)VOCABULARY * sizeof(float)) +
			    STREAM_ROW_BYTES,
		    (double)median / NANOSECONDS_PER_MICROSECOND,
		    (double)step_nanoseconds[program][0] /
			    NANOSECONDS_PER_MICROSECOND);
    }
    step_driver_destroy(driver);
    for (uint32_t card = 0; card < card_count; card++) {
	    backend_cuda_value.card_close(cards[card], WAIT_LIMIT);
    }
    const bool passed =
	    status == STATUS_OK &&
	    (is_broken ? wrong_count == answered_count && card_count > 1 :
			 wrong_count == 0);
    (void)printf("step_card_test: %s: %" PRIu32 " cards, %" PRIu32
		 " answers, %" PRIu32 " wrong%s (%s)\n",
		 passed ? "passed" : "FAILED", card_count, answered_count,
		 wrong_count,
		 is_broken    ? ", the logits hop left out" :
		 streams_only ? ", hops of the streams alone" :
				"",
		 status_text(status));
    return passed ? TEST_PASSED : TEST_FAILED;
}
