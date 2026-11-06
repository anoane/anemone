/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef TEST_STEP_FIXTURE_H
#define TEST_STEP_FIXTURE_H
#include <stddef.h>
#ifdef TEST_STEP_FIXTURE_IMPLEMENTATION
/*
 * step_test: the step driver against step.h on a three-card host-memory model:
 *     card 0: embed (stream = token * EMBED_MULTIPLIER + position + 1), a layer; it reads both n-gram tables
 *     card 1: a layer that also writes the pool entry of every emitting row (ratio 2), then a second segment
 *     card 2: a layer, then the sampler (token = stream % VOCABULARY, logit = stream % LOGIT_MODULUS); table 1
 * A layer is stream * LAYER_MULTIPLIER + its card + 1. The streams go from card to card by hand-offs of every row,
 * the pool to card 2's mirror by a hand-off with an entry ratio. Programs of 1 and 4 rows mark their launches with
 * their row count in the grid. The n-gram rows are held to engram_pack.h's tables and engram_row_ids_entry.
 *
 *   - Steps: tokens/logits/NO_TOKEN, smallest program, every card's step/n-gram areas, byte-equal pool mirror,
 *     settings uploaded once, session-run copies and work order, generation/context bounds and staging cleanup.
 *     Downloads poison staging until completion; uploads may keep reading until then. Early reuse must fail.
 *   - Every refusal of create (a program not thin or of 9 rows, a view), session_set and a step, nothing queued and no
 *     history changed; each area create measures taken where it ends at its arena's end, refused one byte further.
 *   - Failures: results that do not answer the step (another step or row count, a missing or extra token, one
 *     beyond the vocabulary), a launch that faults, a wait that runs out (destroy then frees the block), an upload
 *     that fails (the block then stays), the backend's limit of copies in flight met with twenty hand-offs.
 */
#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_records.h"
#include "log_expectation.h"
#include "anemone/engram.h"
#include "anemone/pack.h"
#include "anemone/step.h"
#include "engram_pack.h"
#include "host_memory_backend.h"
#include "prepare_common.h"

#define CARDS 3u
#define LAST_CARD 2u
#define ARENA_BYTES 65536u
#define CONTEXT 64u
#define SESSIONS 2u
#define ROW_CAPACITY 4u /* the larger program's rows */
#define LIMIT_ROWS 8u /* the plan's row capacity: the arrays' */
#define VOCABULARY TEST_ENGRAM_PACK_VOCABULARY
#define STEP_OFFSET 0u
#define SESSION_OFFSET 1024u
#define RESULT_OFFSET 2048u
#define STREAMS_OFFSET 4096u
#define POOL_OFFSET 8192u
#define ENGRAM_OFFSET 16384u
#define FAR_OFFSET 32768u /* where the many hand-offs of the limit case land */
#define BACK_OFFSET 24576u /* card 0's copy of card 1's streams */
#define ENTRY_BYTES 8ul
#define POOL_RATIO 2u
#define POOL_STRIDE (CONTEXT / POOL_RATIO * ENTRY_BYTES)
#define EMBED_MULTIPLIER 1000003ul
#define LAYER_MULTIPLIER 6364136223846793005ul
#define LOGIT_MODULUS 1000u
#define PROGRAMS 2u
#define LAUNCHES_PER_CARD \
	4u /* at most: card 1's where it also embeds (a refusal) */
#define SEGMENTS 5u
#define SEGMENT_FIELDS 2u /* a segment's first launch and launch count */
#define HAND_OFFS 4u
#define MANY_HAND_OFFS 20u /* more than BACKEND_MAXIMUM_UPLOADS_IN_FLIGHT */
#define WAIT_LIMIT 1000000000ull
#define SHORT_WAIT_LIMIT 1000ull
#define CLOSE_LIMIT 1000000000ull
#define TABLES TEST_ENGRAM_PACK_TABLES
#define CAPTURE_CAPACITY 4096u
#define MOST_COPIES_SEEN 4096u /* that the wrapping backend holds at once */
#define MOST_HELD_BYTES 65536u
#define POISON_BYTE 0xA5u /* where a download's bytes have not arrived */
#define SESSION_SEED 77ull
#define SESSION_TEMPERATURE 0.5f
#define SESSION_TOP_P 0.75f
#define LATER_SEED 91ull /* slot 0's settings, set after a step */
#define LATER_TEMPERATURE 1.5f
#define LATER_TOP_P 0.5f
#define POLLS_TO_TIME_OUT \
	2u /* a wait answers TIMED_OUT once (host_memory_backend.h) */
#define FAULTING_LAUNCH 1u
#define FAILING_TICKET \
	5u /* card 2's result download, the fifth copy of its step */
#define BAD_FLAG 0x20u /* no flag of step_records.h */
#define BAD_PAUSE 7u
#define TOO_FAR 0xFFFFFFFFFFFFull
#define FIRST_STEP_ROWS 3u
#define AGAIN_ROWS 2u
#define WRONG_PAIR_ROWS 2u
#define POSITION_TWO 2u
#define POSITION_THREE 3u
#define POSITION_FOUR 4u
#define POSITION_FIVE 5u
#define POOL_RUNS 2u /* of the step of two sessions */
#define LAUNCH_RUNS \
	4u /* card 0; card 1's first segment, its other two; card 2 */
#define UPLOADS_TO_THE_LAST_CARD \
	4u /* step area, n-gram rows, streams, pool: no settings */
#define SESSION_RANGES 2u
#define FRONT_SEGMENTS \
	2u /* a front step's: the embedding, card 1's first layer */
#define FRONT_REFUSALS \
	3u /* rows of both kinds, a front row with logits, programs with no front */
#define FRONT_DEPTH 2u /* the pipeline of the front steps' test */
#define REFUSALS_OF_ARGUMENTS 7u /* create's without a program to blame */
#define FAULT_REACHES \
	2u /* step_driver_memory: the driver, then its histories */
#define STEP_REFUSALS_BESIDE_THE_TABLES \
	7u /* no row, too many, three missing, a slot, no settings */
#define MOST_RUNS_SEEN 4096u /* launches the stand-in logs */

/* a layer with a mirror: the stream after the layer, xor every pool entry the row sees in the card's mirror, plus the
 * row's position (step_interleaved.h) */
enum operation {
	OPERATION_EMBED = 1,
	OPERATION_LAYER = 2,
	OPERATION_SAMPLE = 3,
	OPERATION_MIRROR_LAYER = 4
};
/* what the stand-in of the sampler answers: as it should; another step's sequence or row count; a token on the rows
 * without logits and none on the others; a token beyond the vocabulary */
enum sample_break {
	SAMPLE_SOUND,
	SAMPLE_OTHER_STEP,
	SAMPLE_OTHER_ROW_COUNT,
	SAMPLE_TOKEN_FLIPPED,
	SAMPLE_TOKEN_BEYOND
};
enum segment_index {
	SEGMENT_EMBED,
	SEGMENT_FIRST_LAYER,
	SEGMENT_SECOND_LAYER,
	SEGMENT_THIRD_LAYER,
	SEGMENT_SAMPLE
};
enum hand_off_index {
	HAND_OFF_TO_SECOND,
	HAND_OFF_POOL,
	HAND_OFF_TO_LAST,
	HAND_OFF_BACK
};
enum area_index {
	AREA_STEP,
	AREA_ENGRAM_ROWS,
	AREA_RESULT,
	AREA_SESSIONS,
	AREA_ROWS_SOURCE,
	AREA_ENTRIES,
	AREAS
};

/* A copy the driver began, as the wrapping backend holds it until a wait sees it finish. */
struct seen_copy {
	const unsigned char *bytes; /* in the staging block */
	unsigned char *
		arrival; /* a download's bytes, where they arrive; NULL: an upload */
	uint64_t byte_count, ticket,
		held_offset; /* in held_bytes: what arrives, or what the upload began with */
	uint32_t card;
	bool is_finished;
};

/* Everything a driver is created from, by value, so that a case changes a copy; fixture_link points it at itself. */
struct fixture {
	struct kernel_launch launches[PROGRAMS][CARDS][LAUNCHES_PER_CARD];
	struct segment_record segments[PROGRAMS][SEGMENTS];
	struct hand_off_record hand_offs[PROGRAMS][MANY_HAND_OFFS];
	struct program_card_record program_cards[PROGRAMS][CARDS];
	struct execution_program programs[PROGRAMS];
	const struct execution_program *pointers[PROGRAMS];
	struct session_range session_ranges
		[SESSION_RANGES]; /* the pool: its owner's and its mirror's */
};

static int failures = 0;
static const char *context_text = "";
static struct device_card *cards[CARDS];
static struct engram_record *engram;
static unsigned char *engram_pack;
static uint32_t
	history[SESSIONS]
	       [CONTEXT]; /* the test's own: what the driver must have used */
static enum sample_break sample_break = SAMPLE_SOUND;
static char capture_path[] = "/tmp/anemone_step_test_XXXXXX";
static int capture_descriptor = -1;
static char captured_text
	[CAPTURE_CAPACITY]; /* what was logged between capture_begin and capture_end */
static int error_stream_descriptor = -1;
/* The backend the drivers get: the one in host memory, its copies and waits seen on the way */
static struct backend_operations checked_backend;
static struct seen_copy copies[MOST_COPIES_SEEN];
static uint32_t copy_count;
static unsigned char held_bytes[MOST_HELD_BYTES];
static uint64_t held_byte_count;
static uint64_t last_ticket[CARDS];
static uint64_t waited_ticket[CARDS];
static uint32_t
	rewritten_uploads; /* staging bytes that changed before their upload was seen */
static uint32_t launch_runs;
/* What the stand-in ran, in order, under the backend's mutex: read with no step in flight */
static struct run_seen {
	uint64_t sequence; /* of the step block of its card */
	uint32_t card, operation;
} runs_seen[MOST_RUNS_SEEN];
static uint32_t runs_seen_count;
static _Atomic uint64_t embedded_sequence; /* the last step card 0 embedded */

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
	expect(capture_descriptor >= 0 && fflush(stderr) == 0 &&
		       ftruncate(capture_descriptor, 0) == 0 &&
		       lseek(capture_descriptor, 0, SEEK_SET) == 0 &&
		       dup2(capture_descriptor, STDERR_FILENO) == STDERR_FILENO,
	       "the error stream is pointed at the temporary file");
}

/* The lines logged since capture_begin, all of them errors; their text in captured_text. */
static uint32_t capture_end(void)
{
	uint32_t line_count = 0;
	const bool is_open = capture_descriptor >= 0 &&
			     error_stream_descriptor >= 0 &&
			     fflush(stderr) == 0;
	const ssize_t byte_count =
		is_open ? pread(capture_descriptor, captured_text,
				sizeof(captured_text) - 1, 0) :
			  -1;
	expect(is_open &&
		       dup2(error_stream_descriptor, STDERR_FILENO) ==
			       STDERR_FILENO &&
		       byte_count >= 0,
	       "the temporary file is read back and the error stream restored");
	captured_text[byte_count >= 0 ? (size_t)byte_count : 0] = '\0';
	for (const char *line = captured_text; *line != '\0'; line++) {
		line_count += *line == '\n' ? 1u : 0u;
	}
	expect(captured_text[0] == '\0' ||
		       test_log_is_expected(captured_text, "", false),
	       "diagnostic output contains errors; release is silent");
	return line_count;
}

/* The arena of an open card of the test; a failure, and zeros in its place, if the card is not open. */
static const unsigned char *arena_of(uint32_t card)
{
	static const unsigned char no_arena[ARENA_BYTES];
	const unsigned char *const arena =
		test_host_memory_backend_arena_of(cards[card]);
	expect(arena != NULL, "the card is open");
	return arena != NULL ? arena : no_arena;
}

static uint32_t card_index_of(const struct device_card *card)
{
	uint32_t index = 0;
	while (index < CARDS && cards[index] != card) {
		index++;
	}
	return index;
}

/* Holds a copy the backend began (it moved the bytes at once): a download's leave the staging block, poison in their
 * place, and an upload's are kept to be compared with the block, until a wait sees the copy finish. */
static void copy_held(const struct device_card *card, const void *bytes,
		      void *arrival, uint64_t byte_count, uint64_t ticket)
{
	const uint32_t index = card_index_of(card);
	const bool has_room = index < CARDS && copy_count < MOST_COPIES_SEEN &&
			      byte_count <= MOST_HELD_BYTES - held_byte_count;
	expect(has_room, "room for the copies a step has in flight");
	if (has_room) {
		copies[copy_count++] =
			(struct seen_copy){ bytes,  arrival,	     byte_count,
					    ticket, held_byte_count, index,
					    false };
		memcpy(held_bytes + held_byte_count, bytes, byte_count);
		held_byte_count += byte_count;
		last_ticket[index] = ticket;
		if (arrival != NULL) {
			memset(arrival, POISON_BYTE, byte_count);
		}
	}
}

MUST_CHECK static result_status
checked_download_begin(struct device_card *card, uint64_t arena_byte_offset,
		       void *destination, uint64_t byte_count,
		       uint64_t *ticket_out)
{
	const result_status status = test_host_memory_backend.download_begin(
		card, arena_byte_offset, destination, byte_count, ticket_out);
	if (status == STATUS_OK) {
		copy_held(card, destination, destination, byte_count,
			  *ticket_out);
	}
	return status;
}

MUST_CHECK static result_status checked_upload_begin(struct device_card *card,
						     uint64_t arena_byte_offset,
						     const void *source,
						     uint64_t byte_count,
						     uint64_t *ticket_out)
{
	const result_status status = test_host_memory_backend.upload_begin(
		card, arena_byte_offset, source, byte_count, ticket_out);
	if (status == STATUS_OK) {
		copy_held(card, source, NULL, byte_count, *ticket_out);
	}
	return status;
}

/* A wait that sees copies finish: their downloads' bytes arrive, their uploads' are compared with what they began
 * with. Once every copy held has finished, the room is free again. */
MUST_CHECK static result_status
checked_upload_wait(struct device_card *card, uint64_t ticket,
		    uint64_t wait_limit_nanoseconds)
{
	const result_status status = test_host_memory_backend.upload_wait(
		card, ticket, wait_limit_nanoseconds);
	const uint32_t index = card_index_of(card);
	bool every_copy_finished = true;
	if (status == STATUS_OK && index < CARDS) {
		waited_ticket[index] = ticket > waited_ticket[index] ?
					       ticket :
					       waited_ticket[index];
		for (uint32_t copy = 0; copy < copy_count; copy++) {
			struct seen_copy *const seen = &copies[copy];
			const bool finishes = seen->card == index &&
					      !seen->is_finished &&
					      seen->ticket <= ticket;
			const unsigned char *const held =
				held_bytes + seen->held_offset;
			if (finishes && seen->arrival != NULL) {
				memcpy(seen->arrival, held, seen->byte_count);
			} else if (finishes) {
				rewritten_uploads += memcmp(seen->bytes, held,
							    seen->byte_count) !=
								     0 ?
							     1u :
							     0u;
			}
			seen->is_finished = seen->is_finished || finishes;
			every_copy_finished = every_copy_finished &&
					      seen->is_finished;
		}
		copy_count = every_copy_finished ? 0 : copy_count;
		held_byte_count = every_copy_finished ? 0 : held_byte_count;
	}
	return status;
}

MUST_CHECK static result_status
checked_launch_run(struct device_card *card,
		   const struct kernel_launch *launches, uint32_t launch_count)
{
	launch_runs++;
	return test_host_memory_backend.launch_run(card, launches,
						   launch_count);
}

static uint64_t stream_after_layer(uint64_t stream, uint32_t card)
{
	return stream * LAYER_MULTIPLIER + card + 1u;
}

/* The stream of a row on card 2 before the sampler, as the model above computes it; and the pool's entry. */
static uint64_t expected_stream(uint32_t token, uint32_t position,
				bool is_pool_entry)
{
	uint64_t stream = (uint64_t)token * EMBED_MULTIPLIER + position + 1u;
	stream = stream_after_layer(stream_after_layer(stream, 0), 1);
	return is_pool_entry ?
		       stream :
		       stream_after_layer(
			       stream_after_layer(stream_after_layer(stream, 1),
						  1),
			       LAST_CARD);
}

/* The fill kernel as the model's kernels: the record's value is the operation, its word count 1 where a layer
 * writes the pool. Rows below the step area's row count only, as a state writer must, at most the plan's (a wide
 * step's launches take its rows from the step area too: test_wide_steps holds its views to their bytes). */
static bool model_stand_in(const struct kernel_launch *launch,
			   uint32_t card_index, unsigned char *arena,
			   uint64_t arena_byte_count, void *context)
{
	struct fill_record record;
	struct step_block block;
	memcpy(&record, launch->argument, sizeof(record));
	memcpy(&block, arena + STEP_OFFSET, sizeof(block));
	(void)context;
	if (arena_byte_count != ARENA_BYTES || block.row_count > LIMIT_ROWS) {
		return false;
	}
	if (runs_seen_count < MOST_RUNS_SEEN) {
		runs_seen[runs_seen_count++] =
			(struct run_seen){ block.sequence, card_index,
					   record.value };
	}
	if (record.value == OPERATION_EMBED) {
		atomic_store(&embedded_sequence, block.sequence);
	}
	for (uint32_t row = 0; row < block.row_count; row++) {
		struct step_row_record step_row;
		uint64_t stream = 0;
		memcpy(&step_row,
		       arena + STEP_OFFSET + STEP_BLOCK_BYTES_VALUE +
			       (size_t)row * STEP_ROW_BYTES_VALUE,
		       sizeof(step_row));
		memcpy(&stream, arena + STREAMS_OFFSET + row * sizeof(stream),
		       sizeof(stream));
		if (record.value == OPERATION_EMBED) {
			stream =
				(uint64_t)step_row.token_id * EMBED_MULTIPLIER +
				(uint64_t)step_row.position + 1u;
		} else if (record.value == OPERATION_MIRROR_LAYER) {
			const uint32_t entries =
				((uint32_t)step_row.position + 1u) / POOL_RATIO;
			uint64_t seen = 0;
			for (uint32_t entry = 0; entry < entries; entry++) {
				uint64_t value = 0;
				memcpy(&value,
				       arena + POOL_OFFSET +
					       step_row.session_slot *
						       POOL_STRIDE +
					       entry * ENTRY_BYTES,
				       sizeof(value));
				seen ^= value;
			}
			stream = (stream_after_layer(stream, card_index) ^
				  seen) +
				 (uint64_t)step_row.position;
		} else if (record.value == OPERATION_LAYER) {
			stream = stream_after_layer(stream, card_index);
			if (record.word_count == 1 &&
			    ((uint32_t)step_row.position + 1u) % POOL_RATIO ==
				    0) {
				memcpy(arena + POOL_OFFSET +
					       step_row.session_slot *
						       POOL_STRIDE +
					       (uint32_t)step_row.position /
						       POOL_RATIO * ENTRY_BYTES,
				       &stream, sizeof(stream));
			}
		} else {
			const bool has_logits =
				(step_row.flags & STEP_ROW_LOGITS) != 0;
			const bool gives_token =
				has_logits !=
				(sample_break == SAMPLE_TOKEN_FLIPPED);
			const struct step_row_result result = {
				.token_id =
					!gives_token ?
						STEP_NO_TOKEN :
					sample_break == SAMPLE_TOKEN_BEYOND ?
						(int32_t)VOCABULARY :
						(int32_t)(stream % VOCABULARY),
				.logit = has_logits ? (float)(stream %
							      LOGIT_MODULUS) :
						      0.0f
			};
			const struct step_result answer = {
				.sequence = block.sequence +
					    (sample_break == SAMPLE_OTHER_STEP ?
						     1u :
						     0u),
				.row_count =
					block.row_count +
					(sample_break == SAMPLE_OTHER_ROW_COUNT ?
						 1u :
						 0u)
			};
			memcpy(arena + RESULT_OFFSET, &answer, sizeof(answer));
			memcpy(arena + RESULT_OFFSET + STEP_RESULT_BYTES +
				       (size_t)row * STEP_ROW_RESULT_BYTES,
			       &result, sizeof(result));
		}
		memcpy(arena + STREAMS_OFFSET + row * sizeof(stream), &stream,
		       sizeof(stream));
	}
	return true;
}

static bool test_step_use_graphs;
static void (*test_step_observe)(void);

#endif
#endif
