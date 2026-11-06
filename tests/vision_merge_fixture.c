/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copy-only oracles use explicit expected token tables, independent of the kernel's token-position formula. */
#include "vision_merge_fixture.h"

#include <stddef.h>
#include <string.h>

#include "anemone/image.h"
#include "anemone/step_records.h"
#include "vision_reference.h"

#define MERGE_PATTERN_PERIOD 64u
#define MERGE_CELL_PATTERN_STEP 128u
#define MERGE_SMALL_START 0x3F00u
#define MERGE_SMALL_NEWLINE 0xBF00u
#define MERGE_SMALL_END 0x3F80u
#define MERGE_SMALL_CELL 0x4000u
#define MERGE_MODEL_FIRST_CELL 0x4100u
#define MERGE_MODEL_SECOND_CELL 0x4200u
#define MERGE_MODEL_FIRST_POSITION 5u
#define MERGE_MODEL_SPAN_TOKENS 4u
#define MERGE_TEXT_POSITION (MERGE_MODEL_FIRST_POSITION - 1u)
#define MERGE_UNTOUCHED_WORD 0xA5A5u
#define MERGE_STEP_LAUNCH (TEST_VISION_MERGE_LAUNCHES - 1u)
#define MERGE_SMALL_CELLS \
	(TEST_VISION_MERGE_SMALL_HEIGHT * TEST_VISION_MERGE_SMALL_COLUMNS)
#define MERGE_WRITTEN_RANGES 5u

static uint16_t pattern(uint32_t base, uint32_t channel)
{
	return (uint16_t)(base + channel % MERGE_PATTERN_PERIOD);
}

static struct linear_range take(uint64_t *next, uint64_t bytes)
{
	const struct linear_range range = { .byte_offset = *next,
					    .byte_count = bytes };
	*next += (bytes + VISION_ALIGNMENT - 1u) / VISION_ALIGNMENT *
		 VISION_ALIGNMENT;
	return range;
}

bool test_vision_merge_fixture_prepare(
	bool include_model, const struct vision_kernels *kernels,
	struct test_vision_merge_fixture *fixture_out)
{
	if (kernels == NULL || fixture_out == NULL) {
		return false;
	}
	struct test_vision_merge_fixture fixture = { .includes_model =
							     include_model };
	uint64_t next = VISION_ALIGNMENT;
	const uint64_t small_row_bytes =
		TEST_VISION_MERGE_SMALL_WIDTH * sizeof(uint16_t);
	fixture.small_weights.image_start = take(&next, small_row_bytes);
	fixture.small_weights.image_newline = take(&next, small_row_bytes);
	fixture.small_weights.image_end = take(&next, small_row_bytes);
	fixture.small_encoded =
		take(&next, (uint64_t)MERGE_SMALL_CELLS * small_row_bytes);
	fixture.small_output =
		take(&next, TEST_VISION_MERGE_SMALL_TOKENS * small_row_bytes);
	fixture.small_copies =
		take(&next, TEST_VISION_MERGE_SMALL_TOKENS * small_row_bytes *
				    VISION_MAXIMUM_COPIES);
	if (next > TEST_VISION_MERGE_SMALL_ARENA_BYTES) {
		return false;
	}
	result_status status = vision_merge_prepare(
		&fixture.small_weights, &fixture.small_encoded,
		&fixture.small_output, TEST_VISION_MERGE_SMALL_WIDTH,
		TEST_VISION_MERGE_SMALL_HEIGHT, TEST_VISION_MERGE_SMALL_COLUMNS,
		1, TEST_VISION_MERGE_SMALL_ARENA_BYTES, kernels,
		&fixture.launches[0]);
	if (status == STATUS_OK) {
		status = vision_merge_prepare(
			&fixture.small_weights, &fixture.small_encoded,
			&fixture.small_copies, TEST_VISION_MERGE_SMALL_WIDTH,
			TEST_VISION_MERGE_SMALL_HEIGHT,
			TEST_VISION_MERGE_SMALL_COLUMNS, VISION_MAXIMUM_COPIES,
			TEST_VISION_MERGE_SMALL_ARENA_BYTES, kernels,
			&fixture.launches[1]);
	}
	fixture.launch_count = TEST_VISION_MERGE_SMALL_LAUNCHES;
	if (include_model && status == STATUS_OK) {
		next = TEST_VISION_MERGE_SMALL_ARENA_BYTES;
		const uint64_t row_bytes =
			(uint64_t)VISION_MAXIMUM_LANGUAGE_WIDTH *
			sizeof(uint16_t);
		fixture.model_weights.image_start = take(&next, row_bytes);
		fixture.model_weights.image_newline = take(&next, row_bytes);
		fixture.model_weights.image_end = take(&next, row_bytes);
		for (uint32_t image = 0; image < TEST_VISION_MERGE_MODEL_IMAGES;
		     image++) {
			fixture.model_encoded[image] = take(&next, row_bytes);
		}
		fixture.embeddings =
			take(&next, (uint64_t)VISION_MAXIMUM_IMAGES *
					    VISION_MAXIMUM_TOKENS * row_bytes);
		fixture.metadata = take(&next, VISION_SESSION_METADATA_BYTES);
		fixture.step =
			take(&next, STEP_BLOCK_BYTES_VALUE +
					    TEST_VISION_MERGE_STEP_ROWS *
						    STEP_ROW_BYTES_VALUE);
		fixture.streams =
			take(&next, (uint64_t)TEST_VISION_MERGE_STEP_ROWS *
					    VISION_MAXIMUM_COPIES * row_bytes);
		if (next > TEST_VISION_MERGE_CARD_ARENA_BYTES) {
			return false;
		}
		for (uint32_t image = 0;
		     image < TEST_VISION_MERGE_MODEL_IMAGES &&
		     status == STATUS_OK;
		     image++) {
			fixture.spans[image] = (struct linear_range){
				.byte_offset = fixture.embeddings.byte_offset +
					       (uint64_t)image *
						       VISION_MAXIMUM_TOKENS *
						       row_bytes,
				.byte_count =
					MERGE_MODEL_SPAN_TOKENS * row_bytes
			};
			status = vision_merge_prepare(
				&fixture.model_weights,
				&fixture.model_encoded[image],
				&fixture.spans[image],
				VISION_MAXIMUM_LANGUAGE_WIDTH, 1, 1, 1,
				TEST_VISION_MERGE_CARD_ARENA_BYTES, kernels,
				&fixture.launches
					 [TEST_VISION_MERGE_SMALL_LAUNCHES +
					  image]);
		}
		if (status == STATUS_OK) {
			status = vision_step_merge_prepare(
				fixture.embeddings, fixture.metadata,
				fixture.step, fixture.streams,
				TEST_VISION_MERGE_STEP_ROWS, 1,
				TEST_VISION_MERGE_CARD_ARENA_BYTES, kernels,
				&fixture.launches[MERGE_STEP_LAUNCH]);
		}
		fixture.launch_count = TEST_VISION_MERGE_LAUNCHES;
	}
	if (status != STATUS_OK) {
		return false;
	}
	*fixture_out = fixture;
	return true;
}

static void fill_row(unsigned char *arena, struct linear_range range,
		     uint32_t row, uint32_t width, uint32_t base)
{
	uint16_t *const values = test_vision_at(
		arena,
		range.byte_offset + (uint64_t)row * width * sizeof(uint16_t));
	for (uint32_t channel = 0; channel < width; channel++) {
		values[channel] = pattern(base, channel);
	}
}

void test_vision_merge_fixture_fill(
	const struct test_vision_merge_fixture *fixture, unsigned char *arena)
{
	const uint64_t bytes = fixture->includes_model ?
				       TEST_VISION_MERGE_CARD_ARENA_BYTES :
				       TEST_VISION_MERGE_SMALL_ARENA_BYTES;
	memset(arena, TEST_VISION_MERGE_SENTINEL, (size_t)bytes);
	fill_row(arena, fixture->small_weights.image_start, 0,
		 TEST_VISION_MERGE_SMALL_WIDTH, MERGE_SMALL_START);
	fill_row(arena, fixture->small_weights.image_newline, 0,
		 TEST_VISION_MERGE_SMALL_WIDTH, MERGE_SMALL_NEWLINE);
	fill_row(arena, fixture->small_weights.image_end, 0,
		 TEST_VISION_MERGE_SMALL_WIDTH, MERGE_SMALL_END);
	for (uint32_t cell = 0; cell < MERGE_SMALL_CELLS; cell++) {
		fill_row(arena, fixture->small_encoded, cell,
			 TEST_VISION_MERGE_SMALL_WIDTH,
			 MERGE_SMALL_CELL + cell * MERGE_CELL_PATTERN_STEP);
	}
	if (!fixture->includes_model) {
		return;
	}
	fill_row(arena, fixture->model_weights.image_start, 0,
		 VISION_MAXIMUM_LANGUAGE_WIDTH, MERGE_SMALL_START);
	fill_row(arena, fixture->model_weights.image_newline, 0,
		 VISION_MAXIMUM_LANGUAGE_WIDTH, MERGE_SMALL_NEWLINE);
	fill_row(arena, fixture->model_weights.image_end, 0,
		 VISION_MAXIMUM_LANGUAGE_WIDTH, MERGE_SMALL_END);
	fill_row(arena, fixture->model_encoded[0], 0,
		 VISION_MAXIMUM_LANGUAGE_WIDTH, MERGE_MODEL_FIRST_CELL);
	fill_row(arena, fixture->model_encoded[1], 0,
		 VISION_MAXIMUM_LANGUAGE_WIDTH, MERGE_MODEL_SECOND_CELL);
	struct vision_session metadata = {
		.image_count = TEST_VISION_MERGE_MODEL_IMAGES
	};
	for (uint32_t image = 0; image < TEST_VISION_MERGE_MODEL_IMAGES;
	     image++) {
		metadata.spans[image] = (struct vision_span){
			.token_begin = MERGE_MODEL_FIRST_POSITION +
				       image * MERGE_MODEL_SPAN_TOKENS,
			.token_count = MERGE_MODEL_SPAN_TOKENS,
			.embedding_row = image * VISION_MAXIMUM_TOKENS
		};
	}
	memset(arena + fixture->metadata.byte_offset, 0,
	       VISION_SESSION_METADATA_BYTES);
	memcpy(arena + fixture->metadata.byte_offset, &metadata,
	       sizeof(metadata));
	const struct step_block block = {
		.sequence = 1, .row_count = TEST_VISION_MERGE_ACTIVE_ROWS
	};
	memcpy(arena + fixture->step.byte_offset, &block, sizeof(block));
	for (uint32_t row = 0; row < TEST_VISION_MERGE_STEP_ROWS; row++) {
		const struct step_row_record value = {
			.position = (int32_t)(MERGE_TEXT_POSITION + row),
			.token_id = row == 0 ? 1 : (int32_t)IMAGE_TOKEN_ID,
			.session_slot = 0,
			.flags = row == 0 ? 0u : STEP_ROW_IMAGE
		};
		memcpy(arena + fixture->step.byte_offset +
			       STEP_BLOCK_BYTES_VALUE +
			       (uint64_t)row * STEP_ROW_BYTES_VALUE,
		       &value, sizeof(value));
	}
}

static bool row_matches(const unsigned char *arena, struct linear_range range,
			uint32_t row, uint32_t width, uint32_t base)
{
	const uint16_t *const values = test_vision_const_at(
		arena,
		range.byte_offset + (uint64_t)row * width * sizeof(uint16_t));
	for (uint32_t channel = 0; channel < width; channel++) {
		const uint16_t expected = base == MERGE_UNTOUCHED_WORD ?
						  MERGE_UNTOUCHED_WORD :
						  pattern(base, channel);
		if (values[channel] != expected) {
			return false;
		}
	}
	return true;
}

bool test_vision_merge_fixture_matches(
	const struct test_vision_merge_fixture *fixture,
	const unsigned char *arena, const unsigned char *initial)
{
	static const uint32_t small_bases[TEST_VISION_MERGE_SMALL_TOKENS] = {
		MERGE_SMALL_START,
		MERGE_SMALL_CELL,
		MERGE_SMALL_CELL + MERGE_CELL_PATTERN_STEP,
		MERGE_SMALL_CELL + 2u * MERGE_CELL_PATTERN_STEP,
		MERGE_SMALL_NEWLINE,
		MERGE_SMALL_CELL + 3u * MERGE_CELL_PATTERN_STEP,
		MERGE_SMALL_CELL + 4u * MERGE_CELL_PATTERN_STEP,
		MERGE_SMALL_CELL + 5u * MERGE_CELL_PATTERN_STEP,
		MERGE_SMALL_NEWLINE,
		MERGE_SMALL_END
	};
	for (uint32_t token = 0; token < TEST_VISION_MERGE_SMALL_TOKENS;
	     token++) {
		if (!row_matches(arena, fixture->small_output, token,
				 TEST_VISION_MERGE_SMALL_WIDTH,
				 small_bases[token])) {
			return false;
		}
		for (uint32_t copy = 0; copy < VISION_MAXIMUM_COPIES; copy++) {
			if (!row_matches(arena, fixture->small_copies,
					 token * VISION_MAXIMUM_COPIES + copy,
					 TEST_VISION_MERGE_SMALL_WIDTH,
					 small_bases[token])) {
				return false;
			}
		}
	}
	if (fixture->includes_model) {
		static const uint32_t
			model_bases[TEST_VISION_MERGE_MODEL_IMAGES]
				   [MERGE_MODEL_SPAN_TOKENS] = {
					   { MERGE_SMALL_START,
					     MERGE_MODEL_FIRST_CELL,
					     MERGE_SMALL_NEWLINE,
					     MERGE_SMALL_END },
					   { MERGE_SMALL_START,
					     MERGE_MODEL_SECOND_CELL,
					     MERGE_SMALL_NEWLINE,
					     MERGE_SMALL_END }
				   };
		static const uint32_t step_bases[TEST_VISION_MERGE_STEP_ROWS] = {
			MERGE_UNTOUCHED_WORD,	 MERGE_SMALL_START,
			MERGE_MODEL_FIRST_CELL,	 MERGE_SMALL_NEWLINE,
			MERGE_SMALL_END,	 MERGE_SMALL_START,
			MERGE_MODEL_SECOND_CELL, MERGE_UNTOUCHED_WORD
		};
		for (uint32_t image = 0; image < TEST_VISION_MERGE_MODEL_IMAGES;
		     image++) {
			for (uint32_t token = 0;
			     token < MERGE_MODEL_SPAN_TOKENS; token++) {
				if (!row_matches(arena, fixture->spans[image],
						 token,
						 VISION_MAXIMUM_LANGUAGE_WIDTH,
						 model_bases[image][token])) {
					return false;
				}
			}
		}
		for (uint32_t row = 0; row < TEST_VISION_MERGE_STEP_ROWS;
		     row++) {
			for (uint32_t copy = 0; copy < VISION_MAXIMUM_COPIES;
			     copy++) {
				if (!row_matches(arena, fixture->streams,
						 row * VISION_MAXIMUM_COPIES +
							 copy,
						 VISION_MAXIMUM_LANGUAGE_WIDTH,
						 step_bases[row])) {
					return false;
				}
			}
		}
	}
	const struct linear_range writes[MERGE_WRITTEN_RANGES] = {
		fixture->small_output, fixture->small_copies, fixture->spans[0],
		fixture->spans[1], fixture->streams
	};
	const uint64_t bytes = fixture->includes_model ?
				       TEST_VISION_MERGE_CARD_ARENA_BYTES :
				       TEST_VISION_MERGE_SMALL_ARENA_BYTES;
	for (uint64_t offset = 0; offset < bytes; offset++) {
		bool written = false;
		for (uint32_t range = 0; range < MERGE_WRITTEN_RANGES;
		     range++) {
			written = written ||
				  (offset >= writes[range].byte_offset &&
				   offset - writes[range].byte_offset <
					   writes[range].byte_count);
		}
		if (!written && arena[offset] != initial[offset]) {
			return false;
		}
	}
	return true;
}
