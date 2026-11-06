/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Shared synthetic merge fixture: small learned spans and two model-width image spans in one session. */
#ifndef TEST_VISION_MERGE_FIXTURE_H
#define TEST_VISION_MERGE_FIXTURE_H

#include <stdbool.h>
#include "anemone/vision.h"

#define TEST_VISION_MERGE_SMALL_ARENA_BYTES 8192u
#define TEST_VISION_MERGE_CARD_ARENA_BYTES (96ull << 20u)
#define TEST_VISION_MERGE_SMALL_WIDTH 16u
#define TEST_VISION_MERGE_SMALL_HEIGHT 2u
#define TEST_VISION_MERGE_SMALL_COLUMNS 3u
#define TEST_VISION_MERGE_SMALL_TOKENS 10u
#define TEST_VISION_MERGE_SMALL_LAUNCHES 2u
#define TEST_VISION_MERGE_MODEL_IMAGES 2u
#define TEST_VISION_MERGE_LAUNCHES 5u
#define TEST_VISION_MERGE_STEP_ROWS 8u
#define TEST_VISION_MERGE_ACTIVE_ROWS 7u
#define TEST_VISION_MERGE_SENTINEL 0xA5u

struct test_vision_merge_fixture {
	struct vision_weights small_weights;
	struct linear_range small_encoded;
	struct linear_range small_output;
	struct linear_range small_copies;
	struct vision_weights model_weights;
	struct linear_range model_encoded[TEST_VISION_MERGE_MODEL_IMAGES];
	struct linear_range embeddings;
	struct linear_range spans[TEST_VISION_MERGE_MODEL_IMAGES];
	struct linear_range metadata;
	struct linear_range step;
	struct linear_range streams;
	struct kernel_launch launches[TEST_VISION_MERGE_LAUNCHES];
	uint32_t launch_count;
	bool includes_model;
};

bool test_vision_merge_fixture_prepare(
	bool include_model, const struct vision_kernels *kernels,
	struct test_vision_merge_fixture *fixture_out);
void test_vision_merge_fixture_fill(
	const struct test_vision_merge_fixture *fixture, unsigned char *arena);
bool test_vision_merge_fixture_matches(
	const struct test_vision_merge_fixture *fixture,
	const unsigned char *arena, const unsigned char *initial);

#endif
