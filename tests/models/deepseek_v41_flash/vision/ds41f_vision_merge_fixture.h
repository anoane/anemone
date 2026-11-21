/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Shared synthetic merge fixture: small learned spans and two model-width image spans in one session. */
#ifndef DS41F_VISION_MERGE_FIXTURE_H
#define DS41F_VISION_MERGE_FIXTURE_H

#include <stdbool.h>
#include "models/deepseek_v41_flash/vision/ds41f_vision.h"

#define DS41F_TEST_VISION_MERGE_SMALL_ARENA_BYTES 8192u
#define DS41F_TEST_VISION_MERGE_CARD_ARENA_BYTES (96ull << 20u)
#define DS41F_TEST_VISION_MERGE_SMALL_WIDTH 16u
#define DS41F_TEST_VISION_MERGE_SMALL_HEIGHT 2u
#define DS41F_TEST_VISION_MERGE_SMALL_COLUMNS 3u
#define DS41F_TEST_VISION_MERGE_SMALL_TOKENS 10u
#define DS41F_TEST_VISION_MERGE_SMALL_LAUNCHES 2u
#define DS41F_TEST_VISION_MERGE_MODEL_IMAGES 2u
#define DS41F_TEST_VISION_MERGE_LAUNCHES 5u
#define DS41F_TEST_VISION_MERGE_STEP_ROWS 8u
#define DS41F_TEST_VISION_MERGE_ACTIVE_ROWS 7u
#define DS41F_TEST_VISION_MERGE_SENTINEL 0xA5u

struct ds41f_test_vision_merge_fixture {
	struct ds41f_vision_weights small_weights;
	struct linear_range small_encoded;
	struct linear_range small_output;
	struct linear_range small_copies;
	struct ds41f_vision_weights model_weights;
	struct linear_range model_encoded[DS41F_TEST_VISION_MERGE_MODEL_IMAGES];
	struct linear_range embeddings;
	struct linear_range spans[DS41F_TEST_VISION_MERGE_MODEL_IMAGES];
	struct linear_range metadata;
	struct linear_range step;
	struct linear_range streams;
	struct kernel_launch launches[DS41F_TEST_VISION_MERGE_LAUNCHES];
	uint32_t launch_count;
	bool includes_model;
};

bool ds41f_test_vision_merge_fixture_prepare(
	bool include_model, const struct ds41f_vision_kernels *kernels,
	struct ds41f_test_vision_merge_fixture *fixture_out);
void ds41f_test_vision_merge_fixture_fill(
	const struct ds41f_test_vision_merge_fixture *fixture,
	unsigned char *arena);
bool ds41f_test_vision_merge_fixture_matches(
	const struct ds41f_test_vision_merge_fixture *fixture,
	const unsigned char *arena, const unsigned char *initial);

#endif
