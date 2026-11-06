/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef TEST_VISION_FIXTURE_H
#define TEST_VISION_FIXTURE_H

#include <stdbool.h>
#include "anemone/vision.h"

#define TEST_VISION_CASES 4u
#define TEST_VISION_ARENA_BYTES 131072u
#define TEST_VISION_GOLDEN_TOLERANCE 0.0078125f
#define TEST_VISION_GOLDEN_ABSOLUTE 0.000244140625f
/* torch.testing._comparison.default_tolerances(torch.bfloat16), pinned in the fixture manifest. */
#define TEST_VISION_DEFAULT_RELATIVE 0.016f
#define TEST_VISION_DEFAULT_ABSOLUTE 1e-5f

struct test_vision_fixture {
	unsigned char *arena;
	struct vision_description description;
	struct vision_weights weights;
	struct vision_kernels kernels;
	struct kernel_launch launches[VISION_MAXIMUM_LAUNCHES];
	uint32_t launch_count;
	uint32_t case_index;
};

bool test_vision_fixture_create(uint32_t case_index,
				struct test_vision_fixture *fixture);
void test_vision_fixture_destroy(struct test_vision_fixture *fixture);
bool test_vision_fixture_math_golden(const struct test_vision_fixture *fixture,
				     const unsigned char *arena);
bool test_vision_fixture_golden(const struct test_vision_fixture *fixture,
				const unsigned char *arena);

#endif
