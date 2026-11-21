/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef DS41F_VISION_FIXTURE_H
#define DS41F_VISION_FIXTURE_H

#include <stdbool.h>
#include "models/deepseek_v41_flash/vision/ds41f_vision.h"

#define DS41F_TEST_VISION_CASES 4u
#define DS41F_TEST_VISION_ARENA_BYTES 131072u
#define DS41F_TEST_VISION_GOLDEN_TOLERANCE 0.0078125f
#define DS41F_TEST_VISION_GOLDEN_ABSOLUTE 0.000244140625f
/* torch.testing._comparison.default_tolerances(torch.bfloat16), pinned in the fixture manifest. */
#define DS41F_TEST_VISION_DEFAULT_RELATIVE 0.016f
#define DS41F_TEST_VISION_DEFAULT_ABSOLUTE 1e-5f

struct ds41f_test_vision_fixture {
	unsigned char *arena;
	struct ds41f_vision_description description;
	struct ds41f_vision_weights weights;
	struct ds41f_vision_kernels kernels;
	struct kernel_launch launches[DS41F_VISION_MAXIMUM_LAUNCHES];
	uint32_t launch_count;
	uint32_t case_index;
};

bool ds41f_test_vision_fixture_create(
	uint32_t case_index, struct ds41f_test_vision_fixture *fixture);
void ds41f_test_vision_fixture_destroy(
	struct ds41f_test_vision_fixture *fixture);
bool ds41f_test_vision_fixture_math_golden(
	const struct ds41f_test_vision_fixture *fixture,
	const unsigned char *arena);
bool ds41f_test_vision_fixture_golden(
	const struct ds41f_test_vision_fixture *fixture,
	const unsigned char *arena);

#endif
