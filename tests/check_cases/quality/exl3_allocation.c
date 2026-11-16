/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Deny either host allocation before this fixture can submit work to a card. */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned test_malloc_calls;
static unsigned test_failed_malloc;
static void *test_case_malloc(size_t bytes)
{
	test_malloc_calls++;
	return test_malloc_calls == test_failed_malloc ? NULL : malloc(bytes);
}

int test_embedded_card_main(void);
#define malloc test_case_malloc
#define main test_embedded_card_main
#include "../../exl3_tc_card_test.c"
#undef main
#undef malloc

#define TEST_ARENA_BYTES (16u << 20u)
#define TEST_WORK_BYTES (8u << 20u)
#define TEST_ALLOCATION_COUNT 2u

int main(int count, char **arguments)
{
	test_failed_malloc = count > 1 && arguments[1][0] == '2' ?
				     TEST_ALLOCATION_COUNT :
				     1u;
	struct test_linear_card card = { .arena_byte_count = TEST_ARENA_BYTES,
					 .work_capacity = TEST_WORK_BYTES,
					 .image = calloc(1, TEST_ARENA_BYTES) };
	if (card.image == NULL) {
		return 1;
	}
	const struct test_case one = { .rows = 1, .live = 1, .inner = 1 };
	const bool ran = case_run(&card, &one, 0);
	free(card.image);
	return !ran && test_malloc_calls == TEST_ALLOCATION_COUNT &&
			       test_linear_failures() == 0 ?
		       0 :
		       1;
}
