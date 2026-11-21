/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Refused preparations must never have their unwritten launch output read. */
#include <string.h>
#include "models/deepseek_v41_flash/moe/ds41f_router.h"

static unsigned test_comparisons;
static int test_unwritten_compare(const void *first, const void *second,
				  size_t size)
{
	(void)first;
	(void)second;
	(void)size;
	test_comparisons++;
	return 1;
}

MUST_CHECK static result_status
test_prepare_failure(const struct router_wide_description *description,
		     uint64_t arena_bytes,
		     const struct router_kernels_record *kernels,
		     struct kernel_launch *launches)
{
	(void)description;
	(void)arena_bytes;
	(void)kernels;
	(void)launches;
	return STATUS_INVALID_ARGUMENT;
}

int test_embedded_router_main(void);
#define memcmp test_unwritten_compare
#define router_wide_small_prepare test_prepare_failure
#define main test_embedded_router_main
#include "../../engine/library/moe/moe_dispatch_test.c"
#undef main
#undef router_wide_small_prepare
#undef memcmp

int main(void)
{
	test_small_launches();
	return test_comparisons == 0 ? 0 : 1;
}
