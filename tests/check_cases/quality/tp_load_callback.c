/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* A backend without private staging slots uses the caller's existing buffers. */
#include "../../../source/engine/storage/load/tp_load.c"

int test_embedded_tp_load_main(void);
#define main test_embedded_tp_load_main
#include "../../engine/storage/load/tp_load_test.c"
#undef main

int main(void)
{
	test_workers = 1;
	retiring_backend = test_host_memory_backend;
	retiring_backend.staging_retire = retire_later;
	retiring_backend.staging_allocate = NULL;
	struct fixture fixture;
	if (!fixture_open(&fixture)) {
		fixture_close(&fixture);
		return 1;
	}
	struct tp_load_result result = { 0 };
	const result_status status = tp_load_run(&fixture.request, &result);
	if (status == STATUS_OK) {
		output_check(&fixture);
	}
	fixture_close(&fixture);
	return status == STATUS_OK && failures == 0 ? 0 : 1;
}
