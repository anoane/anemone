/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* An engine-open failure must close the snapshot descriptor. */
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include "../../serving_fixture.h"

static int test_snapshot_descriptor = -1;
static int test_mkstemp(char *path)
{
	test_snapshot_descriptor = mkstemp(path);
	return test_snapshot_descriptor;
}

MUST_CHECK static result_status
test_open_failure(uint32_t context_tokens, const uint32_t *const *scripts,
		  uint32_t script_count, struct inference_engine **output)
{
	(void)context_tokens;
	(void)scripts;
	(void)script_count;
	(void)output;
	return STATUS_OUT_OF_MEMORY;
}

int test_embedded_engine_main(void);
#define mkstemp test_mkstemp
#define test_serving_open test_open_failure
#define main test_embedded_engine_main
#include "../../engine_test.c"
#undef main
#undef test_serving_open
#undef mkstemp

int main(void)
{
	test_cone();
	const bool closed = test_snapshot_descriptor >= 0 &&
			    fcntl(test_snapshot_descriptor, F_GETFD) < 0;
	if (!closed && test_snapshot_descriptor >= 0) {
		(void)close(test_snapshot_descriptor);
	}
	return closed ? 0 : 1;
}
