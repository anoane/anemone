/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Compile this isolated fixture in place of server_sessions.c. */
#include <stdio.h>
#include <stdlib.h>

static unsigned test_token_allocations;
static void *test_snapshot_malloc(size_t bytes)
{
	test_token_allocations++;
	return malloc(bytes);
}

#define malloc test_snapshot_malloc
#include "../../../source/engine/server/server_sessions.c"
#undef malloc

#define TEST_PATH_BYTES 4096u
#define TEST_FILE_MODE 0600

int main(void)
{
	char directory[] = "/tmp/anemone_snapshot_size_XXXXXX";
	if (mkdtemp(directory) == NULL) {
		return 1;
	}
	struct session_store *const store = calloc(1, sizeof(*store));
	if (store == NULL) {
		(void)rmdir(directory);
		return 1;
	}
	store->identity = 1;
	store->context_limit = 1;
	const int copied = snprintf(store->directory, sizeof(store->directory),
				    "%s", directory);
	char path[TEST_PATH_BYTES];
	const int length =
		snprintf(path, sizeof(path), "%s/wrapped.snapshot", directory);
	bool passed = copied >= 0 &&
		      (size_t)copied < sizeof(store->directory) &&
		      length >= 0 && (size_t)length < sizeof(path);
	struct snapshot_head head = { .identity = 1,
				      .payload_bytes = UINT64_MAX -
						       sizeof(uint32_t) + 1u,
				      .version = SNAPSHOT_VERSION,
				      .token_count = 1,
				      .reach_count = 1,
				      .cache_namespace = { .byte_count = 1 } };
	memcpy(head.magic, SNAPSHOT_MAGIC, sizeof(head.magic));
	const int descriptor = passed ? open(path, O_WRONLY | O_CREAT | O_EXCL,
					     TEST_FILE_MODE) :
					-1;
	passed = passed && descriptor >= 0;
	if (descriptor >= 0) {
		const bool written = write(descriptor, &head, sizeof(head)) ==
				     (ssize_t)sizeof(head);
		const bool closed = close(descriptor) == 0;
		passed = passed && written && closed;
	}
	if (passed) {
		snapshot_read(store, "wrapped.snapshot");
		passed = store->snapshot_count == 0 &&
			 test_token_allocations == 0;
	}
	(void)unlink(path);
	free(store);
	(void)rmdir(directory);
	if (!passed) {
		(void)fprintf(stderr,
			      "wrapped snapshot allocated %u token buffers\n",
			      test_token_allocations);
	}
	return passed ? 0 : 1;
}
