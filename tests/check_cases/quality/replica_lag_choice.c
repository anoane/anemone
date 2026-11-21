/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Isolated private-helper regression (turbo engram review, findings 15 and 17): a copy run that writes no target
 * retains no source buffer (one a worker and one more, and no memory refusal whatever the memory); one that writes
 * retains the lag, and is refused when the memory given does not hold it with the reserve (no host's MemAvailable is
 * read); the lag test passes over a failed target, and with relax it takes a chunk whose lag has passed before the
 * oldest. */
#include "../../../source/engine/storage/replicas/replica_prepare.c"

#define TEST_CHUNKS 4u
#define TEST_CHUNK_BYTES 4096u
#define TEST_WORKERS 2u
#define TEST_LAG_BYTES 100u
#define TEST_CHUNK_WRITTEN 10u
#define TEST_WRITTEN_FAR 1000u /* far past any chunk's lag */

static unsigned failures;

static void expect(bool passed, const char *message)
{
	if (!passed) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

#define TEST_MEMORY_NONE 0u
#define TEST_MEMORY_PLENTY UINT64_MAX

static uint32_t buffers_for(struct prepare_copy_options *options,
			    bool second_final, uint64_t available)
{
	static struct copy_shared shared;
	memset(&shared, 0, sizeof(shared));
	shared.options = options;
	shared.alignment = TEST_CHUNK_BYTES;
	shared.targets[0].finalized = true;
	shared.targets[1].finalized = second_final;
	const bool opened = buffers_open(&shared, available);
	const uint32_t count = opened ? shared.buffer_count : 0u;
	buffers_close(&shared);
	return count;
}

int main(void)
{
	struct prepare_manifest *const manifest = calloc(1, sizeof(*manifest));
	if (manifest == NULL)
		return 1;
	manifest->chunk_bytes = TEST_CHUNK_BYTES;
	manifest->chunks = manifest->known = TEST_CHUNKS;
	manifest->bytes = (uint64_t)TEST_CHUNKS * TEST_CHUNK_BYTES;
	struct prepare_target targets[PREPARE_TARGETS_MOST] = { 0 };
	targets[0].included = targets[1].included = true;
	struct prepare_copy_options options = { .manifest = manifest,
						.targets = targets,
						.count = 2,
						.workers = TEST_WORKERS };
	/* finding 15: finals only, nothing written: one buffer a worker and one more, with no memory at all */
	expect(buffers_for(&options, true, TEST_MEMORY_NONE) ==
		       TEST_WORKERS + 1u,
	       "a run over finals only retains no buffer and checks no memory");
	/* a target to write: the lag (here the whole copy) as well, when the memory holds it */
	expect(buffers_for(&options, false, TEST_MEMORY_PLENTY) ==
		       TEST_CHUNKS + TEST_WORKERS + 1u,
	       "a run that writes retains the lag");
	targets[0].reason = targets[1].reason = NULL;
	expect(buffers_for(&options, false, TEST_MEMORY_NONE) == 0u &&
		       targets[1].reason != NULL &&
		       strcmp(targets[1].reason, "insufficient_memory") == 0,
	       "a run that writes is refused when the memory does not hold its buffers");
	/* finding 17 */
	static struct copy_shared shared;
	memset(&shared, 0, sizeof(shared));
	shared.options = &options;
	shared.lag_bytes = TEST_LAG_BYTES;
	struct retained_chunk retained[2];
	memset(retained, 0, sizeof(retained));
	shared.retained = retained;
	/* a chunk written to both targets; the second failed and is written no more */
	retained[0].bytes = TEST_CHUNK_WRITTEN;
	retained[0].targets[0] = retained[0].targets[1] = true;
	shared.retained_count = 1;
	shared.target_written[0] = TEST_WRITTEN_FAR;
	shared.target_written[1] = TEST_CHUNK_WRITTEN;
	shared.targets[1].failed = true;
	bool inside = true;
	expect(retained_ready(&shared, false, &inside) == 0 && !inside,
	       "a failed target does not hold a chunk's lag");
	/* the oldest chunk went to a target written no further, the next to one far past its lag: relax takes the
	 * lagged one first, outside the lag */
	shared.targets[1].failed = false;
	retained[0].targets[1] = false;
	retained[0].written_at[0] = TEST_WRITTEN_FAR;
	retained[1].bytes = TEST_CHUNK_WRITTEN;
	retained[1].targets[1] = true;
	shared.retained_count = 2;
	shared.target_written[1] = TEST_WRITTEN_FAR;
	inside = true;
	expect(retained_ready(&shared, true, &inside) == 1 && !inside,
	       "relax takes a lagged chunk before the oldest");
	inside = false;
	expect(retained_ready(&shared, false, &inside) == 1,
	       "without relax the lagged chunk is ready too");
	retained[1].targets[1] = false;
	retained[1].targets[0] = true;
	retained[1].written_at[0] = TEST_WRITTEN_FAR;
	expect(retained_ready(&shared, true, &inside) == 0 && inside,
	       "with no lagged chunk relax takes the oldest, inside the lag");
	expect(retained_ready(&shared, false, &inside) == -1,
	       "and without relax none is ready");
	free(manifest);
	if (failures != 0)
		return 1;
	(void)puts("replica_lag_choice: passed");
	return 0;
}
