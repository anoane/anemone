/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "anemone/official_index.h"

#define TEST_ARENA_BYTES (1ull << 20u)
#define TEST_STEP_OFFSET 4096u
#define TEST_KEYS_OFFSET (1ull << 16u)
#define TEST_QUERIES_OFFSET (4ull << 16u)
#define TEST_WEIGHTS_OFFSET (5ull << 16u)
#define TEST_SCORES_OFFSET (6ull << 16u)
#define TEST_SELECTION_OFFSET (7ull << 16u)
#define TEST_CANDIDATES_OFFSET (8ull << 16u)
#define TEST_HANDLE 17u
#define TEST_UNTOUCHED 0xa5
#define TEST_UNADMITTED_ROWS 2u
#define TEST_UNADMITTED_RATIO 3u
#define TEST_CONTEXT_BEYOND_SELECTION (MODEL_SELECTED * 2u)

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", message);
	}
}

static struct index_scan_description description_of(uint32_t ratio,
						    uint32_t role)
{
	return (struct index_scan_description){
		.step = { TEST_STEP_OFFSET,
			  STEP_BLOCK_BYTES_VALUE + STEP_ROW_BYTES_VALUE },
		.keys = { TEST_KEYS_OFFSET,
			  (uint64_t)MODEL_SELECTED / ratio * MODEL_KEY_BYTES },
		.queries = { TEST_QUERIES_OFFSET,
			     MODEL_INDEX_QUERIES_ROW_BYTES },
		.weights = { TEST_WEIGHTS_OFFSET,
			     MODEL_INDEX_WEIGHTS_ROW_BYTES },
		.scores = { TEST_SCORES_OFFSET,
			    (uint64_t)MODEL_CONTEXT_WITHOUT_CANDIDATES *
				    INDEX_SCORE_BYTES },
		.selection = { TEST_SELECTION_OFFSET,
			       MODEL_SELECTION_ROW_BYTES },
		.candidates = { TEST_CANDIDATES_OFFSET,
				MODEL_CANDIDATES_ROW_BYTES },
		.context_token_limit = MODEL_SELECTED,
		.session_capacity = 1,
		.ratio = ratio,
		.row_count = 1,
		.candidate_role = role
	};
}

static void refused(const struct index_scan_description *description,
		    uint64_t arena_bytes, uint32_t kernel)
{
	struct kernel_launch launch;
	memset(&launch, TEST_UNTOUCHED, sizeof(launch));
	const struct kernel_launch before = launch;
	expect(official_index_prepare(description, arena_bytes, kernel,
				      &launch) == STATUS_INVALID_ARGUMENT,
	       "unsupported or unsafe all-visible descriptor refused");
	expect(memcmp(&launch, &before, sizeof(launch)) == 0,
	       "refusal preserves the published launch");
}

int main(void)
{
	for (uint32_t role = INDEX_CANDIDATES_NONE;
	     role <= INDEX_CANDIDATES_READER; role++) {
		for (uint32_t ratio = MODEL_DIRECT_RATIO;
		     ratio <= MODEL_POOLED_RATIO; ratio++) {
			if (role != INDEX_CANDIDATES_NONE &&
			    ratio != MODEL_DIRECT_RATIO)
				continue;
			const struct index_scan_description description =
				description_of(ratio, role);
			struct kernel_launch launch;
			const result_status status = official_index_prepare(
				&description, TEST_ARENA_BYTES, TEST_HANDLE,
				&launch);
			expect(status == STATUS_OK,
			       "source and reader descriptions retain bounded selection");
			if (status != STATUS_OK)
				continue;
			struct official_index_arguments arguments;
			memcpy(&arguments, launch.argument, sizeof(arguments));
			expect(arguments.step_offset == TEST_STEP_OFFSET &&
				       arguments.selection_offset ==
					       TEST_SELECTION_OFFSET &&
				       arguments.context_token_limit ==
					       MODEL_SELECTED &&
				       arguments.ratio == ratio &&
				       arguments.candidates_offset ==
					       (role == INDEX_CANDIDATES_SOURCE ?
							TEST_CANDIDATES_OFFSET :
							INDEX_NO_CANDIDATES),
			       "only the candidate source owns candidate writes");
			expect(launch.kernel == TEST_HANDLE &&
				       launch.grid[0] == 1 &&
				       launch.grid[1] == 1 &&
				       launch.block[0] ==
					       OFFICIAL_INDEX_THREADS &&
				       launch.dynamic_shared_byte_count == 0 &&
				       launch.argument_byte_count ==
					       OFFICIAL_INDEX_ARGUMENT_BYTES,
			       "one prepared pass reads no score scratch");
		}
	}
	struct index_scan_description description =
		description_of(MODEL_DIRECT_RATIO, INDEX_CANDIDATES_SOURCE);
	refused(NULL, TEST_ARENA_BYTES, TEST_HANDLE);
	refused(&description, TEST_ARENA_BYTES, BACKEND_NO_KERNEL);
	refused(&description, 1, TEST_HANDLE);
	description.selection.byte_count--;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of(1, INDEX_CANDIDATES_SOURCE);
	description.candidates.byte_count--;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of(1, INDEX_CANDIDATES_SOURCE);
	description.selection = description.candidates;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of(1, INDEX_CANDIDATES_SOURCE);
	description.row_count = TEST_UNADMITTED_ROWS;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of(1, INDEX_CANDIDATES_SOURCE);
	description.session_capacity = TEST_UNADMITTED_ROWS;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of(1, INDEX_CANDIDATES_SOURCE);
	description.context_token_limit = TEST_CONTEXT_BEYOND_SELECTION;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of(1, INDEX_CANDIDATES_SOURCE);
	description.step_row_first = 1;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of(1, INDEX_CANDIDATES_SOURCE);
	description.split_rows = TEST_UNADMITTED_ROWS;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE);
	description = description_of(1, INDEX_CANDIDATES_SOURCE);
	description.ratio = TEST_UNADMITTED_RATIO;
	refused(&description, TEST_ARENA_BYTES, TEST_HANDLE);
	(void)printf("official_index_test: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
