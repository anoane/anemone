/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Exact legacy/split selection comparison, including raw FP16 order keys and deterministic scalar oracles.
 * --deep adds 512k/1M score rows; --wide adds 16/32-row deep cases. No keys or model weights are allocated.
 * Both output arrays and every inactive row start as canaries before every pair of launches. */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "models/deepseek_v41_flash/ds41f_geometry.h"
#include "engine/program/step_records.h"
#include "ds41f_index_select_split_fixture.h"

#define ARENA_BYTES (32ull << 20u)
#define WIDE_ARENA_BYTES (96ull << 20u)
#define THIN_ROWS LINEAR_MAXIMUM_ROWS
#define MAX_ROWS 32u
#define ORDER_KEYS 65536u
#define HALF_SIGN 0x8000u
#define HALF_MASK 0xFFFFu
#define HALF_INFINITY 0x7C00u
#define POSITIVE_MASK 0x7BFFu
#define TIE_LEVELS 17u
#define HALF_NEGATIVE_ZERO 0x8000u
#define PATTERN_SEED 0xA54FF53Au
#define HASH_SHIFT_FIRST 16u
#define HASH_SHIFT_LAST 15u
#define HASH_MULTIPLIER_FIRST 0x7FEB352Du
#define HASH_MULTIPLIER_SECOND 0x846CA68Bu
#define MIXED_SESSION_STRIDE 3u
#define MAPPED_BLOCK_STRIDE 3u
#define SMALL_ENTRIES 32768u
#define DEEP_HALF_ENTRIES 524288u
#define DEEP_ENTRIES 1048576u
#define PROCESS_SECONDS 900u
#define NEGATIVE_EXTENT 1024u
#define NEGATIVE_BLOCK 3000u
#define HALF_COUNT_DIVISOR 2u
#define WHOLE_WORD_BYTES sizeof(uint32_t)

enum pattern {
	RANDOM_WORDS,
	ALL_EQUAL,
	CUT_TIES,
	INFINITY_TAIL,
	NEWEST_POOR,
	SIGNED_ZEROS,
	PATTERNS
};
static const uint32_t edges[] = { 0,	 1,	7,     8,     9,    255,
				  256,	 511,	512,   513,   1023, 1024,
				  16383, 16384, 16385, 32767, 32768 };
static const uint32_t wide_rows[] = { 16, 32 };
static uint32_t case_count;

static uint32_t mix(uint32_t value)
{
	value ^= value >> HASH_SHIFT_FIRST;
	value *= HASH_MULTIPLIER_FIRST;
	value ^= value >> HASH_SHIFT_LAST;
	value *= HASH_MULTIPLIER_SECOND;
	return value ^ (value >> HASH_SHIFT_FIRST);
}

static uint32_t order_key(uint16_t word)
{
	return (word & HALF_SIGN) != 0 ? ((uint32_t)word ^ HALF_MASK) :
					 ((uint32_t)word ^ HALF_SIGN);
}

static uint32_t element_key(const uint16_t *scores, uint32_t element,
			    uint32_t count, bool is_block)
{
	if (!is_block) {
		return order_key(scores[element]);
	}
	if (element + 1u == count) {
		return HALF_MASK;
	}
	uint32_t largest = 0;
	for (uint32_t entry = 0; entry < DS41F_KEYS_PER_TILE; entry++) {
		const uint32_t key = order_key(
			scores[element * DS41F_KEYS_PER_TILE + entry]);
		largest = key > largest ? key : largest;
	}
	return largest;
}

/* Independent single full-key histogram; no split, byte cuts, CUDA voting or warp-prefix logic. */
static void oracle(const uint16_t *scores, uint32_t extent, bool is_block,
		   const uint32_t *mapped, uint32_t *output)
{
	uint32_t bins[ORDER_KEYS] = { 0 };
	const uint32_t capacity = is_block ? DS41F_CANDIDATE_BLOCKS :
					     DS41F_SELECTED;
	const uint32_t count = is_block ? (extent + DS41F_KEYS_PER_TILE - 1u) /
						  DS41F_KEYS_PER_TILE :
					  extent;
	for (uint32_t element = 0; element < count; element++) {
		bins[element_key(scores, element, count, is_block)]++;
	}
	uint32_t need = count < capacity ? count : capacity,
		 cut = ORDER_KEYS - 1u;
	while (cut != 0 && bins[cut] < need) {
		need -= bins[cut--];
	}
	uint32_t written = 0;
	for (uint32_t element = 0; element < count; element++) {
		const uint32_t key =
			element_key(scores, element, count, is_block);
		const bool tied = key == cut && need != 0;
		if (key > cut || tied) {
			DS41F_TEST_SPLIT_REQUIRE(written < capacity);
			output[written++] =
				mapped == NULL ?
					element :
					mapped[element / DS41F_KEYS_PER_TILE] *
							DS41F_KEYS_PER_TILE +
						element % DS41F_KEYS_PER_TILE;
			need -= tied ? 1u : 0u;
		}
	}
	while (written < capacity) {
		output[written++] = UINT32_MAX;
	}
}

static uint16_t score_word(enum pattern pattern, uint32_t row, uint32_t entry,
			   uint32_t extent)
{
	const uint32_t hash = mix(PATTERN_SEED + row * SMALL_ENTRIES + entry);
	switch (pattern) {
	case RANDOM_WORDS:
		return (uint16_t)hash;
	case ALL_EQUAL:
		return 0;
	case CUT_TIES:
		return (uint16_t)(entry % TIE_LEVELS);
	case INFINITY_TAIL:
		return entry >= extent / HALF_COUNT_DIVISOR ?
			       DS41F_INDEX_NEGATIVE_INFINITY_VALUE :
			       (uint16_t)(hash & POSITIVE_MASK);
	case NEWEST_POOR:
		return entry / DS41F_KEYS_PER_TILE ==
				       (extent - 1u) / DS41F_KEYS_PER_TILE ?
			       0 :
			       (uint16_t)(1u + entry % POSITIVE_MASK);
	case SIGNED_ZEROS:
		return (entry & 1u) != 0 ? HALF_NEGATIVE_ZERO : 0;
	case PATTERNS:
		break;
	}
	DS41F_TEST_SPLIT_REQUIRE(false);
	return 0;
}

static void compare_case(struct ds41f_test_split_fixture *fixture,
			 uint32_t entries, uint32_t rows, uint32_t live,
			 uint32_t ratio, uint32_t extent, bool distinct,
			 enum pattern first_pattern, bool is_reader)
{
	DS41F_TEST_SPLIT_REQUIRE(rows != 0 && rows <= MAX_ROWS && live <= rows);
	const uint32_t context = is_reader ? DEEP_ENTRIES : entries * ratio;
	const uint32_t score_entries =
		is_reader ? DS41F_CONTEXT_WITHOUT_CANDIDATES : entries;
	ds41f_test_split_layout(fixture, rows, score_entries);
	struct {
		struct step_block block;
		struct step_row_record rows[MAX_ROWS];
	} step = { .block = { .row_count = live } };
	uint16_t *scores = malloc((size_t)fixture->scores.byte_count);
	uint32_t *legacy = malloc((size_t)fixture->legacy.byte_count);
	uint32_t *split = malloc((size_t)fixture->split.byte_count);
	uint32_t mappings[MAX_ROWS][DS41F_CANDIDATE_BLOCKS];
	uint32_t expected[DS41F_CANDIDATE_BLOCKS];
	uint32_t extents[MAX_ROWS] = { 0 };
	if (scores == NULL || legacy == NULL || split == NULL) {
		(void)fprintf(stderr, "score/output host allocation failed\n");
		exit(EXIT_FAILURE);
	}
	for (uint32_t row = 0; row < rows; row++) {
		/* Adjacent positions have distinct causal extents; zero remains legal only at ratio two. */
		extents[row] = extent > row ? extent - row : extent;
		const uint32_t logical_extent = is_reader ? context - row :
							    extents[row];
		step.rows[row] = (struct step_row_record){
			.position =
				row >= live ?
					-1 :
				logical_extent == 0 ?
					0 :
					(int32_t)(logical_extent * ratio - 1u),
			.session_slot = row >= live ? UINT32_MAX :
					distinct ? row * MIXED_SESSION_STRIDE :
						   0
		};
		if (is_reader) {
			extents[row] = score_entries;
		}
		const enum pattern pattern = (enum pattern)(
			((uint32_t)first_pattern + row) % PATTERNS);
		for (uint32_t entry = 0; entry < score_entries; entry++) {
			scores[(size_t)row * score_entries + entry] =
				entry < extents[row] ?
					score_word(pattern, row, entry,
						   extents[row]) :
					DS41F_INDEX_NEGATIVE_INFINITY_VALUE;
		}
		for (uint32_t block = 0; block < DS41F_CANDIDATE_BLOCKS;
		     block++) {
			mappings[row][block] =
				block * MAPPED_BLOCK_STRIDE + row;
		}
	}
	ds41f_test_split_upload(fixture, fixture->step.byte_offset, &step,
				fixture->step.byte_count);
	ds41f_test_split_upload(fixture, fixture->scores.byte_offset, scores,
				fixture->scores.byte_count);
	ds41f_test_split_upload(fixture, fixture->candidates.byte_offset,
				mappings, fixture->candidates.byte_count);
	uint64_t score_digest = 0, score_after = 0;
	DS41F_TEST_SPLIT_REQUIRE(
		backend_cuda_value.digest(
			fixture->card, fixture->scores.byte_offset,
			fixture->scores.byte_count, DS41F_TEST_SPLIT_WAIT_NS,
			&score_digest) == STATUS_OK);
	for (uint32_t candidate = 0; candidate <= 1u; candidate++) {
		const bool is_block = candidate != 0;
		if (is_block && (ratio != 1u || is_reader)) {
			continue;
		}
		const uint32_t capacity = is_block ? DS41F_CANDIDATE_BLOCKS :
						     DS41F_SELECTED;
		ds41f_test_split_fill(fixture, fixture->legacy,
				      DS41F_TEST_SPLIT_GUARD_BYTE);
		ds41f_test_split_fill(fixture, fixture->split,
				      DS41F_TEST_SPLIT_GUARD_BYTE);
		ds41f_test_split_fill(fixture, fixture->scratch,
				      DS41F_TEST_SPLIT_GUARD_BYTE);
		struct kernel_launch legacy_launch,
			launches[DS41F_INDEX_SPLIT_LAUNCH_COUNT];
		ds41f_test_split_launches(fixture, rows, context, ratio,
					  is_block, is_reader,
					  DS41F_INDEX_SPLIT_SEGMENTS,
					  &legacy_launch, launches);
		ds41f_test_split_run(fixture, &legacy_launch, 1);
		ds41f_test_split_run(fixture, launches,
				     DS41F_INDEX_SPLIT_LAUNCH_COUNT);
		ds41f_test_split_download(fixture, fixture->legacy.byte_offset,
					  legacy, fixture->legacy.byte_count);
		ds41f_test_split_download(fixture, fixture->split.byte_offset,
					  split, fixture->split.byte_count);
		if (memcmp(legacy, split, (size_t)fixture->legacy.byte_count) !=
		    0) {
			(void)fprintf(
				stderr,
				"case entries=%u rows=%u live=%u ratio=%u extent=%u pattern=%u block=%u\n",
				entries, rows, live, ratio, extent,
				(uint32_t)first_pattern, candidate);
		}
		DS41F_TEST_SPLIT_REQUIRE(
			memcmp(legacy, split,
			       (size_t)fixture->legacy.byte_count) == 0);
		for (uint32_t row = 0; row < live; row++) {
			oracle(scores + (size_t)row * score_entries,
			       extents[row], is_block,
			       is_reader ? mappings[row] : NULL, expected);
			DS41F_TEST_SPLIT_REQUIRE(
				memcmp(split + (size_t)row * capacity, expected,
				       capacity * WHOLE_WORD_BYTES) == 0);
		}
		for (uint32_t word = live * capacity;
		     word < rows * DS41F_CANDIDATE_BLOCKS; word++) {
			DS41F_TEST_SPLIT_REQUIRE(split[word] ==
						 DS41F_TEST_SPLIT_GUARD_WORD);
		}
		ds41f_test_split_check_guards(fixture);
		if (live < rows) {
			const uint64_t inactive_bytes =
				(uint64_t)(rows - live) *
				DS41F_INDEX_SPLIT_ROW_BYTES;
			unsigned char *const inactive =
				malloc((size_t)inactive_bytes);
			if (inactive == NULL) {
				(void)fprintf(
					stderr,
					"inactive scratch allocation failed\n");
				exit(EXIT_FAILURE);
			}
			ds41f_test_split_download(
				fixture,
				fixture->scratch.byte_offset +
					(uint64_t)live *
						DS41F_INDEX_SPLIT_ROW_BYTES,
				inactive, inactive_bytes);
			for (uint64_t byte = 0; byte < inactive_bytes; byte++) {
				DS41F_TEST_SPLIT_REQUIRE(
					inactive[byte] ==
					DS41F_TEST_SPLIT_GUARD_BYTE);
			}
			free(inactive);
		}
		case_count++;
	}
	DS41F_TEST_SPLIT_REQUIRE(
		backend_cuda_value.digest(
			fixture->card, fixture->scores.byte_offset,
			fixture->scores.byte_count, DS41F_TEST_SPLIT_WAIT_NS,
			&score_after) == STATUS_OK);
	DS41F_TEST_SPLIT_REQUIRE(score_digest == score_after);
	free(scores);
	free(legacy);
	free(split);
}

/* A changed input must break equality with a saved legacy result: one score, then only the visible extent. */
static void negative_controls(struct ds41f_test_split_fixture *fixture)
{
	ds41f_test_split_layout(fixture, 1, SMALL_ENTRIES);
	struct {
		struct step_block block;
		struct step_row_record row;
	} step = { .block = { .row_count = 1 },
		   .row = { .position = NEGATIVE_EXTENT - 1u } };
	ds41f_test_split_fill(fixture, fixture->scores, 0);
	ds41f_test_split_upload(fixture, fixture->step.byte_offset, &step,
				sizeof(step));
	struct kernel_launch legacy, split[DS41F_INDEX_SPLIT_LAUNCH_COUNT];
	ds41f_test_split_launches(fixture, 1, SMALL_ENTRIES, 1, false, false,
				  DS41F_INDEX_SPLIT_SEGMENTS, &legacy, split);
	uint32_t expected[DS41F_SELECTED], actual[DS41F_SELECTED];
	ds41f_test_split_run(fixture, &legacy, 1);
	ds41f_test_split_download(fixture, fixture->legacy.byte_offset,
				  expected, sizeof(expected));
	const uint16_t high = HALF_INFINITY;
	ds41f_test_split_upload(fixture,
				fixture->scores.byte_offset +
					(NEGATIVE_EXTENT - 1u) * sizeof(high),
				&high, sizeof(high));
	ds41f_test_split_run(fixture, split, DS41F_INDEX_SPLIT_LAUNCH_COUNT);
	ds41f_test_split_download(fixture, fixture->split.byte_offset, actual,
				  sizeof(actual));
	DS41F_TEST_SPLIT_REQUIRE(
		memcmp(expected, actual, sizeof(actual)) != 0 &&
		actual[DS41F_SELECTED - 1u] == NEGATIVE_EXTENT - 1u);
	step.row.position = DS41F_SELECTED - HALF_COUNT_DIVISOR;
	ds41f_test_split_upload(fixture, fixture->step.byte_offset, &step,
				sizeof(step));
	ds41f_test_split_run(fixture, split, DS41F_INDEX_SPLIT_LAUNCH_COUNT);
	ds41f_test_split_download(fixture, fixture->split.byte_offset, actual,
				  sizeof(actual));
	DS41F_TEST_SPLIT_REQUIRE(memcmp(expected, actual, sizeof(actual)) !=
					 0 &&
				 actual[DS41F_SELECTED - 1u] == UINT32_MAX);
	uint32_t expected_blocks[DS41F_CANDIDATE_BLOCKS];
	uint32_t actual_blocks[DS41F_CANDIDATE_BLOCKS];
	step.row.position = SMALL_ENTRIES - 1u;
	ds41f_test_split_fill(fixture, fixture->scores, 0);
	ds41f_test_split_upload(fixture, fixture->step.byte_offset, &step,
				sizeof(step));
	ds41f_test_split_launches(fixture, 1, SMALL_ENTRIES, 1, true, false,
				  DS41F_INDEX_SPLIT_SEGMENTS, &legacy, split);
	ds41f_test_split_run(fixture, &legacy, 1);
	ds41f_test_split_download(fixture, fixture->legacy.byte_offset,
				  expected_blocks, sizeof(expected_blocks));
	ds41f_test_split_upload(fixture,
				fixture->scores.byte_offset +
					(uint64_t)NEGATIVE_BLOCK *
						DS41F_KEYS_PER_TILE *
						sizeof(high),
				&high, sizeof(high));
	ds41f_test_split_run(fixture, split, DS41F_INDEX_SPLIT_LAUNCH_COUNT);
	ds41f_test_split_download(fixture, fixture->split.byte_offset,
				  actual_blocks, sizeof(actual_blocks));
	bool found = false;
	for (uint32_t block = 0; block < DS41F_CANDIDATE_BLOCKS; block++) {
		found = found || actual_blocks[block] == NEGATIVE_BLOCK;
	}
	DS41F_TEST_SPLIT_REQUIRE(found && memcmp(expected_blocks, actual_blocks,
						 sizeof(actual_blocks)) != 0);
	ds41f_test_split_fill(fixture, fixture->scores, 0);
	step.row.position -= (int32_t)DS41F_KEYS_PER_TILE;
	ds41f_test_split_upload(fixture, fixture->step.byte_offset, &step,
				sizeof(step));
	ds41f_test_split_run(fixture, split, DS41F_INDEX_SPLIT_LAUNCH_COUNT);
	ds41f_test_split_download(fixture, fixture->split.byte_offset,
				  actual_blocks, sizeof(actual_blocks));
	DS41F_TEST_SPLIT_REQUIRE(memcmp(expected_blocks, actual_blocks,
					sizeof(actual_blocks)) != 0 &&
				 actual_blocks[DS41F_CANDIDATE_BLOCKS - 1u] ==
					 SMALL_ENTRIES / DS41F_KEYS_PER_TILE -
						 HALF_COUNT_DIVISOR);
	(void)printf(
		"NEGATIVE ordinary and candidate score/extent controls passed\n");
}

int main(int argc, char **argv)
{
	bool deep = false, native = false, wide = false;
	for (int argument = 1; argument < argc; argument++) {
		if (strcmp(argv[argument], "--deep") == 0) {
			deep = true;
		} else if (strcmp(argv[argument], "--native") == 0) {
			native = true;
		} else if (strcmp(argv[argument], "--wide") == 0) {
			wide = true;
		} else {
			(void)fprintf(
				stderr,
				"usage: %s [--deep] [--wide] [--native]\n",
				argv[0]);
			return EXIT_FAILURE;
		}
	}
	(void)alarm(PROCESS_SECONDS);
	struct card_description_record description;
	DS41F_TEST_SPLIT_REQUIRE(
		backend_cuda_value.card_describe(0, &description) == STATUS_OK);
	const uint32_t arithmetic_class =
		native ? ARITHMETIC_CLASS_SM_120_NATIVE :
			 description.arithmetic_class;
	struct ds41f_test_split_fixture fixture;
	ds41f_test_split_open(&fixture, wide ? WIDE_ARENA_BYTES : ARENA_BYTES,
			      arithmetic_class);
	for (uint32_t rows = 1; rows <= THIN_ROWS; rows++) {
		for (uint32_t edge = 0; edge < sizeof(edges) / sizeof(edges[0]);
		     edge++) {
			const uint32_t ratio = edges[edge] == 0 ?
						       DS41F_POOLED_RATIO :
						       DS41F_DIRECT_RATIO;
			compare_case(&fixture, SMALL_ENTRIES, rows, rows, ratio,
				     edges[edge], (edge & 1u) != 0,
				     (enum pattern)(edge % PATTERNS), false);
		}
		compare_case(&fixture, DS41F_CONTEXT_WITHOUT_CANDIDATES, rows,
			     rows, 1, DS41F_CONTEXT_WITHOUT_CANDIDATES, true,
			     RANDOM_WORDS, true);
		if (deep) {
			for (uint32_t distinct = 0; distinct <= 1u;
			     distinct++) {
				for (uint32_t pattern = RANDOM_WORDS;
				     pattern < PATTERNS; pattern++) {
					compare_case(&fixture,
						     DEEP_HALF_ENTRIES, rows,
						     rows, 1, DEEP_HALF_ENTRIES,
						     distinct != 0,
						     (enum pattern)pattern,
						     false);
					compare_case(&fixture,
						     DEEP_HALF_ENTRIES, rows,
						     rows, DS41F_POOLED_RATIO,
						     DEEP_HALF_ENTRIES,
						     distinct != 0,
						     (enum pattern)pattern,
						     false);
					compare_case(&fixture, DEEP_ENTRIES,
						     rows, rows, 1,
						     DEEP_ENTRIES,
						     distinct != 0,
						     (enum pattern)pattern,
						     false);
				}
			}
		}
	}
	for (uint32_t live = 0; live <= THIN_ROWS; live++) {
		compare_case(&fixture, SMALL_ENTRIES, THIN_ROWS, live, 1,
			     SMALL_ENTRIES, true, CUT_TIES, false);
	}
	if (wide) {
		for (uint32_t index = 0;
		     index < sizeof(wide_rows) / sizeof(wide_rows[0]);
		     index++) {
			const uint32_t rows = wide_rows[index];
			for (uint32_t distinct = 0; distinct <= 1u;
			     distinct++) {
				compare_case(&fixture, DEEP_HALF_ENTRIES, rows,
					     rows, 1, DEEP_HALF_ENTRIES,
					     distinct != 0, RANDOM_WORDS,
					     false);
				compare_case(&fixture, DEEP_HALF_ENTRIES, rows,
					     rows, DS41F_POOLED_RATIO,
					     DEEP_HALF_ENTRIES, distinct != 0,
					     RANDOM_WORDS, false);
				compare_case(&fixture, DEEP_ENTRIES, rows, rows,
					     1, DEEP_ENTRIES, distinct != 0,
					     RANDOM_WORDS, false);
			}
		}
	}
	negative_controls(&fixture);
	ds41f_test_split_close(&fixture);
	(void)printf(
		"index_select_split_card_test passed: class=%u comparisons=%u deep=%u wide=%u\n",
		arithmetic_class, case_count, deep ? 1u : 0u, wide ? 1u : 0u);
	return EXIT_SUCCESS;
}
