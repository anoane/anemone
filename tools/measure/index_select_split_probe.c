/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* C11 score-only selection probe. Kernel preparation and execution use the engine backend; the private card
 * header supplies its stream for CUDA events. Every sample starts with a 192 MiB digest eviction outside the
 * event interval. Split timing includes all five stages and their eager submission gaps. */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/model_layout.h"
#include "anemone/step_records.h"
#include "backend_cuda_card.h"
#include "index_select_split_fixture.h"

enum {
	SAMPLES = 9,
	METHODS = 2,
	LEGACY_METHOD = 0,
	SPLIT_METHOD = 1,
	FLUSH_BYTES = 192 << 20,
	ARENA_BYTES = 384 << 20,
	PROCESS_SECONDS = 240,
	MICROSECONDS_PER_MILLISECOND = 1000,
	MEDIAN_DIVISOR = 2,
	DECIMAL_BASE = 10,
	ARGUMENTS_MINIMUM = 2,
	HASH_SHIFT_FIRST = 16,
	HASH_SHIFT_MIDDLE = 15,
	SCORE_BASE = 0x3000,
	SCORE_MASK = 0x0FFF,
	MOST_ROWS = 1024,
	DEPTH_HALF = 524288,
	DEPTH_FULL = 1048576,
	TUNE_ROWS = 8
};
#define HASH_MULTIPLIER_FIRST 0x7FEB352Du
#define HASH_MULTIPLIER_SECOND 0x846CA68Bu
#define SCORE_SEED 0xA54FF53Au
#define DRIVER(call) TEST_SPLIT_REQUIRE((call) == CUDA_SUCCESS)

static const uint32_t thin_rows[] = { 1, 2, 4, 6, 8 };
static const uint32_t wide_rows[] = { 16, 32, 64, 128, 256, 1024 };
static const uint32_t shallow_entries[] = { 8192, 16384, 32768, 65536, 131072 };
static const uint32_t deep_entries[] = { DEPTH_HALF, DEPTH_FULL };
static CUevent start_event, stop_event;
static uint32_t arithmetic_class;
static uint32_t segments = INDEX_SPLIT_SEGMENTS;

static uint32_t random_word(uint32_t value)
{
	value ^= value >> HASH_SHIFT_FIRST;
	value *= HASH_MULTIPLIER_FIRST;
	value ^= value >> HASH_SHIFT_MIDDLE;
	value *= HASH_MULTIPLIER_SECOND;
	return value ^ (value >> HASH_SHIFT_FIRST);
}

static void fill_scores(struct test_split_fixture *fixture)
{
	for (uint64_t done = 0; done < fixture->scores.byte_count;
	     done += TEST_SPLIT_STAGING_BYTES) {
		const uint64_t left = fixture->scores.byte_count - done;
		const uint64_t bytes = left < TEST_SPLIT_STAGING_BYTES ?
					       left :
					       TEST_SPLIT_STAGING_BYTES;
		uint16_t *const words = (uint16_t *)fixture->staging;
		for (uint64_t word = 0; word < bytes / sizeof(uint16_t);
		     word++) {
			words[word] =
				(uint16_t)(SCORE_BASE |
					   (random_word(
						    SCORE_SEED +
						    (uint32_t)(done / sizeof(uint16_t) +
							       word)) &
					    SCORE_MASK));
		}
		test_split_upload(fixture, fixture->scores.byte_offset + done,
				  fixture->staging, bytes);
	}
}

static void measure_pair(struct test_split_fixture *fixture,
			 const struct kernel_launch *legacy,
			 const struct kernel_launch *split, uint32_t entries,
			 uint32_t rows, bool candidate)
{
	double samples[METHODS][SAMPLES];
	const uint64_t flush_offset = ARENA_BYTES - FLUSH_BYTES;
	for (uint32_t sample = 0; sample < SAMPLES; sample++) {
		for (uint32_t order = 0; order < METHODS; order++) {
			const uint32_t method = (sample + order) % METHODS;
			uint64_t digest = 0;
			TEST_SPLIT_REQUIRE(backend_cuda_value.digest(
						   fixture->card, flush_offset,
						   FLUSH_BYTES,
						   TEST_SPLIT_WAIT_NS,
						   &digest) == STATUS_OK);
			DRIVER(cuEventRecord(start_event,
					     fixture->card->state.stream));
			TEST_SPLIT_REQUIRE(
				backend_cuda_value.launch_run(
					fixture->card,
					method == LEGACY_METHOD ? legacy :
								  split,
					method == LEGACY_METHOD ?
						1u :
						INDEX_SPLIT_LAUNCH_COUNT) ==
				STATUS_OK);
			DRIVER(cuEventRecord(stop_event,
					     fixture->card->state.stream));
			TEST_SPLIT_REQUIRE(
				backend_cuda_value.launch_wait(
					fixture->card, TEST_SPLIT_WAIT_NS) ==
				STATUS_OK);
			float milliseconds = 0;
			DRIVER(cuEventElapsedTime(&milliseconds, start_event,
						  stop_event));
			samples[method][sample] = (double)milliseconds *
						  MICROSECONDS_PER_MILLISECOND;
		}
	}
	for (uint32_t method = 0; method < METHODS; method++) {
		for (uint32_t first = 0; first < SAMPLES; first++) {
			for (uint32_t second = first + 1u; second < SAMPLES;
			     second++) {
				if (samples[method][second] <
				    samples[method][first]) {
					const double held =
						samples[method][first];
					samples[method][first] =
						samples[method][second];
					samples[method][second] = held;
				}
			}
		}
		(void)printf(
			"TIME class=%u entries=%u rows=%u op=%s method=%s segments=%u "
			"us=%.3f min=%.3f max=%.3f\n",
			arithmetic_class, entries, rows,
			candidate ? "candidates" : "ordinary",
			method == LEGACY_METHOD ? "legacy" : "split", segments,
			samples[method][SAMPLES / MEDIAN_DIVISOR],
			samples[method][0], samples[method][SAMPLES - 1u]);
	}
	(void)fflush(stdout);
}

static void run_case(struct test_split_fixture *fixture, uint32_t entries,
		     uint32_t rows)
{
	/* Keep the full case resident. No wave sums substitute for a simultaneous launch. */
	const uint64_t wanted =
		(uint64_t)rows *
			(entries * sizeof(uint16_t) + INDEX_SPLIT_ROW_BYTES +
			 MODEL_CANDIDATES_ROW_BYTES * TEST_SPLIT_FIRST_PASS +
			 MODEL_CANDIDATES_ROW_BYTES + STEP_ROW_BYTES_VALUE) +
		STEP_BLOCK_BYTES_VALUE +
		TEST_SPLIT_RANGES * TEST_SPLIT_GUARD_BYTES * METHODS;
	if (wanted > ARENA_BYTES - FLUSH_BYTES) {
		(void)printf(
			"SKIP entries=%u rows=%u reason=resident_memory_limit\n",
			entries, rows);
		return;
	}
	test_split_layout(fixture, rows, entries);
	TEST_SPLIT_REQUIRE(fixture->used_bytes <= ARENA_BYTES - FLUSH_BYTES);
	struct {
		struct step_block block;
		struct step_row_record rows[MOST_ROWS];
	} step = { .block = { .row_count = rows } };
	for (uint32_t row = 0; row < rows; row++) {
		step.rows[row].position = (int32_t)(entries - 1u - row);
		step.rows[row].session_slot = 0;
	}
	test_split_upload(fixture, fixture->step.byte_offset, &step,
			  fixture->step.byte_count);
	fill_scores(fixture);
	uint32_t *const expected = malloc((size_t)fixture->legacy.byte_count);
	uint32_t *const actual = malloc((size_t)fixture->split.byte_count);
	TEST_SPLIT_REQUIRE(expected != NULL && actual != NULL);
	for (uint32_t candidate = 0; candidate < METHODS; candidate++) {
		const uint64_t output_bytes =
			(uint64_t)rows * (candidate != 0 ?
						  MODEL_CANDIDATES_ROW_BYTES :
						  MODEL_SELECTION_ROW_BYTES);
		struct kernel_launch legacy, split[INDEX_SPLIT_LAUNCH_COUNT];
		test_split_launches(fixture, rows, entries, 1, candidate != 0,
				    false, segments, &legacy, split);
		test_split_run(fixture, &legacy, 1);
		test_split_run(fixture, split, INDEX_SPLIT_LAUNCH_COUNT);
		test_split_download(fixture, fixture->legacy.byte_offset,
				    expected, output_bytes);
		test_split_download(fixture, fixture->split.byte_offset, actual,
				    output_bytes);
		TEST_SPLIT_REQUIRE(
			memcmp(expected, actual, (size_t)output_bytes) == 0);
		measure_pair(fixture, &legacy, split, entries, rows,
			     candidate != 0);
	}
	free(expected);
	free(actual);
}

int main(int argc, char **argv)
{
	TEST_SPLIT_REQUIRE(argc >= ARGUMENTS_MINIMUM);
	char *end = NULL;
	const unsigned long parsed = strtoul(argv[1], &end, DECIMAL_BASE);
	TEST_SPLIT_REQUIRE(end != argv[1] && *end == '\0' &&
			   (parsed == ARITHMETIC_CLASS_SM_80 ||
			    parsed == ARITHMETIC_CLASS_SM_120 ||
			    parsed == ARITHMETIC_CLASS_SM_120_NATIVE));
	arithmetic_class = (uint32_t)parsed;
	bool pilot = false, crossover = false;
	for (int argument = ARGUMENTS_MINIMUM; argument < argc; argument++) {
		if (strcmp(argv[argument], "--pilot") == 0) {
			pilot = true;
		} else if (strcmp(argv[argument], "--crossover") == 0) {
			crossover = true;
		} else if (strncmp(argv[argument], "--segments=",
				   sizeof("--segments=") - 1u) == 0) {
			segments = (uint32_t)strtoul(
				argv[argument] + sizeof("--segments=") - 1u,
				&end, DECIMAL_BASE);
			TEST_SPLIT_REQUIRE(*end == '\0' && segments != 0 &&
					   segments <= INDEX_SPLIT_SEGMENTS);
		} else {
			TEST_SPLIT_REQUIRE(false);
		}
	}
	(void)alarm(PROCESS_SECONDS);
	struct test_split_fixture fixture;
	test_split_open(&fixture, ARENA_BYTES, arithmetic_class);
	DRIVER(cuCtxSetCurrent(fixture.card->context));
	CUdevice device;
	int l2_bytes = 0;
	DRIVER(cuCtxGetDevice(&device));
	DRIVER(cuDeviceGetAttribute(&l2_bytes,
				    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device));
	TEST_SPLIT_REQUIRE(l2_bytes > 0 && l2_bytes < FLUSH_BYTES);
	DRIVER(cuEventCreate(&start_event, CU_EVENT_DEFAULT));
	DRIVER(cuEventCreate(&stop_event, CU_EVENT_DEFAULT));
	const struct linear_range flush = { ARENA_BYTES - FLUSH_BYTES,
					    FLUSH_BYTES };
	test_split_fill(&fixture, flush, 0);
	(void)printf(
		"METHOD samples=%u flush_bytes=%u l2_bytes=%d score_words=0x3000|hash_low12 "
		"timing=events_around_eager_backend_launches\n",
		SAMPLES, FLUSH_BYTES, l2_bytes);
	if (crossover) {
		for (uint32_t depth = 0;
		     depth <
		     sizeof(shallow_entries) / sizeof(shallow_entries[0]);
		     depth++) {
			run_case(&fixture, shallow_entries[depth], 1);
			run_case(&fixture, shallow_entries[depth], TUNE_ROWS);
			for (uint32_t count = 0;
			     count < sizeof(wide_rows) / sizeof(wide_rows[0]);
			     count++) {
				run_case(&fixture, shallow_entries[depth],
					 wide_rows[count]);
			}
		}
		for (uint32_t depth = 0;
		     depth < sizeof(deep_entries) / sizeof(deep_entries[0]);
		     depth++) {
			for (uint32_t count = 0;
			     count < sizeof(wide_rows) / sizeof(wide_rows[0]);
			     count++) {
				run_case(&fixture, deep_entries[depth],
					 wide_rows[count]);
			}
		}
	} else {
		for (uint32_t depth = 0;
		     depth < sizeof(deep_entries) / sizeof(deep_entries[0]);
		     depth++) {
			for (uint32_t count = 0;
			     count < sizeof(thin_rows) / sizeof(thin_rows[0]);
			     count++) {
				if (!pilot || count == 0) {
					run_case(&fixture, deep_entries[depth],
						 thin_rows[count]);
				}
			}
		}
	}
	DRIVER(cuEventDestroy(start_event));
	DRIVER(cuEventDestroy(stop_event));
	DRIVER(cuCtxSetCurrent(NULL));
	test_split_close(&fixture);
	return EXIT_SUCCESS;
}
