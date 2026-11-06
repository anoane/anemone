/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Prepared CUDA graphs: ordered kernels, captured argument ownership, changed arena inputs, eager equivalence,
 * capacity, invalid tables and sticky driver failures. Every process sees only its selected physical card.
 * The fill poisons the decoder output before it reads original FP8 bytes: removing the dependency breaks it.
 * Full-arena comparisons cover output bounds as well as the BF16 values. No kernel intentionally faults. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_records.h"
#include "anemone/backend_seam.h"
#include "anemone/engram_rows_records.h"
#include "anemone/hooks.h"

#define ARENA_BYTES 4096u
#define OUTPUT_OFFSET 1024u
#define OUTPUT_BYTES (ENGRAM_ROWS_VALUES * sizeof(uint16_t))
#define OUTPUT_WORDS (OUTPUT_BYTES / FILL_WORD_BYTES)
#define WAIT_NANOSECONDS 10000000000ull
#define ALARM_SECONDS 300u
#define UNTOUCHED_GRAPH UINT32_MAX
#define CANARY_BYTE 0xA5
#define POISON_WORD 0xDEADBEEFu
#define FP8_ONE 0x38u
#define FP8_TWO 0x40u
#define BF16_ONE 0x3F80u
#define BF16_TWO 0x4000u
#define E8M0_ONE 127u
#define REPLAY_COUNT 8u

enum launch_index { POISON_LAUNCH, DECODE_LAUNCH, LAUNCH_COUNT };

static const struct backend_operations *const backend = &backend_cuda_value;
static int failures = 0;
static unsigned char *staging = NULL;
static unsigned char mirror[ARENA_BYTES];

static void expect(bool condition, const char *what)
{
    if (!condition) { (void)fprintf(stderr, "FAILED: CUDA graph: %s\n", what); failures++; }
}

static void expect_status(result_status actual, result_status expected,
			  const char *what)
{
    if (actual != expected) {
	    (void)fprintf(stderr, "FAILED: CUDA graph: %s: %s, expected %s\n",
			  what, status_text(actual), status_text(expected));
	    failures++;
    }
}

static struct device_card *open_card(uint32_t card_index)
{
	struct device_card *card = NULL;
	expect_status(backend->card_open(card_index, ARENA_BYTES, &card),
		      STATUS_OK, "open");
	return card;
}

static void prepare(struct device_card *card,
		    struct kernel_launch launches[LAUNCH_COUNT])
{
	const struct fill_record fill = { .arena_byte_offset = OUTPUT_OFFSET,
					  .word_count = OUTPUT_WORDS,
					  .value = POISON_WORD };
	const struct engram_values_arguments decode = {
		.rows_offset = 0, .values_offset = OUTPUT_OFFSET
	};
	memset(launches, 0, sizeof(*launches) * LAUNCH_COUNT);
	for (uint32_t index = 0; index < LAUNCH_COUNT; index++) {
		for (uint32_t dimension = 0; dimension < LAUNCH_DIMENSIONS;
		     dimension++) {
			launches[index].grid[dimension] = 1;
			launches[index].block[dimension] = 1;
		}
    }
    launches[POISON_LAUNCH].block[0] = FILL_THREADS_PER_BLOCK_VALUE;
    launches[POISON_LAUNCH].argument_byte_count = sizeof(fill);
    memcpy(launches[POISON_LAUNCH].argument, &fill, sizeof(fill));
    expect_status(backend->kernel_prepare(card, KERNEL_FILE_FILL,
					  KERNEL_FILL_WORDS, sizeof(fill),
					  &launches[POISON_LAUNCH].kernel),
		  STATUS_OK, "prepare fill");
    launches[DECODE_LAUNCH].block[0] = ENGRAM_ROWS_VALUES;
    launches[DECODE_LAUNCH].argument_byte_count = sizeof(decode);
    memcpy(launches[DECODE_LAUNCH].argument, &decode, sizeof(decode));
    expect_status(backend->kernel_prepare(card, KERNEL_FILE_ENGRAM_ROWS,
					  KERNEL_ENGRAM_VALUES, sizeof(decode),
					  &launches[DECODE_LAUNCH].kernel),
		  STATUS_OK, "prepare FP8 decode");
}

static void upload(struct device_card *card, unsigned char code)
{
    uint64_t ticket = 0;
    unsigned char *const buffer = staging;
    if (buffer == NULL) { expect(false, "upload staging exists"); return; }
    memset(buffer, CANARY_BYTE, ARENA_BYTES);
    memset(buffer, code, ENGRAM_ROWS_VALUES);
    memset(buffer + ENGRAM_ROWS_VALUES, E8M0_ONE,
	   ENGRAM_ROWS_ROW_BYTES - ENGRAM_ROWS_VALUES);
    memcpy(mirror, buffer, ARENA_BYTES);
    expect_status(backend->upload_begin(card, 0, buffer, ARENA_BYTES, &ticket),
		  STATUS_OK, "upload");
    expect_status(backend->upload_wait(card, ticket, WAIT_NANOSECONDS),
		  STATUS_OK, "upload completes");
}

static void compare(struct device_card *card, const char *what)
{
    uint64_t ticket = 0;
    unsigned char *const buffer = staging;
    if (buffer == NULL) { expect(false, "download staging exists"); return; }
    expect_status(backend->download_begin(card, 0, buffer, ARENA_BYTES,
					  &ticket),
		  STATUS_OK, "download");
    expect_status(backend->upload_wait(card, ticket, WAIT_NANOSECONDS),
		  STATUS_OK, "download completes");
    expect(memcmp(buffer, mirror, ARENA_BYTES) == 0, what);
}

static void expect_decoded(struct device_card *card, uint16_t expected)
{
	for (uint32_t index = 0; index < ENGRAM_ROWS_VALUES; index++) {
		memcpy(mirror + OUTPUT_OFFSET + index * sizeof(expected),
		       &expected, sizeof(expected));
	}
    compare(card, "decoded values and all surrounding bytes match");
}

static void test_refusals(struct device_card *card,
			  const struct kernel_launch good[LAUNCH_COUNT])
{
	struct kernel_launch bad[LAUNCH_COUNT];
	uint32_t graph = UNTOUCHED_GRAPH;
	expect_status(backend->graph_capture(NULL, good, LAUNCH_COUNT, &graph),
		      STATUS_INVALID_ARGUMENT, "no card capture");
	expect_status(backend->graph_run(NULL, 0), STATUS_INVALID_ARGUMENT,
		      "no card replay");
	expect_status(backend->graph_capture(card, good, LAUNCH_COUNT, NULL),
		      STATUS_INVALID_ARGUMENT, "no output");
	expect_status(backend->graph_capture(card, NULL, LAUNCH_COUNT, &graph),
		      STATUS_INVALID_ARGUMENT, "no table");
	expect_status(backend->graph_capture(card, good, 0, &graph),
		      STATUS_INVALID_ARGUMENT, "empty table");
	for (uint32_t dimension = 0; dimension < LAUNCH_DIMENSIONS;
	     dimension++) {
		memcpy(bad, good, sizeof(bad));
		bad[DECODE_LAUNCH].block[dimension] = 0;
		expect_status(
			backend->graph_capture(card, bad, LAUNCH_COUNT, &graph),
			STATUS_INVALID_ARGUMENT, "late zero block dimension");
		memcpy(bad, good, sizeof(bad));
		bad[DECODE_LAUNCH].grid[dimension] = 0;
		expect_status(
			backend->graph_capture(card, bad, LAUNCH_COUNT, &graph),
			STATUS_INVALID_ARGUMENT, "late zero grid dimension");
	}
    memcpy(bad, good, sizeof(bad));
    bad[DECODE_LAUNCH].kernel = BACKEND_NO_KERNEL;
    expect_status(backend->graph_capture(card, bad, LAUNCH_COUNT, &graph),
		  STATUS_INVALID_ARGUMENT, "late unknown kernel");
    memcpy(bad, good, sizeof(bad));
    bad[DECODE_LAUNCH].argument_byte_count = UINT32_MAX;
    expect_status(backend->graph_capture(card, bad, LAUNCH_COUNT, &graph),
		  STATUS_INVALID_ARGUMENT,
		  "record cannot cause an out-of-bounds validation read");
    memcpy(bad, good, sizeof(bad));
    bad[DECODE_LAUNCH].argument[LAUNCH_ARGUMENT_BYTES - 1] = 1;
    expect_status(backend->graph_capture(card, bad, LAUNCH_COUNT, &graph),
		  STATUS_INVALID_ARGUMENT, "late trailing record byte");
    memcpy(bad, good, sizeof(bad));
    bad[DECODE_LAUNCH].branch = LAUNCH_SIDE_MORE + 1u;
    expect_status(backend->graph_capture(card, bad, LAUNCH_COUNT, &graph),
		  STATUS_INVALID_ARGUMENT, "late reserved byte");
    memcpy(bad, good, sizeof(bad));
    bad[DECODE_LAUNCH].dynamic_shared_byte_count = UINT32_MAX;
    expect_status(backend->graph_capture(card, bad, LAUNCH_COUNT, &graph),
		  STATUS_INVALID_ARGUMENT, "late excessive shared memory");
    expect_status(backend->graph_run(card, 0), STATUS_INVALID_ARGUMENT,
		  "uncaptured handle");
    expect(graph == UNTOUCHED_GRAPH, "refusals do not write the output handle");
    compare(card, "invalid tables execute nothing, including their valid prefix");
}

static void test_replay(uint32_t card_index)
{
	struct kernel_launch launches[LAUNCH_COUNT];
	struct device_card *const card = open_card(card_index);
	uint32_t graph = UNTOUCHED_GRAPH;
	if (card == NULL) {
		return;
	}
    prepare(card, launches);
    upload(card, FP8_ONE);
    test_refusals(card, launches);
    expect_status(backend->graph_capture(card, launches, LAUNCH_COUNT, &graph),
		  STATUS_OK, "capture");
    expect(graph == 0, "first captured handle");
    compare(card, "graph construction executes no kernel");
    /* A graph must own its parameters, rather than referring back to a caller's mutable table. */
    memset(launches, CANARY_BYTE, sizeof(launches));
    for (uint32_t iteration = 0; iteration < REPLAY_COUNT; iteration++) {
        const bool one = (iteration & 1u) == 0;
        upload(card, one ? FP8_ONE : FP8_TWO);
	expect_status(backend->graph_run(card, graph), STATUS_OK, "replay");
	expect_status(backend->launch_wait(card, WAIT_NANOSECONDS), STATUS_OK,
		      "replay completion");
	expect_decoded(card, one ? BF16_ONE : BF16_TWO);
    }
    prepare(card, launches);
    upload(card, FP8_TWO);
    expect_status(backend->launch_run(card, launches, LAUNCH_COUNT), STATUS_OK,
		  "eager same table");
    expect_status(backend->launch_wait(card, WAIT_NANOSECONDS), STATUS_OK,
		  "eager completion");
    expect_decoded(card, BF16_TWO);
    for (uint32_t index = 1; index < BACKEND_MAXIMUM_GRAPHS; index++) {
	    expect_status(backend->graph_capture(card, launches, LAUNCH_COUNT,
						 &graph),
			  STATUS_OK, "all graph slots usable");
	    expect(graph == index, "handles are distinct");
    }
    graph = UNTOUCHED_GRAPH;
    expect_status(backend->graph_capture(card, launches, LAUNCH_COUNT, &graph),
		  STATUS_BUSY_VALUE, "capacity");
    expect(graph == UNTOUCHED_GRAPH, "capacity refusal leaves output untouched");
    expect_status(backend->graph_run(card, BACKEND_MAXIMUM_GRAPHS),
		  STATUS_INVALID_ARGUMENT, "past final handle");
    backend->card_close(card, WAIT_NANOSECONDS);
    expect_status(backend->graph_run(card, 0), STATUS_INVALID_ARGUMENT,
		  "closed card");
    expect_status(backend->graph_capture(card, launches, LAUNCH_COUNT, &graph),
		  STATUS_INVALID_ARGUMENT, "closed capture");
}

#if CHECKING_HOOKS_ENABLED
static void test_faults(uint32_t card_index)
{
    static const char *const faults[] = {
        "backend_cuda_cuGraphCreate", "backend_cuda_cuGraphAddKernelNode",
        "backend_cuda_cuGraphInstantiateWithFlags", "backend_cuda_cuGraphUpload", "backend_cuda_cuGraphLaunch"};
    for (uint32_t index = 0; index < sizeof(faults) / sizeof(faults[0]); index++) {
	    struct device_card *const card = open_card(card_index);
	    struct kernel_launch launches[LAUNCH_COUNT];
	    uint32_t graph = UNTOUCHED_GRAPH;
	    if (card == NULL) {
		    return;
	    }
        prepare(card, launches);
        if (index == sizeof(faults) / sizeof(faults[0]) - 1) {
		expect_status(backend->graph_capture(card, launches,
						     LAUNCH_COUNT, &graph),
			      STATUS_OK, "capture before replay fault");
		expect(fault_arm(faults[index], 1), "arm replay fault");
		expect_status(backend->graph_run(card, graph),
			      STATUS_DEVICE_ERROR, "replay failure");
	} else {
		expect(fault_arm(faults[index], 1), "arm capture fault");
		expect_status(backend->graph_capture(card, launches,
						     LAUNCH_COUNT, &graph),
			      STATUS_DEVICE_ERROR, "capture failure");
		expect(graph == UNTOUCHED_GRAPH,
		       "failed capture leaves output untouched");
        }
	expect(fault_disarm() == 1, "exact fault reached");
	expect_status(backend->launch_run(card, launches, LAUNCH_COUNT),
		      STATUS_DEVICE_ERROR,
		      "eager work cannot bypass failed graph state");
	expect_status(backend->launch_wait(card, 0), STATUS_DEVICE_ERROR,
		      "failure remains sticky");
	expect_status(backend->graph_capture(card, launches, LAUNCH_COUNT,
					     &graph),
		      STATUS_DEVICE_ERROR, "no capture after failure");
	if (graph != UNTOUCHED_GRAPH) {
		expect_status(backend->graph_run(card, graph),
			      STATUS_DEVICE_ERROR, "no replay after failure");
	}
        backend->card_close(card, WAIT_NANOSECONDS);
    }
}
#endif

int main(void)
{
    uint32_t card_count = 0;
    (void)alarm(ALARM_SECONDS);
    expect(backend->graph_capture != NULL && backend->graph_run != NULL, "graph seam populated");
    if (backend->graph_capture == NULL || backend->graph_run == NULL) { return EXIT_FAILURE; }
    expect_status(backend->card_count(&card_count), STATUS_OK,
		  "card enumeration");
    expect(card_count != 0, "real card required");
    void *memory = NULL;
    expect_status(backend->staging_allocate(ARENA_BYTES, &memory), STATUS_OK,
		  "staging");
    staging = memory;
    if (staging == NULL) { return EXIT_FAILURE; }
    for (uint32_t index = 0; index < card_count; index++) {
        test_replay(index);
#if CHECKING_HOOKS_ENABLED
	test_faults(index);
#endif
        /* Reopen after all partial-graph cleanup paths; graph handles belong to the new lifetime alone. */
        test_replay(index);
    }
    backend->staging_free(staging);
    (void)printf("CUDA graphs: %s (%u visible cards)\n", failures == 0 ? "passed" : "FAILED", card_count);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
