/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Server-only dispatch probe through the real C backend. A graph is a chain of tiny, ordered fill kernels.
 * Reports CPU enqueue and CUDA-event stream span separately, in microseconds per table, over warmed batches.
 * This deliberately measures dispatch-bound work, not GEMM speed or model token throughput. */
#include <cuda.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "backend_cuda_card.h"
#include "anemone/backend_records.h"

#define ARENA_BYTES 4096u
#define MAXIMUM_LAUNCHES 256u
#define WORD_COUNT 32u
#define BATCH_REPLAYS 32u
#define WARM_REPLAYS 8u
#define ROUNDS 21u
#define WAIT_NANOSECONDS 10000000000ull
#define NANOSECONDS_PER_SECOND 1000000000ull
#define NANOSECONDS_PER_MICROSECOND 1000.0
#define MICROSECONDS_PER_MILLISECOND 1000.0
#define ALARM_SECONDS 300u

static const struct backend_operations *const backend = &backend_cuda_value;

static void require_status(result_status status)
{
	if (status != STATUS_OK) {
		(void)fprintf(stderr, "graph probe backend: %s\n",
			      status_text(status));
		exit(EXIT_FAILURE);
	}
}

static void require_driver(CUresult status)
{
    if (status != CUDA_SUCCESS) {
        (void)fprintf(stderr, "graph probe driver status: %d\n", (int)status);
        exit(EXIT_FAILURE);
    }
}

static uint64_t nanoseconds(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) { exit(EXIT_FAILURE); }
    return (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND + (uint64_t)now.tv_nsec;
}

static void record_event(struct device_card *card, CUevent event)
{
	result_status status = STATUS_OK;
	struct open_card_state *const state =
		cuda_card_enter(card, true, &status);
	require_status(status);
	if (state == NULL) {
		exit(EXIT_FAILURE);
	}
    require_driver(cuEventRecord(event, state->stream));
    cuda_card_leave(card);
}

static void enqueue(struct device_card *card,
		    const struct kernel_launch *launches, uint32_t count,
		    uint32_t graph, bool use_graph)
{
    require_status(use_graph ? backend->graph_run(card, graph) : backend->launch_run(card, launches, count));
}

int main(void)
{
    static const uint32_t sizes[] = {1, 8, 64, MAXIMUM_LAUNCHES};
    static struct kernel_launch launches[MAXIMUM_LAUNCHES];
    struct device_card *card = NULL;
    struct card_description_record description;
    uint32_t card_count = 0;
    uint32_t kernel = 0;
    CUevent first = NULL;
    CUevent last = NULL;
    result_status status = STATUS_OK;
    void *memory = NULL;
    (void)alarm(ALARM_SECONDS);
    require_status(backend->card_count(&card_count));
    if (card_count != 1) { (void)fprintf(stderr, "graph probe requires one visible GPU\n"); return EXIT_FAILURE; }
    require_status(backend->card_describe(0, &description));
    require_status(backend->card_open(0, ARENA_BYTES, &card));
    require_status(backend->staging_allocate(ARENA_BYTES, &memory));
    require_status(
	    backend->kernel_prepare(card, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
				    sizeof(struct fill_record), &kernel));
    if (cuda_card_enter(card, true, &status) == NULL) {
	    require_status(status);
	    return EXIT_FAILURE;
    }
    require_driver(cuEventCreate(&first, CU_EVENT_DEFAULT));
    require_driver(cuEventCreate(&last, CU_EVENT_DEFAULT));
    cuda_card_leave(card);
    for (uint32_t index = 0; index < MAXIMUM_LAUNCHES; index++) {
	    const struct fill_record record = { .word_count = WORD_COUNT,
						.value = index };
	    launches[index] = (struct kernel_launch){
		    .kernel = kernel,
		    .grid = { 1, 1, 1 },
		    .block = { FILL_THREADS_PER_BLOCK_VALUE, 1, 1 },
		    .argument_byte_count = sizeof(record)
	    };
	    memcpy(launches[index].argument, &record, sizeof(record));
    }
    for (uint32_t size_index = 0; size_index < sizeof(sizes) / sizeof(sizes[0]); size_index++) {
        const uint32_t count = sizes[size_index];
        uint32_t graph = 0;
        require_status(backend->graph_capture(card, launches, count, &graph));
        for (uint32_t mode = 0; mode <= 1; mode++) {
            for (uint32_t warm = 0; warm < WARM_REPLAYS; warm++) { enqueue(card, launches, count, graph, mode != 0); }
            require_status(backend->launch_wait(card, WAIT_NANOSECONDS));
        }
        for (uint32_t round = 0; round < ROUNDS; round++) {
            /* Alternate eager/graph order across rounds to reduce systematic thermal/clock drift. */
            for (uint32_t turn = 0; turn <= 1; turn++) {
                const bool use_graph = ((round + turn) & 1u) != 0;
                record_event(card, first);
                const uint64_t start = nanoseconds();
                for (uint32_t replay = 0; replay < BATCH_REPLAYS; replay++) {
                    enqueue(card, launches, count, graph, use_graph);
                }
                const uint64_t queued = nanoseconds();
                record_event(card, last);
                require_status(backend->launch_wait(card, WAIT_NANOSECONDS));
                const uint64_t finished = nanoseconds();
                float milliseconds = 0.0f;
		if (cuda_card_enter(card, true, &status) == NULL) {
			require_status(status);
			return EXIT_FAILURE;
		}
		require_driver(cuEventElapsedTime(&milliseconds, first, last));
		cuda_card_leave(card);
		uint64_t ticket = 0;
                require_status(backend->download_begin(card, 0, memory, WORD_COUNT * sizeof(uint32_t), &ticket));
                require_status(backend->upload_wait(card, ticket, WAIT_NANOSECONDS));
                const uint32_t *const words = memory;
                for (uint32_t word = 0; word < WORD_COUNT; word++) {
                    if (words[word] != count - 1) { (void)fprintf(stderr, "graph probe order mismatch\n");
                        return EXIT_FAILURE; }
                }
                (void)printf("{\"arithmetic_class\":%" PRIu32 ",\"nodes\":%" PRIu32 ",\"graph\":%s,"
                             "\"round\":%" PRIu32 ",\"batch\":%u,\"enqueue_us\":%.6f,\"stream_us\":%.6f,"
                             "\"wall_us\":%.6f}\n", description.arithmetic_class, count,
                             use_graph ? "true" : "false", round, BATCH_REPLAYS,
                             (double)(queued - start) / NANOSECONDS_PER_MICROSECOND / BATCH_REPLAYS,
                             (double)milliseconds * MICROSECONDS_PER_MILLISECOND / BATCH_REPLAYS,
                             (double)(finished - start) / NANOSECONDS_PER_MICROSECOND / BATCH_REPLAYS);
            }
        }
    }
    if (cuda_card_enter(card, true, &status) == NULL) {
	    require_status(status);
	    return EXIT_FAILURE;
    }
    require_driver(cuEventDestroy(last));
    require_driver(cuEventDestroy(first));
    cuda_card_leave(card);
    backend->card_close(card, WAIT_NANOSECONDS);
    backend->staging_free(memory);
    return EXIT_SUCCESS;
}
