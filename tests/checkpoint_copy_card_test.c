/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Same-card checkpoint copies, eager and graph, with a quarter-MiB arena. No model weights. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/exchange_records.h"
#include "anemone/hooks.h"
#include "anemone/model_layout.h"

#define ARENA_BYTES (256u << 10u)
#define DESTINATION_OFFSET (ARENA_BYTES / 2u)
#define COPY_BYTES ((uint64_t)(MODEL_WINDOW - 1u) * MODEL_STATE_ROW_BYTES)
#define WAIT_LIMIT 1000000000ull
#define JOB_LIMIT_SECONDS 60u
#define PATTERN_PRIME 251u

static bool copy_checked(const unsigned char *bytes, bool both_halves)
{
	for (uint32_t index = 0; index < COPY_BYTES; index++) {
		if (bytes[DESTINATION_OFFSET + index] !=
			    (unsigned char)(index % PATTERN_PRIME) ||
		    (both_halves &&
		     bytes[index] != (unsigned char)(index % PATTERN_PRIME))) {
			return false;
		}
	}
	return true;
}

int main(void)
{
	(void)alarm(JOB_LIMIT_SECONDS);
	const struct backend_operations *const backend = &backend_cuda_value;
	struct card_description_record description;
	struct device_card *card = NULL;
	void *block = NULL;
	uint64_t ticket = 0;
	uint32_t graph = 0;
	result_status status = backend->card_describe(0, &description);
	if (status == STATUS_OK) {
		(void)printf("checkpoint copy: %s\n", description.name);
		status = backend->card_open(0, ARENA_BYTES, &card);
	}
	status = status != STATUS_OK ?
			 status :
			 backend->staging_allocate(ARENA_BYTES, &block);
	if (status == STATUS_OK) {
		unsigned char *const bytes = block;
		memset(bytes, 0, ARENA_BYTES);
		for (uint32_t index = 0; index < COPY_BYTES; index++) {
			bytes[index] = (unsigned char)(index % PATTERN_PRIME);
		}
		status = backend->upload_begin(card, 0, bytes, ARENA_BYTES,
					       &ticket);
		status = status != STATUS_OK ?
				 status :
				 backend->upload_wait(card, ticket, WAIT_LIMIT);
		struct exchange_copy_arguments copy = {
			{ EXCHANGE_OWN_ARENA, 0 },
			{ EXCHANGE_OWN_ARENA, DESTINATION_OFFSET },
			COPY_BYTES
		};
		struct kernel_launch launch = {
			.kernel = BACKEND_COPY,
			.argument_byte_count = EXCHANGE_COPY_ARGUMENT_BYTES
		};
		memcpy(launch.argument, &copy, sizeof(copy));
		status = status != STATUS_OK ?
				 status :
				 backend->launch_run(card, &launch, 1);
		status = status != STATUS_OK ?
				 status :
				 backend->launch_wait(card, WAIT_LIMIT);
		status = status != STATUS_OK ?
				 status :
				 backend->download_begin(card, 0, bytes,
							 ARENA_BYTES, &ticket);
		status = status != STATUS_OK ?
				 status :
				 backend->upload_wait(card, ticket, WAIT_LIMIT);
		if (status == STATUS_OK && !copy_checked(bytes, true)) {
			status = STATUS_DEVICE_ERROR;
		}
		memset(bytes, 0, DESTINATION_OFFSET);
		status = status != STATUS_OK ?
				 status :
				 backend->upload_begin(card, 0, bytes,
						       DESTINATION_OFFSET,
						       &ticket);
		status = status != STATUS_OK ?
				 status :
				 backend->upload_wait(card, ticket, WAIT_LIMIT);
		copy.source.offset = DESTINATION_OFFSET;
		copy.destination.offset = 0;
		memcpy(launch.argument, &copy, sizeof(copy));
		status = status != STATUS_OK ?
				 status :
				 backend->graph_capture(card, &launch, 1,
							&graph);
		status = status != STATUS_OK ? status :
					       backend->graph_run(card, graph);
		status = status != STATUS_OK ?
				 status :
				 backend->launch_wait(card, WAIT_LIMIT);
		status = status != STATUS_OK ?
				 status :
				 backend->download_begin(card, 0, bytes,
							 ARENA_BYTES, &ticket);
		status = status != STATUS_OK ?
				 status :
				 backend->upload_wait(card, ticket, WAIT_LIMIT);
		if (status == STATUS_OK && !copy_checked(bytes, true)) {
			status = STATUS_DEVICE_ERROR;
		}
#if CHECKING_HOOKS_ENABLED
		if (status == STATUS_OK) {
			if (!fault_arm("backend_cuda_cuMemcpyDtoDAsync", 1)) {
				status = STATUS_INVALID_ARGUMENT;
			} else {
				const result_status failed =
					backend->launch_run(card, &launch, 1);
				const uint64_t calls = fault_disarm();
				if (failed != STATUS_DEVICE_ERROR ||
				    calls != 1 ||
				    backend->launch_run(card, &launch, 1) !=
					    STATUS_DEVICE_ERROR ||
				    backend->launch_wait(card, WAIT_LIMIT) !=
					    STATUS_DEVICE_ERROR) {
					status = STATUS_INVALID_ARGUMENT;
				}
			}
		}
#endif
	}
	backend->card_close(card, WAIT_LIMIT);
	backend->staging_free(block);
	if (status != STATUS_OK) {
		(void)fprintf(stderr, "checkpoint copy failed: %u\n",
			      (unsigned int)status);
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
