/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "anemone/hooks.h"
#include "step_driver.h"
#include "step_runs.h"
#define OK STATUS_OK

/* Explicit startup work only. The unpublished table never controls a step; card resources are never driver-owned.
 * A first BUSY stops attempts on that card, including later programs, with deterministic eager fallback. */
result_status step_graphs_prepare(struct step_driver *driver)
{
	size_t segment_total = 0;
	uint32_t full_cards = 0, captured_count = 0, reused_count = 0;
	if (driver != NULL && driver->failure != OK) {
		return driver->failure;
	}
	if (driver == NULL || driver->graphs_attempted ||
	    driver->sequence != 0 || driver->pipeline != NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	driver->graphs_attempted = true;
	if (driver->backend->graph_capture == NULL ||
	    driver->backend->graph_run == NULL) {
		return STATUS_UNSUPPORTED;
	}
	for (uint32_t index = 0; index < driver->program_count; index++) {
		const size_t count = driver->programs[index]->segment_count;
		if (count >
		    SIZE_MAX / sizeof(*driver->run_graphs) - segment_total) {
			return STATUS_OUT_OF_MEMORY;
		}
		segment_total += count;
	}
	/* A table must contain at least one run before storage is requested. */
	if (segment_total == 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	const size_t bank_stride = segment_total;
	const uint32_t banks =
		driver->flight_bank_count != 0 ? driver->flight_bank_count : 1u;
	if (segment_total > SIZE_MAX / sizeof(*driver->run_graphs) / banks)
		return STATUS_OUT_OF_MEMORY;
	segment_total *= banks;
	uint32_t *const graphs = FAULT_POINT(step_graph_memory) ?
					 NULL :
					 calloc(segment_total, sizeof(*graphs));
	if (graphs == NULL) {
		return STATUS_OUT_OF_MEMORY;
	}
	const uint64_t began =
		driver->flight_bank_count != 0 ? step_clock_ns() : 0;
	uint32_t *program_graphs = graphs;
	result_status status = OK, completion = OK;
	for (uint32_t bank = 0; bank < banks && status == OK; bank++) {
		uint32_t bank_captured = 0, bank_eager = 0;
		for (uint32_t index = 0;
		     index < driver->program_count && status == OK; index++) {
			const struct execution_program *const program =
				driver->flight_bank_count != 0 ?
					driver->flight_programs
						[(size_t)bank *
							 driver->program_count +
						 index] :
					driver->programs[index];
			if (bank != 0 && program == driver->programs[index]) {
				const uint32_t *const shared =
					program_graphs -
					(size_t)bank * bank_stride;
				memcpy(program_graphs, shared,
				       (size_t)program->segment_count *
					       sizeof(*program_graphs));
				uint32_t next_hand_off = 0;
				for (uint32_t segment = 0;
				     segment < program->segment_count;) {
					const uint32_t last =
						step_run_last_companions(
							program, segment,
							next_hand_off,
							driver->companion_segments);
					if (shared[segment] == 0)
						bank_eager++;
					else
						reused_count++;
					while (next_hand_off <
						       program->hand_off_count &&
					       program->hand_offs[next_hand_off]
							       .after_segment ==
						       last)
						next_hand_off++;
					segment = last + 1u;
				}
				program_graphs += program->segment_count;
				continue;
			}
			uint32_t hand_off = 0;
			for (uint32_t segment = 0, last = 0;
			     segment < program->segment_count && status == OK;
			     segment = ++last) {
				const struct segment_record *const first =
					&program->segments[segment];
				last = step_run_last_companions(
					program, segment, hand_off,
					driver->companion_segments);
				if ((full_cards >> first->card & 1u) == 0) {
					uint32_t graph = UINT32_MAX;
					/* Capture uploads on the card's stream.
				 * An error may leave work queued. */
					driver->uncovered_work[first->card] =
						true;
					status = driver->backend->graph_capture(
						driver->cards[first->card],
						&program->cards[first->card].launches
							 [first->first_launch],
						step_run_launch_count(
							program, segment, last),
						&graph);
					if (status == OK &&
					    graph >= BACKEND_MAXIMUM_GRAPHS) {
						status = STATUS_DEVICE_ERROR;
					}
					if (status == STATUS_BUSY_VALUE) {
						full_cards |= 1u << first->card;
						status = OK;
					} else if (status == OK) {
						program_graphs[segment] =
							graph + 1u;
						captured_count++;
						bank_captured++;
					}
				}
				bank_eager +=
					program_graphs[segment] == 0 ? 1u : 0u;
				while (hand_off < program->hand_off_count &&
				       program->hand_offs[hand_off]
						       .after_segment == last) {
					hand_off++;
				}
			}
			program_graphs += program->segment_count;
		}
		if (driver->flight_bank_count != 0)
			(void)fprintf(
				stderr,
				"flight program graphs: bank=%u captured=%u eager=%u status=%u; images=eager\n",
				bank, bank_captured, bank_eager,
				(unsigned)status);
	}
	/* Do not short-circuit: a failure on one card does not prove another card's upload retired. */
	for (uint32_t card = 0; card < driver->programs[0]->card_count;
	     card++) {
		const result_status waited = card_wait(driver, card);
		completion = completion != OK ? completion : waited;
	}
	status = completion != OK ? completion : status;
	if (status == OK && captured_count != 0) {
		driver->run_graphs = graphs;
		driver->flight_graph_stride = bank_stride;
	} else {
		free(graphs);
	}
	if (driver->flight_bank_count != 0) {
		const uint64_t ended = step_clock_ns();
		(void)fprintf(
			stderr,
			"flight graph preparation: captured=%u reused=%u table_bytes=%zu elapsed_ns=%" PRIu64
			"\n",
			captured_count, reused_count,
			segment_total * sizeof(*graphs),
			began != 0 && ended >= began ? ended - began : 0);
	}
	/* Any lost completion proof or device error is failclosed. Other refusals can keep the eager driver usable. */
	driver->failure =
		completion != OK || status == STATUS_DEVICE_ERROR ? status : OK;
	COUNTER_ADD(step_graphs_prepared, status == OK ? captured_count : 0u);
	return status;
}
