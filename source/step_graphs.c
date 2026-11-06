/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "anemone/hooks.h"
#include "step_driver.h"
#include "step_runs.h"
#define OK STATUS_OK

/* Explicit startup work only. The unpublished table never controls a step; card resources are never driver-owned.
 * A first BUSY stops attempts on that card, including later programs, with deterministic eager fallback. */
result_status step_graphs_prepare(struct step_driver *driver)
{
	size_t segment_total = 0;
	uint32_t full_cards = 0, captured_count = 0;
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
	uint32_t *const graphs = FAULT_POINT(step_graph_memory) ?
					 NULL :
					 calloc(segment_total, sizeof(*graphs));
	if (graphs == NULL) {
		return STATUS_OUT_OF_MEMORY;
	}
	uint32_t *program_graphs = graphs;
	result_status status = OK, completion = OK;
	for (uint32_t index = 0; index < driver->program_count && status == OK;
	     index++) {
		const struct execution_program *const program =
			driver->programs[index];
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
				driver->uncovered_work[first->card] = true;
				status = driver->backend->graph_capture(
					driver->cards[first->card],
					&program->cards[first->card]
						 .launches[first->first_launch],
					step_run_launch_count(program, segment,
							      last),
					&graph);
				if (status == OK &&
				    graph >= BACKEND_MAXIMUM_GRAPHS) {
					status = STATUS_DEVICE_ERROR;
				}
				if (status == STATUS_BUSY_VALUE) {
					full_cards |= 1u << first->card;
					status = OK;
				} else if (status == OK) {
					program_graphs[segment] = graph + 1u;
					captured_count++;
				}
			}
			while (hand_off < program->hand_off_count &&
			       program->hand_offs[hand_off].after_segment ==
				       last) {
				hand_off++;
			}
		}
		program_graphs += program->segment_count;
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
	} else {
		free(graphs);
	}
	/* Any lost completion proof or device error is failclosed. Other refusals can keep the eager driver usable. */
	driver->failure =
		completion != OK || status == STATUS_DEVICE_ERROR ? status : OK;
	COUNTER_ADD(step_graphs_prepared, status == OK ? captured_count : 0u);
	return status;
}
