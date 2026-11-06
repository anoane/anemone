/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * card_timing.h: how the card tests of the fused launches time today's launches of a part against the fused ones.
 * Each form is a ring of one or more graphs: run r of a sample takes graph (first + r) % graph_count, first counting on
 * from sample to sample, so that a run may read weights that the runs before it did not. The two forms are sampled in
 * turn between two events on the card's stream (the card's own clock, no host in the time), the form that goes first
 * alternating from sample to sample, after warm runs of each; the saving of a pair is the first form's sample less the
 * second's beside it, which slow drift and other work on the card shift alike. L2 is never flushed. Test support: no
 * part of the engine.
 *
 * Contract:
 *   - test_card_timing_pair runs each form warm_run_count times, waits for them, then takes sample_count pairs of
 *     samples of run_count runs each: sample s of form f into samples_out[f * sample_count + s], in microseconds a run,
 *     and form 0's less form 1's into savings_out[s]. It answers false behind a failure line (linear_card_fixture.h)
 *     when the card refuses a graph, an event or a wait, or for a count of 0; the outputs are then unspecified.
 *   - test_card_timing_print sorts values in place and prints " <name>_p10_us=<a> <name>_median_us=<b>
 *     <name>_p90_us=<c>" of them, each divided by divisor (the repetitions of a part in one run of its graph).
 */
#ifndef TEST_CARD_TIMING_H
#define TEST_CARD_TIMING_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/backend_seam.h"

#define TEST_CARD_TIMING_FORMS 2u               /* today's launches, the fused ones */

struct test_card_timing_form {
    const uint32_t *graphs;                     /* graph handles of the card (backend_seam.h's graph_capture) */
    uint32_t graph_count;
};

bool test_card_timing_pair(
	struct device_card *card,
	const struct test_card_timing_form forms[TEST_CARD_TIMING_FORMS],
	uint32_t warm_run_count, uint32_t run_count, uint32_t sample_count,
	double *samples_out, double *savings_out);
void test_card_timing_print(double *values, uint32_t count, double divisor, const char *name);

#endif
