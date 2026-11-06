/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * card_timing.c: the paired timing of card_timing.h. The two events are made for a call and destroyed behind it; a
 * sample records the first event, queues its runs and records the second on the card's stream (backend_cuda_card.h).
 */
#include "card_timing.h"

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "backend_cuda_card.h"
#include "linear_card_fixture.h"

#define FORMS TEST_CARD_TIMING_FORMS
#define MICROSECONDS_PER_MILLISECOND 1000.0
#define PERCENTILE_TEN 10u
#define PERCENTILE_FIFTY 50u
#define PERCENTILE_NINETY 90u
#define PERCENT 100u

enum sample_event { START_EVENT, END_EVENT, SAMPLE_EVENTS };

static const struct backend_operations *const backend = &backend_cuda_value;

/* The events made (or destroyed) on the card; false behind a failure line. */
static bool events_make(struct device_card *card, CUevent events[SAMPLE_EVENTS])
{
	result_status status = STATUS_OK;
	struct open_card_state *const state =
		cuda_card_enter(card, true, &status);
	bool is_made = state != NULL;
	for (uint32_t event = 0; event < SAMPLE_EVENTS; event++) {
		events[event] = NULL;
		is_made = is_made &&
			  cuEventCreate(&events[event], CU_EVENT_DEFAULT) ==
				  CUDA_SUCCESS;
	}
    if (state != NULL) {
	    cuda_card_leave(card);
    }
    return test_linear_expect(is_made, "two events are made on the card");
}
static void events_destroy(struct device_card *card,
			   CUevent events[SAMPLE_EVENTS])
{
	result_status status = STATUS_OK;
	struct open_card_state *const state =
		cuda_card_enter(card, true, &status);
	for (uint32_t event = 0; state != NULL && event < SAMPLE_EVENTS;
	     event++) {
		if (events[event] != NULL) {
			(void)cuEventDestroy(events[event]);
		}
    }
    if (state != NULL) {
	    cuda_card_leave(card);
    }
}

static bool event_record(struct device_card *card, CUevent event)
{
	result_status status = STATUS_OK;
	struct open_card_state *const state =
		cuda_card_enter(card, true, &status);
	const bool is_recorded = state != NULL &&
				 cuEventRecord(event, state->stream) ==
					 CUDA_SUCCESS;
	if (state != NULL) {
		cuda_card_leave(card);
	}
    return test_linear_expect(is_recorded, "an event is recorded on the card's stream");
}

/* The milliseconds between the two events once the second has passed. */
static bool event_elapsed(struct device_card *card,
			  const CUevent events[SAMPLE_EVENTS],
			  float *milliseconds_out)
{
	result_status status = STATUS_OK;
	struct open_card_state *const state =
		cuda_card_enter(card, true, &status);
	const bool is_measured =
		state != NULL &&
		cuEventSynchronize(events[END_EVENT]) == CUDA_SUCCESS &&
		cuEventElapsedTime(milliseconds_out, events[START_EVENT],
				   events[END_EVENT]) == CUDA_SUCCESS;
	if (state != NULL) {
		cuda_card_leave(card);
	}
    return test_linear_expect(is_measured, "the time between two events is measured");
}

/* count runs of a form from graph first on; false behind a failure line. */
static bool form_runs(struct device_card *card,
		      const struct test_card_timing_form *form, uint32_t first,
		      uint32_t count)
{
    bool is_queued = true;
    for (uint32_t run = 0; run < count && is_queued; run++) {
	    is_queued = test_linear_expect_status(
		    backend->graph_run(
			    card,
			    form->graphs[(first + run) % form->graph_count]),
		    STATUS_OK, "a graph is queued");
    }
    return is_queued;
}

/* One sample: microseconds a run, or a negative number behind a failure line. */
static double form_sample(struct device_card *card,
			  const CUevent events[SAMPLE_EVENTS],
			  const struct test_card_timing_form *form,
			  uint32_t first, uint32_t run_count)
{
    float milliseconds = 0.0f;
    const bool is_timed = event_record(card, events[START_EVENT]) && form_runs(card, form, first, run_count) &&
                          event_record(card, events[END_EVENT]) && event_elapsed(card, events, &milliseconds);
    return is_timed ? (double)milliseconds * MICROSECONDS_PER_MILLISECOND / (double)run_count : -1.0;
}

bool test_card_timing_pair(
	struct device_card *card,
	const struct test_card_timing_form forms[TEST_CARD_TIMING_FORMS],
	uint32_t warm_run_count, uint32_t run_count, uint32_t sample_count,
	double *samples_out, double *savings_out)
{
    CUevent events[SAMPLE_EVENTS] = {NULL, NULL};
    bool is_timed = test_linear_expect(run_count != 0 && sample_count != 0 && forms[0].graph_count != 0 &&
                                           forms[1].graph_count != 0,
                                       "a timing has its counts") &&
                    events_make(card, events);
    for (uint32_t form = 0; form < FORMS && is_timed; form++) {
        is_timed = form_runs(card, &forms[form], 0, warm_run_count);
    }
    is_timed = is_timed &&
	       test_linear_expect_status(
		       backend->launch_wait(card,
					    TEST_LINEAR_WAIT_LIMIT_NANOSECONDS),
		       STATUS_OK, "the warm runs finish");
    for (uint32_t sample = 0, first = 0; sample < sample_count && is_timed; sample++) {
        for (uint32_t turn = 0; turn < FORMS && is_timed; turn++, first += run_count) {
            const uint32_t form = (sample + turn) % FORMS;
            samples_out[form * sample_count + sample] = form_sample(card, events, &forms[form], first, run_count);
            is_timed = samples_out[form * sample_count + sample] >= 0.0;
        }
        savings_out[sample] = is_timed ? samples_out[sample] - samples_out[sample_count + sample] : 0.0;
    }
    if (events[START_EVENT] != NULL || events[END_EVENT] != NULL) {
        events_destroy(card, events);
    }
    return is_timed;
}

static int doubles_compare(const void *first, const void *second)
{
    const double first_value = *(const double *)first, second_value = *(const double *)second;
    return (first_value > second_value) - (first_value < second_value);
}

void test_card_timing_print(double *values, uint32_t count, double divisor, const char *name)
{
    qsort(values, count, sizeof(values[0]), doubles_compare);
    (void)printf(" %s_p10_us=%.3f %s_median_us=%.3f %s_p90_us=%.3f", name,
                 values[count * PERCENTILE_TEN / PERCENT] / divisor, name,
                 values[count * PERCENTILE_FIFTY / PERCENT] / divisor, name,
                 values[count * PERCENTILE_NINETY / PERCENT] / divisor);
}
