/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Fixed poll-thread counter history; readers use private endpoints and never consume samples. */
#include "anemone/live_metrics.h"

#include <stddef.h>
#include <string.h>

#define COUNTER_BOUNDARY_POINTS 2u
#define COUNTER_POINT_CAPACITY (LIVE_SAMPLE_CAPACITY + COUNTER_BOUNDARY_POINTS)

struct counter_point {
	uint64_t time_ns, count;
};

static uint64_t next_tick(uint64_t now_ns)
{
	const uint64_t advance =
		LIVE_SAMPLE_PERIOD_NS - now_ns % LIVE_SAMPLE_PERIOD_NS;
	return now_ns <= UINT64_MAX - advance ? now_ns + advance : UINT64_MAX;
}

static uint32_t sample_index(const struct live_sampler_record *sampler,
			     uint32_t position)
{
	return (sampler->next + LIVE_SAMPLE_CAPACITY - sampler->count +
		position) %
	       LIVE_SAMPLE_CAPACITY;
}

static bool valid_inputs(const struct live_sampler_record *sampler,
			 const struct engine_stats *engine)
{
	if (sampler == NULL || engine == NULL ||
	    sampler->next >= LIVE_SAMPLE_CAPACITY ||
	    sampler->count > LIVE_SAMPLE_CAPACITY ||
	    (sampler->count < LIVE_SAMPLE_CAPACITY &&
	     sampler->next != sampler->count) ||
	    engine->slots_total > STATS_SLOT_CAPACITY ||
	    engine->slots_busy > engine->slots_total ||
	    engine->card_count > BACKEND_MAXIMUM_CARDS ||
	    engine->step_ring_next >= STATS_STEP_RING ||
	    engine->step_ring_count > STATS_STEP_RING ||
	    engine->started_ns > engine->now_ns) {
		return false;
	}
	for (uint32_t slot = 0; slot < engine->slots_total; slot++) {
		const struct engine_slot_stats *const current =
			&engine->slots[slot];
		if (current->token_ring_next >= STATS_TOKEN_RING ||
		    current->token_ring_count > STATS_TOKEN_RING ||
		    current->submitted_ns > engine->now_ns ||
		    current->started_ns > engine->now_ns ||
		    current->first_token_ns > engine->now_ns ||
		    current->finished_ns > engine->now_ns ||
		    (current->submitted_ns != 0 &&
		     (current->slot >= engine->slots_total ||
		      current->prompt_processed < current->reused_tokens)) ||
		    (current->started_ns != 0 &&
		     current->started_ns < current->submitted_ns) ||
		    (current->first_token_ns != 0 &&
		     (current->started_ns == 0 ||
		      current->first_token_ns < current->started_ns))) {
			return false;
		}
	}
	if (sampler->count == 0) {
		return sampler->next_sample_ns == 0;
	}
	const struct live_counter_sample *const latest =
		&sampler->samples[sample_index(sampler, sampler->count - 1u)];
	if (latest->now_ns > engine->now_ns ||
	    sampler->next_sample_ns != next_tick(latest->now_ns)) {
		return false;
	}
	for (uint32_t card = 0; card < engine->card_count; card++) {
		const struct engine_card_stats *const current =
			&engine->cards[card];
		if (latest->cards[card].available && current->busy_available &&
		    current->busy_observed_ns != 0 &&
		    current->busy_observed_ns <
			    latest->cards[card].observed_ns) {
			return false;
		}
	}
	return true;
}

/* Only readers inspect older observations. The writer validates its bounds and latest point in constant time. */
static bool valid_history(const struct live_sampler_record *sampler,
			  const struct engine_stats *engine)
{
	uint64_t previous_ns = 0;
	uint64_t observed_ns[BACKEND_MAXIMUM_CARDS] = { 0 };
	for (uint32_t position = 0; position < sampler->count; position++) {
		const struct live_counter_sample *const sample =
			&sampler->samples[sample_index(sampler, position)];
		if ((position != 0 && sample->now_ns <= previous_ns) ||
		    sample->now_ns > engine->now_ns) {
			return false;
		}
		previous_ns = sample->now_ns;
		for (uint32_t card = 0; card < engine->card_count; card++) {
			if (sample->cards[card].available) {
				if (sample->cards[card].observed_ns == 0 ||
				    sample->cards[card].observed_ns <
					    observed_ns[card]) {
					return false;
				}
				observed_ns[card] =
					sample->cards[card].observed_ns;
			}
		}
	}
	for (uint32_t card = 0; card < engine->card_count; card++) {
		const struct engine_card_stats *const current =
			&engine->cards[card];
		if (current->busy_available && current->busy_observed_ns != 0 &&
		    current->busy_observed_ns < observed_ns[card]) {
			return false;
		}
	}
	return true;
}

result_status live_sampler_take(struct live_sampler_record *sampler,
				const struct engine_stats *engine)
{
	if (!valid_inputs(sampler, engine)) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (sampler->count != 0 &&
	    (engine->now_ns < sampler->next_sample_ns ||
	     engine->now_ns ==
		     sampler->samples[sample_index(sampler, sampler->count - 1u)]
			     .now_ns)) {
		return STATUS_OK;
	}
	struct live_counter_sample sample = {
		.now_ns = engine->now_ns,
		.generated = engine->generated_tokens,
		.processed = engine->prefill_tokens,
		.steps = engine->steps_landed
	};
	for (uint32_t slot = 0; slot < engine->slots_total; slot++) {
		const struct engine_slot_stats *const current =
			&engine->slots[slot];
		sample.slots[slot] = (struct live_counter_slot){
			.submitted_ns = current->submitted_ns,
			.request_tag = current->request_tag,
			.processed = current->prompt_processed >=
						     current->reused_tokens ?
					     current->prompt_processed -
						     current->reused_tokens :
					     0,
			.generated = current->generated_tokens
		};
	}
	for (uint32_t card = 0; card < engine->card_count; card++) {
		const struct engine_card_stats *const current =
			&engine->cards[card];
		sample.cards[card] = (struct live_counter_card){
			.observed_ns = current->busy_observed_ns,
			.busy_ns = current->busy_ns,
			.available = current->busy_available &&
				     current->busy_observed_ns != 0
		};
	}
	sampler->samples[sampler->next] = sample;
	sampler->next = (sampler->next + 1u) % LIVE_SAMPLE_CAPACITY;
	sampler->count += sampler->count < LIVE_SAMPLE_CAPACITY ? 1u : 0u;
	sampler->next_sample_ns = next_tick(engine->now_ns);
	return STATUS_OK;
}

/* Duplicate timestamps take the current value; capacity includes a phase origin and the current endpoint. */
static void point_added(struct counter_point points[COUNTER_POINT_CAPACITY],
			uint32_t *count, uint64_t time_ns, uint64_t value)
{
	const uint32_t index = *count != 0 && points[*count - 1u].time_ns ==
						       time_ns ?
				       *count - 1u :
				       (*count)++;
	points[index] = (struct counter_point){ time_ns, value };
}

static struct live_measurement measured(const struct counter_point *points,
					uint32_t count, uint64_t window_ns)
{
	struct live_measurement result = { 0 };
	if (count == 0) {
		return result;
	}
	const struct counter_point *const end = &points[count - 1u];
	uint64_t cutoff = end->time_ns > window_ns ? end->time_ns - window_ns :
						     0;
	cutoff = cutoff < points[0].time_ns ? points[0].time_ns : cutoff;
	if (cutoff >= end->time_ns) {
		return result;
	}
	uint32_t lower = 0;
	while (lower + 1u < count && points[lower + 1u].time_ns <= cutoff) {
		lower++;
	}
	result.elapsed_ns = end->time_ns - cutoff;
	result.count = (double)(end->count - points[lower].count);
	if (cutoff > points[lower].time_ns && lower + 1u < count) {
		const struct counter_point *const upper = &points[lower + 1u];
		const double fraction =
			(double)(cutoff - points[lower].time_ns) /
			(double)(upper->time_ns - points[lower].time_ns);
		result.count -=
			(double)(upper->count - points[lower].count) * fraction;
	}
	return result;
}

enum global_counter { GLOBAL_PREFILL, GLOBAL_DECODE, GLOBAL_STEPS };

static struct live_measurement
global_measured(const struct live_sampler_record *sampler,
		const struct engine_stats *engine, uint64_t window_ns,
		enum global_counter counter)
{
	struct counter_point points[COUNTER_POINT_CAPACITY];
	uint32_t count = 0;
	for (uint32_t position = 0; position < sampler->count; position++) {
		const struct live_counter_sample *const sample =
			&sampler->samples[sample_index(sampler, position)];
		const uint64_t value =
			counter == GLOBAL_PREFILL ? sample->processed :
			counter == GLOBAL_DECODE  ? sample->generated :
						    sample->steps;
		point_added(points, &count, sample->now_ns, value);
	}
	const uint64_t value =
		counter == GLOBAL_PREFILL ? engine->prefill_tokens :
		counter == GLOBAL_DECODE  ? engine->generated_tokens :
					    engine->steps_landed;
	point_added(points, &count, engine->now_ns, value);
	return measured(points, count, window_ns);
}

static struct live_measurement
slot_measured(const struct live_sampler_record *sampler,
	      const struct engine_stats *engine, uint32_t slot,
	      uint64_t window_ns, bool decode)
{
	const struct engine_slot_stats *const current = &engine->slots[slot];
	const uint64_t began = decode ? current->first_token_ns :
					current->started_ns;
	if (current->submitted_ns == 0 || began == 0) {
		return (struct live_measurement){ 0 };
	}
	struct counter_point points[COUNTER_POINT_CAPACITY];
	uint32_t count = 0;
	const uint64_t oldest =
		sampler->count != 0 ?
			sampler->samples[sample_index(sampler, 0)].now_ns :
			engine->now_ns;
	if (began >= oldest) {
		point_added(points, &count, began, 0);
	}
	for (uint32_t position = 0; position < sampler->count; position++) {
		const struct live_counter_sample *const sample =
			&sampler->samples[sample_index(sampler, position)];
		const struct live_counter_slot *const previous =
			&sample->slots[slot];
		if (sample->now_ns >= began &&
		    previous->request_tag == current->request_tag &&
		    previous->submitted_ns == current->submitted_ns) {
			const uint64_t value =
				decode ? (previous->generated != 0 ?
						  previous->generated - 1u :
						  0) :
					 previous->processed;
			point_added(points, &count, sample->now_ns, value);
		}
	}
	const uint64_t value =
		decode ? (current->generated_tokens != 0 ?
				  current->generated_tokens - 1u :
				  0) :
			 current->prompt_processed - current->reused_tokens;
	point_added(points, &count, engine->now_ns, value);
	return measured(points, count, window_ns);
}

static struct live_measurement
card_measured(const struct live_sampler_record *sampler,
	      const struct engine_stats *engine, uint32_t card,
	      uint64_t window_ns)
{
	const struct engine_card_stats *const current = &engine->cards[card];
	if (!current->busy_available || current->busy_observed_ns == 0) {
		return (struct live_measurement){ 0 };
	}
	struct counter_point points[COUNTER_POINT_CAPACITY];
	uint32_t count = 0;
	for (uint32_t position = 0; position < sampler->count; position++) {
		const struct live_counter_card *const previous =
			&sampler->samples[sample_index(sampler, position)]
				 .cards[card];
		if (previous->available) {
			point_added(points, &count, previous->observed_ns,
				    previous->busy_ns);
		}
	}
	point_added(points, &count, current->busy_observed_ns,
		    current->busy_ns);
	return measured(points, count, window_ns);
}

result_status
live_sampler_read(const struct live_sampler_record *sampler,
		  const struct engine_stats *engine,
		  struct live_window_record windows_out[LIVE_WINDOW_COUNT])
{
	if (windows_out == NULL || !valid_inputs(sampler, engine) ||
	    !valid_history(sampler, engine)) {
		return STATUS_INVALID_ARGUMENT;
	}
	static const uint64_t seconds[LIVE_WINDOW_COUNT] = {
		LIVE_WINDOW_SHORT_SECONDS, LIVE_WINDOW_LONG_SECONDS
	};
	struct live_window_record windows[LIVE_WINDOW_COUNT] = { 0 };
	for (uint32_t window = 0; window < LIVE_WINDOW_COUNT; window++) {
		const uint64_t duration =
			seconds[window] * LIVE_SAMPLE_PERIOD_NS;
		windows[window].prefill = global_measured(
			sampler, engine, duration, GLOBAL_PREFILL);
		windows[window].decode = global_measured(
			sampler, engine, duration, GLOBAL_DECODE);
		windows[window].steps = global_measured(sampler, engine,
							duration, GLOBAL_STEPS);
		for (uint32_t slot = 0; slot < engine->slots_total; slot++) {
			windows[window].slots[slot].prefill = slot_measured(
				sampler, engine, slot, duration, false);
			windows[window].slots[slot].decode = slot_measured(
				sampler, engine, slot, duration, true);
		}
		for (uint32_t card = 0; card < engine->card_count; card++) {
			windows[window].cards[card] =
				card_measured(sampler, engine, card, duration);
		}
	}
	memcpy(windows_out, windows, sizeof(windows));
	return STATUS_OK;
}
