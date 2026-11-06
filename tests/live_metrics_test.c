/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "anemone/live_metrics.h"

#define TEST_SECONDS 80u
#define GENERATION_PER_SECOND 4u
#define PREFILL_PER_SECOND 8u
#define STEPS_PER_SECOND 2u
#define SHORT_GENERATED 40.0
#define LONG_GENERATED 240.0
#define SHORT_PREFILLED 80.0
#define LONG_PREFILLED 480.0
#define SHORT_STEPS 20.0
#define LONG_STEPS 120.0
#define WARMUP_SECONDS 3u
#define WARMUP_GENERATED 12.0
#define HALF_SECOND_NS 500000000ull
#define QUARTER_SECOND_NS 250000000ull
#define FIRST_SECONDS 2u
#define SECOND_SECONDS 6u
#define FRACTIONAL_SECONDS 12u
#define FRACTIONAL_CURRENT 13u
#define FRACTIONAL_GENERATED 25u
#define SPARSE_FIRST_TOKENS 8u
#define SPARSE_LAST_TOKENS 20u
#define FRACTIONAL_EXPECTED 23.0
#define LARGE_CLOCK_SECONDS 1000u
#define WRAP_BASE (UINT64_MAX - 5u)
#define WRAP_END 4u
#define WRAP_DELTA 10.0
#define REQUEST_TAG 17u
#define REPLACEMENT_TAG 18u
#define REUSED_TOKENS 1000u
#define REQUEST_PREFILLED 200u
#define REQUEST_SECOND_PREFILL 400u
#define REQUEST_FIRST_SECONDS 4u
#define REQUEST_SECOND_SECONDS 5u
#define REQUEST_DONE_SECONDS 6u
#define REQUEST_LATE_SECONDS 20u
#define REQUEST_GENERATED 5u
#define REQUEST_LATER_GENERATED 9u
#define REQUEST_TIMED_GENERATED 4.0
#define REQUEST_REPLACED_SECONDS 21u
#define REQUEST_REPLACEMENT_WORK 10u
#define REQUEST_RESUBMITTED_SECONDS 23u
#define CARD_QUARTER_BUSY_NS 250000000ull
#define CARD_HALF_BUSY_NS 500000000ull
#define CARD_WARMUP_SECONDS 5u
#define CARD_SHORT_END_SECONDS 15u
#define CARD_COUNT 2u
#define TOLERANCE 0.000001
#define OUTPUT_SENTINEL 0xA5

static int failures;

static void expect(bool condition, const char *message)
{
	if (!condition) {
		(void)printf("FAILED: %s\n", message);
		failures++;
	}
}

static uint64_t seconds_ns(uint64_t seconds)
{
	return seconds * LIVE_SAMPLE_PERIOD_NS;
}

static bool equals(struct live_measurement measurement, double count,
		   uint64_t duration_ns)
{
	return measurement.elapsed_ns == duration_ns &&
	       fabs(measurement.count - count) <= TOLERANCE;
}

/* Failure and observer contracts preserve the original object's bytes, including its existing padding. */
static bool unchanged_bytes(const unsigned char *before, const void *object,
			    size_t count)
{
	const unsigned char *const bytes = object;
	for (size_t index = 0; index < count; index++) {
		if (bytes[index] != before[index]) {
			return false;
		}
	}
	return true;
}

static bool
same_windows(const struct live_window_record first[LIVE_WINDOW_COUNT],
	     const struct live_window_record second[LIVE_WINDOW_COUNT])
{
	for (uint32_t window = 0; window < LIVE_WINDOW_COUNT; window++) {
		if (!equals(first[window].prefill, second[window].prefill.count,
			    second[window].prefill.elapsed_ns) ||
		    !equals(first[window].decode, second[window].decode.count,
			    second[window].decode.elapsed_ns) ||
		    !equals(first[window].steps, second[window].steps.count,
			    second[window].steps.elapsed_ns)) {
			return false;
		}
		for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS; card++) {
			if (!equals(first[window].cards[card],
				    second[window].cards[card].count,
				    second[window].cards[card].elapsed_ns)) {
				return false;
			}
		}
		for (uint32_t slot = 0; slot < STATS_SLOT_CAPACITY; slot++) {
			if (!equals(first[window].slots[slot].prefill,
				    second[window].slots[slot].prefill.count,
				    second[window]
					    .slots[slot]
					    .prefill.elapsed_ns) ||
			    !equals(first[window].slots[slot].decode,
				    second[window].slots[slot].decode.count,
				    second[window]
					    .slots[slot]
					    .decode.elapsed_ns)) {
				return false;
			}
		}
	}
	return true;
}

static void take(struct live_sampler_record *sampler,
		 const struct engine_stats *engine)
{
	expect(live_sampler_take(sampler, engine) == STATUS_OK,
	       "fake-clock sample admitted");
}

static void read_windows(const struct live_sampler_record *sampler,
			 const struct engine_stats *engine,
			 struct live_window_record windows[LIVE_WINDOW_COUNT])
{
	expect(live_sampler_read(sampler, engine, windows) == STATUS_OK,
	       "fake-clock observer reads windows");
}

static void test_regular_windows(void)
{
	struct live_sampler_record sampler = { 0 };
	struct engine_stats engine = { 0 };
	struct live_window_record windows[LIVE_WINDOW_COUNT] = { 0 };
	take(&sampler, &engine);
	read_windows(&sampler, &engine, windows);
	expect(windows[LIVE_WINDOW_SHORT].decode.elapsed_ns == 0,
	       "one observation has no elapsed coverage");
	for (uint64_t second = 1; second <= TEST_SECONDS; second++) {
		engine.now_ns = seconds_ns(second);
		engine.generated_tokens = second * GENERATION_PER_SECOND;
		engine.prefill_tokens = second * PREFILL_PER_SECOND;
		engine.steps_landed = second * STEPS_PER_SECOND;
		take(&sampler, &engine);
		if (second == WARMUP_SECONDS) {
			read_windows(&sampler, &engine, windows);
			expect(equals(windows[LIVE_WINDOW_SHORT].decode,
				      WARMUP_GENERATED,
				      seconds_ns(WARMUP_SECONDS)) &&
				       equals(windows[LIVE_WINDOW_LONG].decode,
					      WARMUP_GENERATED,
					      seconds_ns(WARMUP_SECONDS)),
			       "warm-up reports actual three-second coverage for both windows");
		}
	}
	read_windows(&sampler, &engine, windows);
	expect(sampler.count == LIVE_SAMPLE_CAPACITY &&
		       equals(windows[LIVE_WINDOW_SHORT].decode,
			      SHORT_GENERATED,
			      seconds_ns(LIVE_WINDOW_SHORT_SECONDS)) &&
		       equals(windows[LIVE_WINDOW_LONG].decode, LONG_GENERATED,
			      seconds_ns(LIVE_WINDOW_LONG_SECONDS)),
	       "ring wrap preserves exact ten- and sixty-second generated counts");
	expect(equals(windows[LIVE_WINDOW_SHORT].prefill, SHORT_PREFILLED,
		      seconds_ns(LIVE_WINDOW_SHORT_SECONDS)) &&
		       equals(windows[LIVE_WINDOW_LONG].prefill, LONG_PREFILLED,
			      seconds_ns(LIVE_WINDOW_LONG_SECONDS)) &&
		       equals(windows[LIVE_WINDOW_SHORT].steps, SHORT_STEPS,
			      seconds_ns(LIVE_WINDOW_SHORT_SECONDS)) &&
		       equals(windows[LIVE_WINDOW_LONG].steps, LONG_STEPS,
			      seconds_ns(LIVE_WINDOW_LONG_SECONDS)),
	       "prefill work and landed steps use their own cumulative counters");
	unsigned char before[sizeof(sampler)];
	memcpy(before, &sampler, sizeof(before));
	struct live_window_record repeated[LIVE_WINDOW_COUNT] = { 0 };
	read_windows(&sampler, &engine, repeated);
	expect(unchanged_bytes(before, &sampler, sizeof(before)) &&
		       same_windows(windows, repeated),
	       "independent observers neither consume nor alter retained history");
}

static void test_sparse_and_fractional(void)
{
	struct live_sampler_record sampler = { 0 };
	struct engine_stats engine = { .now_ns = seconds_ns(FIRST_SECONDS) };
	struct live_window_record windows[LIVE_WINDOW_COUNT] = { 0 };
	take(&sampler, &engine);
	engine.now_ns = seconds_ns(SECOND_SECONDS);
	engine.generated_tokens = SPARSE_FIRST_TOKENS;
	take(&sampler, &engine);
	engine.now_ns = seconds_ns(FRACTIONAL_SECONDS);
	engine.generated_tokens = SPARSE_LAST_TOKENS;
	take(&sampler, &engine);
	engine.now_ns = seconds_ns(FRACTIONAL_CURRENT);
	engine.generated_tokens = FRACTIONAL_GENERATED;
	read_windows(&sampler, &engine, windows);
	expect(equals(windows[LIVE_WINDOW_SHORT].decode, FRACTIONAL_EXPECTED,
		      seconds_ns(LIVE_WINDOW_SHORT_SECONDS)),
	       "cutoff between sparse samples interpolates before adding the fresh endpoint");
	expect(sampler.count == WARMUP_SECONDS,
	       "reads never append missed ticks");
	engine.now_ns = seconds_ns(LARGE_CLOCK_SECONDS) + HALF_SECOND_NS;
	take(&sampler, &engine);
	expect(sampler.count == WARMUP_SECONDS + 1u &&
		       sampler.next_sample_ns ==
			       seconds_ns(LARGE_CLOCK_SECONDS + 1u),
	       "a long pause appends one observation and keeps the next tick on the second grid");
	unsigned char before[sizeof(sampler)];
	memcpy(before, &sampler, sizeof(before));
	engine.now_ns += QUARTER_SECOND_NS;
	take(&sampler, &engine);
	expect(unchanged_bytes(before, &sampler, sizeof(before)),
	       "an extra take before the grid deadline leaves history untouched");
	engine.now_ns = seconds_ns(LARGE_CLOCK_SECONDS + 1u);
	take(&sampler, &engine);
	expect(sampler.count == WARMUP_SECONDS + FIRST_SECONDS,
	       "the next grid boundary admits one sample");

	sampler = (struct live_sampler_record){ 0 };
	engine = (struct engine_stats){ .now_ns = LIVE_SAMPLE_PERIOD_NS +
						  HALF_SECOND_NS };
	take(&sampler, &engine);
	engine.now_ns = seconds_ns(FRACTIONAL_CURRENT);
	engine.generated_tokens = FRACTIONAL_GENERATED;
	read_windows(&sampler, &engine, windows);
	const double expected =
		(double)FRACTIONAL_GENERATED *
		(double)LIVE_WINDOW_SHORT_SECONDS /
		((double)(engine.now_ns - LIVE_SAMPLE_PERIOD_NS -
			  HALF_SECOND_NS) /
		 (double)LIVE_SAMPLE_PERIOD_NS);
	expect(equals(windows[LIVE_WINDOW_SHORT].decode, expected,
		      seconds_ns(LIVE_WINDOW_SHORT_SECONDS)),
	       "fractional count is retained when cutoff lies between the initial point and current endpoint");
}

static void test_counter_wrap(void)
{
	struct live_sampler_record sampler = { 0 };
	struct engine_stats engine = { .now_ns = LIVE_SAMPLE_PERIOD_NS,
				       .generated_tokens = WRAP_BASE,
				       .prefill_tokens = WRAP_BASE,
				       .steps_landed = WRAP_BASE };
	struct live_window_record windows[LIVE_WINDOW_COUNT] = { 0 };
	take(&sampler, &engine);
	engine.now_ns = seconds_ns(FIRST_SECONDS);
	engine.generated_tokens = engine.prefill_tokens = engine.steps_landed =
		WRAP_END;
	read_windows(&sampler, &engine, windows);
	expect(equals(windows[LIVE_WINDOW_SHORT].decode, WRAP_DELTA,
		      LIVE_SAMPLE_PERIOD_NS) &&
		       equals(windows[LIVE_WINDOW_SHORT].prefill, WRAP_DELTA,
			      LIVE_SAMPLE_PERIOD_NS) &&
		       equals(windows[LIVE_WINDOW_SHORT].steps, WRAP_DELTA,
			      LIVE_SAMPLE_PERIOD_NS),
	       "global counter subtraction preserves modulo-uint64 wrap without losing low bits");
}

static void test_request_timeline(void)
{
	struct live_sampler_record sampler = { 0 };
	struct engine_stats engine = { .now_ns = LIVE_SAMPLE_PERIOD_NS,
				       .slots_total = 1 };
	struct live_window_record windows[LIVE_WINDOW_COUNT] = { 0 };
	struct engine_slot_stats *const slot = &engine.slots[0];
	*slot = (struct engine_slot_stats){ .request_tag = REQUEST_TAG,
					    .submitted_ns =
						    LIVE_SAMPLE_PERIOD_NS,
					    .reused_tokens = REUSED_TOKENS,
					    .prompt_processed = REUSED_TOKENS };
	take(&sampler, &engine);
	engine.now_ns = seconds_ns(FIRST_SECONDS);
	slot->started_ns = engine.now_ns;
	take(&sampler, &engine);
	engine.now_ns = seconds_ns(REQUEST_FIRST_SECONDS);
	slot->prompt_processed += REQUEST_PREFILLED;
	take(&sampler, &engine);
	read_windows(&sampler, &engine, windows);
	expect(equals(windows[LIVE_WINDOW_SHORT].slots[0].prefill,
		      REQUEST_PREFILLED, seconds_ns(FIRST_SECONDS)) &&
		       windows[LIVE_WINDOW_SHORT].slots[0].decode.elapsed_ns ==
			       0,
	       "prefill starts at actual work start, subtracts reused tokens and precedes timed decode");
	engine.now_ns = seconds_ns(REQUEST_SECOND_SECONDS);
	slot->prompt_processed = REUSED_TOKENS + REQUEST_SECOND_PREFILL;
	slot->first_token_ns = engine.now_ns;
	slot->generated_tokens = 1;
	take(&sampler, &engine);
	read_windows(&sampler, &engine, windows);
	expect(windows[LIVE_WINDOW_SHORT].slots[0].decode.elapsed_ns == 0,
	       "first generated token alone has no timed decode interval");
	engine.now_ns = seconds_ns(REQUEST_DONE_SECONDS);
	slot->generated_tokens = REQUEST_GENERATED;
	take(&sampler, &engine);
	read_windows(&sampler, &engine, windows);
	expect(equals(windows[LIVE_WINDOW_SHORT].slots[0].decode,
		      REQUEST_TIMED_GENERATED, LIVE_SAMPLE_PERIOD_NS),
	       "decode excludes the first token and measures only time since its publication");
	engine.now_ns = seconds_ns(REQUEST_LATE_SECONDS);
	slot->generated_tokens = REQUEST_LATER_GENERATED;
	take(&sampler, &engine);
	read_windows(&sampler, &engine, windows);
	expect(equals(windows[LIVE_WINDOW_SHORT].slots[0].prefill, 0,
		      seconds_ns(LIVE_WINDOW_SHORT_SECONDS)),
	       "live prefill decays to zero after the prompt finishes");
	engine.now_ns = seconds_ns(REQUEST_REPLACED_SECONDS);
	*slot = (struct engine_slot_stats){
		.request_tag = REPLACEMENT_TAG,
		.submitted_ns = seconds_ns(REQUEST_LATE_SECONDS),
		.started_ns = seconds_ns(REQUEST_LATE_SECONDS),
		.reused_tokens = REUSED_TOKENS,
		.prompt_processed = REUSED_TOKENS + REQUEST_REPLACEMENT_WORK
	};
	read_windows(&sampler, &engine, windows);
	expect(equals(windows[LIVE_WINDOW_SHORT].slots[0].prefill,
		      REQUEST_REPLACEMENT_WORK, LIVE_SAMPLE_PERIOD_NS),
	       "a newly observed tag starts from its own zero-work boundary, never the old slot counters");
	take(&sampler, &engine);
	engine.now_ns = seconds_ns(REQUEST_RESUBMITTED_SECONDS);
	slot->submitted_ns = slot->started_ns =
		seconds_ns(REQUEST_REPLACED_SECONDS + 1u);
	slot->prompt_processed = REUSED_TOKENS + 1u;
	read_windows(&sampler, &engine, windows);
	expect(equals(windows[LIVE_WINDOW_SHORT].slots[0].prefill, 1,
		      LIVE_SAMPLE_PERIOD_NS),
	       "a reused tag with a different submission time cannot reuse the prior identity's history");
}

static void test_cards(void)
{
	struct live_sampler_record sampler = { 0 };
	struct engine_stats engine = { .now_ns = LIVE_SAMPLE_PERIOD_NS,
				       .card_count = CARD_COUNT };
	struct live_window_record windows[LIVE_WINDOW_COUNT] = { 0 };
	for (uint32_t card = 0; card < CARD_COUNT; card++) {
		engine.cards[card].busy_available = true;
		engine.cards[card].busy_observed_ns =
			engine.now_ns + HALF_SECOND_NS;
	}
	take(&sampler, &engine);
	engine.now_ns = seconds_ns(CARD_WARMUP_SECONDS);
	for (uint32_t card = 0; card < CARD_COUNT; card++) {
		engine.cards[card].busy_observed_ns =
			engine.now_ns + HALF_SECOND_NS;
		engine.cards[card].busy_ns =
			(CARD_WARMUP_SECONDS - 1u) *
			(card == 0 ? CARD_QUARTER_BUSY_NS : CARD_HALF_BUSY_NS);
	}
	take(&sampler, &engine);
	read_windows(&sampler, &engine, windows);
	expect(equals(windows[LIVE_WINDOW_SHORT].cards[0],
		      (double)((CARD_WARMUP_SECONDS - 1u) *
			       CARD_QUARTER_BUSY_NS),
		      seconds_ns(CARD_WARMUP_SECONDS - 1u)) &&
		       equals(windows[LIVE_WINDOW_SHORT].cards[1],
			      (double)((CARD_WARMUP_SECONDS - 1u) *
				       CARD_HALF_BUSY_NS),
			      seconds_ns(CARD_WARMUP_SECONDS - 1u)),
	       "coherent card timestamps give known 25 and 50 percent busy windows despite engine-clock skew");
	engine.now_ns = seconds_ns(CARD_SHORT_END_SECONDS);
	for (uint32_t card = 0; card < CARD_COUNT; card++) {
		engine.cards[card].busy_observed_ns =
			engine.now_ns + HALF_SECOND_NS;
		engine.cards[card].busy_ns =
			(CARD_SHORT_END_SECONDS - 1u) *
			(card == 0 ? CARD_QUARTER_BUSY_NS : CARD_HALF_BUSY_NS);
	}
	read_windows(&sampler, &engine, windows);
	expect(equals(windows[LIVE_WINDOW_SHORT].cards[0],
		      (double)(LIVE_WINDOW_SHORT_SECONDS * CARD_QUARTER_BUSY_NS),
		      seconds_ns(LIVE_WINDOW_SHORT_SECONDS)),
	       "a card window's endpoint uses the current coherent observation");
	engine.cards[0].busy_available = false;
	read_windows(&sampler, &engine, windows);
	expect(windows[LIVE_WINDOW_SHORT].cards[0].elapsed_ns == 0 &&
		       windows[LIVE_WINDOW_SHORT].cards[1].elapsed_ns != 0,
	       "unavailable current card data remains unknown without hiding another card");
	take(&sampler, &engine);
	engine.now_ns += LIVE_SAMPLE_PERIOD_NS;
	engine.cards[0].busy_available = true;
	engine.cards[0].busy_observed_ns = engine.now_ns + HALF_SECOND_NS;
	engine.cards[0].busy_ns += CARD_QUARTER_BUSY_NS;
	engine.cards[1].busy_observed_ns = engine.now_ns + HALF_SECOND_NS;
	engine.cards[1].busy_ns += CARD_HALF_BUSY_NS;
	read_windows(&sampler, &engine, windows);
	expect(equals(windows[LIVE_WINDOW_SHORT].cards[0],
		      (double)(LIVE_WINDOW_SHORT_SECONDS * CARD_QUARTER_BUSY_NS),
		      seconds_ns(LIVE_WINDOW_SHORT_SECONDS)),
	       "valid observations bracket a missing historical publication without treating it as zero busy time");
	sampler = (struct live_sampler_record){ 0 };
	engine.cards[0].busy_available = false;
	take(&sampler, &engine);
	engine.now_ns += LIVE_SAMPLE_PERIOD_NS;
	engine.cards[0].busy_available = true;
	engine.cards[0].busy_observed_ns = engine.now_ns + HALF_SECOND_NS;
	engine.cards[1].busy_observed_ns = engine.now_ns + HALF_SECOND_NS;
	read_windows(&sampler, &engine, windows);
	expect(windows[LIVE_WINDOW_SHORT].cards[0].elapsed_ns == 0,
	       "one coherent endpoint cannot fabricate a historical card baseline");
}

static void expect_invalid(struct live_sampler_record *sampler,
			   const struct engine_stats *engine)
{
	unsigned char before[sizeof(*sampler)];
	memcpy(before, sampler, sizeof(before));
	struct live_window_record windows[LIVE_WINDOW_COUNT];
	memset(windows, OUTPUT_SENTINEL, sizeof(windows));
	unsigned char before_windows[sizeof(windows)];
	memcpy(before_windows, windows, sizeof(windows));
	expect(live_sampler_take(sampler, engine) == STATUS_INVALID_ARGUMENT &&
		       unchanged_bytes(before, sampler, sizeof(before)),
	       "invalid sampling leaves every history byte untouched");
	expect(live_sampler_read(sampler, engine, windows) ==
			       STATUS_INVALID_ARGUMENT &&
		       unchanged_bytes(before_windows, windows,
				       sizeof(before_windows)),
	       "invalid observations leave every output byte untouched");
}

static void test_invalid_inputs(void)
{
	struct live_sampler_record sampler = { 0 };
	struct engine_stats engine = { .now_ns = seconds_ns(FIRST_SECONDS),
				       .slots_total = 1 };
	struct live_window_record windows[LIVE_WINDOW_COUNT];
	expect(live_sampler_take(NULL, &engine) == STATUS_INVALID_ARGUMENT &&
		       live_sampler_read(NULL, &engine, windows) ==
			       STATUS_INVALID_ARGUMENT &&
		       live_sampler_read(&sampler, &engine, NULL) ==
			       STATUS_INVALID_ARGUMENT,
	       "null sampler and output pointers are refused");
	expect_invalid(&sampler, NULL);
	sampler.next = LIVE_SAMPLE_CAPACITY;
	expect_invalid(&sampler, &engine);
	sampler = (struct live_sampler_record){ .count = LIVE_SAMPLE_CAPACITY +
							 1u };
	expect_invalid(&sampler, &engine);
	sampler = (struct live_sampler_record){ 0 };
	engine.card_count = BACKEND_MAXIMUM_CARDS + 1u;
	expect_invalid(&sampler, &engine);
	engine.card_count = 0;
	engine.slots_total = STATS_SLOT_CAPACITY + 1u;
	expect_invalid(&sampler, &engine);
	engine.slots_total = 1;
	engine.slots[0].token_ring_next = STATS_TOKEN_RING;
	expect_invalid(&sampler, &engine);
	engine.slots[0].token_ring_next = 0;
	engine.step_ring_count = STATS_STEP_RING + 1u;
	expect_invalid(&sampler, &engine);
	engine.step_ring_count = 0;
	take(&sampler, &engine);
	engine.now_ns = LIVE_SAMPLE_PERIOD_NS;
	expect_invalid(&sampler, &engine);
	engine.now_ns = seconds_ns(FIRST_SECONDS);
	engine.slots[0].started_ns = engine.now_ns + 1u;
	expect_invalid(&sampler, &engine);
	engine.slots[0].started_ns = 0;
	engine.card_count = 1;
	engine.cards[0].busy_available = true;
	engine.cards[0].busy_observed_ns = engine.now_ns;
	engine.now_ns += LIVE_SAMPLE_PERIOD_NS;
	take(&sampler, &engine);
	engine.cards[0].busy_observed_ns--;
	expect_invalid(&sampler, &engine);
}

int main(void)
{
	test_regular_windows();
	test_sparse_and_fractional();
	test_counter_wrap();
	test_request_timeline();
	test_cards();
	test_invalid_inputs();
	(void)printf("live_metrics_test: %d failures\n", failures);
	return failures == 0 ? 0 : 1;
}
