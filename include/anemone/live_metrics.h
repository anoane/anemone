/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Poll-thread sampling of engine cumulative counters. No allocation, device work, observer state or token-path
 * updates. take() appends at most once per monotonic second in fixed storage, O(1) in elapsed time; missed ticks
 * are not fabricated. read() never mutates history. The caller supplies the clock through engine.now_ns.
 * Windows end at the supplied snapshot and use up to the last 10/60 seconds available. A fractional boundary
 * interpolates adjacent cumulative samples. Per-request history is keyed by tag AND submission time, clips to
 * the phase start, excludes reused prompt tokens and the first generated token from timed decode intervals.
 * Unavailable/zero-duration measurements have elapsed_ns=0; zero work in an observed interval has count=0.
 * Card windows use their own coherent busy observation timestamps; unavailable current observations stay unknown.
 * Zero-initialize a sampler before its first use. Null inputs, invalid ring/count/card/slot bounds or a backwards
 * clock are INVALID_ARGUMENT and leave outputs/history untouched. Counters may wrap modulo 2^64.
 */
#ifndef LIVE_METRICS_H
#define LIVE_METRICS_H

#include "anemone/engine_stats.h"

#define LIVE_SAMPLE_PERIOD_NS 1000000000ull
#define LIVE_SAMPLE_CAPACITY 64u
#define LIVE_WINDOW_COUNT 2u
#define LIVE_WINDOW_SHORT 0u
#define LIVE_WINDOW_LONG 1u
#define LIVE_WINDOW_SHORT_SECONDS 10u
#define LIVE_WINDOW_LONG_SECONDS 60u

/* count is tokens for decode/prefill, steps for steps, and nanoseconds for card busy time. */
struct live_measurement {
	double count;
	uint64_t elapsed_ns;
};

struct live_slot_window {
	struct live_measurement prefill, decode;
};

struct live_window_record {
	struct live_measurement prefill, decode, steps;
	struct live_measurement cards[BACKEND_MAXIMUM_CARDS];
	struct live_slot_window slots[STATS_SLOT_CAPACITY];
};

struct live_counter_slot {
	uint64_t submitted_ns;
	uint32_t request_tag, processed, generated;
};

struct live_counter_card {
	uint64_t observed_ns, busy_ns;
	bool available;
};

struct live_counter_sample {
	uint64_t now_ns, generated, processed, steps;
	struct live_counter_card cards[BACKEND_MAXIMUM_CARDS];
	struct live_counter_slot slots[STATS_SLOT_CAPACITY];
};

struct live_sampler_record {
	uint32_t next, count;
	uint64_t next_sample_ns;
	struct live_counter_sample samples[LIVE_SAMPLE_CAPACITY];
};

MUST_CHECK result_status live_sampler_take(struct live_sampler_record *sampler,
					   const struct engine_stats *engine);
MUST_CHECK result_status
live_sampler_read(const struct live_sampler_record *sampler,
		  const struct engine_stats *engine,
		  struct live_window_record windows_out[LIVE_WINDOW_COUNT]);

#endif
