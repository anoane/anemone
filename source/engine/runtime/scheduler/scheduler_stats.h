/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Release live counters. All times are monotonic nanoseconds, counters wrap modulo 2^64.
 * Snapshot copies fixed records under the engine's existing lock; no formatting or device work under it.
 * Per-card records use the pipeline's existing lock. Device free memory is refreshed after both copies.
 * No snapshot resets counters: independent observers cannot consume one another's interval.
 */
#ifndef SCHEDULER_STATS_H
#define SCHEDULER_STATS_H

#include "engine/backend/backend_seam.h"
#include "front_end_seam.h"

#define STATS_TOKEN_RING 64u
#define STATS_STEP_RING 64u
#define STATS_ITERATION_BUCKETS 13u
#define STATS_THIN_KINDS 8u
#define STATS_WIDE_KIND STATS_THIN_KINDS
#define STATS_FULL_KIND (STATS_WIDE_KIND + 1u)
#define STATS_DRAFT_KIND (STATS_FULL_KIND + 1u)
#define STATS_STEP_KINDS (STATS_DRAFT_KIND + 1u)
#define STATS_BATCH_PROGRAMS SERVING_BATCH_COSTS

enum stats_phase {
	STATS_IDLE,
	STATS_QUEUED,
	STATS_PROMPT,
	STATS_DECODE,
	STATS_FINISHING
};

enum stats_speculation_mode {
	STATS_SPECULATION_OFF,
	STATS_SPECULATION_LOOKUP,
	STATS_SPECULATION_DEVICE,
	STATS_SPECULATION_HYBRID
};

struct engine_slot_stats {
	uint32_t slot, request_tag, phase, operation;
	uint32_t prompt_tokens, reused_tokens, prompt_processed,
		generated_tokens;
	uint32_t position, context_capacity, reach, decoder_from, checkpoint;
	uint32_t kept_lag; /* fixed supported reuse distance behind reach */
	uint32_t reach_lag; /* rows written beyond current processed position */
	uint32_t chosen_depth, draft_source, speculation_mode;
	uint32_t confidence_bins[DRAFT_CALIBRATION_POSITIONS];
	bool has_confidence, reused_checkpoint;
	uint64_t submitted_ns, started_ns, phase_since_ns, first_token_ns,
		last_token_ns, finished_ns;
	uint64_t prefill_ns, decode_ns;
	uint64_t chunks[STATS_STEP_KINDS];
	uint64_t token_times[STATS_TOKEN_RING];
	uint32_t token_ring_next, token_ring_count;
	uint64_t probes, device_draft_tails;
	struct draft_statistics speculation;
	uint32_t finish_kind, last_error;
};

struct engine_card_stats {
	uint32_t engine_index, backend_index;
	struct card_description_record description;
	uint64_t layer_mask, head_mask, arena_bytes;
	bool holds_draft, free_memory_available, busy_available;
	uint64_t steps, visits, busy_ns;
	uint64_t busy_observed_ns; /* coherent busy counter's own refresh time; zero while publication contends */
};

struct stats_step_sample {
	uint64_t landed_ns, latency_ns;
};

/* A BATCH program of the step plan (description batch_step_us): its rows and verifiers, the steps it took, their
 * summed wall time (submission to landing) and the plan's current cost of a step. */
struct stats_batch_program {
	uint32_t rows, verifiers, cost_us;
	uint64_t steps, step_ns;
};

/* Persistent across generations and snapshot operations. Position is committed/processed context; reach may
 * include speculative cache rows. Idle starts when the final event is TAKEN, initially at engine startup. */
struct engine_session_stats {
	uint32_t position, reach, context_capacity, phase, operation;
	bool active;
	uint64_t idle_since_ns;
	uint64_t snapshots_saved, snapshots_loaded, snapshot_save_bytes,
		snapshot_load_bytes;
	uint64_t snapshot_save_ns, snapshot_load_ns;
};

struct engine_stats {
	uint64_t now_ns, started_ns;
	uint32_t slots_total, slots_busy, queued_commands, event_queue_depth,
		flights;
	uint32_t speculation_mode; /* 0 off, 1 lookup, 2 DSpark, 3 hybrid */
	uint64_t prompt_tokens, cached_tokens, generated_tokens, prefill_tokens;
	uint64_t served, failed, cancelled, backpressure_stalls;
	uint64_t steps_landed, steps[STATS_STEP_KINDS];
	/* Computed prompt + emitted generation tokens per landed engine iteration; metrics opt-in. */
	uint64_t iteration_tokens[STATS_ITERATION_BUCKETS + 1u],
		iteration_token_sum;
	uint64_t checkpoints_saved, checkpoints_restored;
	uint64_t snapshots_saved, snapshots_loaded, snapshot_save_bytes,
		snapshot_load_bytes;
	uint64_t snapshot_save_ns, snapshot_load_ns;
	uint64_t probes, device_draft_tails;
	struct draft_statistics speculation;
	struct stats_step_sample step_samples[STATS_STEP_RING];
	uint32_t step_ring_next, step_ring_count, card_count;
	uint32_t batch_program_count; /* 0: no step plan */
	struct stats_batch_program batch_programs[STATS_BATCH_PROGRAMS];
	struct engine_card_stats cards[BACKEND_MAXIMUM_CARDS];
	/* By session slot, slot_room entries each (engine_stats_tables); a copy fills slots_total of them. A record
	 * owns its tables: copy one with engine_stats_copy, never by assignment. */
	uint32_t slot_room;
	struct engine_slot_stats *slots;
	struct engine_session_stats *sessions;
};

/* Zeroed tables of slot_count entries for a record that copies (engine_stats_counters, engine_stats_copy): taken at
 * create, never on the token path. engine_stats_tables_free releases them (NULL-safe, any record). */
MUST_CHECK result_status engine_stats_tables(struct engine_stats *stats,
					     uint32_t slot_count);
void engine_stats_tables_free(struct engine_stats *stats);
/* Every field of input, and slots_total entries of each table into output's own tables, which output keeps.
 * INVALID_ARGUMENT, output untouched: a NULL record, or output's tables shorter than slots_total. */
MUST_CHECK result_status engine_stats_copy(struct engine_stats *output,
					   const struct engine_stats *input);

/* Enable before the first command; BUSY_VALUE after any admission or with queued commands.
 * Disabling is allowed at any time. Uses only published records under the existing lock; no token hook. */
MUST_CHECK result_status
engine_stats_metrics_enable(struct inference_engine *engine, bool enabled);

/* NULL inputs or output tables shorter than the engine's slots: INVALID_ARGUMENT, output untouched. Callable while
 * commands run. Fixed memory and bounded copy; card_describe may query device memory only here, never on the token
 * path. */
MUST_CHECK result_status engine_stats_snapshot(struct inference_engine *engine,
					       struct engine_stats *stats_out);
/* Same cumulative/session/busy records without refreshing device free memory. Poll-thread samplers use this
 * fixed, bounded copy; it never calls a backend or device API. */
MUST_CHECK result_status engine_stats_counters(struct inference_engine *engine,
					       struct engine_stats *stats_out);
/* Retains the last generation record after final TAKEN until the slot receives another generation. No card work. */
MUST_CHECK result_status engine_stats_slot(struct inference_engine *engine,
					   uint32_t session_slot,
					   struct engine_slot_stats *stats_out);
/* Startup metadata, before submitting commands. Zero count removes metadata; other inputs describe engine order.
 * Backend indices select card_describe, descriptions retain the chosen arithmetic class. */
MUST_CHECK result_status
engine_stats_cards(struct inference_engine *engine,
		   const struct engine_card_stats *cards, uint32_t card_count);

#endif
