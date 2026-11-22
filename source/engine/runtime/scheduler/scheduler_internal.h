/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* SOURCE ONLY: the engine's records (scheduler.c), which scheduler_sessions.c shares; one engine's, guarded as
 * noted. */
#ifndef SCHEDULER_INTERNAL_H
#define SCHEDULER_INTERNAL_H

#include <pthread.h>
#include <stdbool.h>

#include "front_end_seam.h"
#include "engine/runtime/step/step.h"
#include "scheduler_stats.h"

#include "engine/runtime/speculation/draft_lookup.h"
#include "serving_cursor.h"

#define EVENT_CAPACITY \
	256u /* events queued for the front end: one step's at most, many times */
#define DRAWING_ROWS \
	1u /* a pipeline's step takes one session that draws: each such session flies in steps apart */
#define ENGINE_LINE_APART_BYTES \
	64u /* a cache line: members this far apart never share one */

enum slot_phase {
	SLOT_IDLE,
	SLOT_COMMANDED,
	SLOT_FINAL_QUEUED
}; /* commanded: queued or running, final not queued */

struct slot {
	uint32_t phase; /* enum slot_phase */
	uint32_t token_count; /* what the session holds, as its last final event taken said */
	bool is_cancelled;
	uint32_t request_tag; /* live until final TAKEN */
	uint32_t processed_count; /* of token_count, those the cards ran: a command may keep them */
	uint32_t decoder_from; /* a command keeping fewer tokens runs a cone first (model_seam.h) */
	uint32_t reach_count; /* the kept lag counts from it (serving_cursor.h) */
	uint32_t checkpoint_count; /* prompt checkpoint's kept boundary; 0: none */
	struct draft_statistics
		statistics; /* last completed generation, under lock */
};

struct queued_event {
	struct generation_event event;
	uint32_t session_slot;
	uint32_t processed_count; /* of a final event: the positions its command ran */
};

/* A queued command: a generation command, or a session command (operation not 0). */
struct queued_command {
	struct generation_command_record generation;
	struct session_command session;
	struct session_clone_command clone;
	uint64_t clone_submitted_ns;
	bool restores_checkpoint;
};

/* A step in flight (a pipeline's): its rows, each owner's together in turn order. */
struct flight {
	uint64_t submitted_ns;
	uint32_t row_count, owner_count;
	uint32_t owners[FLIGHT_MOST_OWNERS];
	uint32_t taken[FLIGHT_MOST_OWNERS]; /* rows, by owner */
	uint32_t proposed
		[FLIGHT_MOST_OWNERS]; /* of them, the proposals behind its anchor, by owner */
	uint32_t proposal_tokens[FLIGHT_MOST_OWNERS][SERVING_DRAFT_MOST];
	bool drafts[FLIGHT_MOST_OWNERS]; /* indexed by owner turn */
	/* BATCH only (its engine takes at most STEP_BATCH_MAXIMUM_SESSIONS slots): indexed by session slot */
	struct step_round rounds[STEP_BATCH_MAXIMUM_SESSIONS];
	uint32_t round_valid_mask; /* of rounds; published only after every owner validates */
	struct step_row_record *rows; /* the engine's row_capacity */
};

/* A speculating owner's proposals for the positions behind its anchor at anchor_position, whose token the cursor
 * knows: at most most of them, their count. The drafter seam: scheduler.c's lookup drafter, the checkpoint's own draft
 * (DSpark, stage 2) behind it next, and a test's scripted drafter in its place before any command. */
typedef uint32_t (*engine_draft_function)(struct inference_engine *engine,
					  const struct command_cursor *cursor,
					  uint32_t anchor_position,
					  uint32_t most,
					  uint32_t *proposals_out);

#define SERVING_BATCH_COST_ROWS \
	(STEP_BATCH_MAXIMUM_SESSIONS * (SERVING_DRAFT_MOST + 1u))

struct inference_engine {
	struct step_driver *driver;
	struct engine_description_record description;
	struct step_row_result *results; /* row_capacity: a landing step's */
	engine_draft_function
		draft; /* with speculative_depth: the stepping thread's */
	struct draft_lookup *
		lookups; /* [session slot] with speculative_depth, else NULL: the stepping thread's */
	pthread_t thread;
	struct draft_policy_record
		draft_policy; /* pinned at create (draft_learned_acceptance: its table learned) */
	/* draft_learned_acceptance: the given table (the prior) and the outcomes of reached DSpark drafts a cell */
	uint32_t acceptance_prior[DRAFT_SAMPLING_MODES]
				 [DRAFT_CALIBRATION_POSITIONS]
				 [DRAFT_CONFIDENCE_BINS];
	uint32_t learned_reached[DRAFT_SAMPLING_MODES]
				[DRAFT_CALIBRATION_POSITIONS]
				[DRAFT_CONFIDENCE_BINS];
	uint32_t learned_matched[DRAFT_SAMPLING_MODES]
				[DRAFT_CALIBRATION_POSITIONS]
				[DRAFT_CONFIDENCE_BINS];
	/* description batch_step_us: the BATCH programs' rows, fewest first, and a step's microseconds on each */
	uint32_t batch_cost_rows[SERVING_BATCH_COSTS],
		batch_cost_us[SERVING_BATCH_COSTS];
	uint32_t batch_cost_verifiers
		[SERVING_BATCH_COSTS]; /* step_batch_verifiers */
	uint32_t batch_cost_count;
	uint32_t batch_cost_prior_us[SERVING_BATCH_COSTS];
	struct draft_cost_cell
		batch_shape_costs[SERVING_BATCH_COST_ROWS + 1u]
				 [STEP_BATCH_MAXIMUM_SESSIONS + 1u];
	uint32_t most_rows;
	uint32_t wide_rows; /* the driver's wide program's (step_wide_rows), else 0 */
	uint32_t full_rows; /* its largest full program's (step_full_rows), else 0 */
	uint32_t row_capacity; /* the most rows a step carries: most_rows, wide_rows or full_rows */
	uint32_t cone_rows; /* the driver's (step_cone_rows) with front steps, else 0 */
	uint32_t draft_positions; /* the device draft's fixed block size; 0 selects the host lookup drafter */
	int event_descriptor;
	bool is_batch; /* explicit BATCH driver: one eager flight, up to64 causal rows */
	pthread_mutex_t lock; /* guards everything below */
	uint64_t metrics_steps, metrics_prefill, metrics_generated;
	pthread_cond_t wake; /* the stepping thread waits on it */
	struct queued_command *commands;
	struct slot *slots;
	/* [session slot], sized at create like slots */
	uint64_t *slot_generations;
	bool *slot_read_leased;
	struct session_clone_metrics *clone_metrics;
	uint32_t *slot_taken; /* the rows a flight takes while flight_taken_off plans it; 0 between flights */
	struct command_cursor
		*cursors; /* worker-private fixed per-slot records */
	uint64_t admissions; /* generation commands taken so far: the next one's admitted */
	struct draft_statistics
		statistics; /* completed generation commands, protected by lock */
	struct flight flights[STEP_PIPELINE_MOST];
	struct engine_stats
		live; /* protected by lock; retained independently of diagnostics */
	struct engine_stats
		worker_live; /* stepping thread's counters, published at existing lock boundaries */
	uint32_t command_first,
		command_count; /* a ring of session_slot_count commands */
	uint32_t event_first, event_count;
	uint32_t next_slot;
	uint32_t head_chunks_in_row; /* prefill_decode_every: the prompt head's chunk steps since a decoding step */
	uint32_t prefill_chunk_limit; /* current flight's exact whole-granule cap; zero: the original plan */
	/* Sliced prefill: extra turns and chunk visits leave the next due decode turn alone. */
	uint32_t decode_next_slot;
	uint32_t decode_rounds_left; /* remaining ordinary rounds in a sliced-prefill burst */
	uint32_t head_bypass_steps; /* prefill_order_wait: the steps of a head other than the oldest prompt in a row */
	result_status failure; /* shared driver poison; protected by lock */
	uint32_t depth; /* steps in flight at most: the driver's pipeline */
	uint32_t flight_first, flight_count; /* worker-private ring */
	struct queued_event events[EVENT_CAPACITY];
	bool metrics_enabled; /* protected by lock; only iteration accounting opts in */
	bool is_ending;
	bool checkpoint_barrier; /* no flight follows a prompt's last row until its checkpoint is copied */
	/* engine_submit_forced's continuation, NULL for every other command: written under the lock by a forced submit
	 * and by its command's end, read by the stepping thread. The two gaps, never written, keep every other byte 64
	 * or more from it: every member before it and the arrays engine_open lays after the struct (the queued commands
	 * first). So the cache line it lies on, which no thread writes while no forced command runs, holds nothing else
	 * at any alignment the engine is allocated at (b6's rule); appended, it moves no member. */
	unsigned char forced_apart[ENGINE_LINE_APART_BYTES];
	const struct engine_forced_continuation *forced;
	unsigned char forced_apart_after[ENGINE_LINE_APART_BYTES];
};

/* Product timestamps; zero if the monotonic clock fails. No optional timing hooks. */
uint64_t engine_now_ns(void);
/* With engine lock held, publish worker-owned fixed records at an existing boundary. */
void engine_live_publish(struct inference_engine *engine);
/* Worker only: records one successfully landed step without taking a lock. */
void engine_live_iteration(struct inference_engine *engine);
void engine_live_step(struct inference_engine *engine,
		      const struct step_row_record *rows, uint32_t row_count,
		      uint64_t submitted_ns);
/* Exact plan: only whole-granule chunks obey prefill_chunk_limit; the reference tail retains its own forms. */
uint32_t engine_exact_chunk_rows(const struct inference_engine *engine,
				 enum program_kind kind, uint32_t rows,
				 bool has_tail);

/* Worker-private draft allocation for an already selected set of owners. caps is indexed by owner turn;
 * UINT32_MAX leaves its local policy in charge. No driver call or allocation. */
void engine_draft_depths(struct inference_engine *engine,
			 const struct flight *flight, const uint32_t *taken,
			 uint32_t row_count, uint32_t *caps);

/* scheduler.c's: an event queued, waiting for room (false once the engine ends); the driver's failure noted; the oldest
 * flight landed. */
bool engine_event_queued(struct inference_engine *engine, uint32_t session_slot,
			 struct generation_event event,
			 uint32_t processed_count);
void engine_driver_failed(struct inference_engine *engine,
			  result_status status);
void engine_flight_landed(struct inference_engine *engine);
/* scheduler_sessions.c's: a session command once no step is in flight, the oldest flight landing first. */
void engine_session_advanced(struct inference_engine *engine,
			     struct command_cursor *cursor);

#define OK STATUS_OK
#define PREFILL_CONE_UNTRIMMED \
	3u /* prefill_cone_skip: 1's cone, the wide trunk products every row (51cd081's 1) */
#define EXACT_PLAN_GRANULE \
	256u /* the exact plan (exact_chunk_rows_of): the reference ladder's middle program */
#define EXACT_PLAN_TAIL_ROWS \
	128u /* prefill_tail_rows: the tail program every one takes (two 64-row chunks) */

/* prefill_dead_halves (front_end_seam.h): the exact plan's full chunks of whole STEP_FULL_GRANULE_ROWS-row
 * granules (a 320-row tail program's are not) lie before a tail of at least EXACT_PLAN_GRANULE rows, so each one's
 * last row lies at least EXACT_PLAN_GRANULE - kept rows before its prompt's cone end's last row (cone_end_of). */
_Static_assert(EXACT_PLAN_GRANULE == STEP_FULL_GRANULE_ROWS,
	       "a full chunk of whole granules lies before a tail");

struct generation_command_record;
bool is_cancelled(struct inference_engine *engine, uint32_t session_slot);
uint32_t command_end(const struct generation_command_record *command);

#endif
