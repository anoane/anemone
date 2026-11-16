/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* SOURCE ONLY: one worker owns these records; no per-turn allocation. */
#ifndef SERVING_CURSOR_H
#define SERVING_CURSOR_H

#include <stddef.h>
#include "draft_cost.h"
#include "reasoning.h"
#include "anemone/front_end_seam.h"
#include "anemone/engine_stats.h"
#include "anemone/model_layout.h"
#include "anemone/model_seam.h"
#include "anemone/step_records.h"

#define SERVING_STABLE_GRANULE 8u /* fixed-capacity prefill slab phase */

#define SERVING_PAIR_ROWS 2u
/* The most session slots a server takes: a bound on slot index arithmetic and allocation sizes, not a serving cap
 * (the fit or --session_slots gives the count). */
#define SERVING_SLOTS_MOST 1024u
/* The most owners one flight carries: one row each at most, so a step's most rows (thin, wide or BATCH). Per-turn
 * tables are this wide whatever the slot count. */
#define FLIGHT_MOST_OWNERS STEP_BATCH_MAXIMUM_ROWS
_Static_assert(FLIGHT_MOST_OWNERS >= LINEAR_MAXIMUM_ROWS,
	       "a thin step's owners fit the per-turn tables");
/* Proposals a speculative round verifies at most: a thin step's rows less its anchor. Its rejected rows write state
 * up to this many positions past the session's committed count. */
#define SERVING_DRAFT_MOST (LINEAR_MAXIMUM_ROWS - 1u)
/* draft_session_acceptance 2 and 3: the reached drafts a command's log-odds shift (and scale) weighs, its last ones */
#define SERVING_SESSION_SHIFT_DRAFTS 64u
_Static_assert(
	SERVING_DRAFT_MOST <= DRAFT_STAT_POSITIONS,
	"draft statistics must cover the complete verification row budget");
/* A command may keep the first tokens of its session's earlier ones (prefix reuse): where no 2-to-1 group of a pool
 * is open, and no further back than the window ring still holds the 127 rows before them. The lag counts back from the
 * slot's reach, the most tokens its cards ran since a command last kept none (a load's: its snapshot's), not from what
 * the session holds now: an earlier, longer command's rows linger in the ring, and a speculative round wrote rows up
 * past its session's end. Rejected rows and the device draft's positions count in the reach too. One that keeps none
 * starts anew, whatever its slot
 * held. */
#define SERVING_KEPT_GRANULE MODEL_POOLED_RATIO
#define SERVING_KEPT_LAG (MODEL_RING_SLOTS - MODEL_WINDOW)
/* A command's prompt rows before its last cone rows (model_seam.h) and this many more run as front steps: the margin
 * leaves the decoder's state whole for every prefix a later command may keep. */
#define SERVING_CONE_MARGIN (SERVING_KEPT_LAG - 1u)
/* Lookup alone: a poorly accepted round is followed by ordinary drawing rows, then a probe at depth one. */
#define SERVING_DRAFT_COOLDOWN 15u
#define SERVING_SESSION_COST_TAIL_COUNTS 2u

struct command_cursor {
	struct generation_command_record command;
	struct session_command session; /* operation 0: a generation command */
	struct session_clone_command clone;
	uint64_t clone_submitted_ns;
	uint64_t admitted; /* the command's turn in the order commands came (engine.c's prompt_head) */
	/* draft_session_acceptance: this command's reached DSpark drafts, matched and expected by the table, decayed;
	 * and their decayed count */
	double session_hits, session_expected, session_reached;
	/* Levels 6 to 9: conditional counts by position, including the first cut draft's hindsight label. */
	double session_position_hits[DRAFT_CALIBRATION_POSITIONS];
	double session_position_reached[DRAFT_CALIBRATION_POSITIONS];
	double session_position_expected
		[DRAFT_CALIBRATION_POSITIONS]; /* residual calibration, level 10 */
	/* Level 10: bounded BATCH exploration, counted only after full verification. */
	uint64_t session_last_full_round;
	/* draft_session_acceptance 2 to 4: the shift of its drafts' table log-odds; 3: their scale less 1 */
	double session_shift, session_slope_offset;
	struct step_round
		draft_round; /* device proposals for this command's next anchor; none before its prompt */
	struct draft_statistics statistics;
	uint64_t live_chunks[STATS_STEP_KINDS];
	uint64_t live_chunk_step; /* the landed step (steps_landed) that last counted one of its prompt chunks */
	uint64_t live_probes, live_dspark_tails;
	uint32_t position, generated, token_id;
	struct reasoning_state reasoning;
	uint32_t front_end; /* its rows below it are front rows (model_seam.h) */
	uint32_t front_tail_end; /* the reference plan's front end (engine.c's exact plan): front_end or below */
	uint32_t in_flight; /* rows of steps in flight (a pipeline's) */
	uint32_t rounds, proposals,
		accepted; /* speculative rounds, the proposals they verified and those accepted */
	uint32_t depth; /* the most proposals the lookup drafter offers next (engine.c's lookup_drafted) */
	uint32_t cooldown; /* drawing opportunities left without proposals, before the next probe */
	uint32_t dspark_depth, rounds_since_dspark;
	float draft_confidence
		[DRAFT_CALIBRATION_POSITIONS]; /* latest returned tail, retained when skipped */
	/* Level 11: one-owner costs by active-tail presence and depth. */
	struct draft_cost_cell session_costs[SERVING_SESSION_COST_TAIL_COUNTS]
					    [DRAFT_DEPTH_COUNTS];
	/* draft_session_acceptance 2 to 4: its last reached drafts' table log-odds (a ring, session_seen drafts put in;
	 * 4: with each round's first unverified draft checked in hindsight), their outcomes in session_matched */
	float session_log_odds[SERVING_SESSION_SHIFT_DRAFTS];
	uint32_t session_seen;
	uint32_t live_chosen_depth;
	uint32_t live_prompt_processed; /* successful prompt-row high-water, including initially reused tokens */
	uint32_t draft_source; /* source selected for the next verifying flight */
	result_status status;
	bool session_full_depth; /* levels 7 to 9: request-local full-depth selector, with hysteresis */
	bool session_had_full_round; /* Level 10: whether session_last_full_round holds a full round */
	/* draft_session_acceptance 2 to 4: those drafts' outcomes, in session_log_odds' ring */
	uint8_t session_matched[SERVING_SESSION_SHIFT_DRAFTS];
	bool has_draft_confidence, dspark_opportunity, needs_dspark_tail;
	bool live_probe_planned;
	bool restores_checkpoint; /* worker restores it before planning or dispatching any row */
	bool images_pending,
		images_started; /* image encoder quanta precede this request's text prefill */
	bool is_ready, sampler_was_set,
		awaits_token; /* the command's row with logits is in flight */
};

/* Startup-only metadata check; the driver still validates all cards/areas/launches. */
static inline bool
serving_program_matches(const struct execution_program *program, uint32_t rows,
			uint32_t row_capacity, uint32_t session_capacity,
			uint32_t context_tokens)
{
	return program != NULL && program->kind == PROGRAM_THIN &&
	       program->row_count == rows &&
	       program->limits.row_capacity == row_capacity &&
	       program->limits.session_capacity == session_capacity &&
	       program->limits.context_token_limit == context_tokens;
}
#endif
