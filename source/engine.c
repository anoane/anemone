/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * engine.c: the engine of front_end_seam.h over one step driver. Contract: front_end_seam.h.
 *
 * The stepping thread takes the oldest command, sets its slot's sampler, reads its new tokens in steps of up to
 * most_rows rows (the head computes the last row's logits only, and none when nothing is to be generated), then steps
 * one row a token: each drawn token is a TOKEN event; the end token, the command's limit and the context's end each
 * end the command with its final event, and a cancel is looked for before every step. One mutex guards the commands,
 * the events and the slots, and one condition wakes the stepping thread (a command, room for an event, the end of
 * the engine): it holds the lock to take a command, to queue an event and to look for a cancel, never across a step.
 * The event descriptor is a semaphore eventfd: written once a queued event and read once a taken event, so it is
 * readable while an event is queued (and, for a moment after a take, once more: a wake that finds none). Fault point:
 * engine_resources.
 * Several sessions, or a driver with a pipeline (step.h), go by flights: steps of the rows of several owners, up to
 * the pipeline's depth of them in flight at once; an owner's prompt rows fly ahead, its next drawn row waits for the
 * token of its last.
 * Speculation (front_end_seam.h) goes by flights too, a round a flight: the owner's anchor and its drafter's proposals
 * fly together, every row drawn, at most one owner with proposals a step; when it lands, its tokens are its draws up
 * to the first that differs from the proposal behind it or the end token (keyed match), one at a time as steps of one
 * drawing row would give them, and the driver's history of the session is cut back to the rows those tokens commit.
 */
#include "anemone/front_end_seam.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include "anemone/chat_images.h"
#include "anemone/step_images.h"
#include "anemone/hooks.h"
#include "anemone/step.h"

#include "engine_internal.h"
#include "draft_session.h"
#include "serving_cursor.h"
#include "step_rounds.h"

#define OK STATUS_OK

static bool token_ends_generation(const struct inference_engine *engine,
				  uint32_t token_id)
{
	return engine->description.ignore_eos == 0 &&
	       token_id == engine->description.end_token_id;
}

#define PROMPT_CHUNK_SNAPSHOT_SALT \
	0x4e3750524f4d5032ull /* prompt program selection and retained turns identify state */
#define NANOSECONDS_PER_MICROSECOND 1000u
#define BATCH_COST_WEIGHT 8u /* a BATCH program's cost moves an eighth of the way to each step it took */
#define ACCEPTANCE_PRIOR_REACHED 16u /* draft_learned_acceptance: the given table weighs as many reached outcomes */
#define ACCEPTANCE_HALVED_REACHED 512u /* a cell's outcomes are halved once it holds as many reached drafts */
#define SESSION_ACCEPTANCE_PRIOR 4.0 /* draft_session_acceptance: expected matches of weight at the ratio 1 */
#define SESSION_ACCEPTANCE_DECAY 0.98 /* a session's counts decay by it a reached draft (about 50 remembered) */
#define SESSION_SHIFT_MODE 2u /* draft_session_acceptance: the table's log-odds shifted per command */
#define SESSION_SHIFT_PRIOR 4.0 /* the shift's prior precision: as many drafts at shift 0 */
#define SESSION_SHIFT_STEPS 4u /* Newton steps a round, from the last shift */
#define SESSION_PROBABILITY_FLOOR 0.001 /* a table acceptance's log-odds from this to 1 less it */
#define SESSION_PLATT_MODE 3u /* draft_session_acceptance: the table's log-odds scaled and shifted per command */
#define SESSION_PLATT_PRIOR 1.0 /* the scale's (less 1) and the shift's prior precisions, at 0 */
#define SESSION_SLOPE_LEAST 0.0 /* the least scale: a draft's acceptance never falls as its table's rises */
#define SESSION_HINDSIGHT_MODE 4u /* 2, with each round's first unverified draft checked against the target's draw */
#define SESSION_FULL_MODE 5u /* every draft verified while the command's drafts (4's checks too) match at the rate */
#define SESSION_FULL_RATE 0.82 /* gsl08a a request: code's reached drafts match at 0.84-0.92, prose's at 0.72-0.80 */
#define SESSION_FULL_LEAST 16.0 /* the decayed reached drafts a command needs before it may verify every draft */
#define SESSION_POSITION_MODE \
	6u /* conditional acceptance learned separately at each draft position */
#define SESSION_POSITION_PRIOR \
	4.0 /* the current confidence estimate's weight in reached drafts */
#define SESSION_SCOUT_MODE 7u
#define SESSION_SCOUT_WARM_ROUNDS \
	4u /* full blocks before command-local decisions */
#define SESSION_SCOUT_LEAST_REACHED \
	2.0 /* effective conditional observations at each middle position */
#define SESSION_SCOUT_ENTER_RATE \
	0.90 /* fixed5: code middle positions average 0.96, prose about 0.72 */
#define SESSION_SCOUT_LEAVE_RATE \
	0.82 /* hysteresis keeps a single rejection from changing the command's regime */
#define SESSION_SECOND_MODE 8u
#define SESSION_NATURAL_MODE 9u
#define SESSION_RESIDUAL_MODE 10u
#define SESSION_COST_MODE 11u
#define SESSION_NATURAL_LEAST_REACHED \
	3.5 /* four observations after decay, with no full-depth warmup */
#define SESSION_SECOND_POSITION 1u
#define SESSION_SECOND_LEAST_REACHED 1.5 /* two observations after decay */
#define SESSION_SECOND_ENTER_RATE 0.90
#define SESSION_SECOND_LEAVE_RATE 0.80
#define SESSION_MIDDLE_LEAST_REACHED 3.5 /* four observations after decay */
#define SESSION_MIDDLE_VETO_RATE \
	0.25 /* repeated middle failures override the second-position selector */
/* The only signals the stepping thread takes, raised by its own faults: blocked, the kernel would end the process
 * before any handler (a sanitizer's) could report. The program's threads take every signal sent to the process. */
static const int fault_signals[] = { SIGSEGV, SIGBUS,  SIGFPE,
				     SIGILL,  SIGTRAP, SIGSYS };

/* Queues an event, waiting for room; false, and nothing queued, once the engine ends. */
bool engine_event_queued(struct inference_engine *engine, uint32_t session_slot,
			 struct generation_event event,
			 uint32_t processed_count)
{
	(void)pthread_mutex_lock(&engine->lock);
	if (engine->event_count == EVENT_CAPACITY && !engine->is_ending) {
		engine->live.backpressure_stalls++;
	}
	while (engine->event_count == EVENT_CAPACITY && !engine->is_ending) {
		(void)pthread_cond_wait(&engine->wake, &engine->lock);
	}
	const bool is_queued = !engine->is_ending;
	if (is_queued) {
		struct engine_slot_stats *const live =
			&engine->live.slots[session_slot];
		const uint64_t now = engine_now_ns();
		engine_live_publish(engine);
		if (engine->cursors[session_slot].session.operation == 0 &&
		    live->request_tag == event.request_tag) {
			if (event.event_kind == GENERATION_EVENT_TOKEN) {
				if (live->generated_tokens == 0) {
					live->first_token_ns = now;
					live->prefill_ns =
						now >= live->started_ns ?
							now - live->started_ns :
							0;
					live->phase = STATS_DECODE;
					live->phase_since_ns = now;
				}
				live->last_token_ns = now;
				live->generated_tokens++;
				live->token_times[live->token_ring_next] = now;
				live->token_ring_next =
					(live->token_ring_next + 1u) %
					STATS_TOKEN_RING;
				live->token_ring_count +=
					live->token_ring_count <
							STATS_TOKEN_RING ?
						1u :
						0u;
				engine->live.generated_tokens++;
			} else {
				live->finished_ns = now;
				live->phase = STATS_FINISHING;
				live->phase_since_ns = now;
				live->finish_kind = event.event_kind;
				live->last_error = event.status;
				live->prefill_ns =
					live->first_token_ns != 0 ?
						live->prefill_ns :
					now >= live->started_ns ?
						now - live->started_ns :
						0;
				engine->live.failed +=
					event.event_kind ==
							GENERATION_EVENT_FAILED ?
						1u :
						0u;
				engine->live.cancelled +=
					event.event_kind ==
							GENERATION_EVENT_CANCELLED ?
						1u :
						0u;
				engine->live.served +=
					event.event_kind == GENERATION_EVENT_FINISHED_AT_END_TOKEN ||
							event.event_kind ==
								GENERATION_EVENT_FINISHED_AT_LIMIT ?
						1u :
						0u;
				engine->live.probes += live->probes;
				engine->live.dspark_tails += live->dspark_tails;
			}
			live->decode_ns =
				live->first_token_ns != 0 &&
						now >= live->first_token_ns ?
					now - live->first_token_ns :
					0;
		}
		const uint64_t one = 1;
		engine->events[(engine->event_first + engine->event_count) %
			       EVENT_CAPACITY] =
			(struct queued_event){ event, session_slot,
					       processed_count };
		engine->event_count++;
		engine->slots[session_slot].phase =
			event.event_kind == GENERATION_EVENT_TOKEN ?
				SLOT_COMMANDED :
				SLOT_FINAL_QUEUED;
		if (write(engine->event_descriptor, &one, sizeof(one)) !=
		    (ssize_t)sizeof(one)) {
			LOG_ERROR(
				"engine: the event descriptor could not be written");
		}
	}
	(void)pthread_mutex_unlock(&engine->lock);
	return is_queued;
}

static bool is_cancelled(struct inference_engine *engine, uint32_t session_slot)
{
	(void)pthread_mutex_lock(&engine->lock);
	const bool cancelled = engine->slots[session_slot].is_cancelled ||
			       engine->is_ending;
	(void)pthread_mutex_unlock(&engine->lock);
	return cancelled;
}

/* Rows actually run, including proposals a later result rejects, bound what remains in the window rings. */
static void rows_reached(struct inference_engine *engine,
			 const struct step_row_record *rows, uint32_t row_count)
{
	(void)pthread_mutex_lock(&engine->lock);
	for (uint32_t index = 0; index < row_count; index++) {
		struct slot *const slot =
			&engine->slots[rows[index].session_slot];
		const uint32_t reach = (uint32_t)rows[index].position + 1u;
		slot->reach_count =
			reach > slot->reach_count ? reach : slot->reach_count;
	}
	(void)pthread_mutex_unlock(&engine->lock);
}

/* The position behind a command's new tokens: the kept tokens and the new ones. */
static uint32_t command_end(const struct generation_command_record *command)
{
	return command->kept_token_count + command->new_token_count;
}

/* The first completion of a generating prompt, before any token event can release its slot. */
static bool checkpoint_due(const struct inference_engine *engine,
			   const struct command_cursor *cursor,
			   uint32_t processed)
{
	return cursor->generated == 0 &&
	       cursor->command.generated_token_limit != 0 &&
	       command_end(&cursor->command) > SERVING_KEPT_GRANULE &&
	       processed >= command_end(&cursor->command) &&
	       step_checkpoint_supported(engine->driver);
}

/* At even G no compressor group is open; future pool and key entries cannot change entries below G. The decoder's
 * whole tail includes G's preceding window. Only the window ring needs a copy (step.h). */
MUST_CHECK static result_status
prompt_checkpoint(struct inference_engine *engine,
		  const struct command_cursor *cursor)
{
	const uint32_t granule = engine->description.prefill_prefix_stable !=
						 0 ?
					 SERVING_STABLE_GRANULE :
					 SERVING_KEPT_GRANULE;
	const uint32_t boundary =
		(command_end(&cursor->command) - 1u) / granule * granule;
	const result_status status = step_checkpoint_save(
		engine->driver, cursor->command.session_slot, boundary);
	(void)pthread_mutex_lock(&engine->lock);
	engine->slots[cursor->command.session_slot].checkpoint_count =
		status == OK ? boundary : 0;
	engine->live.checkpoints_saved += status == OK ? 1u : 0u;
	(void)pthread_mutex_unlock(&engine->lock);
	return status;
}

/* The token at position of a cursor's session: below its kept tokens the session's committed history (a cone run
 * again, cone_planned), then a new one, else the last it drew. */
static uint32_t command_token(const struct inference_engine *engine,
			      const struct command_cursor *cursor,
			      uint32_t position)
{
	const struct generation_command_record *const command =
		&cursor->command;
	return position < command->kept_token_count ?
		       step_history_token(engine->driver, command->session_slot,
					  position) :
	       position < command_end(command) ?
		       command->new_token_ids[position -
					      command->kept_token_count] :
		       cursor->token_id;
}

/* The flags of a cursor's row at position: logits for the last of its prompt, front below its front end. */
static uint32_t command_flags(const struct command_cursor *cursor,
			      uint32_t position)
{
	uint32_t flags =
		position < cursor->front_end ? STEP_ROW_FRONT :
		position + 1 >= command_end(&cursor->command) &&
				cursor->command.generated_token_limit != 0 ?
					       STEP_ROW_LOGITS :
					       0;
	const struct chat_images *const images = cursor->command.images;
	for (uint32_t index = 0; images != NULL && index < images->count;
	     index++) {
		const struct chat_image_input *const image =
			&images->images[index];
		if (position >= image->token_begin &&
		    position - image->token_begin < image->plan.token_count) {
			flags |= STEP_ROW_IMAGE;
		}
	}
	return flags;
}

/* The count of tokens read at which a command ends at the context's end: the token drawn at the last position is the
 * last (front_end_seam.h), several sessions stop one read earlier. */
static uint32_t held_limit(const struct inference_engine *engine)
{
	return engine->description.context_token_limit -
	       (engine->description.session_slot_count > 1 ? 1u : 0u);
}

/* engine.c's drafter: the owner's lookup drafter (draft_lookup.h) takes its tokens up to the anchor's, those it drew
 * from the driver's history (the rows that committed them), and proposes at most the cursor's depth: 1 at a command's
 * first round, twice the last behind a round all accepted (to the draft positions at most), else the last round's
 * accepted proposals and one (engine_flight_landed). A wrong proposal costs a row, which at a long context
 * scans every key (MAP B 5.3); the depth never changes a token (keyed match). */
static uint32_t lookup_drafted(struct inference_engine *engine,
			       const struct command_cursor *cursor,
			       uint32_t anchor_position, uint32_t most,
			       uint32_t *proposals_out)
{
	const uint32_t session_slot = cursor->command.session_slot;
	const uint32_t end = command_end(&cursor->command);
	struct draft_lookup *const lookup = &engine->lookups[session_slot];
	for (uint32_t position = lookup->count; position <= anchor_position;
	     position++) {
		draft_lookup_append(
			lookup,
			position >= end && position < anchor_position ?
				step_history_token(engine->driver, session_slot,
						   position) :
				command_token(engine, cursor, position));
	}
	struct draft_lookup query = *lookup;
	if (engine->description.hybrid_lookup_min > query.ngram_least) {
		query.ngram_least = engine->description.hybrid_lookup_min;
	}
	const uint32_t depth = engine->draft_positions == 0 ? cursor->depth : most;
	return draft_lookup_propose(&query, depth < most ? depth : most,
				    proposals_out);
}

/* The trained five-position block stays fixed; only its verified prefix changes. */
static uint32_t device_drafted(struct inference_engine *engine,
			       const struct command_cursor *cursor,
			       uint32_t anchor_position, uint32_t most,
			       uint32_t *proposals_out)
{
	(void)engine; /* The callback only uses the engine for diagnostic output. */
	const struct step_round *const round = &cursor->draft_round;
#if TIMING_HOOKS_ENABLED
	if (engine->description.draft_trace != 0) /* diagnostics: the round as the host sees it */
		(void)fprintf(stderr,
			      "draft round session %" PRIu32 " anchor %" PRId32 " token %" PRId32
			      " (host %" PRIu32 " %" PRIu32 ") proposals %" PRId32 " %" PRId32 " %" PRId32
			      " %" PRId32 " %" PRId32 " count %" PRIu32 "\n",
			      round->session_slot, round->anchor_position, round->anchor_token, anchor_position,
			      cursor->token_id, round->proposals[0], round->proposals[1], round->proposals[2],
			      round->proposals[3], round->proposals[4], round->proposal_count);
#endif
	if (round->proposal_count == 0 ||
	    round->anchor_position != (int32_t)anchor_position ||
	    round->anchor_token != (int32_t)cursor->token_id ||
	    round->session_slot != cursor->command.session_slot) {
		return 0;
	}
	uint32_t depth = cursor->dspark_depth < most ? cursor->dspark_depth : most;
	depth = depth < round->proposal_count ? depth : round->proposal_count;
	uint32_t count = 0;
	while (count < depth) {
		proposals_out[count] = (uint32_t)round->proposals[count];
		count++;
	}
	return count;
}

/* draft_session_acceptance: the factor of a session's table acceptances, (its matches + SESSION_ACCEPTANCE_PRIOR) /
 * (its expected matches + SESSION_ACCEPTANCE_PRIOR) of its decayed reached drafts; else 1. */
static double session_factor(const struct inference_engine *engine,
			     const struct command_cursor *cursor)
{
	return engine->description.draft_session_acceptance == 0 ? 1.0 :
		(cursor->session_hits + SESSION_ACCEPTANCE_PRIOR) /
			(cursor->session_expected + SESSION_ACCEPTANCE_PRIOR);
}

/* A table acceptance (per thousand) times factor, at most the scale. */
static uint32_t acceptance_scaled(uint32_t acceptance, double factor)
{
	const double scaled = (double)acceptance * factor;
	return scaled >= (double)DRAFT_PROBABILITY_SCALE ?
		       DRAFT_PROBABILITY_SCALE :
		       (uint32_t)(scaled + 0.5);
}

/* The log-odds of a table acceptance (per thousand), its probability held from the floor to 1 less it. */
static double acceptance_log_odds(uint32_t acceptance)
{
	double probability = (double)acceptance / DRAFT_PROBABILITY_SCALE;
	probability = probability < SESSION_PROBABILITY_FLOOR ? SESSION_PROBABILITY_FLOOR :
		probability > 1.0 - SESSION_PROBABILITY_FLOOR ? 1.0 - SESSION_PROBABILITY_FLOOR : probability;
	return log(probability / (1.0 - probability));
}

/* draft_session_acceptance: a table acceptance (per thousand) for this command: 1, times its factor; 2, its log-odds
 * plus the command's shift (the confidence head's bins are not calibrated across content: code drafts of a middle
 * bin match far more often than prose's); 3, its log-odds times the command's scale plus its shift; 5, 1 while the
 * command's reached drafts match at SESSION_FULL_RATE (code's and structured text's: every draft verified, as the fixed
 * policy), else as it is; 6, position-local conditional counts lift underestimated bins; 0, as it is. */
static uint32_t acceptance_session(const struct inference_engine *engine,
				   const struct command_cursor *cursor,
				   uint32_t position, uint32_t acceptance)
{
	const uint32_t mode = engine->description.draft_session_acceptance;
	if (mode == SESSION_RESIDUAL_MODE || mode == SESSION_COST_MODE)
		return draft_session_probability(
			acceptance, cursor->session_position_hits[position],
			cursor->session_position_expected[position],
			cursor->session_position_reached[position]);
	if (mode == SESSION_SCOUT_MODE || mode == SESSION_SECOND_MODE ||
	    mode == SESSION_NATURAL_MODE) {
		const uint64_t rounds =
			cursor->statistics.source[DRAFT_DSPARK].rounds;
		const bool second_mode = mode == SESSION_SECOND_MODE ||
					 mode == SESSION_NATURAL_MODE;
		const double least_reached =
			mode == SESSION_NATURAL_MODE ?
				SESSION_NATURAL_LEAST_REACHED :
				SESSION_SECOND_LEAST_REACHED;
		const bool second_high =
			second_mode &&
			cursor->session_position_reached[SESSION_SECOND_POSITION] >=
				least_reached &&
			cursor->session_position_hits[SESSION_SECOND_POSITION] >=
				SESSION_SECOND_ENTER_RATE *
					cursor->session_position_reached
						[SESSION_SECOND_POSITION];
		const bool probe =
			(mode != SESSION_NATURAL_MODE &&
			 rounds < SESSION_SCOUT_WARM_ROUNDS) ||
			((mode == SESSION_SCOUT_MODE || second_high) &&
			 rounds % engine->draft_policy.probe_rounds == 0);
		return probe || cursor->session_full_depth ?
			       DRAFT_PROBABILITY_SCALE :
			       acceptance;
	}
	if (mode == SESSION_POSITION_MODE) {
		/* Keep a confident draft's estimate, and let command-local evidence lift underestimated bins. The
		 * shared table can otherwise hide a position's high acceptance behind other positions' top bins. */
		const double probability =
			(cursor->session_position_hits[position] *
				 DRAFT_PROBABILITY_SCALE +
			 SESSION_POSITION_PRIOR * (double)acceptance) /
			(cursor->session_position_reached[position] +
			 SESSION_POSITION_PRIOR);
		const uint32_t learned = (uint32_t)lround(probability);
		return learned > acceptance ? learned : acceptance;
	}
	if (mode == SESSION_FULL_MODE)
		return cursor->session_reached >= SESSION_FULL_LEAST &&
				       cursor->session_hits >=
					       SESSION_FULL_RATE *
						       cursor->session_reached ?
			       DRAFT_PROBABILITY_SCALE :
			       acceptance;
	if (mode != SESSION_SHIFT_MODE && mode != SESSION_PLATT_MODE && mode != SESSION_HINDSIGHT_MODE)
		return acceptance_scaled(acceptance, session_factor(engine, cursor));
	const double slope = mode == SESSION_PLATT_MODE ? 1.0 + cursor->session_slope_offset : 1.0;
	const double shifted =
		1.0 / (1.0 + exp(-(slope * acceptance_log_odds(acceptance) + cursor->session_shift)));
	return (uint32_t)lround(shifted * DRAFT_PROBABILITY_SCALE);
}

/* draft_session_acceptance 2: the shift d of the command's last reached drafts, the maximum of their likelihood under
 * acceptances sigmoid(log-odds + d) with a Gaussian prior of precision SESSION_SHIFT_PRIOR at 0, by Newton steps
 * from the last shift. */
static void session_shift_update(struct command_cursor *cursor)
{
	const uint32_t count = cursor->session_seen <
					       SERVING_SESSION_SHIFT_DRAFTS ?
				       cursor->session_seen :
				       SERVING_SESSION_SHIFT_DRAFTS;
	double shift = cursor->session_shift;
	for (uint32_t step = 0; step < SESSION_SHIFT_STEPS; step++) {
		double gradient = SESSION_SHIFT_PRIOR * shift, curvature = SESSION_SHIFT_PRIOR;
		for (uint32_t draft = 0; draft < count; draft++) {
			const double log_odds = (double)cursor->session_log_odds[draft] + shift;
			const double probability = 1.0 / (1.0 + exp(-log_odds));
			gradient += probability - (double)cursor->session_matched[draft];
			curvature += probability * (1.0 - probability);
		}
		shift -= gradient / curvature;
	}
	cursor->session_shift = shift;
}

/* draft_session_acceptance: one reached draft of the command (position, confidence bin), matched or not, into its
 * decayed counts and its ring of the last ones' table log-odds and outcomes. */
static void session_draft_seen(const struct inference_engine *engine,
			       struct command_cursor *cursor, uint32_t mode,
			       uint32_t position, uint32_t bin, bool matched)
{
	const uint32_t acceptance = engine->draft_policy.acceptance[mode][position][bin];
	cursor->session_expected =
		cursor->session_expected * SESSION_ACCEPTANCE_DECAY +
		(double)acceptance / DRAFT_PROBABILITY_SCALE;
	cursor->session_hits = cursor->session_hits * SESSION_ACCEPTANCE_DECAY + (matched ? 1.0 : 0.0);
	cursor->session_reached = cursor->session_reached * SESSION_ACCEPTANCE_DECAY + 1.0;
	if (engine->description.draft_session_acceptance ==
		    SESSION_POSITION_MODE ||
	    engine->description.draft_session_acceptance ==
		    SESSION_SCOUT_MODE ||
	    engine->description.draft_session_acceptance ==
		    SESSION_SECOND_MODE ||
	    engine->description.draft_session_acceptance ==
		    SESSION_NATURAL_MODE ||
	    (engine->description.draft_session_acceptance ==
		     SESSION_RESIDUAL_MODE ||
	     engine->description.draft_session_acceptance ==
		     SESSION_COST_MODE)) {
		cursor->session_position_hits[position] =
			cursor->session_position_hits[position] *
				SESSION_ACCEPTANCE_DECAY +
			(matched ? 1.0 : 0.0);
		cursor->session_position_reached[position] =
			cursor->session_position_reached[position] *
				SESSION_ACCEPTANCE_DECAY +
			1.0;
		if ((engine->description.draft_session_acceptance ==
			     SESSION_RESIDUAL_MODE ||
		     engine->description.draft_session_acceptance ==
			     SESSION_COST_MODE))
			cursor->session_position_expected[position] =
				cursor->session_position_expected[position] *
					SESSION_ACCEPTANCE_DECAY +
				(double)acceptance / DRAFT_PROBABILITY_SCALE;
	}
	const uint32_t place =
		cursor->session_seen++ % SERVING_SESSION_SHIFT_DRAFTS;
	cursor->session_log_odds[place] = (float)acceptance_log_odds(acceptance);
	cursor->session_matched[place] = matched ? 1u : 0u;
}

/* Selection bias makes a pooled reached-draft rate mostly describe the first two positions. Give each middle
 * position equal weight instead. The first draft is always needed; the final draft has few reached observations.
 * Full probes at the existing probe interval let a request whose content changes supply new middle observations. */
static void session_full_update(const struct inference_engine *engine,
				struct command_cursor *cursor)
{
	const uint32_t positions = engine->draft_positions <
						   DRAFT_CALIBRATION_POSITIONS ?
					   engine->draft_positions :
					   DRAFT_CALIBRATION_POSITIONS;
	if (engine->description.draft_session_acceptance ==
		    SESSION_SECOND_MODE ||
	    engine->description.draft_session_acceptance ==
		    SESSION_NATURAL_MODE ||
	    (engine->description.draft_session_acceptance ==
		     SESSION_RESIDUAL_MODE ||
	     engine->description.draft_session_acceptance ==
		     SESSION_COST_MODE)) {
		/* The second proposal discriminates the fixed donor's code/prose (1.00/0.63) without waiting for
		 * three observations at every deeper position. It remains observable in a depth-one round's valid
		 * hindsight, so a low-acceptance request needs no periodic full-depth override to detect a change.
		 * Strong second-position evidence with a deeper veto can still probe, so the veto cannot trap it. */
		const double least_reached =
			engine->description.draft_session_acceptance ==
					SESSION_NATURAL_MODE ?
				SESSION_NATURAL_LEAST_REACHED :
				SESSION_SECOND_LEAST_REACHED;
		const double reached =
			cursor->session_position_reached[SESSION_SECOND_POSITION];
		const double threshold = cursor->session_full_depth ?
						 SESSION_SECOND_LEAVE_RATE :
						 SESSION_SECOND_ENTER_RATE;
		cursor->session_full_depth =
			positions > SESSION_SECOND_POSITION &&
			reached >= least_reached &&
			cursor->session_position_hits[SESSION_SECOND_POSITION] >=
				threshold * reached;
		for (uint32_t position = SESSION_SECOND_POSITION + 1u;
		     position + 1u < positions; position++)
			if (cursor->session_position_reached[position] >=
				    SESSION_MIDDLE_LEAST_REACHED &&
			    cursor->session_position_hits[position] <
				    SESSION_MIDDLE_VETO_RATE *
					    cursor->session_position_reached
						    [position])
				cursor->session_full_depth = false;
		return;
	}
	double rate = 0.0;
	uint32_t observed_positions = 0;
	for (uint32_t position = 1; position + 1u < positions; position++) {
		if (cursor->session_position_reached[position] <
		    SESSION_SCOUT_LEAST_REACHED) {
			cursor->session_full_depth = false;
			return;
		}
		rate += cursor->session_position_hits[position] /
			cursor->session_position_reached[position];
		observed_positions++;
	}
	if (observed_positions != 0) {
		rate /= (double)observed_positions;
		cursor->session_full_depth = rate >=
					     (cursor->session_full_depth ?
						      SESSION_SCOUT_LEAVE_RATE :
						      SESSION_SCOUT_ENTER_RATE);
	}
}

/* draft_session_acceptance 3: the scale a (kept as a - 1) and shift d of the command's last reached drafts, the
 * maximum of their likelihood under acceptances sigmoid(a log-odds + d) with Gaussian priors of precision
 * SESSION_PLATT_PRIOR at a = 1 and d = 0 (Platt's scaling a command at a time), by Newton steps from the last ones;
 * a at least SESSION_SLOPE_LEAST. */
static void session_platt_update(struct command_cursor *cursor)
{
	const uint32_t count = cursor->session_seen <
					       SERVING_SESSION_SHIFT_DRAFTS ?
				       cursor->session_seen :
				       SERVING_SESSION_SHIFT_DRAFTS;
	double offset = cursor->session_slope_offset, shift = cursor->session_shift;
	for (uint32_t step = 0; step < SESSION_SHIFT_STEPS; step++) {
		double gradient_offset = SESSION_PLATT_PRIOR * offset, gradient_shift = SESSION_PLATT_PRIOR * shift;
		double curvature_offset = SESSION_PLATT_PRIOR, curvature_shift = SESSION_PLATT_PRIOR;
		double curvature_both = 0.0;
		for (uint32_t draft = 0; draft < count; draft++) {
			const double log_odds = (double)cursor->session_log_odds[draft];
			const double probability = 1.0 / (1.0 + exp(-((1.0 + offset) * log_odds + shift)));
			const double residual = probability - (double)cursor->session_matched[draft];
			const double weight = probability * (1.0 - probability);
			gradient_offset += residual * log_odds;
			gradient_shift += residual;
			curvature_offset += weight * log_odds * log_odds;
			curvature_both += weight * log_odds;
			curvature_shift += weight;
		}
		const double determinant = curvature_offset * curvature_shift - curvature_both * curvature_both;
		offset -= (curvature_shift * gradient_offset - curvature_both * gradient_shift) / determinant;
		shift -= (curvature_offset * gradient_shift - curvature_both * gradient_offset) / determinant;
		offset = offset < SESSION_SLOPE_LEAST - 1.0 ? SESSION_SLOPE_LEAST - 1.0 : offset;
	}
	cursor->session_slope_offset = offset;
	cursor->session_shift = shift;
}

/* Exploration/full-depth selection is a C1 choice, not a claim that every conditional probability is one. */
static bool session_full_requested(const struct inference_engine *engine,
				   const struct command_cursor *cursor)
{
	const uint64_t rounds = cursor->statistics.source[DRAFT_DSPARK].rounds;
	const double reached =
		cursor->session_position_reached[SESSION_SECOND_POSITION];
	const bool second_high =
		reached >= SESSION_SECOND_LEAST_REACHED &&
		cursor->session_position_hits[SESSION_SECOND_POSITION] >=
			SESSION_SECOND_ENTER_RATE * reached;
	return rounds < SESSION_SCOUT_WARM_ROUNDS ||
	       cursor->session_full_depth ||
	       (second_high && rounds % engine->draft_policy.probe_rounds == 0);
}

/* Positive configured prices seed unseen single-owner depths. Each cell is a whole successful round;
 * shared flights never supply a request's duration. The tail dimension distinguishes ordinary decode. */
static uint32_t draft_affine_cost(const struct inference_engine *engine,
				  uint32_t depth, uint32_t tails)
{
	const uint64_t cost = (uint64_t)engine->draft_policy.base_us +
			      (uint64_t)tails * engine->draft_policy.tail_us +
			      (uint64_t)depth * engine->draft_policy.row_us;
	return cost > DRAFT_MOST_COST_US ? DRAFT_MOST_COST_US : (uint32_t)cost;
}

static uint32_t session_cost_depth(const struct inference_engine *engine,
				   const struct command_cursor *cursor,
				   const struct draft_policy_record *policy,
				   uint32_t anchor, uint32_t most, bool sampled)
{
	uint32_t prices[DRAFT_DEPTH_COUNTS];
	for (uint32_t depth = 0; depth <= most; depth++) {
		const bool wants_tail =
			depth != 0 || engine->draft_policy.probe_rounds == 1u ||
			draft_probe_due(&engine->draft_policy,
					cursor->rounds_since_dspark + 1u);
		const uint32_t tails =
			wants_tail &&
					cursor->command.generated_token_limit -
							cursor->generated >
						1u &&
					(uint64_t)anchor + depth + 1u +
							engine->draft_positions <=
						engine->description
							.context_token_limit ?
				1u :
				0u;
		prices[depth] = draft_cost_estimate(
			&cursor->session_costs[tails][depth],
			draft_affine_cost(engine, depth, tails));
	}
	return draft_rate_depth(policy, cursor->draft_confidence, most, sampled,
				prices, NULL);
}

/* The proposals of an owner whose rows in the step so far end with anchor_position: none unless that row draws; else
 * the drafter's, at most the description's draft positions, the rows left in the step and as many as leave the
 * anchor's draw and theirs within the command's limit and the context (a round never runs past its last token), and
 * no further than the first that is no token of the vocabulary. Their count. */
static uint32_t proposals_taken(struct inference_engine *engine,
				struct command_cursor *cursor,
				uint32_t anchor_position, uint32_t rows_left,
				bool step_planned, bool one_owner,
				uint32_t *proposals_out)
{
	cursor->dspark_opportunity = false;
	cursor->live_probe_planned = false;
	cursor->needs_dspark_tail = false;
	cursor->dspark_depth = 0;
	const uint32_t left =
		cursor->command.generated_token_limit -
		cursor->generated; /* a live command's: 1 or more */
	const uint32_t held = held_limit(engine);
	if ((command_flags(cursor, anchor_position) & STEP_ROW_LOGITS) == 0 ||
	    anchor_position + 1u >= held) {
		return 0;
	}
	uint32_t most = engine->description.speculative_depth < rows_left ?
				engine->description.speculative_depth :
				rows_left;
	most = left - 1u < most ? left - 1u : most;
	most = held - anchor_position - 1u < most ?
		       held - anchor_position - 1u :
		       most;
	const bool adaptive = engine->draft_policy.depth_policy ==
			      DRAFT_DEPTH_ADAPTIVE;
	if ((engine->draft_positions == 0 || adaptive) && most != 0 && cursor->cooldown != 0) {
		cursor->cooldown--;
		return 0;
	}
	cursor->draft_source = engine->draft_positions != 0 ? DRAFT_DSPARK :
							      DRAFT_LOOKUP;
	uint32_t proposed = 0;
	if (most != 0) {
		const bool sampled = cursor->command.sampler_temperature > 0.0f;
		const bool probe = !cursor->has_draft_confidence ||
				   draft_probe_due(&engine->draft_policy,
						   cursor->rounds_since_dspark);
		/* a BATCH step plan of several owners (rows_left its cap) stands for the policy's per-owner depth;
		 * draft_session_acceptance: the policy's table scaled by the session's factor */
		struct draft_policy_record session_policy;
		const struct draft_policy_record *policy =
			&engine->draft_policy;
		if (engine->description.draft_session_acceptance != 0 && !step_planned) {
			session_policy = engine->draft_policy;
			const uint32_t mode = sampled ? 1u : 0u;
			for (uint32_t position = 0;
			     position < DRAFT_CALIBRATION_POSITIONS; position++)
				for (uint32_t bin = 0;
				     bin < DRAFT_CONFIDENCE_BINS; bin++)
					session_policy.acceptance
						[mode][position]
						[bin] = acceptance_session(
						engine, cursor, position,
						engine->draft_policy.acceptance
							[mode][position][bin]);
			policy = &session_policy;
		}
		uint32_t economic_depth =
			engine->draft_positions == 0 ? 0 :
			step_planned		     ? most :
				       draft_depth(policy,
						   cursor->draft_confidence,
						   most, sampled);
		if (engine->description.draft_session_acceptance ==
			    SESSION_COST_MODE &&
		    one_owner && !step_planned &&
		    policy->depth_policy == DRAFT_DEPTH_RATE &&
		    engine->draft_positions != 0 && most != 0)
			economic_depth = session_cost_depth(engine, cursor,
							    policy,
							    anchor_position,
							    most, sampled);
		if ((engine->description.draft_session_acceptance ==
			     SESSION_RESIDUAL_MODE ||
		     engine->description.draft_session_acceptance ==
			     SESSION_COST_MODE) &&
		    one_owner &&
		    (policy->depth_policy == DRAFT_DEPTH_COST ||
		     policy->depth_policy == DRAFT_DEPTH_RATE) &&
		    (engine->description.draft_session_acceptance ==
				     SESSION_RESIDUAL_MODE ?
			     session_full_requested(engine, cursor) :
			     (cursor->statistics.source[DRAFT_DSPARK].rounds <
				      SESSION_SCOUT_WARM_ROUNDS ||
			      !cursor->session_had_full_round ||
			      cursor->statistics.source[DRAFT_DSPARK].rounds -
					      cursor->session_last_full_round >=
				      engine->draft_policy.probe_rounds - 1u)))
			economic_depth = most;
		if (engine->draft_positions != 0) {
			cursor->dspark_opportunity = true;
			cursor->dspark_depth = probe ? 1u : economic_depth;
			if (adaptive && cursor->depth < cursor->dspark_depth) {
				cursor->dspark_depth = cursor->depth;
			}
			cursor->live_probe_planned = probe;
		}
		proposed = engine->draft(engine, cursor, anchor_position, most, proposals_out);
		/* A forced verification need not produce another tail when the retained estimate still says no. */
		cursor->needs_dspark_tail =
			engine->draft_positions != 0 &&
			(economic_depth != 0 ||
			 engine->draft_policy.probe_rounds == 1u ||
			 (proposed == 0 &&
			  (probe ||
			   draft_probe_due(&engine->draft_policy,
					   cursor->rounds_since_dspark + 1u))));
		if (engine->description.hybrid_lookup_min != 0 && !probe) {
			uint32_t lookup_tokens[DRAFT_STAT_POSITIONS];
			const uint32_t available = lookup_drafted(engine, cursor, anchor_position, most, lookup_tokens);
			const uint32_t depth = draft_hybrid_depth(
				&engine->draft_policy, cursor->draft_confidence,
				available, sampled,
				cursor->dspark_depth < available ?
					cursor->dspark_depth :
					available,
				false,
				&cursor->statistics.source[DRAFT_LOOKUP]);
			if (depth != 0) {
				memcpy(proposals_out, lookup_tokens, depth * sizeof(*proposals_out));
				proposed = depth;
				cursor->draft_source = DRAFT_LOOKUP;
			}
		}
	}
	uint32_t count = 0;
	while (count < proposed && count < most &&
	       proposals_out[count] < engine->description.vocabulary_size) {
		count++;
	}
	return count;
}

#define PREFILL_CONE_TIGHT 2u /* prefill_cone_skip: the prompt's last row's cone alone (cone_end_of) */
#define PREFILL_CONE_UNTRIMMED 3u /* prefill_cone_skip: 1's cone, the wide trunk products every row (51cd081's 1) */
#define EXACT_PLAN_GRANULE 256u /* the exact plan (exact_chunk_rows_of): the reference ladder's middle program */
#define EXACT_PLAN_TAIL_FROM 512u /* fewer rows the reference plan halves into 64-row chunks alone */
#define EXACT_PLAN_TAIL_MOST_ROWS 448u /* prefill_tail_rows: a tail program's rows at most (seven 64-row chunks) */
#define EXACT_PLAN_TAIL_LONG_ROWS 320u /* prefill_tail_rows=448: the tail program beside it (five 64-row chunks) */
#define EXACT_PLAN_TAIL_ROWS 128u /* prefill_tail_rows: the tail program every one takes (two 64-row chunks) */
/* prefill_margin (the exact plan, chunks_in_flight 1): the rows a prompt's full region gives up to its front, the
 * first whole multiples of EXACT_PLAN_GRANULE of the reference plan's full rows before that region's tail (engine.c's
 * exact plan below), as far as cone + prefill_margin rows stay full rows. Each row keeps its form (a chunk of more
 * than 64 rows): the same bits in the cone, whose rows read no row a front step leaves short of the decoder's layers;
 * the decoder's state then begins later (decoder_from), so a later command that keeps a prefix ending among those
 * rows runs a cone again. The reference front region's own tail stays at its end (front_tail_end). The fast plan
 * (prefill_fast_plan) keeps no tail: its full region is cone + prefill_margin rows. */
static uint32_t exact_front_moved(const struct inference_engine *engine,
				  const struct command_cursor *cursor,
				  uint32_t end)
{
	const uint32_t margin = engine->description.prefill_margin, cone = engine->cone_rows;
	const uint32_t front = cursor->front_end;
	const uint32_t plain_end = cursor->command.generated_token_limit != 0 && end != 0 ? end - 1u : end;
	if (engine->description.chunks_in_flight != 1u || margin == 0 ||
	    margin >= SERVING_CONE_MARGIN || plain_end <= front ||
	    end <= cone + margin || end - cone - margin <= front)
		return 0;
	if (engine->description.prefill_fast_plan != 0)
		return end - cone - margin - front; /* the fast plan: no tail to keep */
	const uint32_t full = plain_end - front;
	const uint32_t tail = full < EXACT_PLAN_TAIL_FROM ? full : EXACT_PLAN_GRANULE + full % EXACT_PLAN_GRANULE;
	const uint32_t most = end - cone - margin - front;
	const uint32_t moved = full - tail < most ? full - tail : most;
	return moved / EXACT_PLAN_GRANULE * EXACT_PLAN_GRANULE;
}

/* prefill_cone_skip: the end whose cone a full step's rows of a prompt of end tokens keep (cone_flags_of), so that a
 * later command keeping at least end - margin - 1 of its tokens runs no cone again (decoder_from): 1 and 3 the plan's
 * margin (prefill_margin's with the exact plan's front rows, else the serving margin's), 2 none (the prompt's last
 * row's cone alone: a later command keeping fewer than end - 1 tokens runs a cone). 0: no rows skip. */
static uint32_t cone_end_of(const struct inference_engine *engine, uint32_t end)
{
	const uint32_t margin = engine->description.prefill_margin;
	const uint32_t kept =
		engine->description.prefill_cone_skip == PREFILL_CONE_TIGHT ?
			0u :
		engine->description.chunks_in_flight == 1u && margin != 0 &&
				margin < SERVING_CONE_MARGIN ?
			margin :
			SERVING_CONE_MARGIN;
	return engine->description.prefill_cone_skip == 0 || engine->cone_rows == 0 || engine->full_rows == 0 ||
			       end <= kept ?
		       0u :
		       end - kept;
}

/* prefill_dead_halves (front_end_seam.h): the exact plan's full chunks of whole MODEL_FULL_CONE_LEAST_ROWS-row
 * granules (a 320-row tail program's are not) lie before a tail of at least EXACT_PLAN_GRANULE rows, so each one's
 * last row lies at least EXACT_PLAN_GRANULE - kept rows before its prompt's cone end's last row (cone_end_of). */
_Static_assert(EXACT_PLAN_GRANULE == MODEL_FULL_CONE_LEAST_ROWS,
	       "a full chunk of whole granules lies before a tail");
uint32_t engine_full_cone_behind(uint32_t front_steps,
				 uint32_t chunks_in_flight,
				 uint32_t prefill_fast_plan,
				 uint32_t prefill_margin,
				 uint32_t prefill_cone_skip)
{
	const bool margin_kept = chunks_in_flight == 1u &&
				 prefill_margin != 0 &&
				 prefill_margin < SERVING_CONE_MARGIN;
	const uint32_t kept = prefill_cone_skip == PREFILL_CONE_TIGHT ? 0u :
			      margin_kept ? prefill_margin :
					    SERVING_CONE_MARGIN;
	return front_steps == 0 || prefill_cone_skip == 0 || chunks_in_flight != 1u || prefill_fast_plan != 0 ||
			       kept >= EXACT_PLAN_GRANULE ?
		       0u :
		       EXACT_PLAN_GRANULE - kept;
}

/* Stable chunks start in the same owner phase. FULL and WIDE share the frontend
 * at a fixed physical capacity; moving the boundary changes only decoder work
 * outside the later prompt's required cone. */
static uint32_t stable_front_end(const struct inference_engine *engine,
				 uint32_t start, uint32_t end)
{
	const uint32_t whole =
		engine->cone_rows + engine->description.prefill_margin;
	const uint32_t rows = engine->wide_rows;
	const uint32_t front =
		rows != 0 && end > whole ? (end - whole) / rows * rows : 0;
	return front > start ? front : start;
}

/* A generation command's first position and front end (model_seam.h's front steps), with the lock held: its prompt
 * rows before the last cone rows and SERVING_CONE_MARGIN more are front rows. Where it keeps fewer tokens than
 * the session's decoder state needs and brings too few new ones for a cone of its own, it first runs the session's
 * history again from where a cone fits, or from 0 where the windows no longer reach that far back. */
static void cone_planned(struct inference_engine *engine,
			 struct command_cursor *cursor)
{
	const uint32_t cone = engine->cone_rows;
	const uint32_t end = command_end(&cursor->command);
	struct slot *const slot = &engine->slots[cursor->command.session_slot];
	uint32_t start = cursor->command.kept_token_count;
	cursor->front_end = start;
	cursor->front_tail_end = start;
	if (cone == 0 || cursor->session.operation != 0) {
		return;
	}
	const bool runs_cone = start < slot->decoder_from && end - start < cone;
	if (runs_cone) {
		const uint32_t fits =
			end > cone ? (end - cone) / SERVING_KEPT_GRANULE *
					     SERVING_KEPT_GRANULE :
				     0;
		const uint32_t oldest =
			slot->reach_count > SERVING_KEPT_LAG ?
				slot->reach_count - SERVING_KEPT_LAG :
				0; /* the ring still holds the window of a row there */
		start = fits >= oldest ? fits : 0;
		if (engine->description.prefill_prefix_stable != 0) {
			start -= start % SERVING_STABLE_GRANULE;
			if (start < oldest)
				start = 0;
		}
	}
	const uint32_t whole = cone + SERVING_CONE_MARGIN;
	cursor->position = start;
	cursor->front_end = end > whole && end - whole > start ? end - whole :
								 start;
	cursor->front_tail_end = cursor->front_end;
	cursor->front_end += exact_front_moved(engine, cursor, end);
	if (engine->description.prefill_fast_plan != 0)
		cursor->front_tail_end = cursor->front_end; /* no reference tail inside the front */
	if (engine->description.prefill_prefix_stable != 0) {
		cursor->front_end = stable_front_end(engine, start, end);
		cursor->front_tail_end = cursor->front_end;
	}
	if (cursor->front_end > start || runs_cone) {
		slot->decoder_from = cursor->front_end == start && start == 0 ?
					     0 :
					     cursor->front_end + cone - 1u;
	}
	/* prefill_cone_skip: the full steps leave the decoder's state whole from the cone end's last row on */
	const uint32_t cone_end = cone_end_of(engine, end);
	if (engine->description.prefill_cone_skip != 0 && end > cursor->front_end + 1u && cone_end > slot->decoder_from)
		slot->decoder_from = cone_end - 1u;
}

/* The worker stops reading the borrowed prompt BEFORE publishing final. Taking
 * final permits a new command, but never revives a failed shared driver. */
static void command_finished(struct inference_engine *engine,
			     struct command_cursor *cursor, uint32_t kind)
{
	const struct generation_event final = {
		cursor->command.request_tag,
		kind,
		0,
		kind == GENERATION_EVENT_FAILED ?
			0 :
			cursor->position + (cursor->generated != 0 ? 1u : 0u),
		(uint32_t)cursor->status,
		0.0f
	};
	const uint32_t session_slot = cursor->command.session_slot;
	cursor->is_ready = false;
	(void)pthread_mutex_lock(&engine->lock);
	engine_live_publish(engine);
	engine->live.slots[session_slot].finish_kind = kind;
	engine->live.slots[session_slot].finished_ns = engine_now_ns();
	engine->slots[session_slot].statistics = cursor->statistics;
	draft_statistics_add(&engine->statistics, &cursor->statistics);
	(void)pthread_mutex_unlock(&engine->lock);
	if (cursor->rounds != 0) {
		LOG_INFORMATION("engine: slot %" PRIu32 ": %" PRIu32
				" tokens, %" PRIu32
				" speculative rounds verified %" PRIu32
				" proposals and accepted %" PRIu32,
				session_slot, cursor->generated, cursor->rounds,
				cursor->proposals, cursor->accepted);
	}
	(void)engine_event_queued(
		engine, session_slot, final,
		kind == GENERATION_EVENT_FAILED ? 0 : cursor->position);
}

/* A ready command gets one step, then the next ready slot gets its turn. The
 * serialized configuration retains its admitted prompt chunk size. */
static void command_advanced(struct inference_engine *engine,
			     struct command_cursor *cursor)
{
	const struct generation_command_record command = cursor->command;
	const uint32_t session_slot = command.session_slot;
	const uint32_t new_count = command_end(&command);
	const uint32_t limit = command.generated_token_limit;
	uint32_t kind = GENERATION_EVENT_TOKEN;
	bool shared_failure = false;
	if (cursor->status != OK) {
		command_finished(engine, cursor, GENERATION_EVENT_FAILED);
		return;
	}
	if (is_cancelled(engine, session_slot)) {
		command_finished(engine, cursor, GENERATION_EVENT_CANCELLED);
		return;
	}
	if (!cursor->sampler_was_set) {
		const struct step_session settings = {
			command.sampler_seed, command.sampler_temperature,
			command.sampler_top_p
		};
		cursor->status = step_session_set(engine->driver, session_slot,
						  &settings);
		cursor->sampler_was_set = true;
		shared_failure = cursor->status != OK;
	}
	if (cursor->status == OK) {
		const uint32_t most_rows =
			engine->description.session_slot_count > 1 ?
				1 :
				engine->most_rows;
		const uint32_t left = cursor->position < new_count ?
					      new_count - cursor->position :
					      1;
		const uint32_t fronts =
			cursor->position < cursor->front_tail_end ? cursor->front_tail_end - cursor->position :
			cursor->position < cursor->front_end ? cursor->front_end - cursor->position :
								  0;
		const uint32_t kind_rows = fronts != 0 ? fronts : left;
		const uint32_t taken_rows = left < kind_rows ? left : kind_rows;
		const uint32_t row_count = taken_rows < most_rows ? taken_rows :
								    most_rows;
		if (row_count == 0 || row_count > LINEAR_MAXIMUM_ROWS) {
			cursor->status = STATUS_DEVICE_ERROR;
			(void)pthread_mutex_lock(&engine->lock);
			engine->failure = cursor->status;
			(void)pthread_mutex_unlock(&engine->lock);
			command_finished(engine, cursor,
					 GENERATION_EVENT_FAILED);
			return;
		}
		struct step_row_record rows[LINEAR_MAXIMUM_ROWS];
		struct step_row_result results[LINEAR_MAXIMUM_ROWS] = { { 0 } };
		for (uint32_t row = 0; row < row_count; row++) {
			const uint32_t position = cursor->position + row;
			const uint32_t token =
				command_token(engine, cursor, position);
			rows[row] = (struct step_row_record){
				(int32_t)position, (int32_t)token, session_slot,
				command_flags(cursor, position)
			};
			cursor->status =
				token >= engine->description.vocabulary_size ?
					STATUS_INVALID_ARGUMENT :
					cursor->status;
		}
		if (cursor->status == OK) {
			const uint64_t submitted_ns = engine_now_ns();
			cursor->status = step_run_entry(engine->driver, rows,
							row_count, results);
			shared_failure = cursor->status != OK;
			if (cursor->status == OK) {
				engine_live_step(engine, rows, row_count,
						 submitted_ns);
				rows_reached(engine, rows, row_count);
			}
		}
		cursor->position += row_count;
		if (cursor->status == OK &&
		    checkpoint_due(engine, cursor, cursor->position)) {
			cursor->status = prompt_checkpoint(engine, cursor);
			shared_failure = cursor->status != OK;
		}
		const struct step_row_result drawn = results[row_count - 1];
		if (cursor->status == OK && cursor->position >= new_count &&
		    limit != 0 &&
		    (drawn.token_id < 0 ||
		     (uint32_t)drawn.token_id >=
			     engine->description.vocabulary_size)) {
			cursor->status = STATUS_DEVICE_ERROR;
			shared_failure = true;
		}
		if (cursor->status == OK && cursor->position >= new_count &&
		    limit == 0) {
			kind = GENERATION_EVENT_FINISHED_AT_LIMIT;
		} else if (cursor->status == OK &&
			   cursor->position >= new_count) {
			cursor->token_id = (uint32_t)drawn.token_id;
			cursor->generated++;
			if (!engine_event_queued(engine, session_slot,
						 (struct generation_event){
							 command.request_tag,
							 GENERATION_EVENT_TOKEN,
							 cursor->token_id,
							 cursor->position + 1,
							 OK, drawn.logit },
						 0)) {
				cursor->is_ready = false;
				return;
			}
			kind = token_ends_generation(engine, cursor->token_id) ?
				       GENERATION_EVENT_FINISHED_AT_END_TOKEN :
			       cursor->generated == limit ||
					       cursor->position ==
						       held_limit(engine) ?
				       GENERATION_EVENT_FINISHED_AT_LIMIT :
				       GENERATION_EVENT_TOKEN;
		}
	}
	if (cursor->status != OK) {
		/* Shared poison includes malformed sampler/results. Even
		 * after every FAILED is taken, unknown card work forbids driver reuse. */
		if (shared_failure) {
			(void)pthread_mutex_lock(&engine->lock);
			engine->failure = engine->failure == OK ?
						  cursor->status :
						  engine->failure;
			(void)pthread_mutex_unlock(&engine->lock);
		}
		kind = GENERATION_EVENT_FAILED;
	}
	if (kind != GENERATION_EVENT_TOKEN) {
		command_finished(engine, cursor, kind);
	}
}

void engine_driver_failed(struct inference_engine *engine, result_status status)
{
	(void)pthread_mutex_lock(&engine->lock);
	engine->failure = engine->failure == OK ? status : engine->failure;
	(void)pthread_mutex_unlock(&engine->lock);
}

/* Whether the owner's row at position may run: inside the context, with a token of the vocabulary. */
static bool row_is_valid(const struct inference_engine *engine,
			 const struct command_cursor *cursor, uint32_t position)
{
	const uint32_t token = command_token(engine, cursor, position);
	return position < engine->description.context_token_limit &&
	       token < engine->description.vocabulary_size;
}

/* The owner's sampler settings set on its driver slot, once; false when that fails (the driver then fails). */
static bool sampler_ready(struct inference_engine *engine,
			  uint32_t session_slot)
{
	struct command_cursor *const cursor = engine->cursors + session_slot;
	if (!cursor->sampler_was_set) {
		const struct step_session settings = {
			cursor->command.sampler_seed,
			cursor->command.sampler_temperature,
			cursor->command.sampler_top_p
		};
		cursor->status = step_session_set(engine->driver, session_slot,
						  &settings);
		cursor->sampler_was_set = cursor->status == OK;
		if (cursor->status != OK) {
			engine_driver_failed(engine, cursor->status);
		}
	}
	return cursor->sampler_was_set;
}

/* chunks_in_flight 1, the exact plan: the reference plan (1,024-row ladder, two chunks in flight: chunk_rows_of below)
 * runs a region's last 256 + rows % 256 rows (all of them below 512) as 64-row chunks and thin steps, whose forms
 * differ from a larger chunk's (--tp_trunk_mma, --exl3_tc_moe take launches of 64 rows and fewer); a chunk of more
 * than 64 rows gives each row the same bits at any row count. So the rows before that tail go in the largest programs
 * of whole EXACT_PLAN_GRANULE multiples (the tail stays the same from chunk to chunk), then the tail as the reference
 * runs it: the reference's bits in fewer, larger chunks. A ladder without such programs falls back to its largest. */
uint32_t engine_exact_chunk_rows(const struct inference_engine *engine,
				 enum program_kind kind, uint32_t rows,
				 bool has_tail)
{
	const uint32_t tail = !has_tail ? 0 :
			      rows < EXACT_PLAN_TAIL_FROM ? rows : EXACT_PLAN_GRANULE + rows % EXACT_PLAN_GRANULE;
	const uint32_t whole = rows - tail;
	/* prefill_tail_rows: a tail program of the kind (full; wide: --prefill_front_tail) takes 64-row chunks of the
	 * tail a step, each in its forms: the largest of tail_rows', EXACT_PLAN_TAIL_LONG_ROWS' (below tail_rows) and
	 * EXACT_PLAN_TAIL_ROWS' whose program the driver has and whose rows are left, else one 64-row chunk */
	const uint32_t tail_rows = engine->description.prefill_tail_rows;
	const uint32_t tail_programs[] = { tail_rows, EXACT_PLAN_TAIL_LONG_ROWS, EXACT_PLAN_TAIL_ROWS };
	uint32_t tail_chunk = MODEL_WIDE_LEAST_ROWS;
	for (uint32_t index = 0; index < sizeof(tail_programs) / sizeof(tail_programs[0]); index++) {
		const uint32_t program = tail_programs[index];
		if (tail_rows != 0 && program <= tail_rows && rows >= program &&
		    step_chunk_rows(engine->driver, kind, program) == program) {
			tail_chunk = program;
			break;
		}
	}
	uint32_t chunk =
		whole == 0 ? step_chunk_rows(engine->driver, kind, tail_chunk) :
			     0;
	const uint32_t limit = engine->description.prefill_decode_rows == 0 ?
				       engine->description.prefill_chunk_rows :
				       engine->prefill_chunk_limit;
	for (uint32_t most = limit != 0 && whole > limit ? limit : whole;
	     most >= EXACT_PLAN_GRANULE && chunk == 0;) {
		const uint32_t program =
			step_chunk_rows(engine->driver, kind, most);
		chunk = program % EXACT_PLAN_GRANULE == 0 ? program : 0;
		most = program != 0 ? program - 1u : 0; /* below a program of no whole granules */
	}
	if (chunk == 0 && whole != 0)
		chunk = step_chunk_rows(engine->driver, kind, whole);
	/* prefill_thin_tail: a full (wide) step of fewer rows than the least chunk (the tail's rows short of a 64-row
	 * chunk, a prompt's row of logits not among them) takes the largest such program of them (main.c's thin-tail
	 * programs, their rows in a thin step's forms: no other has fewer than 64 rows); else they go in thin steps */
	if (chunk > rows && rows < MODEL_WIDE_LEAST_ROWS) {
		const uint32_t program =
			step_chunk_rows(engine->driver, kind, rows);
		chunk = program > LINEAR_MAXIMUM_ROWS ? program : 0;
	}
	return chunk <= rows ? chunk : 0;
}

/* The rows of the chunk an owner's next `rows` rows go in with programs of kind (wide or full): its largest program of
 * at most a chunks_in_flight-th of them (the description's), or its smallest, when that fits them; 0: none. A chunk
 * takes its program's cards in turn, so a prompt in fewer chunks leaves the pipeline's cards idle (agent L). The exact
 * plan's rows: those of a region (has_tail) or of the front rows prefill_margin gave up (none of them in a tail). */
static uint32_t chunk_rows_of(const struct inference_engine *engine,
			      enum program_kind kind, uint32_t rows,
			      bool has_tail)
{
	if (engine->description.prefill_prefix_stable != 0) {
		const uint32_t capacity = kind == PROGRAM_WIDE ?
						  engine->wide_rows :
						  engine->full_rows;
		return rows < capacity ? rows : capacity;
	}
	if (engine->description.chunks_in_flight == 1u && engine->description.prefill_fast_plan == 0)
		return engine_exact_chunk_rows(engine, kind, rows, has_tail);
	const uint32_t share =
		rows / (engine->description.chunks_in_flight != 0 ?
				engine->description.chunks_in_flight :
				1u);
	const uint32_t smallest =
		step_chunk_rows(engine->driver, kind, MODEL_WIDE_LEAST_ROWS);
	const uint32_t chunk = step_chunk_rows(
		engine->driver, kind, share > smallest ? share : smallest);
	return chunk <= rows ? chunk : 0;
}

/* A wide step (model_seam.h) for the owner at first_slot, the one whose turn it is: its next rows, as many as
 * chunk_rows_of gives, when it is ready and they are all front rows of its prompt, into flight. false, and nothing
 * changed but the owner's sampler settings, when it has no such rows. */
static bool wide_flight_taken(struct inference_engine *engine,
			      uint32_t first_slot, struct flight *flight)
{
	struct command_cursor *const cursor = engine->cursors + first_slot;
	const uint32_t position = cursor->position + cursor->in_flight;
	const bool is_tail_region = position < cursor->front_tail_end; /* else the rows prefill_margin moved */
	const uint32_t wide =
		engine->wide_rows == 0 || position >= cursor->front_end ?
			0 :
			chunk_rows_of(engine, PROGRAM_WIDE,
				      (is_tail_region ? cursor->front_tail_end :
							cursor->front_end) -
					      position,
				      is_tail_region);
	bool is_wide = wide != 0 && cursor->is_ready &&
		       !cursor->images_pending && !cursor->awaits_token &&
		       cursor->status == OK &&
		       !is_cancelled(engine, first_slot);
	for (uint32_t row = 0; row < wide && is_wide; row++) {
		is_wide = row_is_valid(engine, cursor, position + row);
	}
	if (!is_wide || !sampler_ready(engine, first_slot)) {
		return false;
	}
	flight->owner_count = 1;
	flight->owners[0] = first_slot;
	flight->taken[0] = wide;
	for (uint32_t row = 0; row < wide; row++) {
		flight->rows[row] = (struct step_row_record){
			(int32_t)(position + row),
			(int32_t)command_token(engine, cursor, position + row),
			first_slot,
			command_flags(cursor, position + row) |
				(engine->description.prefill_prefix_stable !=
						 0 ?
					 STEP_ROW_PADDED :
					 0u)
		};
	}
	engine->next_slot =
		(first_slot + 1) % engine->description.session_slot_count;
	return true;
}

/* prefill_cone_skip: the flags of a full step's prompt row at position for the cone of a prompt ending at end
 * (cone_end_of; its last row, end - 1, draws): the cone's count of decoder layers whose output of the row is read
 * (model_layout.h: layer 20 + j reads the window of its 127 rows before, so the last row needs layer 39, a row 1..127
 * before it layers to 38, and so on); a row at end - 1 or later all of them. trunks: CONE_TRUNKS too. */
static uint32_t cone_flags_of(uint32_t position, uint32_t end, bool trunks)
{
	const uint32_t behind = position + 1u < end ? end - 1u - position : 0u;
	const uint32_t layers = MODEL_CONE_LAYERS(behind);
	const uint32_t trunk_flag = trunks ? STEP_ROW_CONE_TRUNKS : 0u;
	return STEP_ROW_CONE | trunk_flag | layers << STEP_ROW_CONE_SHIFT;
}

/* A full step (model_seam.h) for the owner at first_slot, the one whose turn it is: its next rows, as many as
 * chunk_rows_of gives, when it is ready and they are all rows of its prompt that are neither front rows nor rows
 * of logits (a full program has no head: the prompt's last row, which draws the first token, goes in a thin step).
 * false, and nothing changed but the owner's sampler settings, when it has no such rows. */
static bool full_flight_taken(struct inference_engine *engine,
			      uint32_t first_slot, struct flight *flight)
{
	struct command_cursor *const cursor = engine->cursors + first_slot;
	const uint32_t position = cursor->position + cursor->in_flight;
	const uint32_t end = command_end(&cursor->command);
	const uint32_t plain_end =
		cursor->command.generated_token_limit != 0 && end != 0 ?
			end - 1u :
			end; /* below the first row of logits */
	const uint32_t full = engine->full_rows == 0 ||
					      position < cursor->front_end ||
					      position >= plain_end ?
				      0 :
				      chunk_rows_of(engine, PROGRAM_FULL,
						    plain_end - position, true);
	bool is_full = full != 0 && cursor->is_ready &&
		       !cursor->images_pending && !cursor->awaits_token &&
		       cursor->status == OK &&
		       !is_cancelled(engine, first_slot);
	for (uint32_t row = 0; row < full && is_full; row++) {
		is_full = row_is_valid(engine, cursor, position + row) &&
			  (command_flags(cursor, position + row) &
			   ~STEP_ROW_IMAGE) == 0;
	}
	if (!is_full || !sampler_ready(engine, first_slot)) {
		return false;
	}
	flight->owner_count = 1;
	flight->owners[0] = first_slot;
	flight->taken[0] = full;
	const uint32_t cone_end = cone_end_of(engine, end);
	/* the bound a program of prefill_dead_halves was built for (engine_full_cone_behind) */
	const uint32_t least_behind =
		engine->description.prefill_prefix_stable != 0 ?
			0u :
			engine_full_cone_behind(
				engine->description.front_steps,
				engine->description.chunks_in_flight,
				engine->description.prefill_fast_plan,
				engine->description.prefill_margin,
				engine->description.prefill_cone_skip);
	ASSERT_TEXT(
		least_behind == 0 || full % MODEL_FULL_CONE_LEAST_ROWS != 0 ||
			cone_end == 0 ||
			cone_end - 1u >= position + full - 1u + least_behind,
		"least_behind == 0 || full % ANEMONE_MODEL_FULL_CONE_LEAST_ROWS != 0 || cone_end == 0 || cone_end - 1u >= position + full - 1u + least_behind"); /* whole granules: no tail chunk */
	const bool trunks = engine->description.prefill_cone_skip != PREFILL_CONE_UNTRIMMED;
	for (uint32_t row = 0; row < full; row++) {
		flight->rows[row] = (struct step_row_record){
			(int32_t)(position + row),
			(int32_t)command_token(engine, cursor, position + row),
			first_slot,
			command_flags(cursor, position + row) |
				(engine->description.prefill_prefix_stable !=
						 0 ?
					 STEP_ROW_PADDED :
					 0u) |
				(cone_end != 0 ?
					 cone_flags_of(position + row, cone_end,
						       trunks) :
					 0u)
		};
	}
	engine->next_slot =
		(first_slot + 1) % engine->description.session_slot_count;
	return true;
}

/* The BATCH program a step of rows rows and drafting sessions takes (step_image_programs.c): the first, fewest rows and
 * then fewest verifiers first, that holds both; batch_cost_count: none. */
static uint32_t batch_step_program(const struct inference_engine *engine,
				   uint32_t rows, uint32_t drafting)
{
	uint32_t index = 0;
	while (index < engine->batch_cost_count &&
	       (engine->batch_cost_rows[index] < rows || engine->batch_cost_verifiers[index] < drafting))
		index++;
	return index;
}

/* A BATCH step's microseconds for rows rows and drafting sessions: its program's; 0: no program holds them. */
static uint32_t batch_step_cost(const struct inference_engine *engine,
				uint32_t rows, uint32_t drafting)
{
	const uint32_t index = batch_step_program(engine, rows, drafting);
	if (index >= engine->batch_cost_count)
		return 0;
	if (engine->description.draft_session_acceptance != SESSION_COST_MODE)
		return engine->batch_cost_us[index];
	const uint64_t unused =
		(uint64_t)(engine->batch_cost_rows[index] - rows) *
			engine->draft_policy.row_us +
		(uint64_t)(engine->batch_cost_verifiers[index] - drafting) *
			engine->draft_policy.tail_us;
	const uint32_t prior = engine->batch_cost_prior_us[index];
	const uint32_t floor = prior < engine->draft_policy.base_us ?
				       prior :
				       engine->draft_policy.base_us;
	const uint32_t seed = unused < prior && prior - unused > floor ?
				      (uint32_t)(prior - unused) :
				      floor;
	return rows <= SERVING_BATCH_COST_ROWS &&
			       drafting <= SERVING_SESSION_CAPACITY ?
		       draft_cost_estimate(
			       &engine->batch_shape_costs[rows][drafting],
			       seed) :
		       seed;
}

/* A landed BATCH step of rows rows that took elapsed_ns: its program's cost moves an eighth of the way to it (the
 * description's costs are the start; a step's cost then follows what this server's steps take). */
static void batch_step_measured(struct inference_engine *engine,
				const struct flight *flight,
				uint64_t elapsed_ns)
{
	const uint64_t elapsed_us = elapsed_ns / NANOSECONDS_PER_MICROSECOND;
	uint32_t drafting = 0;
	for (uint32_t row = 0; row < flight->row_count; row++)
		drafting += (flight->rows[row].flags & STEP_ROW_DRAFTS) != 0 ?
				    1u :
				    0u;
	const uint32_t index = batch_step_program(engine, flight->row_count, drafting);
#if TIMING_HOOKS_ENABLED
	const uint32_t shape_us =
		batch_step_cost(engine, flight->row_count, drafting);
	const uint32_t shape_samples =
		flight->row_count <= SERVING_BATCH_COST_ROWS &&
				drafting <= SERVING_SESSION_CAPACITY ?
			engine->batch_shape_costs[flight->row_count][drafting]
				.observations :
			0u;
#endif
	if (engine->description.draft_session_acceptance == SESSION_COST_MODE &&
	    flight->owner_count != 0 &&
	    flight->row_count <= SERVING_BATCH_COST_ROWS &&
	    drafting <= SERVING_SESSION_CAPACITY &&
	    index < engine->batch_cost_count) {
		bool decode_only = true;
		uint32_t expected_rows = flight->owner_count;
		for (uint32_t turn = 0; turn < flight->owner_count; turn++) {
			decode_only = decode_only &&
				      engine->cursors[flight->owners[turn]]
						      .generated != 0;
			expected_rows += flight->proposed[turn];
		}
		if (decode_only && expected_rows == flight->row_count) {
			struct draft_cost_cell *const cell =
				&engine->batch_shape_costs[flight->row_count]
							  [drafting];
			const uint32_t seed = batch_step_cost(
				engine, flight->row_count, drafting);
			(void)draft_cost_observe(cell, seed, elapsed_us);
			if (flight->owner_count == 1u && drafting <= 1u &&
			    flight->proposed[0] < DRAFT_DEPTH_COUNTS) {
				struct command_cursor *const cursor =
					&engine->cursors[flight->owners[0]];
				(void)draft_cost_observe(
					&cursor->session_costs
						 [drafting][flight->proposed[0]],
					draft_affine_cost(engine,
							  flight->proposed[0],
							  drafting),
					elapsed_us);
			}
		}
	}

#if TIMING_HOOKS_ENABLED
	if (engine->description.draft_trace != 0) {
		bool decode_only = true, known = true;
		double expected_tokens = (double)flight->owner_count;
		uint32_t matched_proposals = 0, anchor = 0;
		for (uint32_t turn = 0; turn < flight->owner_count; turn++) {
			const struct command_cursor *const cursor =
				&engine->cursors[flight->owners[turn]];
			const struct step_round *const round =
				&cursor->draft_round;
			decode_only = decode_only && cursor->generated != 0;
			known = known && (flight->proposed[turn] == 0 ||
					  cursor->draft_source == DRAFT_DSPARK);
			double reach = 1.0;
			bool matched = true;
			for (uint32_t position = 0;
			     position < flight->proposed[turn]; position++) {
				if (cursor->draft_source == DRAFT_DSPARK) {
					const uint32_t mode =
						cursor->command.sampler_temperature >
								0.0f ?
							1u :
							0u;
					const uint32_t bin =
						draft_confidence_bin(
							round->confidence_logits
								[position]);
					reach *=
						(double)acceptance_session(
							engine, cursor,
							position,
							engine->draft_policy.acceptance
								[mode][position]
								[bin]) /
						DRAFT_PROBABILITY_SCALE;
					expected_tokens += reach;
				}
				const int32_t drawn =
					engine->results[anchor + position]
						.token_id;
				matched = matched &&
					  !token_ends_generation(
						  engine, (uint32_t)drawn) &&
					  drawn == flight->rows[anchor +
								position + 1u]
							   .token_id;
				matched_proposals += matched ? 1u : 0u;
			}
			anchor += flight->taken[turn];
		}
		if (decode_only) {
			const uint64_t affine_us =
				(uint64_t)engine->draft_policy.base_us +
				(uint64_t)drafting *
					engine->draft_policy.tail_us +
				(uint64_t)(flight->row_count -
					   flight->owner_count) *
					engine->draft_policy.row_us;
			(void)fprintf(
				stderr,
				"draft cost rows=%u owners=%u tails=%u menu_us=%u affine_us=%" PRIu64
				" elapsed_us=%" PRIu64
				" expected_tokens=%.6f known=%u matched=%u shape_us=%u shape_samples=%u\n",
				flight->row_count, flight->owner_count,
				drafting,
				index < engine->batch_cost_count ?
					engine->batch_cost_us[index] :
					0u,
				affine_us, elapsed_us, expected_tokens,
				(unsigned int)known, matched_proposals,
				shape_us, shape_samples);
		}
	}
#endif
	if (index < engine->batch_cost_count && elapsed_us != 0 &&
	    elapsed_us < DRAFT_MOST_COST_US) {
		const uint64_t cost = engine->batch_cost_us[index];
		engine->batch_cost_us[index] =
			(uint32_t)((cost * (BATCH_COST_WEIGHT - 1u) + elapsed_us + BATCH_COST_WEIGHT / 2u) /
				   BATCH_COST_WEIGHT);
		struct stats_batch_program *const live =
			&engine->worker_live.batch_programs[index];
		live->steps++;
		live->step_ns += elapsed_ns;
		live->cost_us = engine->batch_cost_us[index];
	}
}

/* Mode11 prices the candidate's live tail count, including generation and context boundaries. */
static uint32_t batch_draft_tails(const struct inference_engine *engine,
				  const struct flight *flight,
				  const uint32_t *depths)
{
	if (engine->description.draft_session_acceptance != SESSION_COST_MODE)
		return flight->owner_count;
	uint32_t tails = 0;
	for (uint32_t turn = 0; turn < flight->owner_count; turn++) {
		const struct command_cursor *const cursor =
			&engine->cursors[flight->owners[turn]];
		const uint32_t depth =
			depths[turn] == UINT32_MAX ? 0u : depths[turn];
		tails +=
			cursor->command.generated_token_limit -
							cursor->generated >
						1u &&
					(uint64_t)cursor->position +
							cursor->in_flight +
							depth + 1u +
							engine->draft_positions <=
						engine->description
							.context_token_limit ?
				1u :
				0u;
	}
	return tails;
}

/* The step plan of a BATCH step whose owners all decode (description batch_step_us): every owner with device drafts
 * for its anchor verifies its first, and of the others the most probable ones (a draft's reach probability: the
 * policy's acceptance of each draft before it and its own, by position and confidence) as long as the step's expected
 * tokens a microsecond grow, a step's cost its program's (batch_plan_ties: the list cut only between drafts of unequal
 * reach). caps[turn]: the owner's most proposals (UINT32_MAX: no plan). Drafts only: the verified tokens are the
 * target's whatever the plan. */
void engine_draft_depths(struct inference_engine *engine,
			 const struct flight *flight, const uint32_t *taken,
			 uint32_t row_count, uint32_t *caps)
{
	struct candidate {
		double reach;
		uint32_t turn;
	} candidates[SERVING_SESSION_CAPACITY * SERVING_DRAFT_MOST], held;
	uint32_t count = 0, rows = row_count, planned[SERVING_SESSION_CAPACITY];
	uint32_t available[SERVING_SESSION_CAPACITY];
	double tokens = 0.0;
	for (uint32_t turn = 0; turn < flight->owner_count; turn++)
		caps[turn] = UINT32_MAX;
	if (engine->batch_cost_count == 0)
		return;
	if (engine->description.draft_session_acceptance == SESSION_COST_MODE &&
	    flight->owner_count == 1u)
		return;
	for (uint32_t turn = 0; turn < flight->owner_count; turn++) {
		const struct command_cursor *const cursor = engine->cursors + flight->owners[turn];
		const uint32_t anchor = cursor->position + cursor->in_flight;
		const struct step_round *const round = &cursor->draft_round;
		if (taken[flight->owners[turn]] != 1u ||
		    (command_flags(cursor, anchor) & STEP_ROW_LOGITS) == 0)
			return; /* a prompt row: the policy's depths */
		tokens += 1.0;
		const bool drafted = cursor->has_draft_confidence && round->proposal_count != 0 &&
				     round->anchor_position == (int32_t)anchor &&
				     round->anchor_token == (int32_t)cursor->token_id &&
				     round->session_slot == cursor->command.session_slot;
		/* An unavailable round chooses its refresh independently; do not price that unknown tail as active. */
		if (engine->description.draft_session_acceptance ==
			    SESSION_COST_MODE &&
		    !drafted)
			return;
		uint32_t most = drafted ? round->proposal_count : 0;
		most = most < engine->description.speculative_depth ? most : engine->description.speculative_depth;
		most = most < DRAFT_CALIBRATION_POSITIONS ?
			       most :
			       DRAFT_CALIBRATION_POSITIONS;
		if ((engine->description.draft_session_acceptance ==
			     SESSION_RESIDUAL_MODE ||
		     engine->description.draft_session_acceptance ==
			     SESSION_COST_MODE)) {
			const uint32_t left =
				cursor->command.generated_token_limit -
				cursor->generated;
			const uint32_t context_held = held_limit(engine);
			const uint32_t context_left =
				anchor < context_held ?
					context_held - anchor - 1u :
					0u;
			most = most < left - 1u ? most : left - 1u;
			most = most < context_left ? most : context_left;
		}
		available[turn] = most;
		const uint32_t mode = cursor->command.sampler_temperature > 0.0f ? 1u : 0u;
		double reach = 1.0;
		planned[turn] = most == 0 ? UINT32_MAX : 1u; /* no drafts: unplanned, it drafts for its next round */
		for (uint32_t position = 0; position < most; position++) {
			const uint32_t bin = draft_confidence_bin(
				round->confidence_logits[position]);
			const uint32_t acceptance = engine->draft_policy.acceptance[mode][position][bin];
			reach *= (double)acceptance_session(
					 engine, cursor, position, acceptance) /
				 DRAFT_PROBABILITY_SCALE;
			if (position == 0) { /* its first: always verified */
				tokens += reach;
				rows++;
			} else {
				candidates[count++] = (struct candidate){ reach, turn };
			}
		}
	}
	for (uint32_t index = 1; index < count; index++) { /* most probable first; an owner's in position order */
		held = candidates[index];
		uint32_t at = index;
		for (; at > 0 && candidates[at - 1u].reach < held.reach; at--)
			candidates[at] = candidates[at - 1u];
		candidates[at] = held;
	}
	uint32_t candidate_depths[SERVING_SESSION_CAPACITY];
	memcpy(candidate_depths, planned,
	       (size_t)flight->owner_count * sizeof(*planned));
	uint32_t drafting = batch_draft_tails(engine, flight, candidate_depths);
	uint32_t best = 0, best_cost = batch_step_cost(engine, rows, drafting);
	double best_tokens = tokens;
	for (uint32_t index = 0; index < count && rows + index + 1u <= engine->most_rows; index++) {
		tokens += candidates[index].reach;
		candidate_depths[candidates[index].turn]++;
		drafting = batch_draft_tails(engine, flight, candidate_depths);
		/* batch_plan_ties: no cut inside a run of drafts of equal reach */
		if (engine->description.batch_plan_ties != 0 && index + 1u < count &&
		    candidates[index + 1u].reach == candidates[index].reach)
			continue;
		const uint32_t cost = batch_step_cost(engine, rows + index + 1u, drafting);
		if (cost != 0 && (best_cost == 0 || tokens * best_cost > best_tokens * cost)) {
			best = index + 1u;
			best_cost = cost;
			best_tokens = tokens;
		}
	}
	for (uint32_t index = 0; index < best; index++)
		planned[candidates[index].turn]++;
	if ((engine->description.draft_session_acceptance ==
		     SESSION_RESIDUAL_MODE ||
	     engine->description.draft_session_acceptance ==
		     SESSION_COST_MODE) &&
	    flight->owner_count > 1u) {
		uint32_t selected_rows = row_count;
		for (uint32_t turn = 0; turn < flight->owner_count; turn++)
			selected_rows += planned[turn] == UINT32_MAX ?
						 0u :
						 planned[turn];
		for (uint32_t turn = 0; turn < flight->owner_count; turn++) {
			const struct command_cursor *const cursor =
				&engine->cursors[flight->owners[turn]];
			const uint64_t rounds =
				cursor->statistics.source[DRAFT_DSPARK].rounds;
			const bool due =
				!cursor->session_had_full_round ||
				rounds - cursor->session_last_full_round >=
					engine->draft_policy.probe_rounds - 1u;
			if (planned[turn] != UINT32_MAX &&
			    available[turn] > planned[turn] && due &&
			    selected_rows <= engine->most_rows &&
			    available[turn] - planned[turn] <=
				    engine->most_rows - selected_rows) {
				planned[turn] = available[turn];
#if TIMING_HOOKS_ENABLED
				if (engine->description.draft_trace != 0)
					(void)fprintf(
						stderr,
						"draft exploration slot=%u round=%" PRIu64
						" depth=%u\n",
						cursor->command.session_slot,
						rounds, planned[turn]);
#endif
				/* One real probe per flight; probability estimates stay unchanged. */
				break;
			}
		}
	}
	for (uint32_t turn = 0; turn < flight->owner_count; turn++)
		caps[turn] = planned[turn];
}

/* prefill_order: the slot of the oldest command (admitted) still reading its prompt, ready for text, whose prompt
 * rows other than those in flight are left; the slot count when none. Only it takes chunks or prompt rows: the other
 * prompts wait, each whole behind it (their chunks unchanged), and its first token comes after its own prompt. With
 * shortest (PREFILL_ORDER_SHORTEST), of those prompts the one with the fewest such rows left in whole
 * EXACT_PLAN_GRANULE granules (ties: the oldest, so prompts of about one size go in their commands' order): a
 * prompt's chunks depend on its own rows alone, so each is the same whichever goes first. */
static uint32_t prompt_head(const struct inference_engine *engine,
			    bool shortest, uint32_t *count_out)
{
	const uint32_t slot_count = engine->description.session_slot_count;
	uint32_t head = slot_count, head_left = 0;
	uint32_t count = 0;
	for (uint32_t session_slot = 0; session_slot < slot_count; session_slot++) {
		const struct command_cursor *const cursor = engine->cursors + session_slot;
		const uint32_t next = cursor->position + cursor->in_flight, end = command_end(&cursor->command);
		const bool is_prompt = cursor->is_ready && !cursor->images_pending && cursor->session.operation == 0 &&
				       next < end;
		const uint32_t left = next < end ? (end - next + EXACT_PLAN_GRANULE - 1u) / EXACT_PLAN_GRANULE : 0;
		count += is_prompt ? 1u : 0u;
		const bool is_older = head == slot_count || cursor->admitted < engine->cursors[head].admitted;
		if (is_prompt && (!shortest ? is_older :
				  head == slot_count || left < head_left || (left == head_left && is_older))) {
			head = session_slot;
			head_left = left;
		}
	}
	if (count_out != NULL)
		*count_out = count;
	return head;
}

/* A decoding owner: past its prompt and not failed or cancelled. include_in_flight retains the cap while its rows
 * land; otherwise it must be able to take a drawing row now. */
static bool decoding_owner(struct inference_engine *engine, uint32_t head,
			   bool include_in_flight)
{
	for (uint32_t session_slot = 0; session_slot < engine->description.session_slot_count; session_slot++) {
		const struct command_cursor *const cursor = engine->cursors + session_slot;
		if (session_slot != head && cursor->is_ready && !cursor->images_pending &&
		    (include_in_flight || (!cursor->awaits_token && cursor->in_flight == 0)) &&
		    cursor->status == OK && cursor->session.operation == 0 &&
		    cursor->position >= command_end(&cursor->command) && !is_cancelled(engine, session_slot))
			return true;
	}
	return false;
}

/* Whether the owner at session_slot still reads its prompt (a row of it not yet in flight). */
static bool reads_prompt(const struct inference_engine *engine,
			 uint32_t session_slot)
{
	const struct command_cursor *const cursor = engine->cursors + session_slot;
	return cursor->is_ready && cursor->position + cursor->in_flight < command_end(&cursor->command);
}

/* The rows of the owner's next wide or full chunk, as wide_flight_taken and full_flight_taken plan it; 0: none (its
 * prompt's tail of thin rows, or no prompt rows). */
static uint32_t next_chunk_rows(const struct inference_engine *engine,
				const struct command_cursor *cursor)
{
	const uint32_t position = cursor->position + cursor->in_flight, end = command_end(&cursor->command);
	const uint32_t plain_end = cursor->command.generated_token_limit != 0 && end != 0 ? end - 1u : end;
	const bool is_tail_region = position < cursor->front_tail_end;
	if (position < cursor->front_end)
		return engine->wide_rows == 0 ?
			       0 :
			       chunk_rows_of(engine, PROGRAM_WIDE,
					     (is_tail_region ?
						      cursor->front_tail_end :
						      cursor->front_end) -
						     position,
					     is_tail_region);
	return engine->full_rows == 0 || position >= plain_end ?
		       0 :
		       chunk_rows_of(engine, PROGRAM_FULL, plain_end - position,
				     true);
}

/* A ready prompt whose next rows belong in one FULL/WIDE chunk. Keep invalid rows on the ordinary failure
 * path: a refused chunk must not strand the command behind a reservation that can never run. */
static bool prompt_chunk_ready(const struct inference_engine *engine,
			       uint32_t session_slot)
{
	const struct command_cursor *const cursor =
		engine->cursors + session_slot;
	const uint32_t rows = next_chunk_rows(engine, cursor);
	if (rows == 0 || !cursor->is_ready || cursor->images_pending ||
	    cursor->awaits_token || cursor->status != OK)
		return false;
	const uint32_t position = cursor->position + cursor->in_flight;
	for (uint32_t row = 0; row < rows; row++) {
		if (!row_is_valid(engine, cursor, position + row) ||
		    (position >= cursor->front_end &&
		     (command_flags(cursor, position + row) &
		      ~STEP_ROW_IMAGE) != 0))
			return false;
	}
	return true;
}

/* Read-only cold-plan fronts. Kept boundaries must preserve both the original front tail and moved granules. */
static void prefix_fronts_planned(const struct inference_engine *engine,
				  struct command_cursor *cursor)
{
	const uint32_t start = cursor->command.kept_token_count,
		       end = command_end(&cursor->command);
	cursor->front_end = start;
	cursor->front_tail_end = start;
	if (engine->cone_rows == 0)
		return;
	if (engine->description.prefill_prefix_stable != 0) {
		cursor->front_end = stable_front_end(engine, start, end);
		cursor->front_tail_end = cursor->front_end;
		return;
	}
	const uint32_t whole = engine->cone_rows + SERVING_CONE_MARGIN;
	cursor->front_end = end > whole && end - whole > start ? end - whole :
								 start;
	cursor->front_tail_end = cursor->front_end;
	cursor->front_end += exact_front_moved(engine, cursor, end);
}

result_status engine_prefix_boundary(const struct inference_engine *engine,
				     uint32_t prompt_tokens, uint32_t maximum,
				     uint32_t decoder_from,
				     uint32_t *boundary_out)
{
	if (engine == NULL || boundary_out == NULL || prompt_tokens == 0 ||
	    prompt_tokens > engine->description.context_token_limit ||
	    maximum >= prompt_tokens ||
	    decoder_from > engine->description.context_token_limit)
		return STATUS_INVALID_ARGUMENT;
	uint32_t best = 0;
	const bool reserved = engine->description.session_slot_count <= 1u ||
			      engine->description.prefill_first != 0 ||
			      engine->description.prefill_order != 0;
	if (maximum != 0 && engine->description.chunks_in_flight == 1u &&
	    engine->description.prefill_fast_plan == 0 && reserved &&
	    engine->description.prefill_chunk_rows == 0 &&
	    engine->description.prefill_decode_rows == 0) {
		if (engine->description.prefill_prefix_stable != 0) {
			best = maximum / SERVING_STABLE_GRANULE *
			       SERVING_STABLE_GRANULE;
			/* Below the complete decoder tail, leave a whole cone to
			 * recompute. Choose its closed phase before cone_planned
			 * can rewind; the retained frontend remains canonical. */
			if (best < decoder_from &&
			    prompt_tokens - best < engine->cone_rows)
				best = prompt_tokens > engine->cone_rows ?
					       (prompt_tokens -
						engine->cone_rows) /
						       SERVING_STABLE_GRANULE *
						       SERVING_STABLE_GRANULE :
					       0u;
			*boundary_out = best;
			return OK;
		}
		struct command_cursor cold = {
			.command = { .new_token_count = prompt_tokens,
				     .generated_token_limit = 1 }
		};
		prefix_fronts_planned(engine, &cold);
		while (cold.position <= maximum) {
			const uint32_t position = cold.position;
			if (position % SERVING_KEPT_GRANULE == 0 &&
			    !(position < decoder_from &&
			      prompt_tokens - position < engine->cone_rows)) {
				struct command_cursor warm = {
					.command = { .kept_token_count =
							     position,
						     .new_token_count =
							     prompt_tokens -
							     position,
						     .generated_token_limit =
							     1 }
				};
				prefix_fronts_planned(engine, &warm);
				if (warm.front_end ==
					    (position > cold.front_end ?
						     position :
						     cold.front_end) &&
				    warm.front_tail_end ==
					    (position > cold.front_tail_end ?
						     position :
						     cold.front_tail_end))
					best = position;
			}
			const uint32_t chunk = next_chunk_rows(engine, &cold);
			const uint32_t advance = chunk != 0 ? chunk : 1u;
			if (advance > maximum - position)
				break;
			cold.position += advance;
		}
	}
	*boundary_out = best;
	return STATUS_OK;
}

/* The next step of several sessions or of a pipeline (the flights; the serial command_advanced is the single-session
 * path): a wide step where the owner whose turn it is has one (wide_flight_taken), else a full one
 * (full_flight_taken); else each ready owner that may take
 * a row gets one, round robin from first_slot (an owner whose next row needs the
 * token of a row in flight takes none, and with a pipeline at most DRAWING_ROWS owners that draw take one); the rows
 * left go to owners still in their prompt, as many consecutive prompt rows as each has. A step's rows are all front
 * rows or none (model_seam.h): the first owner's row decides. An owner's rows lie together
 * in ascending positions, as the hand-offs of new pool entries read them. With speculation, the first owner in turn
 * whose rows end with its row that draws once every owner's prompt rows are counted takes its proposals from the rows
 * left (proposals_taken), behind that row: at most one owner of a step verifies. An owner that failed or was cancelled
 * takes none and ends once nothing of it is in flight. Without a pipeline the step flies when it lands. false when no
 * owner takes a row. */
static bool flight_taken_off(struct inference_engine *engine,
			     uint32_t first_slot)
{
	const uint32_t slot_count = engine->description.session_slot_count;
	struct flight *const flight =
		&engine->flights[(engine->flight_first + engine->flight_count) %
				 engine->depth];
	uint32_t taken[SERVING_SESSION_CAPACITY] = { 0 };
	uint32_t proposals[SERVING_DRAFT_MOST];
	uint32_t verifier = SERVING_SESSION_CAPACITY,
		 proposed =
			 0; /* the turn of the owner with proposals (none), their count */
	bool drafts = false;
	memset(flight->proposed, 0, sizeof(flight->proposed));
	if (engine->is_batch) {
		memset(flight->drafts, 0, sizeof(flight->drafts));
		flight->round_valid_mask = 0;
	}
	const uint32_t decode_rows = engine->description.prefill_decode_rows;
	uint32_t prompt_count = 0;
	const uint32_t pending_prompt =
		engine->description.prefill_order != 0 || decode_rows != 0 ?
			prompt_head(engine, false, &prompt_count) :
			slot_count;
	const bool has_decoder =
		(decode_rows != 0 ||
		 engine->description.prefill_decode_every != 0) &&
		pending_prompt < slot_count &&
		decoding_owner(engine, slot_count, false);
	const bool slices =
		decode_rows != 0 && pending_prompt < slot_count &&
		(has_decoder || decoding_owner(engine, slot_count, true));
	const bool shares_prefill =
		slices && engine->description.prefill_decode_waiters != 0 &&
		prompt_count > 1u;
	if (shares_prefill && engine->decode_rounds_left > 1u)
		engine->decode_rounds_left = 1u;
	engine->prefill_chunk_limit = engine->description.prefill_chunk_rows;
	if (slices && !shares_prefill &&
	    (engine->prefill_chunk_limit == 0 ||
	     decode_rows < engine->prefill_chunk_limit))
		engine->prefill_chunk_limit = decode_rows;
	/* prefill_order: the head prompt's chunk alone (prompt_head): in any turn with prefill_first, else in its own turn
	 * or a waiting prompt's (a decoding owner's turn takes a thin or BATCH step) */
	const uint32_t oldest = engine->description.prefill_order != 0 ?
					pending_prompt :
					slot_count;
	/* prefill_order_wait: after so many steps in a row of a shorter prompt, the oldest's */
	const uint32_t wait = engine->description.prefill_order_wait;
	const bool shortest = engine->description.prefill_order >=
				      PREFILL_ORDER_SHORTEST &&
			      (wait == 0 || engine->head_bypass_steps < wait);
	const uint32_t head = shortest ? prompt_head(engine, true, NULL) :
					 oldest;
	const bool head_chunks = head < slot_count && engine->cursors[head].status == OK &&
				 !is_cancelled(engine, head) && next_chunk_rows(engine, engine->cursors + head) != 0;
	/* prefill_decode_every: after so many of the head's chunk steps in a row, a step of the decoding owners first */
	const uint32_t every = engine->description.prefill_decode_every;
	const uint32_t interval = every != 0 ? every : 1u;
	const bool starts_burst = slices &&
				  engine->head_chunks_in_row >= interval;
	const bool decode_due =
		has_decoder &&
		(slices ? starts_burst || engine->decode_rounds_left != 0 :
			  every != 0 && head_chunks &&
				  engine->head_chunks_in_row >= every);
	/* Retire the pending decoder before another chunk consumes its remaining burst. */
	const bool decode_wait =
		slices && !has_decoder &&
		(starts_burst || engine->decode_rounds_left != 0);
	/* A pipeline may land a flight on the head's turn. Slices supply its next prompt turn independently of that
	 * cursor; decode_due still gives the decoder its turn between chunks. */
	const bool head_turn =
		head < slot_count && !decode_due && !decode_wait &&
		(engine->description.prefill_first != 0 || slices ||
		 first_slot == head || reads_prompt(engine, first_slot));
	bool is_wide =
		!decode_due && !decode_wait &&
		(head < slot_count ?
			 head_turn &&
				 (wide_flight_taken(engine, head, flight) ||
				  full_flight_taken(engine, head, flight)) :
			 wide_flight_taken(engine, first_slot, flight) ||
				 full_flight_taken(engine, first_slot, flight));
	for (uint32_t distance = 1;
	     engine->description.prefill_first != 0 && head == slot_count &&
	     !is_wide && !decode_due && !decode_wait && distance < slot_count;
	     distance++) { /* prefill_first: any owner's chunk, in turn order from first_slot */
		const uint32_t session_slot = (first_slot + distance) % slot_count;
		is_wide = wide_flight_taken(engine, session_slot, flight) ||
			  full_flight_taken(engine, session_slot, flight);
	}
	engine->head_chunks_in_row = is_wide && (head < slot_count || slices) ?
					     engine->head_chunks_in_row + 1u :
					     0u;
	if (!slices)
		engine->decode_rounds_left = 0;
	else {
		if (starts_burst)
			engine->decode_rounds_left =
				shares_prefill ?
					1u :
				engine->description.prefill_decode_rounds != 0 ?
					engine->description
						.prefill_decode_rounds :
					1u;
		if (decode_due)
			engine->decode_rounds_left--;
	}
	if (!is_wide && slices)
		first_slot = decode_due ? engine->decode_next_slot :
			     head_turn && !head_chunks ? head :
							 first_slot;
	uint32_t row_count = is_wide ? flight->taken[0] : 0, drawing = 0,
		 front = 0, image_rows = 0;
	flight->owner_count = is_wide ? flight->owner_count : 0;
	for (uint32_t distance = 0;
	     distance < slot_count && row_count < engine->most_rows && !is_wide;
	     distance++) {
		const uint32_t session_slot =
			(first_slot + distance) % slot_count;
		struct command_cursor *const cursor =
			engine->cursors + session_slot;
		const uint32_t position = cursor->position + cursor->in_flight;
		const bool is_prompt = position < command_end(&cursor->command);
		/* A short tail shares only an ordinary thin flight, never a chunk or a decoder's reserved turn. */
		const bool short_prompt =
			engine->description.prefill_order ==
				PREFILL_ORDER_SHORT_ROWS &&
			head_turn && !head_chunks && is_prompt &&
			command_end(&cursor->command) - position <=
				engine->most_rows &&
			next_chunk_rows(engine, cursor) == 0;
		if (!cursor->is_ready || cursor->images_pending ||
		    cursor->awaits_token ||
		    (!is_prompt && cursor->in_flight != 0) ||
		    (is_prompt &&
		     (decode_due || decode_wait ||
		      (engine->description.prefill_prompt_chunks != 0 &&
		       prompt_chunk_ready(engine, session_slot)) ||
		      (head < slot_count &&
		       (head_chunks ||
			(session_slot != head && !short_prompt)))) &&
		     cursor->status == OK &&
		     !is_cancelled(engine, session_slot))) {
			/* Reserved chunks never donate a foreign row to BATCH. Other prompt rows wait unless order 3
			 * admits a short tail; cancelled/failed commands end here. Decoding owners may still share. */
			continue;
		}
		cursor->status = cursor->status == OK &&
						 !row_is_valid(engine, cursor,
							       position) ?
					 STATUS_INVALID_ARGUMENT :
					 cursor->status;
		if (cursor->status != OK ||
		    is_cancelled(engine, session_slot)) {
			if (cursor->in_flight == 0) {
				command_finished(
					engine, cursor,
					cursor->status != OK ?
						GENERATION_EVENT_FAILED :
						GENERATION_EVENT_CANCELLED);
			}
			continue;
		}
		if ((!engine->is_batch && !is_prompt && engine->depth > 1 &&
		     drawing == DRAWING_ROWS) ||
		    (row_count != 0 && (command_flags(cursor, position) &
					STEP_ROW_FRONT) != front) ||
		    (row_count != 0 && (command_flags(cursor, position) &
					STEP_ROW_IMAGE) != image_rows)) {
			continue;
		}
		front = command_flags(cursor, position) & STEP_ROW_FRONT;
		image_rows = command_flags(cursor, position) & STEP_ROW_IMAGE;
		if (!sampler_ready(engine, session_slot)) {
			continue;
		}
		drawing += is_prompt ? 0u : 1u;
		flight->owners[flight->owner_count++] = session_slot;
		taken[session_slot] = 1;
		row_count++;
		engine->next_slot = (session_slot + 1) % slot_count;
		/* Only the scheduled round advances its own cursor. */
		if (slices && decode_due && !is_prompt)
			engine->decode_next_slot = engine->next_slot;
	}
	/* the rows left: more prompt rows of the owners still reading their prompt */
	for (uint32_t turn = 0;
	     turn < flight->owner_count && row_count < engine->most_rows &&
	     !is_wide;
	     turn++) {
		const struct command_cursor *const cursor =
			engine->cursors + flight->owners[turn];
		uint32_t *const own = &taken[flight->owners[turn]];
		const uint32_t next = cursor->position + cursor->in_flight;
		while (row_count < engine->most_rows &&
		       next + *own < command_end(&cursor->command) &&
		       (next >= cursor->front_tail_end ||
			next + *own < cursor->front_tail_end) &&
		       (command_flags(cursor, next + *own) & STEP_ROW_FRONT) ==
			       front &&
		       (command_flags(cursor, next + *own) & STEP_ROW_IMAGE) ==
			       image_rows &&
		       row_is_valid(engine, cursor, next + *own)) {
			(*own)++;
			row_count++;
		}
	}
	/* then the first verifier's proposals: they never take a row a prompt could (a BATCH step plan: at most caps) */
	uint32_t caps[SERVING_SESSION_CAPACITY];
	if (engine->is_batch && !is_wide)
		engine_draft_depths(engine, flight, taken, row_count, caps);
	for (uint32_t turn = 0;
	     turn < flight->owner_count &&
	     (engine->is_batch || verifier == SERVING_SESSION_CAPACITY) &&
	     engine->description.speculative_depth != 0 && !is_wide;
	     turn++) {
		struct command_cursor *const cursor =
			engine->cursors + flight->owners[turn];
		uint32_t *const own = &taken[flight->owners[turn]];
		const uint32_t anchor =
			cursor->position + cursor->in_flight + *own - 1u;
		const uint32_t rows_left = engine->most_rows - row_count;
		const bool step_planned = engine->is_batch && caps[turn] != UINT32_MAX && flight->owner_count > 1u;
		proposed = proposals_taken(
			engine, cursor, anchor,
			engine->is_batch && caps[turn] < rows_left ?
				caps[turn] :
				rows_left,
			step_planned, flight->owner_count == 1u, proposals);
		drafts = engine->draft_positions != 0 &&
			 (cursor->needs_dspark_tail ||
			  (proposed != 0 &&
			   cursor->draft_source == DRAFT_LOOKUP)) &&
			 (command_flags(cursor, anchor) & STEP_ROW_LOGITS) !=
				 0 &&
			 cursor->command.generated_token_limit -
					 cursor->generated >
				 1u &&
			 anchor + proposed + 1u + engine->draft_positions <=
				 engine->description.context_token_limit;
		if (engine->is_batch) {
			flight->proposed[turn] = proposed;
			flight->drafts[turn] = drafts;
			memcpy(flight->proposal_tokens[turn], proposals,
			       (size_t)proposed * sizeof(*proposals));
		}
		verifier = proposed != 0 || drafts ? turn : verifier;
		*own += proposed;
		row_count += proposed;
	}
	if (row_count == 0) {
		return false;
	}
	if (head < slot_count && (is_wide || taken[head] != 0)) /* the head's step: prefill_order_wait's count */
		engine->head_bypass_steps = head == oldest ? 0 : engine->head_bypass_steps + 1u;
	for (uint32_t turn = 0, row = 0; turn < flight->owner_count && !is_wide;
	     turn++) {
		struct command_cursor *const cursor =
			engine->cursors + flight->owners[turn];
		flight->taken[turn] = taken[flight->owners[turn]];
		flight->proposed[turn] = engine->is_batch ?
						 flight->proposed[turn] :
					 turn == verifier ? proposed :
							    0;
		const uint32_t *const own_proposals =
			engine->is_batch ? flight->proposal_tokens[turn] :
					   proposals;
		const bool own_drafts = engine->is_batch ?
						flight->drafts[turn] :
						drafts && turn == verifier;
		/* the owner's rows, its proposals last: each drawn */
		const uint32_t first_proposal =
			flight->taken[turn] - flight->proposed[turn];
		for (uint32_t own = 0; own < flight->taken[turn]; own++) {
			const uint32_t position =
				cursor->position + cursor->in_flight + own;
			flight->rows[row++] = (struct step_row_record){
				(int32_t)position,
				(int32_t)(own < first_proposal ?
						  command_token(engine, cursor,
								position) :
						  own_proposals[own -
								first_proposal]),
				flight->owners[turn],
				own < first_proposal ?
					command_flags(cursor, position) |
						(own_drafts && own + 1u ==
									 first_proposal ?
							 STEP_ROW_DRAFTS :
							 0u) :
					STEP_ROW_LOGITS | STEP_ROW_PROPOSED
			};
		}
	}
	flight->submitted_ns = engine_now_ns();
	const result_status status =
		engine->depth > 1 ?
			step_submit(engine->driver, flight->rows, row_count) :
			OK;
	flight->row_count = row_count;
	if (status == OK && engine->depth > 1) {
		rows_reached(engine, flight->rows, row_count);
	}
	engine->flight_count += status == OK ? 1u : 0u;
	for (uint32_t row = 0; row < row_count && status == OK; row++) {
		const struct step_row_record *const flown = &flight->rows[row];
		engine->checkpoint_barrier =
			engine->checkpoint_barrier ||
			checkpoint_due(engine,
				       &engine->cursors[flown->session_slot],
				       (uint32_t)flown->position + 1u);
	}
	for (uint32_t turn = 0, row = 0; turn < flight->owner_count; turn++) {
		struct command_cursor *const cursor =
			engine->cursors + flight->owners[turn];
		row += flight->taken[turn];
		cursor->status = status == OK ? cursor->status : status;
		cursor->live_chosen_depth = flight->proposed[turn];
		cursor->in_flight += status == OK ? flight->taken[turn] : 0u;
		cursor->awaits_token =
			status == OK && row - 1u < engine->row_capacity &&
			(flight->rows[row - 1u].flags & STEP_ROW_LOGITS) != 0;
	}
	if (status != OK) {
		engine_driver_failed(engine, status);
	}
	return true;
}

/* The target's draws decide the accepted prefix before any owner publishes a token. The device merely reports that
 * same decision and uses it to build the next draft; malformed metadata fails the whole flight. */
MUST_CHECK static result_status round_checked(struct inference_engine *engine,
					      const struct flight *flight,
					      struct step_round *round_out)
{
	for (uint32_t anchor = 0; anchor < flight->row_count; anchor++) {
		const struct step_row_record *const row = &flight->rows[anchor];
		if ((row->flags & STEP_ROW_DRAFTS) == 0) {
			continue;
		}
		result_status status =
			step_round_take(engine->driver, round_out);
		uint32_t emitted = 1;
		while (anchor + emitted < flight->row_count &&
		       flight->rows[anchor + emitted].session_slot ==
			       row->session_slot &&
		       (flight->rows[anchor + emitted].flags &
			STEP_ROW_PROPOSED) != 0 &&
		       !token_ends_generation(
			       engine,
			       (uint32_t)engine->results[anchor + emitted - 1u]
				       .token_id) &&
		       engine->results[anchor + emitted - 1u].token_id ==
			       flight->rows[anchor + emitted].token_id) {
			emitted++;
		}
		if (status == OK &&
		    (round_out->session_slot != row->session_slot ||
		     round_out->emitted_count != emitted ||
		     round_out->anchor_position !=
			     row->position + (int32_t)emitted ||
		     round_out->anchor_token !=
			     engine->results[anchor + emitted - 1u].token_id ||
		     round_out->proposal_count != engine->draft_positions)) {
			status = STATUS_DEVICE_ERROR;
		}
		if (status == OK) {
			const struct step_row_record last = {
				round_out->anchor_position +
					(int32_t)engine->draft_positions - 1,
				0, row->session_slot, 0
			};
			rows_reached(engine, &last, 1);
		}
		return status;
	}
	return OK;
}

/* BATCH has already validated every device result in the eager driver before host history commits.
 * Copy every active round before any owner publishes reached rows, statistics, checkpoints or events. */
MUST_CHECK static result_status
batch_rounds_checked(struct inference_engine *engine, struct flight *flight)
{
	struct step_round pending[SERVING_SESSION_CAPACITY] = { 0 };
	uint32_t mask = 0;
	for (uint32_t slot = 0; slot < SERVING_SESSION_CAPACITY; slot++)
		pending[slot].session_slot = STEP_NO_SESSION;
	for (uint32_t anchor = 0; anchor < flight->row_count; anchor++) {
		const struct step_row_record *const row = &flight->rows[anchor];
		if ((row->flags & STEP_ROW_DRAFTS) == 0)
			continue;
		struct step_round *const next = &pending[row->session_slot];
		const result_status status = step_round_take_for_session(
			engine->driver, row->session_slot, next);
		if (status != OK)
			return status;
		uint32_t emitted = 1;
		while (anchor + emitted < flight->row_count &&
		       flight->rows[anchor + emitted].session_slot ==
			       row->session_slot &&
		       (flight->rows[anchor + emitted].flags &
			STEP_ROW_PROPOSED) != 0 &&
		       !token_ends_generation(
			       engine,
			       (uint32_t)engine->results[anchor + emitted - 1u]
				       .token_id) &&
		       engine->results[anchor + emitted - 1u].token_id ==
			       flight->rows[anchor + emitted].token_id)
			emitted++;
		if (next->session_slot != row->session_slot ||
		    next->emitted_count != emitted ||
		    next->anchor_position != row->position + (int32_t)emitted ||
		    next->anchor_token !=
			    engine->results[anchor + emitted - 1u].token_id ||
		    next->proposal_count != engine->draft_positions)
			return STATUS_DEVICE_ERROR;
		mask |= 1u << row->session_slot;
	}
	memcpy(flight->rounds, pending, sizeof(pending));
	flight->round_valid_mask = mask;
	return OK;
}

static void batch_rows_reached(struct inference_engine *engine,
			       const struct flight *flight)
{
	rows_reached(engine, flight->rows, flight->row_count);
	for (uint32_t slot = 0; slot < engine->description.session_slot_count;
	     slot++) {
		if ((flight->round_valid_mask & (1u << slot)) == 0)
			continue;
		const struct step_round *const next = &flight->rounds[slot];
		const struct step_row_record last = {
			next->anchor_position +
				(int32_t)engine->draft_positions - 1,
			0, slot, 0
		};
		rows_reached(engine, &last, 1);
	}
}

/* draft_learned_acceptance: a reached DSpark draft's outcome joins its cell (sampling mode, position, confidence bin),
 * whose acceptance becomes (matched + the prior's of ACCEPTANCE_PRIOR_REACHED outcomes) / (reached + those), per
 * thousand, rounded; a cell holding ACCEPTANCE_HALVED_REACHED reached drafts halves its counts (it follows the
 * text). */
static void acceptance_learned(struct inference_engine *engine, uint32_t mode,
			       uint32_t position, uint32_t bin, bool matched)
{
	if (position >= DRAFT_CALIBRATION_POSITIONS ||
	    bin >= DRAFT_CONFIDENCE_BINS)
		return;
	uint32_t *const reached = &engine->learned_reached[mode][position][bin];
	uint32_t *const hits = &engine->learned_matched[mode][position][bin];
	*reached += 1u;
	*hits += matched ? 1u : 0u;
	if (*reached >= ACCEPTANCE_HALVED_REACHED) {
		*reached /= 2u;
		*hits /= 2u;
	}
	const uint64_t weight = (uint64_t)*reached + ACCEPTANCE_PRIOR_REACHED;
	engine->draft_policy.acceptance[mode][position][bin] =
		(uint32_t)(((uint64_t)*hits * DRAFT_PROBABILITY_SCALE +
			    (uint64_t)engine->acceptance_prior[mode][position]
							      [bin] *
				    ACCEPTANCE_PRIOR_REACHED +
			    weight / 2u) /
			   weight);
}

/* Only the reached prefix supplies conditional acceptance observations. Target draws after a rejection were
 * conditioned on rejected history and cannot label the remaining draft proposals. */
static void round_statistics(struct inference_engine *engine,
			     struct command_cursor *cursor,
			     const struct flight *flight, uint32_t anchor,
			     uint32_t proposed, const struct step_round *next)
{
	const uint32_t mode = cursor->command.sampler_temperature > 0.0f ? 1u :
									   0u;
	struct draft_statistics *const statistics = &cursor->statistics;
	statistics->drawing_rounds++;
	statistics->depth_rounds[proposed]++;
	if (cursor->dspark_opportunity) {
		if (proposed != 0 && cursor->draft_source == DRAFT_DSPARK) {
			cursor->rounds_since_dspark = 0;
		} else if (cursor->rounds_since_dspark <
			   engine->draft_policy.probe_rounds) {
			cursor->rounds_since_dspark++;
		}
		cursor->dspark_opportunity = false;
	}
	if (proposed != 0) {
		cursor->live_probes += cursor->live_probe_planned &&
						       cursor->draft_source ==
							       DRAFT_DSPARK ?
					       1u :
					       0u;
		struct draft_source_statistics *const source =
			&statistics->source[cursor->draft_source];
		source->rounds++;
		if ((engine->description.draft_session_acceptance ==
			     SESSION_RESIDUAL_MODE ||
		     engine->description.draft_session_acceptance ==
			     SESSION_COST_MODE) &&
		    cursor->draft_source == DRAFT_DSPARK &&
		    proposed == engine->draft_positions) {
			cursor->session_had_full_round = true;
			cursor->session_last_full_round = source->rounds;
		}
		bool reached = true;
		for (uint32_t position = 0; position < proposed; position++) {
			const int32_t drawn =
				engine->results[anchor + position].token_id;
			const bool matched =
				reached &&
				!token_ends_generation(engine,
						       (uint32_t)drawn) &&
				drawn == flight->rows[anchor + position + 1u]
						 .token_id;
			source->proposed[position]++;
			source->reached[position] += reached ? 1u : 0u;
			source->accepted[position] += matched ? 1u : 0u;
			if (cursor->draft_source == DRAFT_DSPARK) {
				const uint32_t bin = draft_confidence_bin(
					cursor->draft_round
						.confidence_logits[position]);
				struct draft_outcomes *const outcome =
					&statistics->confidence[mode][position]
							       [bin];
				outcome->verified++;
				outcome->reached += reached ? 1u : 0u;
				outcome->matched += matched ? 1u : 0u;
				if (reached && engine->description.draft_session_acceptance != 0)
					session_draft_seen(engine, cursor, mode, position, bin, matched);
				if (reached && engine->description.draft_learned_acceptance != 0)
					acceptance_learned(engine, mode, position, bin, matched);
			}
			reached = matched;
		}
		/* draft_session_acceptance 4 to 11: a round whose verified drafts all matched also checks its first
		 * unverified draft against the target's draw after them (the bonus token): the drafts the policy cuts
		 * are seen too, not only the ones it verifies (4: the shift's samples would else lack every cut draft;
		 * 5: the rate would else count the confident drafts alone; gsl10c: prose reached the full depth too) */
		const struct step_round *const round = &cursor->draft_round;
		if (reached && cursor->draft_source == DRAFT_DSPARK &&
		    (engine->description.draft_session_acceptance ==
			     SESSION_HINDSIGHT_MODE ||
		     engine->description.draft_session_acceptance ==
			     SESSION_FULL_MODE ||
		     engine->description.draft_session_acceptance ==
			     SESSION_POSITION_MODE ||
		     engine->description.draft_session_acceptance ==
			     SESSION_SCOUT_MODE ||
		     engine->description.draft_session_acceptance ==
			     SESSION_SECOND_MODE ||
		     engine->description.draft_session_acceptance ==
			     SESSION_NATURAL_MODE ||
		     (engine->description.draft_session_acceptance ==
			      SESSION_RESIDUAL_MODE ||
		      engine->description.draft_session_acceptance ==
			      SESSION_COST_MODE)) &&
		    proposed < round->proposal_count &&
		    proposed < DRAFT_CALIBRATION_POSITIONS) {
			const int32_t drawn = engine->results[anchor + proposed].token_id;
			session_draft_seen(
				engine, cursor, mode, proposed,
				draft_confidence_bin(
					round->confidence_logits[proposed]),
				!token_ends_generation(engine,
						       (uint32_t)drawn) &&
					drawn == round->proposals[proposed]);
		}
		if (cursor->draft_source == DRAFT_DSPARK &&
		    (engine->description.draft_session_acceptance ==
			     SESSION_SHIFT_MODE ||
		     engine->description.draft_session_acceptance ==
			     SESSION_HINDSIGHT_MODE))
			session_shift_update(cursor);
		if (cursor->draft_source == DRAFT_DSPARK &&
		    engine->description.draft_session_acceptance ==
			    SESSION_PLATT_MODE)
			session_platt_update(cursor);
		if (cursor->draft_source == DRAFT_DSPARK &&
		    (engine->description.draft_session_acceptance ==
			     SESSION_SCOUT_MODE ||
		     engine->description.draft_session_acceptance ==
			     SESSION_SECOND_MODE ||
		     engine->description.draft_session_acceptance ==
			     SESSION_NATURAL_MODE ||
		     (engine->description.draft_session_acceptance ==
			      SESSION_RESIDUAL_MODE ||
		      engine->description.draft_session_acceptance ==
			      SESSION_COST_MODE)))
			session_full_update(engine, cursor);
#if TIMING_HOOKS_ENABLED
		if (engine->description.draft_trace != 0 &&
		    cursor->draft_source == DRAFT_DSPARK) {
			(void)fprintf(stderr,
				      "draft selector slot=%u round=%" PRIu64
				      " depth=%u full=%u positions",
				      cursor->command.session_slot,
				      source->rounds, proposed,
				      (unsigned int)cursor->session_full_depth);
			for (uint32_t position = 0;
			     position < engine->draft_positions; position++)
				(void)fprintf(
					stderr, " %.3f/%.3f@%.9g",
					cursor->session_position_hits[position],
					cursor->session_position_reached
						[position],
					(double)(position < round->proposal_count ?
							 round->confidence_logits
								 [position] :
							 0.0f));
			(void)fputc('\n', stderr);
		}
#endif
	}
	if (next->session_slot == cursor->command.session_slot) {
		for (uint32_t position = 0; position < next->proposal_count;
		     position++) {
			const uint32_t bin = draft_confidence_bin(
				next->confidence_logits[position]);
			statistics->confidence[mode][position][bin].produced++;
			cursor->draft_confidence[position] = next->confidence_logits[position];
		}
		cursor->has_draft_confidence = next->proposal_count != 0 || cursor->has_draft_confidence;
	}
}

/* The oldest flight back: the whole batch is checked before any token is published (no successful row retires
 * shared failed work), then each owner's rows join its position and the owner of a row with logits takes its token:
 * a verifier, its anchor's draw and, while each draw is the proposal of the row behind it and no end token, that
 * row's draw too (keyed match), each a token as a step of one row gives it, its row joining the position; then the
 * driver's history of it is cut back to its position. Owners of a failed step fail; they end once nothing of theirs
 * is in flight. */
void engine_flight_landed(struct inference_engine *engine)
{
	struct step_round round = { .session_slot = STEP_NO_SESSION };
	struct step_row_result *const results = engine->results;
	memset(results, 0, engine->row_capacity * sizeof(*results));
	struct flight *const flight = &engine->flights[engine->flight_first];
	result_status status =
		engine->depth > 1 ? step_finish(engine->driver, results) :
				    step_run_entry(engine->driver, flight->rows,
						   flight->row_count, results);
#if TIMING_HOOKS_ENABLED && !CHECKING_HOOKS_ENABLED
	LOG_WARNING_ENTRY("records_round rows=%" PRIu32 " elapsed_ns=%" PRIu64,
			  flight->row_count,
			  engine_now_ns() - flight->submitted_ns);
#endif
	if (status == OK && !engine->is_batch) {
		engine_live_step(engine, flight->rows, flight->row_count,
				 flight->submitted_ns);
	}
	engine->flight_first = (engine->flight_first + 1) % engine->depth;
	engine->flight_count--;
	if (status == OK && engine->depth == 1 && !engine->is_batch) {
		rows_reached(engine, flight->rows, flight->row_count);
	}
	for (uint32_t row = 0; row < flight->row_count && status == OK; row++) {
		const bool draws =
			(flight->rows[row].flags & STEP_ROW_LOGITS) != 0;
		if ((draws && (results[row].token_id < 0 ||
			       (uint32_t)results[row].token_id >=
				       engine->description.vocabulary_size)) ||
		    (!draws && results[row].token_id != STEP_NO_TOKEN)) {
			status = STATUS_DEVICE_ERROR;
		}
	}
	status = status == OK ?
			 (engine->is_batch ?
				  batch_rounds_checked(engine, flight) :
				  round_checked(engine, flight, &round)) :
			 status;
	if (status == OK && engine->is_batch) {
		batch_rows_reached(engine, flight);
		engine_live_step(engine, flight->rows, flight->row_count,
				 flight->submitted_ns);
		const uint64_t now = engine_now_ns();
		if (flight->submitted_ns != 0 && now > flight->submitted_ns)
			batch_step_measured(engine, flight, now - flight->submitted_ns);
	}
	if (status != OK) {
		engine_driver_failed(engine, status);
	}
	/* The barrier makes a prompt-ending flight the last in flight, after every earlier flight has landed. */
	for (uint32_t turn = 0, row = 0;
	     turn < flight->owner_count && status == OK; turn++) {
		const struct command_cursor *const cursor =
			engine->cursors + flight->owners[turn];
		row += flight->taken[turn];
		if (checkpoint_due(engine, cursor,
				   (uint32_t)flight->rows[row - 1u].position +
					   1u)) {
			status = prompt_checkpoint(engine, cursor);
		}
	}
	engine->checkpoint_barrier = engine->flight_count != 0 &&
				     engine->checkpoint_barrier;
	if (status != OK) {
		engine_driver_failed(engine, status);
	}
	for (uint32_t turn = 0, row = 0; turn < flight->owner_count; turn++) {
		struct command_cursor *const cursor =
			engine->cursors + flight->owners[turn];
		const uint32_t limit = cursor->command.generated_token_limit;
		const uint32_t proposed = flight->proposed[turn];
		const struct step_round *const owner_round =
			engine->is_batch ?
				&flight->rounds[flight->owners[turn]] :
				&round;
		row += flight->taken[turn];
		cursor->in_flight -= flight->taken[turn];
		cursor->awaits_token =
			false; /* an owner's row with logits is its last in flight */
		cursor->status = cursor->status == OK ? status : cursor->status;
		if (cursor->status != OK || !cursor->is_ready ||
		    row - 1u >= engine->row_capacity) {
			continue;
		}
		const uint32_t anchor_row = row - 1u - proposed;
		if ((flight->rows[anchor_row].flags & STEP_ROW_LOGITS) != 0) {
			round_statistics(engine, cursor, flight, anchor_row,
					 proposed, owner_round);
		}
		cursor->position += flight->taken[turn] - proposed;
		uint32_t kind =
			cursor->position >= command_end(&cursor->command) &&
					limit == 0 ?
				GENERATION_EVENT_FINISHED_AT_LIMIT :
				GENERATION_EVENT_TOKEN;
		bool goes_on =
			kind == GENERATION_EVENT_TOKEN &&
			(flight->rows[anchor_row].flags & STEP_ROW_LOGITS) != 0;
		uint32_t accepted = 0;
		for (uint32_t index = 0; goes_on; index++) {
			const struct step_row_result *const drawn =
				&results[anchor_row + index];
			cursor->position += index != 0 ? 1u : 0u;
			cursor->token_id = (uint32_t)drawn->token_id;
			cursor->generated++;
			if (!engine_event_queued(
				    engine, flight->owners[turn],
				    (struct generation_event){
					    cursor->command.request_tag,
					    GENERATION_EVENT_TOKEN,
					    cursor->token_id,
					    cursor->position + 1, OK,
					    drawn->logit },
				    0)) {
				cursor->is_ready = false; /* the engine ends */
				break;
			}
			kind = token_ends_generation(engine, cursor->token_id) ?
				       GENERATION_EVENT_FINISHED_AT_END_TOKEN :
			       cursor->generated == limit ||
					       cursor->position ==
						       held_limit(engine) ?
				       GENERATION_EVENT_FINISHED_AT_LIMIT :
				       GENERATION_EVENT_TOKEN;
			goes_on = kind == GENERATION_EVENT_TOKEN &&
				  index < proposed &&
				  drawn->token_id ==
					  flight->rows[anchor_row + index + 1u]
						  .token_id;
			accepted += goes_on ? 1u : 0u;
		}
		if (!cursor->is_ready) {
			continue;
		}
		cursor->draft_round = owner_round->session_slot ==
						      flight->owners[turn] ?
					      *owner_round :
					      (struct step_round){ 0 };
		if (proposed != 0) {
			const uint32_t doubled = cursor->depth + cursor->depth;
			cursor->rounds++;
			cursor->proposals += proposed;
			cursor->accepted += accepted;
			if (engine->draft_positions == 0 ||
			    engine->draft_policy.depth_policy ==
				    DRAFT_DEPTH_ADAPTIVE) {
				cursor->depth =
					accepted + accepted < proposed ? 1u :
					accepted < proposed ? accepted + 1u :
					doubled < engine->description.speculative_depth ? doubled :
					engine->description.speculative_depth;
				cursor->cooldown =
					accepted + accepted < proposed ?
						SERVING_DRAFT_COOLDOWN :
						0;
			}
			COUNTER_ADD(speculative_rounds, 1u);
			COUNTER_ADD(speculative_proposals, proposed);
			COUNTER_ADD(speculative_accepted, accepted);
		}
		const result_status cut =
			proposed == 0 ?
				OK :
				step_history_truncate(engine->driver,
						      flight->owners[turn],
						      cursor->position);
		if (cut != OK) {
			cursor->status = cut;
			engine_driver_failed(engine, cut);
			kind = GENERATION_EVENT_FAILED;
		}
		if (kind != GENERATION_EVENT_TOKEN) {
			command_finished(engine, cursor, kind);
		}
	}
}

/* A step takes off while one may and an owner takes a row, else the oldest in flight lands. */
static void flights_advanced(struct inference_engine *engine,
			     uint32_t first_slot)
{
	if (!engine->checkpoint_barrier &&
	    engine->flight_count < engine->depth &&
	    flight_taken_off(engine, first_slot)) {
		return;
	}
	if (engine->flight_count != 0) {
		engine_flight_landed(engine);
	}
}

static void *stepping_thread(void *argument)
{
	struct inference_engine *const engine = argument;
	const uint32_t slot_count = engine->description.session_slot_count;
	if (slot_count == 0 || slot_count > SERVING_SESSION_CAPACITY) {
		return NULL;
	}
	(void)pthread_mutex_lock(&engine->lock);
	while (!engine->is_ending) {
		while (engine->command_count != 0) {
			const struct queued_command queued =
				engine->commands[engine->command_first];
			const struct session_command *const session =
				&queued.session;
			const uint32_t held =
				session->operation == SESSION_SAVE ?
					engine->slots[session->session_slot]
						.token_count :
					0;
			const struct generation_command_record command =
				session->operation == 0 ?
					queued.generation :
					(struct generation_command_record){
						.session_slot =
							session->session_slot,
						.kept_token_count = held,
						.request_tag =
							session->request_tag
					};
			engine->command_first =
				(engine->command_first + 1) % slot_count;
			engine->command_count--;
			engine->cursors
				[command.session_slot] = (struct command_cursor){
				.command = command,
				.session = *session,
				.clone = queued.clone,
				.clone_submitted_ns = queued.clone_submitted_ns,
				.position = command.kept_token_count,
				.admitted = engine->admissions++,
				.live_prompt_processed =
					command.kept_token_count,
				.depth = 1,
				.rounds_since_dspark =
					engine->description.draft_start_economic !=
							0 ?
						0u :
						engine->draft_policy
							.probe_rounds,
				.restores_checkpoint =
					queued.restores_checkpoint,
				.status = engine->failure,
				.images_pending =
					session->operation == 0 &&
					(command.images != NULL ||
					 step_images_supported(engine->driver)),
				.is_ready = true
			};
			if (session->operation == 0) {
				struct engine_slot_stats *const live =
					&engine->live
						 .slots[command.session_slot];
				live->started_ns = engine_now_ns();
				live->phase = STATS_PROMPT;
				live->phase_since_ns = live->started_ns;
			}
			if (!queued.restores_checkpoint) {
				cone_planned(
					engine,
					&engine->cursors[command.session_slot]);
			}
			if (engine->lookups !=
			    NULL) { /* what the drafter holds past the kept tokens is forgotten */
				draft_lookup_keep(
					&engine->lookups[command.session_slot],
					command.kept_token_count);
			}
		}
		uint32_t restoring = slot_count;
		for (uint32_t session_slot = 0; session_slot < slot_count;
		     session_slot++) {
			if (engine->cursors[session_slot].is_ready &&
			    engine->cursors[session_slot].restores_checkpoint) {
				restoring = session_slot;
				break;
			}
		}
		if (restoring < slot_count) {
			struct command_cursor *const cursor =
				&engine->cursors[restoring];
			(void)pthread_mutex_unlock(&engine->lock);
			if (engine->flight_count != 0) {
				engine_flight_landed(engine);
			} else {
				cursor->status =
					cursor->status != OK ?
						cursor->status :
						step_checkpoint_restore(
							engine->driver,
							restoring,
							cursor->command
								.kept_token_count);
				if (cursor->status != OK) {
					engine_driver_failed(engine,
							     cursor->status);
				}
				(void)pthread_mutex_lock(&engine->lock);
				/* Only G's window was repaired. The old reach still bounds earlier prefixes. */
				engine->slots[restoring].decoder_from =
					cursor->command.kept_token_count;
				engine->live.checkpoints_restored +=
					cursor->status == OK ? 1u : 0u;
				cursor->restores_checkpoint = false;
				cone_planned(engine, cursor);
				(void)pthread_mutex_unlock(&engine->lock);
			}
			(void)pthread_mutex_lock(&engine->lock);
			continue;
		}
		uint32_t pending =
			slot_count; /* a session command waits for no other */
		for (uint32_t session_slot = 0; session_slot < slot_count;
		     session_slot++) {
			pending = engine->cursors[session_slot].is_ready &&
						  engine->cursors[session_slot]
								  .session
								  .operation !=
							  0 ?
					  session_slot :
					  pending;
		}
		if (pending < slot_count) {
			(void)pthread_mutex_unlock(&engine->lock);
			engine_session_advanced(engine,
						engine->cursors + pending);
			(void)pthread_mutex_lock(&engine->lock);
			continue;
		}
		uint32_t selected = slot_count;
		for (uint32_t distance = 0; distance < slot_count; distance++) {
			const uint32_t session_slot =
				(engine->next_slot + distance) % slot_count;
			if (engine->cursors[session_slot].is_ready) {
				selected = session_slot;
				break;
			}
		}
		if (selected == slot_count) {
			(void)pthread_cond_wait(&engine->wake, &engine->lock);
			continue;
		}
		/* Landing a full pipeline consumes no owner's turn. Successful submissions advance the cursor themselves. */
		const bool landing_only =
			engine->flight_count != 0 &&
			(engine->flight_count >= engine->depth ||
			 engine->checkpoint_barrier);
		if (engine->description.prefill_prompt_chunks == 0 ||
		    (!landing_only && !prompt_chunk_ready(engine, selected)))
			engine->next_slot = (selected + 1) % slot_count;
		if (engine->failure != OK) {
			engine->cursors[selected].status = engine->failure;
		}
		(void)pthread_mutex_unlock(&engine->lock);
		struct command_cursor *const selected_cursor =
			engine->cursors + selected;
		if (selected_cursor->images_pending) {
			/* Drain the finite existing pipeline before borrowing card 0. A decoder cannot
			 * refill it between these waits and starve an image's next quantum. */
			while (engine->flight_count != 0) {
				engine_flight_landed(engine);
			}
			(void)pthread_mutex_lock(&engine->lock);
			selected_cursor->status =
				engine->failure != OK ? engine->failure :
							selected_cursor->status;
			(void)pthread_mutex_unlock(&engine->lock);
			if (selected_cursor->status != OK ||
			    is_cancelled(engine, selected)) {
				step_images_cancel(engine->driver, selected);
				command_finished(
					engine, selected_cursor,
					selected_cursor->status != OK ?
						GENERATION_EVENT_FAILED :
						GENERATION_EVENT_CANCELLED);
			} else {
				bool finished = false;
				result_status image_status = OK;
				if (!selected_cursor->images_started) {
					image_status = step_images_begin(
						engine->driver, selected,
						selected_cursor->command.images);
					selected_cursor->images_started =
						image_status == OK;
				}
				if (image_status == OK) {
					image_status = step_images_advance(
						engine->driver, selected,
						&finished);
				}
				if (image_status != OK &&
				    image_status != STATUS_BUSY_VALUE) {
					selected_cursor->status = image_status;
					engine_driver_failed(engine,
							     image_status);
				}
				selected_cursor->images_pending = !finished;
			}
		} else if (engine->is_batch || engine->depth > 1 ||
			   engine->wide_rows != 0 || engine->full_rows != 0 ||
			   engine->description.speculative_depth != 0 ||
			   (slot_count > 1 && engine->most_rows > 1)) {
			flights_advanced(engine, selected);
		} else {
			command_advanced(engine, engine->cursors + selected);
		}
		(void)pthread_mutex_lock(&engine->lock);
		engine_live_publish(engine);
	}
	(void)pthread_mutex_unlock(&engine->lock);
	return NULL;
}

result_status engine_draft_statistics(struct inference_engine *engine,
				      uint32_t session_slot,
				      struct draft_statistics *command_out,
				      struct draft_statistics *total_out)
{
	if (engine == NULL ||
	    session_slot >= engine->description.session_slot_count ||
	    (command_out == NULL && total_out == NULL) ||
	    (command_out != NULL && command_out == total_out)) {
		return STATUS_INVALID_ARGUMENT;
	}
	(void)pthread_mutex_lock(&engine->lock);
	const result_status status = engine->slots[session_slot].phase ==
						     SLOT_COMMANDED ?
					     STATUS_BUSY_VALUE :
					     OK;
	if (status == OK) {
		if (command_out != NULL) {
			*command_out = engine->slots[session_slot].statistics;
		}
		if (total_out != NULL) {
			*total_out = engine->statistics;
		}
	}
	(void)pthread_mutex_unlock(&engine->lock);
	return status;
}

/* A cap needs a prepared whole-granule program in every admitted chunk kind. Refuse sparse custom ladders instead
 * of silently falling back to a larger chunk or to the tail's different forms. */
static bool chunk_limit_supported(const struct step_driver *driver,
				  uint32_t limit)
{
	if (limit == 0)
		return true;
	const enum program_kind kinds[] = { PROGRAM_WIDE, PROGRAM_FULL };
	for (uint32_t index = 0; index < sizeof(kinds) / sizeof(kinds[0]);
	     index++) {
		if (step_chunk_rows(driver, kinds[index], UINT32_MAX) == 0)
			continue;
		uint32_t program = step_chunk_rows(driver, kinds[index], limit);
		while (program >= EXACT_PLAN_GRANULE &&
		       program % EXACT_PLAN_GRANULE != 0)
			program = step_chunk_rows(driver, kinds[index],
						  program - 1u);
		if (program < EXACT_PLAN_GRANULE)
			return false;
	}
	return true;
}

result_status engine_create(struct step_driver *driver,
			    const struct engine_description_record *description,
			    uint32_t most_rows,
			    struct inference_engine **engine_out)
{
	const bool is_batch = step_rounds_is_batch(driver);
	/* batch_step_us: a cost for each of the driver's BATCH programs, fewest rows (then verifiers) first */
	uint32_t batch_rows[SERVING_BATCH_COSTS],
		batch_verifiers[SERVING_BATCH_COSTS], cost_count = 0;
	const uint32_t batch_count =
		driver != NULL && is_batch ?
			step_batch_layout(driver, batch_rows, batch_verifiers,
					  SERVING_BATCH_COSTS) :
			0;
	while (description != NULL && cost_count < SERVING_BATCH_COSTS &&
	       description->batch_step_us[cost_count] != 0)
		cost_count++;
	const uint32_t decode_capacity =
		is_batch ? step_chunk_rows(driver, PROGRAM_BATCH, UINT32_MAX) :
			   LINEAR_MAXIMUM_ROWS;
	if (driver == NULL || description == NULL || engine_out == NULL ||
	    most_rows == 0 || most_rows > decode_capacity ||
	    (is_batch && (most_rows > STEP_BATCH_MAXIMUM_ROWS ||
			  step_pipeline_depth(driver) != 1u)) ||
	    description->session_slot_count == 0 ||
	    description->session_slot_count > SERVING_SESSION_CAPACITY ||
	    description->context_token_limit == 0 ||
	    description->end_token_id >= description->vocabulary_size ||
	    description->ignore_eos > 1u ||
	    (description->ignore_eos != 0 &&
	     (description->speculative_depth != 0 || is_batch)) ||
	    description->front_steps > 1 || description->prefill_first > 1 ||
	    description->prefill_prompt_chunks > 1u ||
	    description->prefix_cold_plan_qualified > 1u ||
	    description->prefill_prefix_stable > 1u ||
	    (description->prefill_prefix_stable != 0 &&
	     (description->chunks_in_flight != 1u ||
	      description->prefill_fast_plan != 0 ||
	      description->prefill_chunk_rows != 0 ||
	      description->prefill_decode_rows != 0 ||
	      description->prefill_cone_skip != 1u ||
	      description->wide_rows == 0 ||
	      description->wide_rows % SERVING_STABLE_GRANULE != 0 ||
	      step_chunk_rows(driver, PROGRAM_WIDE, UINT32_MAX) !=
		      description->wide_rows ||
	      step_chunk_rows(driver, PROGRAM_FULL, UINT32_MAX) !=
		      description->wide_rows ||
	      (description->prefill_margin != 0 &&
	       description->prefill_margin < SERVING_STABLE_GRANULE - 1u) ||
	      (description->session_slot_count > 1u &&
	       description->prefill_first == 0 &&
	       description->prefill_order == 0))) ||
	    description->prefill_order > PREFILL_ORDER_SHORT_ROWS ||
	    description->prefill_decode_waiters > 1u ||
	    description->prefill_decode_rounds > PREFILL_DECODE_MOST_ROUNDS ||
	    (description->prefill_decode_every != 0 &&
	     description->prefill_order == 0 &&
	     description->prefill_decode_rows == 0) ||
	    ((description->prefill_chunk_rows != 0 ||
	      description->prefill_decode_rows != 0) &&
	     (description->chunks_in_flight != 1u ||
	      description->prefill_fast_plan != 0 ||
	      description->prefill_chunk_rows % EXACT_PLAN_GRANULE != 0 ||
	      description->prefill_decode_rows % EXACT_PLAN_GRANULE != 0 ||
	      description->prefill_chunk_rows > description->wide_rows ||
	      description->prefill_decode_rows > description->wide_rows ||
	      !chunk_limit_supported(driver, description->prefill_chunk_rows) ||
	      !chunk_limit_supported(driver,
				     description->prefill_decode_rows))) ||
	    (description->prefill_order_wait != 0 &&
	     description->prefill_order < PREFILL_ORDER_SHORTEST) ||
	    description->prefill_cone_skip > PREFILL_CONE_UNTRIMMED ||
	    (description->prefill_tail_rows != 0 &&
	     (description->prefill_tail_rows % MODEL_WIDE_LEAST_ROWS != 0 ||
	      description->prefill_tail_rows % EXACT_PLAN_GRANULE == 0 ||
	      description->prefill_tail_rows < EXACT_PLAN_TAIL_ROWS ||
	      description->prefill_tail_rows > EXACT_PLAN_TAIL_MOST_ROWS ||
	      description->chunks_in_flight != 1u ||
	      description->prefill_fast_plan != 0)) ||
	    (cost_count != 0 && cost_count != batch_count) ||
	    (description->draft_policy != NULL &&
	     !draft_policy_valid(description->draft_policy)) ||
	    (description->hybrid_lookup_min != 0 &&
	     (step_draft_positions(driver) == 0 ||
	      description->speculative_depth == 0 ||
	      description->hybrid_lookup_min > description->lookup_ngram_most ||
	      description->lookup_ngram_least == 0 ||
	      description->lookup_ngram_least > description->lookup_ngram_most ||
	      description->lookup_ngram_most > DRAFT_LOOKUP_MOST_NGRAM)) ||
	    (description->front_steps != 0 && step_cone_rows(driver) == 0) ||
	    (description->wide_rows != 0 &&
	     (description->wide_rows != step_wide_rows(driver) ||
	      description->front_steps == 0)) ||
	    (description->speculative_depth != 0 &&
	     (description->speculative_depth >= most_rows ||
	      step_draft_positions(driver) > DRAFT_CALIBRATION_POSITIONS ||
	      description->speculative_depth > SERVING_DRAFT_MOST ||
	      (step_draft_positions(driver) != 0 ?
		       description->speculative_depth >
				       step_draft_positions(driver) ||
			       (!is_batch &&
				description->session_slot_count > 1u &&
				step_pipeline_depth(driver) == 1u) :
		       description->lookup_ngram_least == 0 ||
			       description->lookup_ngram_least >
				       description->lookup_ngram_most ||
			       description->lookup_ngram_most >
				       DRAFT_LOOKUP_MOST_NGRAM)))) {
		LOG_ERROR(
			"engine: refused: no driver, description or output, rows out of range, or a description "
			"without slots, context or an end token inside its vocabulary, or with a cone or wide "
			"rows not its programs', or draft positions or n-gram lengths out of range, or BATCH step costs "
			"not one a BATCH program");
		return STATUS_INVALID_ARGUMENT;
	}
	if (is_batch) {
		const result_status attached = step_rounds_domain_attach(
			driver, description->vocabulary_size,
			description->end_token_id);
		if (attached != OK)
			return attached;
	}
	const size_t slot_count = description->session_slot_count;
	const uint32_t device_positions = step_draft_positions(driver);
	const bool needs_lookup = device_positions == 0 || description->hybrid_lookup_min != 0;
	const uint32_t bucket_bits =
		!needs_lookup ?
			0 :
			draft_lookup_bucket_bits(
				description->context_token_limit,
				description->lookup_ngram_least,
				description->lookup_ngram_most);
	const size_t draft_words =
		description->speculative_depth == 0 || !needs_lookup ?
			0 :
			slot_count * ((size_t)description->context_token_limit +
				      ((size_t)1u << bucket_bits));
	const uint32_t full_rows =
		step_chunk_rows(driver, PROGRAM_FULL, UINT32_MAX);
	const uint32_t chunk_rows = description->wide_rows > full_rows ?
					    description->wide_rows :
					    full_rows;
	const uint32_t row_capacity =
		chunk_rows > most_rows ? chunk_rows : most_rows;
	struct inference_engine *const engine =
		FAULT_POINT(engine_resources) ?
			NULL :
			calloc(1,
			       sizeof(*engine) +
				       slot_count *
					       (sizeof(struct queued_command) +
						sizeof(struct command_cursor) +
						sizeof(struct draft_lookup) +
						sizeof(struct slot)) +
				       (size_t)row_capacity *
					       (STEP_PIPELINE_MOST *
							sizeof(struct step_row_record) +
						sizeof(struct step_row_result)) +
				       draft_words * sizeof(uint32_t));
	if (engine == NULL) {
		return STATUS_OUT_OF_MEMORY;
	}
	engine->driver = driver;
	engine->is_batch = is_batch;
	engine->batch_cost_count = cost_count;
	engine->worker_live.batch_program_count = cost_count;
	for (uint32_t index = 0; index < cost_count; index++) {
		engine->batch_cost_rows[index] = batch_rows[index];
		engine->batch_cost_us[index] = description->batch_step_us[index];
		engine->batch_cost_prior_us[index] =
			description->batch_step_us[index];
		engine->batch_cost_verifiers[index] = batch_verifiers[index];
		engine->worker_live.batch_programs[index] =
			(struct stats_batch_program){
				batch_rows[index], batch_verifiers[index],
				description->batch_step_us[index], 0, 0
			};
	}
	engine->draft_positions =
		description->speculative_depth != 0 ? device_positions : 0;
	engine->description = *description;
	engine->live.started_ns = engine_now_ns();
	engine->live.speculation_mode =
		description->speculative_depth == 0 ? STATS_SPECULATION_OFF :
		description->hybrid_lookup_min != 0 ? STATS_SPECULATION_HYBRID :
		device_positions != 0		    ? STATS_SPECULATION_DSPARK :
						      STATS_SPECULATION_LOOKUP;
	draft_policy_default(&engine->draft_policy);
	if (description->draft_policy != NULL) {
		engine->draft_policy = *description->draft_policy;
	}
	engine->description.draft_policy = &engine->draft_policy;
	memcpy(engine->acceptance_prior, engine->draft_policy.acceptance, sizeof(engine->acceptance_prior));
	engine->cone_rows =
		description->front_steps != 0 ? step_cone_rows(driver) : 0;
	engine->description.snapshot_identity =
		step_session_identity(driver) ^ description->snapshot_identity ^
		(description->prefill_prompt_chunks != 0 ?
			 PROMPT_CHUNK_SNAPSHOT_SALT :
			 0);
	engine->depth = step_pipeline_depth(driver);
	engine->most_rows = is_batch				 ? most_rows :
			    most_rows - 1u < LINEAR_MAXIMUM_ROWS ? most_rows :
								   1u;
	engine->wide_rows = description->wide_rows;
	engine->full_rows = full_rows;
	engine->row_capacity = row_capacity;
	engine->commands = (struct queued_command *)(engine + 1);
	engine->cursors =
		(struct command_cursor *)(engine->commands + slot_count);
	struct draft_lookup *const lookups =
		(struct draft_lookup *)(engine->cursors + slot_count);
	engine->slots = (struct slot *)(lookups + slot_count);
	struct step_row_record *const flight_rows =
		(struct step_row_record *)(engine->slots + slot_count);
	for (uint32_t index = 0; index < STEP_PIPELINE_MOST; index++) {
		engine->flights[index].rows =
			flight_rows + (size_t)index * row_capacity;
	}
	engine->results =
		(struct step_row_result *)(flight_rows +
					   (size_t)STEP_PIPELINE_MOST *
						   row_capacity);
	engine->draft = engine->draft_positions != 0 ? device_drafted :
						       lookup_drafted;
	engine->lookups = draft_words != 0 ? lookups : NULL;
	if (draft_words != 0) {
		uint32_t *draft_memory =
			(uint32_t *)(engine->results + row_capacity);
		for (uint32_t session_slot = 0; session_slot < slot_count;
		     session_slot++) {
			draft_lookup_begin(
				&engine->lookups[session_slot], draft_memory,
				description->context_token_limit,
				draft_memory + description->context_token_limit,
				bucket_bits, description->lookup_ngram_least,
				description->lookup_ngram_most);
			draft_memory += draft_words / slot_count;
		}
	}
	engine->event_descriptor =
		FAULT_POINT(engine_resources) ?
			-1 :
			eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC | EFD_SEMAPHORE);
	bool is_running = false;
	if (engine->event_descriptor >= 0 &&
	    pthread_mutex_init(&engine->lock, NULL) == 0) {
		if (pthread_cond_init(&engine->wake, NULL) == 0) {
			sigset_t sent_signals, before;
			bool is_masked = sigfillset(&sent_signals) == 0;
			for (uint32_t index = 0;
			     index <
			     sizeof(fault_signals) / sizeof(fault_signals[0]);
			     index++) {
				is_masked = is_masked &&
					    sigdelset(&sent_signals,
						      fault_signals[index]) ==
						    0;
			}
			is_masked = is_masked &&
				    pthread_sigmask(SIG_SETMASK, &sent_signals,
						    &before) == 0;
			is_running =
				is_masked && !FAULT_POINT(engine_resources) &&
				pthread_create(&engine->thread, NULL,
					       stepping_thread, engine) == 0;
			if (is_masked) {
				(void)pthread_sigmask(SIG_SETMASK, &before,
						      NULL);
			}
			if (!is_running) {
				(void)pthread_cond_destroy(&engine->wake);
			}
		}
		if (!is_running) {
			(void)pthread_mutex_destroy(&engine->lock);
		}
	}
	if (!is_running) {
		LOG_ERROR(
			"engine: no event descriptor, lock or stepping thread could be had");
		if (engine->event_descriptor >= 0) {
			(void)close(engine->event_descriptor);
		}
		free(engine);
		return STATUS_OUT_OF_MEMORY;
	}
	*engine_out = engine;
	return OK;
}

void engine_destroy(struct inference_engine *engine)
{
	if (engine == NULL) {
		return;
	}
	(void)pthread_mutex_lock(&engine->lock);
	engine->is_ending = true;
	(void)pthread_cond_signal(&engine->wake);
	(void)pthread_mutex_unlock(&engine->lock);
	(void)pthread_join(engine->thread, NULL);
	(void)pthread_cond_destroy(&engine->wake);
	(void)pthread_mutex_destroy(&engine->lock);
	(void)close(engine->event_descriptor);
	free(engine);
}

result_status engine_describe(const struct inference_engine *engine,
			      struct engine_description_record *description_out)
{
	if (engine == NULL || description_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	*description_out = engine->description;
	return OK;
}

result_status engine_submit(struct inference_engine *engine,
			    const struct generation_command_record *command)
{
	if (engine == NULL || command == NULL ||
	    command->session_slot >= engine->description.session_slot_count ||
	    command->new_token_ids == NULL || command->new_token_count == 0 ||
	    command->reserved_must_be_zero != 0 ||
	    (engine->description.prefill_prefix_stable != 0 &&
	     (command->kept_token_count % SERVING_STABLE_GRANULE != 0 ||
	      command->images != NULL)) ||
	    (command->images != NULL &&
	     !step_images_supported(engine->driver)) ||
	    !isfinite(command->sampler_temperature) ||
	    command->sampler_temperature < 0.0f ||
	    !(command->sampler_top_p > 0.0f &&
	      command->sampler_top_p <= 1.0f)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint32_t context = engine->description.context_token_limit;
	(void)pthread_mutex_lock(&engine->lock);
	struct slot *const slot = &engine->slots[command->session_slot];
	const bool restores_checkpoint =
		command->kept_token_count != 0 &&
		slot->checkpoint_count == command->kept_token_count &&
		slot->reach_count >= command->kept_token_count &&
		slot->reach_count - command->kept_token_count >
			SERVING_KEPT_LAG;
	bool duplicate_tag = false;
	for (uint32_t other = 0; other < engine->description.session_slot_count;
	     other++) {
		duplicate_tag = duplicate_tag ||
				(engine->slots[other].phase != SLOT_IDLE &&
				 engine->slots[other].request_tag ==
					 command->request_tag);
	}
	const result_status status =
		engine->failure != OK ? engine->failure :
		engine->is_ending     ? STATUS_BUSY_VALUE :
		slot->phase != SLOT_IDLE ||
				engine->slot_read_leased[command->session_slot] ||
				engine->command_count ==
					engine->description.session_slot_count ?
				    STATUS_BUSY_VALUE :
		duplicate_tag ? STATUS_INVALID_ARGUMENT :
		command->kept_token_count > slot->token_count ||
				command->kept_token_count > context ||
				command->new_token_count >
					context - command->kept_token_count ?
				STATUS_INVALID_ARGUMENT :
		command->kept_token_count > slot->processed_count ||
				command->kept_token_count %
						SERVING_KEPT_GRANULE !=
					0 ||
				(command->kept_token_count != 0 &&
				 !restores_checkpoint &&
				 slot->reach_count - command->kept_token_count >
					 SERVING_KEPT_LAG) ?
				STATUS_INVALID_ARGUMENT :
		engine->description.session_slot_count > 1 &&
				command->generated_token_limit != 0 &&
				command_end(command) == context ?
				STATUS_INVALID_ARGUMENT :
				OK;
	if (status == OK &&
	    engine->slot_generations[command->session_slot] == UINT64_MAX) {
		(void)pthread_mutex_unlock(&engine->lock);
		return STATUS_UNSUPPORTED;
	}
	if (status == OK) {
		engine->slot_generations[command->session_slot]++;
		engine->clone_metrics[command->session_slot] =
			(struct session_clone_metrics){ 0 };
		engine->live.slots[command->session_slot] =
			(struct engine_slot_stats){
				.slot = command->session_slot,
				.request_tag = command->request_tag,
				.phase = STATS_QUEUED,
				.prompt_tokens = command_end(command),
				.reused_tokens = command->kept_token_count,
				.prompt_processed = command->kept_token_count,
				.position = command->kept_token_count,
				.context_capacity = context,
				.speculation_mode =
					engine->live.speculation_mode,
				.reused_checkpoint = restores_checkpoint,
				.submitted_ns = engine_now_ns()
			};
		engine->live.slots[command->session_slot].phase_since_ns =
			engine->live.slots[command->session_slot].submitted_ns;
		engine->live.sessions[command->session_slot].operation = 0;
		engine->live.sessions[command->session_slot].idle_since_ns = 0;
		engine->live.prompt_tokens += command_end(command);
		engine->live.cached_tokens += command->kept_token_count;
		engine->commands[(engine->command_first + engine->command_count) %
				 engine->description.session_slot_count] =
			(struct queued_command){ .generation = *command,
						 .restores_checkpoint =
							 restores_checkpoint };
		engine->command_count++;
		*slot = (struct slot){ SLOT_COMMANDED,
				       slot->token_count,
				       false,
				       command->request_tag,
				       slot->processed_count,
				       slot->decoder_from,
				       command->kept_token_count != 0 ?
					       slot->reach_count :
					       0,
				       0, slot->statistics };
		(void)pthread_cond_signal(&engine->wake);
	}
	(void)pthread_mutex_unlock(&engine->lock);
	return status;
}

result_status engine_cancel(struct inference_engine *engine,
			    uint32_t session_slot)
{
	if (engine == NULL ||
	    session_slot >= engine->description.session_slot_count) {
		return STATUS_INVALID_ARGUMENT;
	}
	(void)pthread_mutex_lock(&engine->lock);
	struct slot *const slot = &engine->slots[session_slot];
	slot->is_cancelled = slot->is_cancelled ||
			     slot->phase == SLOT_COMMANDED;
	(void)pthread_mutex_unlock(&engine->lock);
	return OK;
}

/* Takes the oldest event, if there is one. */
static bool event_taken(struct inference_engine *engine,
			struct generation_event *event_out)
{
	(void)pthread_mutex_lock(&engine->lock);
	const bool is_taken = engine->event_count != 0;
	if (is_taken) {
		const struct queued_event taken =
			engine->events[engine->event_first];
		engine->event_first =
			(engine->event_first + 1) % EVENT_CAPACITY;
		engine->event_count--;
		if (taken.event.event_kind != GENERATION_EVENT_TOKEN) {
			engine->live.sessions[taken.session_slot].idle_since_ns =
				engine_now_ns();
			engine->live.sessions[taken.session_slot].operation = 0;
			struct slot *const slot =
				&engine->slots[taken.session_slot];
			*slot = (struct slot){
				SLOT_IDLE,
				taken.event.session_token_count,
				false,
				0,
				taken.processed_count,
				slot->decoder_from,
				slot->reach_count > taken.processed_count ?
					slot->reach_count :
					taken.processed_count,
				slot->checkpoint_count, slot->statistics
			};
		}
		if (engine->event_count == EVENT_CAPACITY - 1) {
			(void)pthread_cond_signal(
				&engine->wake); /* room again for an event */
		}
		*event_out = taken.event;
	}
	(void)pthread_mutex_unlock(&engine->lock);
	return is_taken;
}

/* A taken event's count off the descriptor (the semaphore eventfd's read takes one): the count never falls below
 * the queue's, so no burst of events between two takes can leave one queued with the descriptor clear. */
result_status engine_next_event(struct inference_engine *engine,
				struct generation_event *event_out,
				bool *event_available_out)
{
	uint64_t one = 0;
	if (engine == NULL || event_out == NULL ||
	    event_available_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	*event_available_out = event_taken(engine, event_out);
	if (*event_available_out &&
	    read(engine->event_descriptor, &one, sizeof(one)) < 0) {
		LOG_ERROR("engine: the event descriptor could not be read");
		return STATUS_INPUT_OUTPUT_ERROR;
	}
	return OK;
}

result_status engine_event_descriptor(struct inference_engine *engine,
				      int *descriptor_out)
{
	if (engine == NULL || descriptor_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	*descriptor_out = engine->event_descriptor;
	return OK;
}
