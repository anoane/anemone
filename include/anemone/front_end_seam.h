/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * front_end_seam.h: how anything that serves requests talks to the engine.
 *
 * Contract:
 *   - A front end (the OpenAI-compatible server is the first) never touches a card, a kernel or the pack's
 *     weights. What it needs from the pack and the model is handed over once by engine_describe. It turns
 *     a request into token ids and hands the engine ONE fixed-size command; the engine answers with a stream of
 *     fixed-size events. No text crosses this seam. Optional immutable image records carry prepared BF16 patches and
 *     their typed token spans; the worker encodes them on card 0 before the request's text prefill.
 *   - A command names a session slot and how many of that session's tokens are kept: everything after them is
 *     forgotten, the new tokens are read, then generation runs. Keeping a prefix is how prefill caching works;
 *     the front end decides what matches, the engine truncates and continues. A kept count at the slot's prompt
 *     checkpoint may exceed the ordinary lag: the engine first restores its window rows (step.h).
 *   - With a description's front_steps set, a prompt's rows before its last cone rows (the driver's programs',
 *     step_cone_rows) and SERVING_CONE_MARGIN
 *     run as front steps (model_seam.h): the same answers, sooner. A command that keeps fewer tokens than its
 *     session's decoder state needs (after a snapshot's load or a cut-short prompt) and brings too few new ones for
 *     a cone of its own first reads the session's own tokens again from where one fits (serving_cursor.h).
 *   - With a description's speculative_depth set, the engine decodes speculatively. The driver selects its device
 *     draft where it has one, otherwise prompt lookup. Device proposals use their round's next anchor and a
 *     confidence-table prefix maximizing estimated saved time, using the latest returned scores before the next
 *     tail is scheduled. Periodic depth-one probes refresh skipped tails and bound lookup takeover; DSpark has no
 *     rejection cooldown. Lookup alone retains adaptive depth and its cooldown. A command's row that draws (its
 *     prompt's last, then the token it drew last: the anchor) goes in one step with up to speculative_depth tokens
 *     that its drafter proposes for the positions behind it (prompt lookup: what followed the latest earlier
 *     occurrence of the longest suffix of lookup_ngram_most down to lookup_ngram_least of the session's tokens), each
 *     row's token drawn by the sampler. The draws are the command's tokens up to the first whose row before it
 *     carried another token than the draw, or the end token (keyed match), and only their rows join what the session
 *     holds: the events, and the tokens kept, are those of steps of one drawing row for every seed. The limit, the
 *     context's end fall between a round's tokens; cancellation is checked between rounds. Rejected rows leave state
 *     past what the session holds, which the next rows write again before any reads it; a later command keeps a
 *     prefix at most SERVING_KEPT_LAG behind its slot's reach (serving_cursor.h), which leaves room for them.
 *   - The sampler settings are data in the command: nothing is prepared ahead. engine_submit copies the
 *     record. The token ids and optional image records are borrowed: they stay valid and unchanged until the
 *     final event of the command has been taken.
 *   - Every generated token is one TOKEN event, the end token included; the end token is the model's, reported by
 *     engine_describe. Every command ends with exactly one final event (any kind but TOKEN), and a command
 *     that reaches the end token on its last allowed step ends at the end token. Every event carries the number
 *     of tokens the session holds after it. The context limit bounds the positions read: the token drawn at the
 *     last position is the last one, FINISHED_AT_LIMIT behind it. Four-session commands stop one read
 *     earlier so prompt plus drawn tokens stay within the admitted short context.
 *   - After a final event the session holds the first session_token_count tokens of the kept, the new and the
 *     generated tokens, in that order: all of them after FINISHED_AT_END_TOKEN and FINISHED_AT_LIMIT, those read
 *     or generated up to the step boundary after CANCELLED, whatever survived, possibly none, after FAILED. The
 *     next command of the slot may keep at most that many.
 *   - A slot takes one command at a time: a submit before the final event of the previous one has been taken
 *     fails with STATUS_BUSY_VALUE. So submit knows what the session holds, and it refuses
 *     (STATUS_INVALID_ARGUMENT) a kept_token_count above what the cards ran of that, off the kept granule or
 *     more than SERVING_KEPT_LAG behind the slot's reach without a checkpoint at that count
 *     (serving_cursor.h), a slot the engine does not
 *     have, a command with no new token, kept and new tokens beyond the context limit, a temperature that is
 *     negative or not finite, a top_p outside (0, 1], and a reserved field that is not zero. A token id at or above
 *     the vocabulary size is found when it is read and ends the command with a FAILED event.
 *   - Commands and events cross through two bounded queues, one pair per engine. Submitting to a full command
 *     queue fails at once with STATUS_BUSY_VALUE: the front end refuses the request. When the event queue is
 *     full the engine stops stepping sessions until an event is taken: no event is ever dropped, and the engine
 *     never grows a queue.
 *   - One thread submits and cancels; one thread, the same or another, takes events. An engine serves one front
 *     end at a time. No function below blocks or touches a card.
 *   - One worker steps one driver. One session retains serial chunking; four sessions admit context<=256 with
 *     prepared thin programs and a startup row limit1/2/4. Default1 retains RR; optional2/4 take at most one next
 *     row per ready owner, in RR order, with actual live3 using program4. All selected sampler settings precede
 *     dispatch; every result is checked before any batch token is published. Numerical equivalence is unqualified.
 *     Tags must differ across live commands until
 *     final TAKEN; a duplicate in a free slot is INVALID_ARGUMENT. A driver failure poisons admission; peers
 *     receive FAILED without more driver calls. Host command finalization does not establish card retirement.
 *     Recreate only after the card owner actually drains/closes all readers and retires quarantined staging.
 *   - The two records are pinned for LP64, the only data model the engine is built for.
 */
#ifndef FRONT_END_SEAM_H
#define FRONT_END_SEAM_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/status.h"
#include "anemone/draft_policy.h"

#define GENERATION_COMMAND_BYTES 56
#define GENERATION_EVENT_BYTES 24

struct generation_command_record {
	const uint32_t
		*new_token_ids; /* read after the kept ones; at least one */
	uint64_t sampler_seed; /* a draw is keyed by seed, position and token id */
	uint32_t session_slot;
	uint32_t kept_token_count; /* tokens of the session that stay; the rest is forgotten */
	uint32_t new_token_count;
	uint32_t generated_token_limit; /* 0: read the new tokens and generate nothing */
	float sampler_temperature; /* 0 selects the highest logit, the lowest index on a tie */
	float sampler_top_p; /* 1 keeps every token */
	uint32_t request_tag; /* returned in every event of this command */
	uint32_t reserved_must_be_zero;
	const struct chat_images *
		images; /* immutable request-owned pixels; borrowed until final TAKEN */
};

enum generation_event_kind {
	GENERATION_EVENT_TOKEN, /* one generated token, the end token included */
	GENERATION_EVENT_FINISHED_AT_END_TOKEN, /* final: the model's end token was generated */
	GENERATION_EVENT_FINISHED_AT_LIMIT, /* final: the limit of the command or of the context */
	GENERATION_EVENT_CANCELLED, /* final: engine_cancel ended it */
	GENERATION_EVENT_FAILED, /* final: status says why */
	GENERATION_EVENT_KIND_COUNT
};

struct generation_event {
	uint32_t request_tag;
	uint32_t event_kind; /* enum generation_event_kind */
	uint32_t token_id; /* of a TOKEN event; 0 in every other */
	uint32_t session_token_count; /* tokens the session holds after this event */
	uint32_t status; /* result_status: STATUS_OK except in a FAILED event */
	float logit; /* of a TOKEN event's token, as the head computed it; else 0 */
};

_Static_assert(
	sizeof(struct generation_command_record) == GENERATION_COMMAND_BYTES,
	"the command record changed size: every front end is built on it");
_Static_assert(sizeof(struct generation_event) == GENERATION_EVENT_BYTES,
	       "the event record changed size: every front end is built on it");

/* A slot's state to or from a file between its generation commands (a snapshot, step.h): saved for the session's
 * first token_count tokens (at most those the cards ran), or loaded with token_ids as the tokens the session then
 * holds, reach_count as its reach and decoder_from as its actual decoder boundary. A slot's reach is the largest
 * row position plus one its cards wrote, including rejected proposals and device draft positions, since a command
 * last kept none. A loaded snapshot supplies the saving slot's reach; a kept prefix counts back from that reach
 * (serving_cursor.h), for its window ring
 * may hold rows of positions beyond what the session holds. The file is the front end's: the engine writes or reads
 * it from its offset on and never closes it. The command takes the slot as a generation command does, waits for
 * every step in flight, and ends with one final event: FINISHED_AT_LIMIT, or FAILED with the status (a failed save
 * leaves the session as it was, a failed load empty). */
enum session_operation {
	SESSION_SAVE = 1,
	SESSION_LOAD = 2,
	SESSION_CLONE = 3 /* submitted only through engine_session_clone */
};
struct session_command {
	const uint32_t *
		token_ids; /* a load's, borrowed until its final event is taken; NULL for a save */
	int descriptor;
	uint32_t session_slot;
	uint32_t token_count;
	uint32_t operation; /* enum session_operation */
	uint32_t request_tag;
	uint32_t reach_count; /* a load's: token_count to the context limit; a save's 0 */
	uint32_t reserved_must_be_zero;
	uint32_t decoder_from; /* a load's actual decoder boundary, at most the context limit; a save's 0 */
};

enum session_clone_source {
	SESSION_CLONE_ORDINARY = 0,
	SESSION_CLONE_CHECKPOINT = 1
};

/* An idle source read lease and independent idle destination, checked atomically against their current generations.
 * The caller proves canonical-prefix/owner eligibility; the engine derives actual reach/decoder/checkpoint metadata.
 * No source cursor, sampler, adaptive draft state or mutable arena storage is shared. */
struct session_clone_command {
	uint64_t source_generation, destination_generation;
	uint32_t source_slot, destination_slot;
	uint32_t token_count, source_kind;
	uint32_t request_tag, reserved_must_be_zero;
};

struct session_clone_metrics {
	uint64_t byte_count; /* physical ranges, including mirrors and an actual checkpoint */
	uint64_t copy_ns; /* submission through every card's completion, with flights already drained */
	uint64_t wait_ns; /* admission/queue/flight-drain time before the copy */
};

/* Idle-slot metadata after the final event is taken. A checkpoint is local to the slot, is replaced after a prompt
 * completes, and is invalidated by a snapshot load; snapshots carry the ordinary ring and decoder boundary only. */
struct session_state {
	uint32_t reach_count;
	uint32_t decoder_from;
	uint32_t checkpoint_count; /* 0: none; else a kept-granule boundary before the prompt's last row */
	uint64_t generation; /* changes on each accepted mutating admission; never a request tag */
	struct session_clone_metrics
		clone; /* last clone, cleared on a later mutating admission */
};

/* The pointers stay valid, and what they point at unchanged, for as long as the engine lives. */
/* The most BATCH programs a driver's step costs name (--batch_step_us). */
#define SERVING_BATCH_COSTS 8u
/* prefill_order's value for the fewest rows left first (struct engine_description_record). */
#define PREFILL_ORDER_SHORTEST 2u
/* Shortest first, with short prompt tails sharing a thin step's existing row budget. */
#define PREFILL_ORDER_SHORT_ROWS 3u

#define PREFILL_DECODE_MOST_ROUNDS 64u

struct engine_description_record {
	const char *model_name; /* the model's one name (model_seam.h) */
	const void *
		front_end_tables; /* the pack's tokenizer and template tables, digest checked at load */
	uint64_t front_end_table_byte_count;
	uint32_t session_slot_count;
	uint32_t context_token_limit;
	uint32_t vocabulary_size;
	uint32_t end_token_id;
	uint32_t front_steps; /* 1: a prompt's rows before its cone are front steps (model_seam.h); 0: none */
	uint32_t wide_rows; /* 0, or the driver's wide program's rows: front rows run in wide steps of that many */
	uint32_t chunks_in_flight; /* a prompt's chunked rows in about so many chunks at least (engine.c); 0: 1 */
	uint32_t prefill_margin; /* chunks_in_flight 1: full rows beyond a prompt's cone (engine.c); 0: the default */
	uint32_t prefill_fast_plan; /* chunks_in_flight 1: 1 the largest chunk that fits, not the exact plan */
	/* Experimental fixed-capacity prefill; aligned fronts, inactive tail
         * rows and an eight-row cache phase. Requires uncapped reserved chunks
         * and compatible padded WIDE/FULL programs; other cold bits than plan
         * 0. */
	uint32_t prefill_prefix_stable;
	/* 1: prompts one at a time in the order their commands came (engine.c's prompt_head): only the oldest command
	 * still in its prompt takes chunks or prompt rows, so its first token comes after its own prompt rather than after
	 * every prompt in flight; 0: the owners' turns (each prompt a chunk in turn). Each prompt's chunks are the same.
	 * PREFILL_ORDER_SHORTEST: of the prompts the one with the fewest rows left goes first (ties: the oldest),
	 * so a short prompt does not wait behind a long one's prefill, and prompts of one size go as with 1.
	 * PREFILL_ORDER_SHORT_ROWS also admits other no-chunk prompt tails whose remaining rows fit a thin
	 * step's row budget. Whole chunks and the decode cadence are unchanged. */
	uint32_t prefill_order;
	/* 0, or with either shortest order: after so many steps in a row of prompts other than the oldest, a step
	 * of the oldest's (no prompt waits for ever behind shorter ones); 0: the fewest rows left alone decide */
	uint32_t prefill_order_wait;
	/* 0, or with prefill_order (M12's decode cadence): while the head prompt has chunks left, after so many of its
	 * chunk steps in a row a step of the decoding owners' rows comes first (each step's programs and rows as without
	 * it: the same bits); 0: the turns alone decide (with prefill_first, the decoding owners wait for the chunks) */
	uint32_t prefill_decode_every;
	/* Exact plan only: cap whole-granule chunks, retaining the original tail. chunk_rows applies always (also for
	 * numerical qualification); decode_rows only while another owner can decode, then gives that owner a round
	 * between chunks. Zero keeps the original plan; nonzero values are whole 256-row granules. */
	uint32_t prefill_chunk_rows, prefill_decode_rows;
	/* With decode_rows: this many ordinary decode rounds between chunk turns; 0 or 1 means one. */
	uint32_t prefill_decode_rounds;
	/* With decode_rows: 1 keeps original chunks and one decode round while multiple prompts remain. */
	uint32_t prefill_decode_waiters;
	/* 1: a full step's prompt rows carry the cone's layer count (step_records.h STEP_ROW_CONE: the decoder
	 * layers whose output of the row a later row or the head reads) for the prompt's end less the plan's
	 * margin (engine.c cone_end_of: prefill_margin's), so the later layers' cores and routed experts leave the
	 * row alone; a later command keeping fewer than end - margin - 1 of the prompt's tokens then runs a cone
	 * again (decoder_from). 2: the cone of the prompt's last row (a margin of 0). Needs front_steps (the
	 * driver's cone); every row read keeps its bits. 1 and 2 mark the rows CONE_TRUNKS too (the decoder layers'
	 * wide trunk products leave them alone as well); 3: 1 without it. 0: no row skips. */
	uint32_t prefill_cone_skip;
	/* 0, or the rows of a full program that takes the exact plan's 64-row tail chunks in their forms (64 a group):
	 * the tail's 64-row chunks go in steps of so many rows (engine.c's exact plan), each row the same bits */
	uint32_t prefill_tail_rows;
	uint32_t speculative_depth; /* 0: no speculation; else the most proposals a round verifies, below most_rows */
	uint32_t lookup_ngram_most; /* host lookup's longest suffix, including hybrid mode */
	uint32_t lookup_ngram_least; /* and its shortest, at least 1 */
	uint64_t snapshot_identity; /* given: what else a snapshot's bits follow (0: none); create mixes in its own */
	uint64_t image_math_identity; /* immutable encoder arithmetic identity, mixed only into nonempty image keys */
	uint32_t hybrid_lookup_min; /* 0: device only; else minimum suffix match; lookup must also beat DSpark's gain */
	uint32_t draft_trace; /* 1: each device round's session, anchor and proposals to stderr (diagnostics); 0: none */
	const struct draft_policy_record *
		draft_policy; /* NULL: defaults; copied at create, need not outlive it */
	/* BATCH drivers: a step's microseconds on each of the driver's BATCH programs, fewest rows first; nonzero: each
	 * step verifies the drafts whose rows buy the most expected tokens a microsecond (engine.c's step plan), 0: the
	 * draft policy's depths alone */
	uint32_t batch_step_us[SERVING_BATCH_COSTS];
	/* 1: a wide or full step of any owner that has one goes before every other step (the prompts' chunks first, so
	 * that streams decode without pauses for later prompts' chunks); 0: an owner's chunk in its turn */
	uint32_t prefill_first;
	/* Reserve runnable prompt chunks from ordinary BATCH packing without reserving the decode turns. */
	uint32_t prefill_prompt_chunks;
	/* 1: the DSpark acceptance table the draft policy and the BATCH step plan read (by sampling mode, position and
	 * confidence bin) follows this server's own outcomes: each cell's matched over reached drafts, the policy's
	 * table the prior (engine.c, ACCEPTANCE_PRIOR_REACHED outcomes of weight), halved past
	 * ACCEPTANCE_HALVED_REACHED, so it follows the running text; 0: the policy's table as given. Drafts only: the
	 * verified tokens are the target's */
	uint32_t draft_learned_acceptance;
	/* 1: a session's table acceptances are scaled by its own ratio of matched to expected drafts (engine.c,
	 * SESSION_ACCEPTANCE_*), for the draft policy's depth and the BATCH step plan (Morrow's per-request
	 * acceptance); 2: the table's log-odds plus a shift fitted to the command's last reached drafts; 3: its
	 * log-odds times a scale plus a shift, both fitted so (Platt's scaling); 4: 2, its samples with each round's
	 * first unverified draft checked against the target's next draw; 5: every draft verified (acceptance 1) while
	 * the command's reached drafts and 4's hindsight checks match at 0.82 or more, else the table;
	 * 6: position-local conditional counts, including hindsight, lift underestimated confidence bins with a
	 * four-observation prior; 7: four full-depth probes, then full depth while middle-position acceptance is high
	 * (0.90 entry, 0.82 exit), otherwise the table unchanged; full probes at the configured probe interval;
	 * 8: four full probes, then second-position acceptance (0.90 entry, 0.80 exit, two observations), vetoed by
	 * repeated middle failures; otherwise the table unchanged, with full recovery probes only while the second
	 * position stays strong; 9: the second-position rule with four observations and no full-depth warmup;
	 * 10: position residual calibration in both directions; the second-position full-depth choice applies
	 * directly at one owner while BATCH planning retains calibrated conditional probabilities; one bounded
	 * full-depth exploration per BATCH flight, only when real row capacity permits;
	 * 0: the table alone. Drafts only */
	/* 11: residual probabilities with observed successful pure-decode row/tail shape costs; one-owner RATE
	 * uses request-local whole-round depth costs after warmup, with bounded full-depth recovery probes. */
	uint32_t draft_session_acceptance;
	/* 1: a command's first DSpark round takes the policy's depth from the drafts' confidences (as every later
	 * round); 0: it is a forced probe of one draft (rounds_since_dspark starts at the probe interval).
	 * Drafts only */
	uint32_t draft_start_economic;
	/* 1: the BATCH step plan cuts its list of drafts (most probable first) only between drafts of unequal reach,
	 * so drafts of equal reach (identical commands' at the same place) are verified all or none and identical
	 * commands keep identical depths; 0: anywhere (the earlier owner's draft first). Drafts only */
	uint32_t batch_plan_ties;
	/* Immutable producer assertion: this actual placement, arithmetic and prefill recipe support exact cold-plan
	 * boundaries. Zero means unknown/unqualified and cannot authorize cross-session slot copies. This capability
	 * is not a user knob and does not select a prefix policy, numerical form or scheduling behavior. */
	uint32_t prefix_cold_plan_qualified;
	/* 1: emit end-token draws and continue to the token/context limit. Non-speculative THIN only. */
	uint32_t ignore_eos;
};

struct inference_engine;
struct step_driver;

/* The program's side. The driver's programs read up to most_rows rows a step (1 to LINEAR_MAXIMUM_ROWS), its
 * host experts attached; the description's pointers outlive the engine, which copies the record. The driver is the
 * engine's until destroy returns. A NULL argument, a slot count of 0, a context limit or vocabulary of 0, an end
 * token outside the vocabulary, front_steps other than 0 or 1 or set where the driver's programs carry no cone
 * (step_cone_rows), wide_rows other than 0 or the rows of the driver's wide program
 * (step_wide_rows, which needs front steps), and speculative_depth not below most_rows or above a thin step's
 * rows less one, or above the driver's nonzero draft block size; that block must fit the five calibration positions.
 * A device draft with several slots needs a pipeline.
 * Host lookup (including hybrid) needs 1 <= least <= most <= 16. Hybrid needs a device draft, speculation enabled,
 * and hybrid_lookup_min <= lookup_ngram_most. A non-NULL draft_policy must pass draft_policy_valid. These are
 * STATUS_INVALID_ARGUMENT; memory, the descriptor or the thread refused
 * STATUS_OUT_OF_MEMORY (fault point engine_resources). destroy ends the command that runs at its next step
 * boundary, with no final event, and joins the thread; it accepts NULL. */
/* prefill_dead_halves: with the exact plan and the cone (chunks_in_flight 1, no fast plan, prefill_cone_skip), the
 * least rows between the last row of a full chunk of whole MODEL_FULL_CONE_LEAST_ROWS-row granules (no tail
 * program's) and its prompt's cone end's last row (engine.c); 0: no such bound (another plan, no cone). A program
 * built for the bound may leave a decoder layer's sums of a half none of whose rows' outputs there the cone reads
 * (model_layout.h's MODEL_CONE_LAYERS of the half's last row's least distance). */
uint32_t engine_full_cone_behind(uint32_t front_steps,
				 uint32_t chunks_in_flight,
				 uint32_t prefill_fast_plan,
				 uint32_t prefill_margin,
				 uint32_t prefill_cone_skip);

MUST_CHECK result_status
engine_create(struct step_driver *driver,
	      const struct engine_description_record *description,
	      uint32_t most_rows, struct inference_engine **engine_out);
void engine_destroy(struct inference_engine *engine);

/* Release usage data. Last completed generation command in this slot, and engine lifetime completed-command totals.
 * Read after its final event, before submitting another command. Either output may be NULL, but not both; outputs
 * must be distinct. NULL engine, bad slot or aliased outputs: INVALID_ARGUMENT; active slot: BUSY. On failure outputs
 * stay untouched. Session save/load commands do not change these statistics. Fixed storage, no card work. */
MUST_CHECK result_status
engine_draft_statistics(struct inference_engine *engine, uint32_t session_slot,
			struct draft_statistics *command_out,
			struct draft_statistics *total_out);

/* First user: the server at start (its tokenizer, the model list, the refusal of a prompt that is too long). */
MUST_CHECK result_status
engine_describe(const struct inference_engine *engine,
		struct engine_description_record *description_out);
/* Last resumable cold-plan boundary <= maximum, for a generating text prompt. The caller proves the retained
 * rows have this same cold-plan provenance and invariant THIN arithmetic. Uses immutable prepared sizes only;
 * checks front seams and declines cone rewinds. Fixed exact plan only; multiple slots require chunk reservation
 * by prefill_first or prefill_order, and both prefill row caps must be zero. Other plans return boundary 0.
 * With prefix-stable
 * prefill this rounds to its owner phase and leaves a whole cone when starting
 * below decoder_from. The caller additionally clips to the growing donor's original
 * headless prompt prefix; generated THIN rows are excluded.
 * No card work.
 * Invalid pointers, empty/out-of-context prompt, maximum >= prompt_tokens or decoder_from beyond context:
 * INVALID_ARGUMENT, output untouched. Same-length provenance is not inferred from matching tokens here. */
MUST_CHECK result_status engine_prefix_boundary(
	const struct inference_engine *engine, uint32_t prompt_tokens,
	uint32_t maximum, uint32_t decoder_from, uint32_t *boundary_out);
MUST_CHECK result_status
engine_submit(struct inference_engine *engine,
	      const struct generation_command_record *command);
/* A session command: refused as submit refuses (a busy slot, a full queue, a tag in use), and for an operation or
 * descriptor that is none, a save of more tokens than the session ran or of none, or with a reach, a load beyond the
 * context or off the kept granule (serving_cursor.h) or whose reach is below its count or beyond the context, and a
 * reserved field that is not zero. */
MUST_CHECK result_status engine_session(struct inference_engine *engine,
					const struct session_command *command);
/* Reserves both current idle generations. Only destination receives a final event; source remains unchanged and
 * cannot accept writers until every copy drains. Success holds exactly token_count with the source's actual reach;
 * cancellation/non-device failure discards destination history for a cold restart. Hardware failure poisons the engine.
 * Stale generations/invalid boundaries return INVALID_ARGUMENT; busy slots/read leases return BUSY with no change. */
MUST_CHECK result_status
engine_session_clone(struct inference_engine *engine,
		     const struct session_clone_command *command);
/* Reads idle-slot metadata; refuses NULL, an absent slot (INVALID_ARGUMENT) or a live command (BUSY).
 * Leaves the output untouched on failure. No card work. */
MUST_CHECK result_status engine_session_state(struct inference_engine *engine,
					      uint32_t session_slot,
					      struct session_state *state_out);
/* Ends the command of a slot at the next step or speculative-round boundary, with one CANCELLED as its final
 * event. A slot with no command, or whose final event is already queued, is left alone. Fails only for a NULL
 * engine or a slot the engine does not have. First users: a matched stop sequence, and a client that went away. */
MUST_CHECK result_status engine_cancel(struct inference_engine *engine,
				       uint32_t session_slot);
/* Takes the next event if one is waiting; event_available_out says whether it did. */
MUST_CHECK result_status engine_next_event(struct inference_engine *engine,
					   struct generation_event *event_out,
					   bool *event_available_out);
/* A descriptor that polls readable while events are waiting, for the poll loop of a server. The front end only
 * polls it: it never reads, writes or closes it, and it lives as long as the engine. This is a semaphore eventfd:
 * the engine writes one count per queued event and engine_next_event reads one per taken event. */
MUST_CHECK result_status
engine_event_descriptor(struct inference_engine *engine, int *descriptor_out);

#endif
