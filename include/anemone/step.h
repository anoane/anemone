/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * step.h: the step driver, which runs a model's programs (model_seam.h) one step at a time and knows no model.
 *
 * Contract:
 *   - create takes thin programs that program_prepare made from one plan, for distinct row counts (a step runs the
 *     one of the fewest rows that holds it), each with the first's segments (but for their launches) and hand-offs;
 *     wide ones (model_seam.h): each the front of the first program, a thin one, in segments and hand-offs, run by a
 *     front step of exactly its rows; and full ones: all of it but its head, less the hand-offs of no entry ratio
 *     behind their last segment, which only the head reads (cone programs are STATUS_UNSUPPORTED until the
 *     device-driven driver of wide_prefill.md). It checks them against one
 *     another and the cards: one plan's cards, limits and offsets; every card, launch range, segment, hand-off and
 *     area names something that exists (a thin program has no views; each card that runs a wide program's launches
 *     has the same views, slabs of its rows one behind the other in the arena, which a step writes and uploads);
 *     every area and every copy at its largest (row_count rows, or every session's entries to the context limit)
 *     lies inside its arenas. It obtains everything a step uses: page-locked staging for the
 *     step area, the n-gram rows, the largest set of hand-offs and the result, and each session's token history
 *     (the context limit of every slot). Nothing is allocated per step, and nothing runs on a card before the first
 *     step. Explicit graph preparation may upload executable graphs before it without running model kernels.
 *   - An explicit BATCH cohort replaces THIN decode programs: up to64 rows/eight sessions, one unchanged88B
 *     round per session slot and <=8-row views above eight rows. A private cold token/EOS domain is required
 *     before work (engine_create attaches it). Every active owner validates before any result/round publication.
 *     Ordinary THIN metadata, its single round and its legacy token-domain contract remain unchanged.
 *   - A step: prepare n-gram rows from committed history plus the step's proposed inputs without changing
 *     committed tokens/counts. The step area and n-gram rows go up; the segments run in order,
 *     the hand-offs after each through staging (a download behind the source card's launches, its wait, an upload
 *     before the destination's next launches); then the result card's result area comes back, and is taken only if
 *     it answers this step. Once all copies and launches complete successfully, the input tokens join
 *     committed host history/counts, including PROPOSED tokens until the caller truncates the rejected suffix.
 *     On any failure those remain unchanged; device state is not rolled back.
 *     The row_count results are valid only on success and must not be used on failure.
 *   - step_run_entry refuses (STATUS_INVALID_ARGUMENT, nothing queued, no history changed) a NULL
 *     argument, a row count of 0 or above the largest program's, a step no program runs (a wide program runs front
 *     steps of exactly its rows, the thin ones up to theirs) and a row that breaks step_records.h or this rule:
 *     a row continues its session (its position is the count of tokens the session holds, with the rows before it
 *     in the step) or, as the session's first row of the step, goes back to a position at or below that count: the
 *     tokens before it stay (0 begins the session again). The caller keeps only what the model's state still holds
 *     behind such a position (the engine: serving_cursor.h's kept granule and lag).
 *   - Every slot starts at seed 0, temperature 0 and top_p 1; these, and settings set later, reach the card that
 *     samples with the next step (with a pipeline: the next step that samples).
 *   - Deferred mirrors. A front step leaves to a mirror sync each hand-off of an entry ratio behind a segment of its
 *     front that goes to a card running a segment of the front before it and none behind it there (DeepSeek-V4.1-
 *     Flash: layer 20's pool entries and keys going back to the cards of earlier front layers in an interleaved
 *     layout), where all of them leave one card behind one segment; state flows forward (model_seam.h), so nothing a
 *     front step runs reads them. Each session keeps the range of positions its front steps wrote since its last
 *     sync. A step of other rows first syncs the range of each of its sessions behind which its own rows lie (once:
 *     the range is then empty), from the card they leave to each card they go to, in pieces through the session block
 *     (step_sessions.c), before any segment reads them: without a pipeline at the step's start; with one, before the
 *     leaving card's visit that those hand-offs follow, the cards they go to standing still. A snapshot's save syncs
 *     its session first; a load leaves its session synced. A layout whose hand-offs back go to no such card defers
 *     nothing.
 *   - Expert pauses (model_seam.h): create holds each to its segment, card and arena, a card's slots to be the same
 *     in all its pauses, at least jobs_per_row times the program's rows and apart from every area and copy above on
 *     that card, the pauses to name the host experts in order, each once, and every program's pauses and host
 *     experts to be the first's. step_host_experts_attach copies every host expert they name out of the host
 *     arena into page-locked staging memory, once: a step of programs with host experts is refused before it. At a
 *     pause the job table of the step's rows comes back; each job that names a host expert by its tag gets the slot
 *     that holds that expert, else the next of the card's in turn that no job of the pause took, and that expert goes
 *     up into it; the job then points there (the same bytes for the same kernels), and the table goes up again
 *     before the next segment. A job whose offsets are neither all below slots_offset nor one tag three times fails
 *     the step with STATUS_DEVICE_ERROR.
 *   - Every wait is bounded by the driver's limit. Failure permanently fails this driver.
 *     Destroy waits for pending work. If staging_retire is supported, pending blocks transfer to backend
 *     ownership until every possible reader retires by successful completion or actual successful drain/close.
 *     Otherwise the legacy quarantine remains allocated. A failed close/drain is not retirement; no device
 *     recovery or rollback is promised. Driver metadata and callbacks are never retained by the backend.
 *   - One thread uses a driver (a pipeline's workers are its own). The cards and the programs are the driver's until
 *     it is destroyed.
 */
#ifndef STEP_H
#define STEP_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/engram.h"
#include "anemone/model_seam.h"
#include "anemone/status.h"
#include "anemone/step_records.h"

#define STEP_MOST_PINNED_EXPERT_BYTES \
	(64ull << 30u) /* page-locked host experts: a 64 GiB card's worth */
#define STEP_PIPELINE_MOST 8u /* steps in flight at once */

struct step_driver;

/* programs: program_count of 1 or more; cards: their card_count entries, numbered as theirs. engram: NULL for
 * programs that read no n-gram rows. On failure driver_out is untouched and nothing is left allocated. */
MUST_CHECK result_status step_driver_create(
	const struct backend_operations *backend,
	struct device_card *const *cards,
	const struct execution_program *const *programs, uint32_t program_count,
	const struct engram_record *engram, uint64_t wait_limit_nanoseconds,
	struct step_driver **driver_out);
void step_driver_destroy(struct step_driver *driver); /* accepts NULL */

/* Optional TP startup schedule; ordinary programs/segment records and their printed bytes are unchanged.
 * Each pair is two adjacent segments of one layer/stage on distinct cards, enqueued before either's outgoing waits.
 * No pair overlaps another, crosses a front/draft boundary, pauses for experts or hands data to its own peer.
 * The descriptors name exactly the driver's programs in their existing order; each is the first's pair prefix.
 * arithmetic_identity is the same nonzero TP rounding/shard recipe stamp for every program. The attachment copies
 * an immutable map at load and changes snapshot identity only for this driver. No caller table is retained.
 * It must precede steps, graphs, pipeline, image alternatives and loaded session history; host experts are refused.
 * TP pipeline_start retains the queue/finish API through one coordinator: complete flights execute serially,
 * every group member queues before peer waits, and all cards retire before the next flight uses shared state.
 * Queue depth is host admission capacity; this mode promises no GPU overlap.
 * Missing/malformed/late attachment is INVALID_ARGUMENT. A failed driver answers its failure. No work is queued.
 */
#define STEP_NO_COMPANION UINT32_MAX
struct step_companion_pair {
	uint32_t first_segment;
	uint32_t second_segment;
};
struct step_companion_program {
	const struct execution_program *program;
	const struct step_companion_pair *pairs;
	uint32_t pair_count;
	uint32_t reserved_zero;
	uint64_t arithmetic_identity;
};
MUST_CHECK result_status step_companions_attach(
	struct step_driver *driver,
	const struct step_companion_program *programs, uint32_t program_count);

/* Wider optional groups, with the same immutable range table: first_segment and second_segment are the first
 * and last members. members is exactly2 or4, every range has that many consecutive distinct-card segments,
 * and no front/draft boundary or intra-group handoff cuts it. As with pairs, all members enqueue before any
 * wait; failure attempts the remaining members before group drain. Ordinary pair attachment still requires2.
 * This is startup-only and inherits companion attachment's lifetime, refusal and output contracts. */
MUST_CHECK result_status
step_groups_attach(struct step_driver *driver,
		   const struct step_companion_program *programs,
		   uint32_t program_count, uint32_t members);

/* After a groups attach: each card's submissions of a group from a thread of its own (a launcher a card), the step's
 * thread waiting for all of them before the group's hand-offs and waits; the same launches on each card in the same
 * order (only the host's order across cards changes: a graph's launch costs the host hundreds of microseconds). A
 * driver without groups, a second call or a thread that does not start is STATUS_INVALID_ARGUMENT (nothing
 * started). The launchers stop with the driver. */
MUST_CHECK result_status step_launchers_start(struct step_driver *driver);

/* After TP groups attach, before graphs, a pipeline or any step: cards with identical n-gram table masks
 * upload from the first such card's gathered staging bytes. Eager and queued TP flights gather cards in ascending
 * order; every flight still owns its immutable block until it retires. No arena, arithmetic or snapshot identity
 * changes. The ordinary path retains separate gathers. NULL, no groups/tables, a repeat or late call is
 * INVALID_ARGUMENT; a failed driver returns its failure. No work or allocation occurs. */
MUST_CHECK result_status step_engram_share(struct step_driver *driver);

/* Before the first step, a BATCH driver: verifiers[program] (program_count, the driver's) the drafting sessions a step
 * of that BATCH program may hold, as its draft walks were prepared (model_v41_tp2_program.h batch_verifiers); a step
 * that draws then takes the fewest-row BATCH program holding its rows and its drafting sessions. 0 for a program not
 * BATCH. A second attach, a driver not BATCH or a count above the sessions is STATUS_INVALID_ARGUMENT. */
MUST_CHECK result_status step_batch_verifiers_attach(struct step_driver *driver,
						     const uint32_t *verifiers,
						     uint32_t program_count);
/* The drafting sessions a step of the driver's BATCH program of rows rows may hold (of several such programs, the
 * fewest); 0: no such program. Several BATCH programs may share a row count once their verifiers are attached
 * (distinct counts): a step takes the one of fewest verifiers that holds its drafting sessions. */
uint32_t step_batch_verifiers(const struct step_driver *driver, uint32_t rows);
/* The driver's BATCH programs as rows_out[i] / verifiers_out[i], fewest rows first and of equal rows fewest
 * verifiers first (the order in which a drawing step looks for the first that holds it); their count, 0 when there
 * are more than most (or none). */
uint32_t step_batch_layout(const struct step_driver *driver, uint32_t *rows_out,
			   uint32_t *verifiers_out, uint32_t most);

/* host_arena: host_byte_count bytes (loader.h); NULL when the programs keep no expert in host memory. A failed driver
 * answers its failure. A NULL driver, a second attach, no arena for host experts and an expert outside it are
 * STATUS_INVALID_ARGUMENT, more than STEP_MOST_PINNED_EXPERT_BYTES of them
 * STATUS_OUT_OF_MEMORY, and staging memory refused its status; then nothing stays allocated. */
MUST_CHECK result_status step_host_experts_attach(struct step_driver *driver,
						  const void *host_arena,
						  uint64_t host_byte_count);

/* A NULL argument, a slot at or above the session capacity, a temperature that is negative or not finite and a
 * top_p outside (0, 1] are STATUS_INVALID_ARGUMENT. */
MUST_CHECK result_status step_session_set(struct step_driver *driver,
					  uint32_t session_slot,
					  const struct step_session *settings);

/* Optional, once before any step: capture coalesced runs in program/create order, then wait for their uploads.
 * A driver never prepared stays eager. BUSY stops captures only on that card: prepared runs use graphs, the
 * remaining runs stay eager. Success can therefore mean zero graphs. No capture executes model kernels.
 * Missing graph operations return UNSUPPORTED. NULL, late and repeated calls return INVALID_ARGUMENT.
 * Every accepted attempt consumes the one startup opportunity, including UNSUPPORTED and allocation failure.
 * Other capture refusals discard the unpublished table; successful completion proof leaves an eager driver.
 * DEVICE_ERROR or a failed bounded completion wait permanently fails it. Every graph, even one abandoned after
 * a failed preparation or driver destruction, stays the card's until card close; repeated drivers share that
 * fixed capacity. Programs remain immutable through driver destruction. Backend graph nodes copy their launch
 * arguments, and graph replay reads changing step data only through stable arena offsets. This API proves no
 * agreement with the original model and selects no serving preset. */
MUST_CHECK result_status step_graphs_prepare(struct step_driver *driver);

/* BATCH is initially eager only: an initial depth1 declaration succeeds without queued workers; other
 * depths, repeated and late declarations refuse. Submit/finish remain unavailable; run validates all owners
 * before committing host history. The engine uses one multi-owner flight, including for one session.
 * Optional, once before any step and after graphs. Companion groups use one coordinator and serial whole flights
 * (above); ordinary programs use the per-card workers described here. Up to depth
 * (2 to STEP_PIPELINE_MOST) of them at once: each takes its card's next ready task, a step's visit of the card
 * (model_seam.h), of the oldest step, so a card visited twice a step runs a later step's first visit while an earlier
 * step's second waits for its inputs (step_pipeline.c: the order, the leases and why it cannot deadlock). A hand-off
 * goes down by its source's worker behind the run that writes it and up by its destination's before the visit that
 * reads it, behind the source's mark, through the staging of the step's slot. Steps may end out of order; a finish
 * takes them in order. Programs with expert pauses are STATUS_UNSUPPORTED; NULL, late and repeated starts
 * and a depth outside the range INVALID_ARGUMENT; then the driver stays as it was. step_run_entry is then a submit
 * and its finish. Every wait of the pipeline is timed: nothing moving for the driver's limit fails it. */
MUST_CHECK result_status step_pipeline_start(struct step_driver *driver,
					     uint32_t depth);
/* Experimental five-card companion-prefix/native-tail pipeline, result card 4. Segments before the tail run
 * on cards 0..3; later segments run on card 4. Internal prefix hand-offs keep their ordered copies; tail hand-offs
 * use each flight's staging. Two workers own disjoint cards: every prefix retires its four cards and downloads
 * before reusing CMP scratch; every tail retires its uploads, kernels and results before the next tail.
 * Front steps retire uploads too.
 * Other topologies, mirrors, BATCH and expert pauses refuse. Same cold/depth contract as pipeline_start; that
 * existing entry keeps serial companion flights. No numerical or performance qualification is implied. */
MUST_CHECK result_status step_pipeline_overlap_start(struct step_driver *driver,
						     uint32_t depth);
uint32_t step_pipeline_depth(
	const struct step_driver *driver); /* 1 without a pipeline */
/* A step queued behind those in flight: step_run_entry's refusals, and BUSY with depth steps in flight. Its
 * tokens join the committed history when it is queued, including PROPOSED tokens until the caller truncates the
 * rejected suffix before its session's next submit: a failed step fails the driver. */
MUST_CHECK result_status step_submit(struct step_driver *driver,
				     const struct step_row_record *rows,
				     uint32_t row_count);
/* Waits for the oldest step in flight and takes its results as step_run_entry does; none in flight, no results
 * or no pipeline: INVALID_ARGUMENT. */
MUST_CHECK result_status step_finish(struct step_driver *driver,
				     struct step_row_result *results_out);

/* The last successfully finished step's round. Without DRAFTS it is zeroed with session_slot NO_SESSION.
 * A NULL argument or no finished step is INVALID_ARGUMENT; a failed driver answers its failure. The submitter's
 * thread calls this after run or finish, before finishing another step. Output is untouched on failure. */
MUST_CHECK result_status step_round_take(const struct step_driver *driver,
					 struct step_round *round_out);

/* Last successful round of this active session, unchanged on refusal. BATCH uses a session-indexed
 * table of unchanged88-byte records; legacy THIN returns its one matching owner. The legacy take
 * above refuses a BATCH result with multiple active owners rather than choosing one implicitly. */
MUST_CHECK result_status step_round_take_for_session(
	const struct step_driver *driver, uint32_t session_slot,
	struct step_round *round_out);
/* The admitted device draft block's positions; 0 without a device drafter or with a NULL driver. */
uint32_t step_draft_positions(const struct step_driver *driver);

/* Snapshots of a session (model_seam.h's session ranges), for the front end to keep and give back: the bytes of
 * the first token_count tokens, each identity once in identity order. A save writes them through write in pieces, a
 * load reads them through read into every range of each identity; the slot then holds token_ids as its committed
 * tokens (a failed load: none). A function that answers false fails the call with STATUS_INPUT_OUTPUT_ERROR.
 * A NULL argument, a slot beyond the capacity, a count beyond the context or above what a saved slot holds, and a
 * step in flight are STATUS_INVALID_ARGUMENT; a card's error fails the driver as a step's does. */
typedef bool (*step_bytes_function)(void *context, unsigned char *bytes,
				    uint64_t byte_count);
uint64_t step_session_bytes(const struct step_driver *driver,
			    uint32_t token_count);
MUST_CHECK result_status step_session_save(struct step_driver *driver,
					   uint32_t session_slot,
					   uint32_t token_count,
					   step_bytes_function write,
					   void *context);
MUST_CHECK result_status step_session_load(struct step_driver *driver,
					   uint32_t session_slot,
					   const uint32_t *token_ids,
					   uint32_t token_count,
					   step_bytes_function read,
					   void *context);
/* Quiescent same-card copies into independent destination ranges on every holding card. The caller owns the
 * source read lease and canonical-prefix proof. A checkpoint source additionally requires an actually captured
 * checkpoint at token_count; its window is copied separately, never manufactured from ordinary state.
 * Pending source mirrors return BUSY without changing either slot. All queued copies are drained on failure.
 * Success publishes destination committed history only after every card completes; no sampler/draft state is copied.
 * byte_count_out counts physical range copies, including mirrors and an optional checkpoint, only on success. */
MUST_CHECK result_status step_session_clone(struct step_driver *driver,
					    uint32_t source_slot,
					    uint32_t destination_slot,
					    uint32_t token_count,
					    bool checkpoint,
					    uint64_t *byte_count_out);
/* Discard a quiescent destination's committed/transient history. Its physical bytes are inaccessible to prefix
 * reuse; its next command must begin cold at position zero, whose existing write-before-read contract replaces
 * them. This is not rollback to an earlier prefix. Device failures remain poisoned and cannot cold-fallback. */
MUST_CHECK result_status step_session_discard(struct step_driver *driver,
					      uint32_t session_slot);
/* What a snapshot fits: the arithmetic of the limits and every identity's units, not the cards they lie on. */
uint64_t step_session_identity(const struct step_driver *driver);
/* Transient prompt-end windows (model_seam.h). No allocation, only same-card copies, with no step in flight.
 * Every submitted step, including other slots' visits, must be finished. Copies and their waits complete before
 * returning, while no worker owns a card; a later submit may then visit any card again.
 * Save copies the min(boundary, window_rows) rows before boundary; restore copies them back into their ring slots.
 * The caller owns the saved boundary and prefix identity, and guarantees each saved row was computed exactly.
 * A NULL driver, absent ranges, invalid slot, boundary beyond committed history/context or off the pooled granule,
 * and steps in flight are INVALID_ARGUMENT. Card failures poison the driver. Snapshots do not carry checkpoints. */
bool step_checkpoint_supported(const struct step_driver *driver);
MUST_CHECK result_status step_checkpoint_save(struct step_driver *driver,
					      uint32_t session_slot,
					      uint32_t boundary);
MUST_CHECK result_status step_checkpoint_restore(struct step_driver *driver,
						 uint32_t session_slot,
						 uint32_t boundary);
/* The rows at a prompt's end its programs run whole while front steps carry the others (model_seam.h); 0: none. */
uint32_t step_cone_rows(const struct step_driver *driver);
/* The rows of its largest wide program (model_seam.h): a front step of exactly that many runs it; 0: none. */
uint32_t step_wide_rows(const struct step_driver *driver);
/* The rows of its largest program of kind (wide, full or explicit batch) of at most most_rows; 0: none. A front step of exactly a wide
 * program's rows runs it; a step of exactly a full program's rows, none a front row and none with logits, runs that and
 * answers no token (a full program has no head, model_v41_program.c). */
uint32_t step_chunk_rows(const struct step_driver *driver,
			 enum program_kind kind, uint32_t most_rows);
/* The committed token at position of a session, for its submitter's thread (a cone run again, front_end_seam.h);
 * position below the session's committed count. */
uint32_t step_history_token(const struct step_driver *driver,
			    uint32_t session_slot, uint32_t position);
/* A speculative round's end (front_end_seam.h): the session's committed tokens cut back to its first token_count, for
 * its submitter's thread, so that the n-gram rows of its next step read none of the tokens its rejected rows carried
 * and a row that continues past the cut is refused. A failed driver answers its failure. A NULL driver, a slot beyond
 * the capacity, a count above the tokens the session holds and a session with a step in flight (a pipeline's) are
 * STATUS_INVALID_ARGUMENT, and nothing changes. The cards keep what the cut rows wrote: a row at a position
 * writes its state before any row reads it (step_records.h), so the next rows write over it. */
MUST_CHECK result_status step_history_truncate(struct step_driver *driver,
					       uint32_t session_slot,
					       uint32_t token_count);

MUST_CHECK result_status step_run_entry(struct step_driver *driver,
					const struct step_row_record *rows,
					uint32_t row_count,
					struct step_row_result *results_out);

#if CHECKING_HOOKS_ENABLED
/* Checking-only observation for a quiescent driver, including a failed one. Outputs remain untouched on refusal. */
MUST_CHECK result_status step_committed_history_read(
	const struct step_driver *driver, uint32_t session_slot,
	uint32_t *history_out, uint64_t history_capacity,
	uint32_t *token_count_out);
#endif

#endif
