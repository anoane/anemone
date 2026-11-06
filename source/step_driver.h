/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Private driver ownership/completion state shared by eager and graph startup. */
#ifndef STEP_DRIVER_H
#define STEP_DRIVER_H
#include <stdbool.h>
#include <stdatomic.h>
#include <time.h>
#include "anemone/step.h"
#include "anemone/engine_stats.h"
#include "step_rounds.h"

#define STEP_PAUSE_REGIONS \
	2u /* per expert pause: its jobs in the block, its experts' copies */
#define STEP_SLOT_FIELDS \
	2u /* per slot: the expert it holds plus 1 (0: none), its pause */
#define STEP_SYNC_FIELDS \
	3u /* per mirror sync: its session slot, its first position, the position behind its last */
/* the session block: a snapshot's and a mirror sync's pieces */
#define STEP_SESSION_STAGING_BYTES (16ull << 20u)
#define STEP_NANOSECONDS_PER_SECOND 1000000000ull
#define STEP_WAIT_SEQUENCE_STRIDE 2u
#define STEP_BUSY_SNAPSHOT_ATTEMPTS 8u
#define STEP_ATOMIC_ALWAYS_LOCK_FREE 2u

_Static_assert(ATOMIC_LLONG_LOCK_FREE == STEP_ATOMIC_ALWAYS_LOCK_FREE,
	       "live card counters require always lock-free 64-bit atomics");

struct step_pipeline; /* step_pipeline.c */
struct step_launchers; /* step_companions.c */
struct step_images;

struct step_driver {
	const struct backend_operations *backend;
	struct step_pipeline
		*pipeline; /* NULL: a step at a time on the caller's thread */

	const struct execution_program **
		programs; /* the first's card areas and hand-offs are every one's */
	const struct execution_program **
		image_programs; /* parallel table; only steps with IMAGE rows select it */
	const struct engram_record *engram;
	struct step_images *images;
	unsigned char *staging;
	uint32_t *run_graphs; /* [program][segment]: graph handle + 1, or eager */
	uint32_t *companion_segments; /* NULL: ordinary PP; first program's immutable segment-to-peer map */
	uint64_t companion_identity; /* TP arithmetic recipe; absent from ordinary PP snapshot identity */
	struct step_launchers *launchers; /* NULL: a group's members submitted in turn on the step's thread */
	uint32_t *histories; /* [session][context limit]; then the per-session counters and sync triples below */
	uint32_t *token_counts; /* [session] */
	uint32_t *next_positions; /* [session]: what a step's check expects */
	/* [session]: the positions front steps wrote that the mirrors lack (step.h), from .. to - 1 */
	uint32_t *mirror_from, *mirror_to;
	uint32_t *mirror_syncs; /* [session][STEP_SYNC_FIELDS]: a step's mirror syncs, without a pipeline */
	uint8_t *deferred; /* [hand-off of programs[0]]: 1 where front steps leave it to the mirror sync */
	uint64_t *hand_off_staging; /* [hand-off] where its bytes lie in the block; then the two */
	uint64_t *pause_staging; /* [pause]: its jobs in the block; then where its experts' */
	uint64_t *pause_copies; /* copies begin in expert_copies; then row_ids */
	uint64_t *row_ids; /* [row][table][row of a token] */
	uint64_t *session_sequences; /* [session]: the last step that carried a row of it */
	unsigned char *expert_copies; /* page-locked; NULL until attached */
	unsigned char *
		session_block; /* page-locked: snapshots' pieces; NULL until the first */
	uint64_t *slots; /* [slot of a card][STEP_SLOT_FIELDS] */
	uint64_t pause_mark; /* the pauses so far: the mark of the one that runs */
	uint32_t first_slot[BACKEND_MAXIMUM_CARDS],
		next_slot[BACKEND_MAXIMUM_CARDS];
	struct device_card *cards[BACKEND_MAXIMUM_CARDS];
	uint64_t last_ticket[BACKEND_MAXIMUM_CARDS];
	uint64_t engram_staging[BACKEND_MAXIMUM_CARDS];
	bool uncovered_work
		[BACKEND_MAXIMUM_CARDS]; /* no later copy ticket proves it complete */
	uint32_t unconfirmed
		[BACKEND_MAXIMUM_CARDS]; /* copies no wait has seen finish */
	uint64_t session_staging, result_staging, view_staging, staging_bytes,
		sequence, wait_limit_nanoseconds;
	uint32_t program_count, largest_row_count, table_count, rows_per_token,
		vocabulary_size; /* 0: no n-grams */
	uint32_t *batch_verifiers; /* [program] a BATCH program's drafting sessions a step (step_batch_verifiers_
				      attach); NULL: its rows / (draft positions + 1) */
	/* the card the deferred hand-offs leave (BACKEND_MAXIMUM_CARDS: none), a bit each card they go to, the
	 * first segment they follow */
	uint32_t mirror_source, mirror_cards, mirror_segment;
	result_status failure;
	struct step_round
		last_round; /* last successfully finished step; NO_SESSION without DRAFTS */
	struct step_round_table
		last_rounds; /* fixed session-indexed BATCH result publication */
	uint32_t round_end_token_id;
	bool round_domain_attached;
	bool batch_eager_started; /* cold depth-one declaration; never starts queued workers */
	bool sessions_changed, graphs_attempted, has_finished;
	bool engram_shared; /* TP only: identical table masks upload the first card's gathered bytes */
	/* A card's worker alone writes these; pipeline publication happens under its existing lock. */
	uint64_t live_busy_ns[BACKEND_MAXIMUM_CARDS];
	uint64_t live_steps[BACKEND_MAXIMUM_CARDS],
		live_visits[BACKEND_MAXIMUM_CARDS];
	/* Even sequence publishes a coherent (completed time, active wait start) pair to refresh readers.
	 * A card has one writer under the existing worker/borrow ownership; no lock is added to its wait. */
	atomic_ullong live_wait_sequence[BACKEND_MAXIMUM_CARDS];
	atomic_ullong live_wait_completed_ns[BACKEND_MAXIMUM_CARDS];
	atomic_ullong live_wait_started_ns[BACKEND_MAXIMUM_CARDS];
};

/* step_launchers_start's threads stopped and joined (step_companions.c); nothing without them. */
void step_launchers_stop(struct step_driver *driver);

static inline uint64_t step_clock_ns(void)
{
	struct timespec now;
	return clock_gettime(CLOCK_MONOTONIC, &now) == 0 && now.tv_sec >= 0 ?
		       (uint64_t)now.tv_sec * STEP_NANOSECONDS_PER_SECOND +
			       (uint64_t)now.tv_nsec :
		       0;
}

/* Waits for copies and work not proven complete by a later copy ticket. */
MUST_CHECK static inline result_status card_wait(struct step_driver *driver,
						 uint32_t card)
{
	const bool has_work = driver->unconfirmed[card] != 0 ||
			      driver->uncovered_work[card];
	if (has_work) {
		(void)atomic_fetch_add_explicit(
			&driver->live_wait_sequence[card], 1,
			memory_order_seq_cst);
	}
	const uint64_t began = has_work ? step_clock_ns() : 0;
	if (has_work) {
		atomic_store_explicit(&driver->live_wait_started_ns[card],
				      began, memory_order_seq_cst);
		(void)atomic_fetch_add_explicit(
			&driver->live_wait_sequence[card], 1,
			memory_order_seq_cst);
	}
	result_status status = driver->unconfirmed[card] == 0 ?
				       STATUS_OK :
				       driver->backend->upload_wait(
					       driver->cards[card],
					       driver->last_ticket[card],
					       driver->wait_limit_nanoseconds);
	driver->unconfirmed[card] =
		status == STATUS_OK ? 0 : driver->unconfirmed[card];
	status = status != STATUS_OK || !driver->uncovered_work[card] ?
			 status :
			 driver->backend->launch_wait(
				 driver->cards[card],
				 driver->wait_limit_nanoseconds);
	driver->uncovered_work[card] =
		status == STATUS_OK ? false : driver->uncovered_work[card];
	if (has_work) {
		(void)atomic_fetch_add_explicit(
			&driver->live_wait_sequence[card], 1,
			memory_order_seq_cst);
		const uint64_t ended = step_clock_ns();
		driver->live_busy_ns[card] +=
			began != 0 && ended >= began ? ended - began : 0;
		atomic_store_explicit(&driver->live_wait_completed_ns[card],
				      driver->live_busy_ns[card],
				      memory_order_seq_cst);
		atomic_store_explicit(&driver->live_wait_started_ns[card], 0,
				      memory_order_seq_cst);
		(void)atomic_fetch_add_explicit(
			&driver->live_wait_sequence[card], 1,
			memory_order_seq_cst);
	}
	return status;
}

/* Refresh only: copies steps/visits under the existing pipeline lock, then samples cumulative busy time including
 * an active wait. Eight nonblocking sequence attempts bound a refresh; contention clears busy_available and leaves
 * its known completed lower bound intact. busy_observed_ns belongs to this sample, separately from engine time. */
void step_live_cards(struct step_driver *driver,
		     struct engine_card_stats *cards, uint32_t card_count);

/* step.c's parts of a step, which step_pipeline.c shares (block: a staging block of the driver's layout). */
MUST_CHECK result_status step_copy_begin(struct step_driver *driver,
					 uint32_t card, uint64_t arena,
					 unsigned char *staging,
					 uint64_t byte_count, bool is_download);
MUST_CHECK result_status
step_hand_off_copies(struct step_driver *driver, unsigned char *block,
		     uint32_t index, const struct step_row_record *rows,
		     uint32_t row_count, bool is_download);
/* n-gram row ids of the rows from committed history and the step's earlier rows: table_count * rows_per_token a row */
MUST_CHECK result_status step_row_ids(const struct step_driver *driver,
				      const struct step_row_record *rows,
				      uint32_t row_count,
				      uint64_t *row_ids_out);
/* A card's n-gram rows gathered from row_ids into its staging in block (the TP coordinator's steps: at submit, behind
 * the step in flight). */
MUST_CHECK result_status step_card_gather(struct step_driver *driver,
					  unsigned char *block,
					  const uint64_t *row_ids,
					  uint32_t row_count, uint32_t card);
/* A card's step area and n-gram rows go up from block, the rows first gathered there from row_ids where it gathers
 * (once a step: a card visited again puts the same bytes up again). */
MUST_CHECK result_status step_card_begin(struct step_driver *driver,
					 unsigned char *block,
					 const uint64_t *row_ids,
					 uint32_t row_count, uint32_t card,
					 bool gathers);
void step_block_write(unsigned char *block, uint64_t sequence,
		      const struct step_row_record *rows, uint32_t row_count);
/* A wide program's views of the step (model_seam.h) into block at the driver's view staging: the same views on every
 * card that has any, so written once a step. */
void step_views_write(const struct step_driver *driver,
		      const struct execution_program *program,
		      unsigned char *block, uint64_t sequence,
		      const struct step_row_record *rows, uint32_t row_count);
/* Those views go up from block to card, one copy: a card's views lie one behind the other. None: nothing. */
MUST_CHECK result_status step_views_begin(
	struct step_driver *driver, const struct execution_program *program,
	unsigned char *block, uint32_t card);
/* The result area in block, taken if it answers step sequence and its rows (step.h). */
MUST_CHECK result_status step_result_take(struct step_driver *driver,
					  const unsigned char *block,
					  uint64_t sequence,
					  const struct step_row_record *rows,
					  uint32_t row_count,
					  struct step_row_result *results_out);
/* step_experts.c's: expert pause index of the step's row_count rows (model_seam.h). */
MUST_CHECK result_status step_experts_pause(struct step_driver *driver,
					    uint32_t index, uint32_t row_count);
/* A front step's results (model_seam.h): no token and +0 for each row. */
void step_front_results(uint32_t row_count,
			struct step_row_result *results_out);
/* step.h's refusal of a step's arguments, logged; OK when it is accepted. */
MUST_CHECK result_status step_rows_check(struct step_driver *driver,
					 const struct step_row_record *rows,
					 uint32_t row_count, bool has_results);
/* The thin program of the fewest rows that holds row_count, or for a front step a wide program of exactly row_count
 * rows, and its graphs (NULL: eager); NULL if none. */
const struct execution_program *
step_program_of(const struct step_driver *driver, uint32_t row_count,
		bool is_front, const uint32_t **run_graphs_out);
/* Resolve the base row shape, then its image alternative if any row is typed IMAGE. Image alternatives are eager;
 * the returned program and graphs are captured in each pipeline slot, never changed globally for a step. */
const struct execution_program *
step_program_for_rows(const struct step_driver *driver,
		      const struct step_row_record *rows, uint32_t row_count,
		      const uint32_t **run_graphs_out);
/* Optional startup companion map: enqueue each complete pair before processing either outgoing boundary. */
MUST_CHECK result_status step_companions_run(
	struct step_driver *driver, const struct execution_program *program,
	const uint32_t *run_graphs, const struct step_row_record *rows,
	uint32_t row_count, struct step_row_result *results_out);
/* Same group schedule into a caller-owned flight block, leaving result parsing to its finish. Caller owns all
 * cards exclusively, has uploaded this flight's step/views/settings, and drains every card before reusing state. */
MUST_CHECK result_status step_companions_enqueue(
	struct step_driver *driver, const struct execution_program *program,
	const uint32_t *run_graphs, const struct step_row_record *rows,
	uint32_t row_count, unsigned char *block);
/* Split pipeline only. Caller owns its side's cards. Prefix downloads remain in block until tail retirement;
 * tail uploads them before its first launch. split_segment is a complete companion-group boundary. */
MUST_CHECK result_status step_companions_prefix_enqueue(
	struct step_driver *driver, const struct execution_program *program,
	const uint32_t *run_graphs, const struct step_row_record *rows,
	uint32_t row_count, unsigned char *block, uint32_t split_segment);
MUST_CHECK result_status step_companions_tail_enqueue(
	struct step_driver *driver, const struct execution_program *program,
	const uint32_t *run_graphs, const struct step_row_record *rows,
	uint32_t row_count, unsigned char *block, uint32_t split_segment);
/* The rows' tokens join the host history and counts. */
void step_history_commit(struct step_driver *driver,
			 const struct step_row_record *rows,
			 uint32_t row_count);
/* step_pipeline.c: its workers end and give the cards back to the driver's thread; its memory is freed with the
 * driver's staging (staging_is_read: some copy may still read it). Accepts a driver without a pipeline. */
void step_pipeline_stop(struct step_driver *driver);
void step_pipeline_free(struct step_driver *driver, bool staging_is_read);
/* No step in flight: true without a pipeline. */
bool step_pipeline_is_idle(const struct step_driver *driver);
/* Step sequence still in flight: false without a pipeline. */
bool step_pipeline_holds(const struct step_driver *driver, uint64_t sequence);
/* step.h's mirror ranges: a front step's rows extend their sessions' ranges; a step of other rows takes the range of
 * each of its sessions, behind which its own rows lie, as (slot, from, to) triples into syncs_out (at most a triple a
 * session slot), and empties it. The triples taken. */
uint32_t step_mirror_ranges(struct step_driver *driver,
			    const struct step_row_record *rows,
			    uint32_t row_count, uint32_t *syncs_out);
/* step_sessions.c: the deferred hand-offs' entries of the positions from .. to - 1 of a session slot, from the card
 * they leave to each card they go to, in pieces through the session block. The calling thread owns those cards. */
MUST_CHECK result_status step_mirror_sync(struct step_driver *driver,
					  uint32_t session_slot, uint32_t from,
					  uint32_t to);

#endif
