/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_seam.h: what the engine needs to know about a model.
 *
 * Contract:
 *   - A model is described ONCE, at load, from its pack: its sizes, the state it needs under a placement, the plan
 *     of every card's arena and the program the cards run. Nothing here is called per token: the step driver
 *     (step.h) walks the program this seam made at load.
 *   - The loader finds the table of a pack's model with model_operations_of and the name in the pack's
 *     header. A model has ONE name, operations.name. Every function of a table answers
 *     STATUS_UNSUPPORTED for a pack that names another model.
 *   - The units placement may move (a unit is what must sit together) are the pack's: the converter, which knows
 *     the model's tensors, wrote them (pack.h). They are told apart in model-neutral terms, so the loader and the
 *     planner never name a tensor. A stage unit runs in sequence and sits whole on one card. An on-demand unit is
 *     read when the stage unit that owns it routes to it: it sits on that unit's card or in host
 *     memory. A host-only unit never goes to a card. Units outside the layers (embedding, head, draft head,
 *     vision) are stage units like the layers.
 *   - A model owns its state layout (what a session keeps per token and per layer, with its mirrors, rings and
 *     workspace) and says in bytes what each card needs under a given placement, so the memory plan is made, and
 *     can fail, before anything is loaded. The plan puts that state behind each card's weights (loader.h).
 *   - A program is what the cards run for steps of one kind and up to one row count: per card its launches in order,
 *     cut into segments, the hand-offs between cards and the pauses for routed experts in host memory. A launch reads
 *     its rows' positions and sessions from the step area (step_records.h) or a view of it, which the step driver
 *     writes with it, so one program runs every step of its kind. The draft step area and its one-row views are
 *     written on each card that drafts (the result card; a tensor-parallel draft's every rank) by its verification
 *     launch, before any launch reads them. A DRAFTS step runs every segment, its draft tail a segment of each card
 *     that drafts; another stops before draft_segment_first (a front step still runs only its front).
 *   - A visit is a card's segments between segments of other cards: a layout may visit a card more than once a step,
 *     and the step driver may run another step's visit of that card between two of one step's (step.h). So each
 *     visit stands on its own: what it reads of the step (the step area, its views, the n-gram rows, the settings) the
 *     driver puts up again at its start, what its layers read of the rotation step its own step segment forms, and the
 *     rows it reads of another card come in by hand-offs before it. State flows forward: what a hand-off with an entry
 *     ratio carries is read by segments behind its after_segment alone, in this step or a later one. A card's rows
 *     that one visit writes or takes in and a later visit of the same card reads (a group's selection, the candidates)
 *     a lease names, which the driver keeps from being overwritten in between.
 *   - The first and today only model is DeepSeek-V4.1-Flash. Another model is another table and, where it needs
 *     arithmetic the present kernels lack, new kernels in every backend; the engine's other host parts do not
 *     name a model.
 */
#ifndef MODEL_SEAM_H
#define MODEL_SEAM_H

#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/status.h"
#include "anemone/step_records.h"

#define MODEL_NAME_CAPACITY \
	64 /* a name with its NUL: the name field of a pack's header */
#define PLACEMENT_IN_HOST_MEMORY UINT32_MAX /* in place of a card number */
#define PROGRAM_NOWHERE \
	UINT64_MAX /* in place of an offset: the card holds no such area */
#define SEGMENT_NO_LAYER UINT32_MAX /* the layer of a segment that runs none */
#define SEGMENT_PAUSE_PAIR_OUTPUTS \
	1u /* a segment ends before a row sum of pairs computed elsewhere */
#define SEGMENT_PAUSE_EXPERT_UPLOADS \
	2u /* a segment ends behind a router: an expert pause follows */
#define EXPERT_PAUSE_MATRICES \
	3u /* gate, up, down: the offsets of a job (linear_records.h) */
#define EXPERT_PAUSE_BYTES 80
#define HAND_OFF_BYTES 48
#define SEGMENT_BYTES 24
#define STEP_VIEW_BYTES 16
#define SESSION_RANGE_BYTES 40
#define WINDOW_CHECKPOINT_BYTES 40
#define VISIT_LEASE_BYTES 12

/* First users: the loader, which logs it, and the engine, which reports the last three to a front end. */
struct model_description_record {
	uint32_t layer_count;
	uint32_t hidden_size;
	uint32_t vocabulary_size;
	uint32_t context_token_limit;
	uint32_t end_token_id; /* generating it ends a reply */
};

/* A pack stores these values: a value is never reused or renumbered; new kinds are appended. */
enum placement_unit_kind {
	PLACEMENT_UNIT_STAGE = 0, /* runs in sequence: whole on one card */
	PLACEMENT_UNIT_ON_DEMAND =
		1, /* read when routed: on its owner's card or in host memory */
	PLACEMENT_UNIT_HOST_ONLY = 2, /* read by the host: never on a card */
	PLACEMENT_UNIT_KIND_COUNT = 3 /* number of kinds; not a kind */
};

/* Where every unit sits. The caller numbers the cards. */
struct placement_record {
	const uint32_t *
		card_of_unit; /* per unit: a card, or PLACEMENT_IN_HOST_MEMORY */
	uint32_t unit_count; /* the pack's */
	uint32_t card_count; /* every card of card_of_unit is below it */
};

/* What the state is sized for and how products add, chosen at load. A row capacity is 1 to
 * MODEL_WIDE_MOST_ROWS (model_layout.h). A valid value that no kernel of this build takes yet (8-bit state; a
 * context past MODEL_CONTEXT_WITHOUT_CANDIDATES) is UNSUPPORTED. A thin step carries at most
 * LINEAR_MAXIMUM_ROWS rows whatever the capacity: a capacity above it is a wide step's chunk. A model with a
 * draft head (the units after its head) plans it only with draft_positions set: its tensors and state then join the
 * plan, on a placement that can run them (else UNSUPPORTED); with 0 the plan is that of a model without one. */
enum state_width_code { STATE_WIDTH_8 = 8, STATE_WIDTH_16 = 16 };
struct model_limits {
	uint32_t context_token_limit; /* tokens one session may hold: at most the model's */
	uint32_t row_capacity; /* the most rows one step carries */
	uint32_t session_capacity; /* session slots */
	enum state_width_code state_width; /* bits of a stored attention value */
	enum linear_profile_code linear_profile;
	enum linear_order_code linear_order;
	enum cmp_order_code
		cmp_order; /* class 80 trunk and routed experts; reference by default */
	uint32_t draft_positions; /* 0: no draft; else 5..7 positions of the draft block */
	uint32_t official_hc_rows; /* 0: existing HC; 1/3/4/8: pinned reference HC shape;
                                                      * experimental original-weight thin PP only */
	uint32_t official_reference; /* 0: retained programs; 1: original serial M1 reference arithmetic. */
};

/* After its segment, byte_count bytes of each row of the step (with a view, of the from card's view's rows) from one
 * card's arena to another's, row r at offset + r byte_count; with an entry_ratio q, of the rows at a position p with
 * (p + 1) % q = 0 only, entry p / q of the row's session at offset + session slot * session_stride + (p / q)
 * byte_count (a pool entry or key a card mirrors). */
struct hand_off_record {
	uint64_t source_offset;
	uint64_t destination_offset;
	uint64_t session_stride; /* 0 where entry_ratio is 0 */
	uint32_t byte_count;
	uint32_t entry_ratio;
	uint32_t from_card;
	uint32_t to_card;
	uint32_t after_segment;
	uint32_t view; /* 0: the step's rows */
};

/* What a program runs (wide_prefill.md 2, 3):
 *   THIN  steps of 1 to row_count rows, at most LINEAR_MAXIMUM_ROWS: every unit over every row (a token of
 *         each session, a short prompt's rows).
 *   WIDE  a chunk: steps of MODEL_WIDE_LEAST_ROWS to row_count rows of one session, consecutive positions:
 *         the units every row of a prompt needs, which the model names (DeepSeek-V4.1-Flash: its layers below 20 and
 *         layer 20's front, model_layout.h); its slabs read views.
 *   CONE  steps of exactly row_count rows, a prompt's last, behind its chunks: the other units, each over the rows a
 *         later unit or the head reads (model_layout.h's cone), through views; then the head on the flagged rows.
 *   FULL  steps as WIDE's: every unit over every row, slabs through views, then the head on the flagged rows. It
 *         serves a prompt or continuation of fewer rows than the cone program's (views lie at fixed rows, so no cone
 *         is cut shorter) and every prompt with the cone off (wide_prefill.md 8, B4).
 * In profile 1 a prompt's chunks and cone, or its full steps, leave the bytes its thin steps leave in all state a later
 * step reads (pools, key tiles, the window rows of the last 128 positions) and in the flagged rows' logits (B3).
 * A thin program may also carry front steps: steps whose rows all have STEP_ROW_FRONT (step_records.h). Such a
 * step runs the segments before front_segment_count and the hand-offs behind them, and no head: the units every row of
 * a prompt needs (WIDE's), whose state every later row reads. A prompt's last cone_rows rows need every unit; a row
 * before them may be a front row, since nothing the other units compute for it is read again (DeepSeek-V4.1-Flash: the
 * decoder's layers, model_layout.h's cone). The front ends on a run of the step driver: no run crosses it. */
/*   BATCH (tp4_v3_claude, staged with tp4_v3's host half; TP plans that ask for it): steps of 9 to row_count rows
 *         (at most 64) of several sessions: every unit over every row through slab views, then on the cards that
 *         sample, slab by slab, the head and the sampler over the slab's flagged rows; the result area is one per
 *         slab: slab s's at the card's result_offset plus s (STEP_RESULT_BYTES + 8 STEP_ROW_RESULT_BYTES),
 *         each the slab view's sequence and row count. */
enum program_kind {
	PROGRAM_THIN = 0,
	PROGRAM_WIDE = 1,
	PROGRAM_CONE = 2,
	PROGRAM_FULL = 3,
	PROGRAM_BATCH = 4,
	PROGRAM_KIND_COUNT = 5 /* number of kinds; not a kind */
};
/* Shared with the step driver's BATCH (step_records.h; tp4_v3's host half): 64 rows, 80-byte slab result areas. */
#define PROGRAM_BATCH_MOST_ROWS STEP_BATCH_MAXIMUM_ROWS
#define PROGRAM_BATCH_SLAB_RESULT_BYTES STEP_BATCH_SLAB_RESULT_BYTES

/* A view of a step: a step block and rows of its own (step_records.h) at block_offset of a card's arena, which the
 * step driver writes with the step area: the step's rows from first_row on, at most row_limit (1 or more) of them,
 * their count (0 where the step has none from first_row on) and the step's sequence. Row r of a view is row
 * first_row + r of the step, so a launch that reads it finds its per-row arrays first_row rows into the step's: a
 * slab of a wide step, a layer of the cone. View 0 is the step area itself, views 1 .. view_count a card's views[]. */
struct step_view {
	uint64_t block_offset;
	uint32_t first_row;
	uint32_t row_limit;
};

/* Launches of one card run as one launch_run; the segments of a stage may run at once, each on its card. A layer is
 * a segment of its own, or two where an expert pause divides it. A visit of a card that holds a layer begins with its
 * step segment (layer SEGMENT_NO_LAYER), what the visit's layers read of a step (a rotation step), run at each
 * step before any other segment of that visit. */
struct segment_record {
	uint32_t card;
	uint32_t stage;
	uint32_t layer; /* or SEGMENT_NO_LAYER */
	uint32_t first_launch;
	uint32_t launch_count;
	uint32_t pause; /* 0, or SEGMENT_PAUSE_* */
};

/* Behind segment after_segment of card, which ends with SEGMENT_PAUSE_EXPERT_UPLOADS, the router of a layer
 * that keeps host_expert_count of its routed experts in host memory has written its job table (linear_records.h) at
 * jobs_offset, jobs_per_row jobs a row. A job names the layer's host expert i by three offsets equal to slots_offset
 * + i LINEAR_OFFSET_ALIGNMENT_BYTES (a tag); its bytes are expert_byte_count from host_expert_offsets[
 * first_host_expert + i] of the host arena (loader.h), its gate, up and down matrices at matrix_offsets in them.
 * Before the next segment the step driver puts each expert a job names so into one of the card's slot_count slots
 * (slot k at slots_offset + k slot_byte_count: the same slots in every pause of the card), uploading it unless the
 * slot holds it already, and points the job's offsets there. Any other job keeps its offsets, all below
 * slots_offset. */
struct expert_pause {
	uint64_t jobs_offset;
	uint64_t slots_offset;
	uint64_t slot_byte_count;
	uint64_t expert_byte_count; /* at most slot_byte_count */
	uint64_t matrix_offsets[EXPERT_PAUSE_MATRICES];
	uint32_t first_host_expert;
	uint32_t host_expert_count;
	uint32_t jobs_per_row;
	uint32_t slot_count; /* at least jobs_per_row times the row count of the program */
	uint32_t card;
	uint32_t after_segment;
};

/* What one session keeps on a card between its steps, which a snapshot of its first n tokens takes (step.h): from
 * offset + slot * session_stride, min(unit_count, ceil(n / unit_tokens)) units of unit_bytes, or all unit_count with
 * unit_tokens 0. identity names it across the plans of one model: a snapshot's range goes back into every range of
 * that identity (a card that mirrors an owner's entries takes the same bytes). */
struct session_range {
	uint64_t offset;
	uint64_t session_stride;
	uint64_t unit_count;
	uint32_t unit_bytes;
	uint32_t unit_tokens;
	uint32_t card;
	uint32_t identity;
};

/* A prompt-end checkpoint's window rows on one card. The ring has ring_rows rows of row_bytes per session,
 * each session at ring_offset + slot * ring_session_stride;
 * checkpoint_offset holds window_rows preceding a kept boundary, packed in position order, per session. Neither
 * range overlaps the other; both are inside the card's arena. These transient bytes are absent from snapshots.
 * A boundary is a multiple of MODEL_POOLED_RATIO: every compressor group is closed, and entries and keys
 * below it remain immutable, so only window rows need copying. */
struct window_checkpoint {
	uint64_t ring_offset;
	uint64_t checkpoint_offset;
	uint64_t ring_session_stride;
	uint32_t ring_rows;
	uint32_t row_bytes;
	uint32_t window_rows;
	uint32_t card;
};

/* What one card runs, and where it keeps what the step driver and the teacher-forced harness move. The n-gram rows
 * of table t for row r lie at engram_rows_offset + (k row capacity + r) rows per token ENGRAM_ROW_BYTES, k
 * the number of the card's tables below t. */
struct program_card_record {
	const struct kernel_launch
		*launches; /* launch_count, in the order they run */
	uint64_t arena_byte_count; /* the card's arena, as arena_byte_count gives it */
	uint64_t step_offset; /* the step area (step_records.h) */
	uint64_t session_offset; /* the session settings, or PROGRAM_NOWHERE */
	uint64_t result_offset; /* the result area, or PROGRAM_NOWHERE */
	uint64_t logits_offset; /* the head's logits (model_layout.h), or NOWHERE */
	uint64_t streams_offset; /* streams its layers update in place (model_layout.h) */
	uint64_t carried_pre_offset; /* the pre a site carries on; both NOWHERE without layers */
	uint64_t engram_rows_offset; /* or PROGRAM_NOWHERE */
	uint32_t engram_tables; /* bit t: the card reads table t's rows (engram.h) */
	uint32_t launch_count;
	const struct step_view
		*views; /* view_count, in their order; none in a thin program */
	uint32_t view_count;
	uint64_t round_offset; /* the draft round record, or PROGRAM_NOWHERE */
};

/* Rows of card that its visit holding first_segment writes or takes in by a hand-off and its later visit holding
 * last_segment reads last: no later step's visit holding first_segment may run before this step's visit holding
 * last_segment has run (step.h). */
struct visit_lease {
	uint32_t card;
	uint32_t first_segment;
	uint32_t last_segment; /* of a later visit of card */
};

struct execution_program {
	const struct program_card_record
		*cards; /* card_count, numbered as the layout's */
	const struct segment_record
		*segments; /* segment_count, in the order they run */
	const struct hand_off_record *
		hand_offs; /* hand_off_count, in the order of their after_segment */
	struct model_limits limits; /* the plan's */
	uint32_t card_count;
	uint32_t segment_count;
	uint32_t hand_off_count;
	uint32_t row_count; /* the most rows a step of this program carries */
	uint32_t result_card; /* the card that samples */
	const struct expert_pause *
		expert_pauses; /* expert_pause_count, in the order of their after_segment */
	const uint64_t *host_expert_offsets; /* host_expert_count */
	uint32_t expert_pause_count;
	uint32_t host_expert_count;
	enum program_kind kind;
	uint32_t lease_count;
	const struct session_range *
		session_ranges; /* session_range_count, by identity ascending */
	uint32_t session_range_count;
	uint32_t front_segment_count; /* a front step's segments come first; 0: no front steps */
	uint32_t cone_rows; /* a prompt's last rows, which no front step may carry */
	uint32_t checkpoint_count;
	const struct window_checkpoint *
		checkpoints; /* checkpoint_count ranges, identical in every program */
	const struct visit_lease *
		leases; /* lease_count; a plan's programs name the same as far as
                                                         their segments reach: the step driver takes the first's */
	uint32_t draft_segment_first; /* DRAFTS-only tail, or segment_count without one */
};

_Static_assert(sizeof(struct hand_off_record) == HAND_OFF_BYTES,
	       "a hand-off has no padding");
_Static_assert(sizeof(struct segment_record) == SEGMENT_BYTES,
	       "a segment has no padding");
_Static_assert(sizeof(struct expert_pause) == EXPERT_PAUSE_BYTES,
	       "an expert pause has no padding");
_Static_assert(sizeof(struct step_view) == STEP_VIEW_BYTES,
	       "a view has no padding");
_Static_assert(sizeof(struct session_range) == SESSION_RANGE_BYTES,
	       "a session range has no padding");
_Static_assert(sizeof(struct window_checkpoint) == WINDOW_CHECKPOINT_BYTES,
	       "a window checkpoint has no padding");
_Static_assert(sizeof(struct visit_lease) == VISIT_LEASE_BYTES,
	       "a lease has no padding");

struct pack_record;
struct load_layout;
struct model_plan_record;

/* Every function refuses a NULL argument, a card index at or above the card count of its placement or plan (but
 * PLACEMENT_IN_HOST_MEMORY, which state_byte_count takes) and a limit out of range
 * (STATUS_INVALID_ARGUMENT, one error line), and leaves its outputs untouched on failure. */
struct model_operations {
	const char *name;
	MUST_CHECK result_status (*describe)(
		const struct pack_record *pack,
		struct model_description_record *description_out);
	/* First user: the planner, before anything is loaded. The bytes of state card_index needs under the placement,
     * or with PLACEMENT_IN_HOST_MEMORY the page-locked host memory the state needs. A placement that
     * pack_placement_is_valid refuses is STATUS_INVALID_ARGUMENT. */
	MUST_CHECK result_status (*state_byte_count)(
		const struct pack_record *pack,
		const struct placement_record *placement,
		const struct model_limits *limits, uint32_t card_index,
		uint64_t *byte_count_out);
	/* The offsets of every array behind each card's weights, for the layout of that placement. */
	MUST_CHECK result_status (*plan_create)(
		const struct pack_record *pack,
		const struct load_layout *layout,
		const struct model_limits *limits,
		struct model_plan_record **plan_out);
	void (*plan_destroy)(struct model_plan_record *plan); /* accepts NULL */
	/* The arena card_open is given: the card's weights and its state, at most the layout's card_byte_count plus
     * state_byte_count under the placement and limits the plan was made for. */
	MUST_CHECK result_status (*arena_byte_count)(
		const struct model_plan_record *plan, uint32_t card_index,
		uint64_t *byte_count_out);
	/* Once the weights are loaded: the frequency words, the expert offset tables, the copies of weights kept in
     * state, every other state byte zero.
     * In relaxed CMP order also checks trunk and resident routed-expert scales, pinning fold or reference fallback
     * before any program is prepared. Programs must be discarded before this operation is repeated. */
	MUST_CHECK result_status (*arena_initialise)(
		struct model_plan_record *plan,
		const struct backend_operations *backend,
		struct device_card *const *cards,
		uint64_t wait_limit_nanoseconds);
	/* Prepares the kernels on every card and the program of a kind for its steps of up to row_count rows (at most
     * the plan's row capacity, and what the kind allows), its slice counts from the plan's profile and order and each
     * card's arithmetic class. Programs of one plan and kind differ only in row_count, launches and the launch ranges
     * of their segments; of other kinds, in segments, views and hand-offs too. A kind outside the enumeration is
     * STATUS_INVALID_ARGUMENT, one this build makes no program of yet STATUS_UNSUPPORTED. */
	MUST_CHECK result_status (*program_prepare)(
		const struct model_plan_record *plan,
		const struct backend_operations *backend,
		struct device_card *const *cards,
		const struct card_description_record *descriptions,
		enum program_kind kind, uint32_t row_count,
		struct execution_program **program_out);
	void (*program_destroy)(
		struct execution_program *program); /* accepts NULL */
};

/* The table of the model a pack's header names, or NULL for a model this build does not carry. It reads at most
 * MODEL_NAME_CAPACITY bytes of model_name: a NULL name, or one with no NUL among them, finds no table. */
const struct model_operations *model_operations_of(const char *model_name);

#endif
