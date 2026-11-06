/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * step_records.h: what the host tells the cards about a step, and what the card that samples answers. C11 and CUDA,
 * nothing of the engine included: for the step driver (step.h), the kernels that read a row's position, token or
 * session, and the tests.
 *
 * Contract:
 *   - A card's step area is one struct step_block and, right behind it, row_capacity row records: row r at
 *     the area's offset plus STEP_BLOCK_BYTES_VALUE plus r STEP_ROW_BYTES_VALUE. row_capacity is the plan's and
 *     the program names the area's offset on each card (model_seam.h). Before a step's first launch the host writes
 *     the block and rows 0 .. row_count - 1 and uploads them to every card; the rows behind keep a row an earlier
 *     step checked, or the zeros the arena began with. The verification launch writes a draft step area and its
 *     one-row views on the result card in the step it answers, before the draft launches read them. The host's
 *     DRAFTS anchor plus its proposals, one emitted row and the admitted draft block fit inside the context limit.
 *     A padded block sets STEP_BLOCK_PADDED in flags, including its empty views: its router writes
 *     invalid expert picks behind row_count, since prefill expert programs scan their physical capacity.
 *   - Kernels read a row's position, token and session here, at offsets their records name, so no record changes
 *     from step to step. A kernel that writes state (a ring, a pool, key tiles, a compressor's slots) or reads at
 *     indices a kernel or a hand-off wrote (the attention core at a selection's entries, in a row behind perhaps
 *     another group's; the experts at the router's jobs) does so for the rows below row_count only. Another may
 *     compute the rows behind, which nothing reads: it addresses its arrays by row and by a position, token or slot
 *     an earlier step checked, so it stays inside them.
 *   - A row: a position from 0 to the context limit less 1; a token id below the vocabulary; a session slot below
 *     the plan's session capacity; flags of STEP_ROW_* only; the rows of one session its next positions,
 *     ascending and consecutive. A step's rows are all front rows or none, and a front row asks for no logits
 *     (model_seam.h: a front step). The step driver holds every row to this before the upload: kernels check nothing.
 *   - The session settings lie at an offset of the card that samples, slot s at s STEP_SESSION_BYTES,
 *     uploaded when they change and read by the sampler for every row of the slot: seed, temperature and top_p as
 *     front_end_seam.h defines them (a draw keyed by seed, position and token; temperature 0 takes the argmax).
 *   - The result area of the card that samples is one struct step_result and, right behind it, row_capacity
 *     row results. The sampler writes the sequence and row count of the step it answers and, for each row below
 *     row_count, the token drawn and its logit, or STEP_NO_TOKEN and +0 where the row's flags ask for no
 *     logits. The driver takes a result only if its sequence and row count are the step's.
 *   - A record has fixed-width members, widest first, and no padding: its size is asserted in both languages.
 */
#ifndef STEP_RECORDS_H
#define STEP_RECORDS_H

#include <stdint.h>

#define STEP_BLOCK_BYTES_VALUE 16u
#define STEP_ROW_BYTES_VALUE 16u
#define STEP_SESSION_BYTES 16u
#define STEP_RESULT_BYTES 16u
#define STEP_ROW_RESULT_BYTES 8u
#define STEP_ROW_LOGITS \
	0x1u /* flag: the head computes this row's logits, the sampler draws */
#define STEP_ROW_FRONT \
	0x2u /* flag: only the program's front runs for the row (model_seam.h) */
#define STEP_ROW_PADDED \
	0x20u /* headless prefill: fixed-capacity chunk, unused rows stay inactive   \
         */
#define STEP_ROW_FLAGS \
	0x1FFFu /* every flag the driver takes (the cone's layers included) */
/* A full step's prompt row (engine.c's prefill_cone_skip): CONE set, CONE_LAYERS(flags) of the decoder's layers (from
 * model_layout.h's MODEL_DECODER_FIRST_LAYER on) have an output of the row that a later row or the head reads;
 * a later decoder layer's core and routed experts may leave the row alone (its output there is never read: the
 * cone); with CONE_TRUNKS too, the wide trunk products of a decoder layer after the row's last one past the first
 * (relaxed_linear_records.h: none of that layer's outputs or window rows of the row is read). Without CONE every layer
 * runs the row. */
#define STEP_ROW_CONE_TRUNKS 0x40u
#define STEP_ROW_CONE 0x80u
#define STEP_ROW_CONE_SHIFT 8u
#define STEP_ROW_CONE_MASK 0x1Fu
#define STEP_ROW_CONE_LAYERS(flags) \
	(((flags) >> STEP_ROW_CONE_SHIFT) & STEP_ROW_CONE_MASK)
/* Whether decoder_layer (its index from the decoder's first layer) may leave a row of these flags alone. */
#define STEP_ROW_CONE_SKIPS(flags, decoder_layer) \
	(((flags) & STEP_ROW_CONE) != 0 &&        \
	 (decoder_layer) >= STEP_ROW_CONE_LAYERS(flags))
/* And whether a wide trunk product that takes decoder_layer's cone (the layer before its own) may. */
#define STEP_ROW_CONE_TRUNK_SKIPS(flags, decoder_layer) \
	(((flags) & STEP_ROW_CONE_TRUNKS) != 0 &&       \
	 STEP_ROW_CONE_SKIPS(flags, decoder_layer))
#define STEP_ROW_IMAGE \
	0x10u /* a typed image-span position; uses VL bias and no Engram */
#define STEP_NO_TOKEN \
	(-1) /* the token of a row whose logits nobody asked for */
/* Speculative rounds: THIN admits one verifier, BATCH one per session. Every flagged row has LOGITS.
 * Each owner's last unproposed row is its anchor; later rows of that owner are PROPOSED, at most seven.
 * DRAFTS is on that anchor alone, with room for its next block. Session rows remain ascending and consecutive. */
#define STEP_ROW_PROPOSED \
	0x4u /* the row's token is a proposal: the verifying session's draw at the
                                                   row before it (position - 1, this step) verifies it */
#define STEP_ROW_DRAFTS \
	0x8u /* on the verifying session's anchor row: the draft segment runs and
                                                   proposes from the token this step accepts */
#define STEP_DRAFT_MOST \
	7u /* proposals of a round: a thin step's rows less the anchor */
#define STEP_NO_SESSION UINT32_MAX
#define STEP_ROUND_BYTES 88u
#define STEP_DRAFT_VIEW_BYTES 32u /* a one-row step block: the block, one row */
#define STEP_DRAFT_RESULT_BYTES \
	32u /* a one-row result area, its 16-byte alignment kept */
#define STEP_BATCH_MAXIMUM_ROWS 64u
#define STEP_BATCH_MAXIMUM_SESSIONS 8u
#define STEP_BATCH_SLAB_ROWS 8u
#define STEP_BATCH_SLAB_RESULT_BYTES \
	(STEP_RESULT_BYTES + STEP_BATCH_SLAB_ROWS * STEP_ROW_RESULT_BYTES)
#define STEP_BATCH_RESULT_SPAN_MOST 4096u
#define STEP_RESULT_SPAN_MOST \
	1024u /* a DRAFTS step's download: the result area to the round's end */

#define STEP_BLOCK_PADDED \
	1u /* inactive router picks carry the no-expert sentinel */

struct step_block {
	uint64_t sequence; /* of the step, from 1; the result repeats it */
	uint32_t row_count; /* 1 to the program's rows, 0 to drain; of a view, 0 or more */
	uint32_t flags; /* 0, or STEP_BLOCK_PADDED; empty views keep it */
};

struct step_row_record {
	int32_t position;
	int32_t token_id;
	uint32_t session_slot;
	uint32_t flags; /* STEP_ROW_* */
};

struct step_session {
	uint64_t seed;
	float temperature;
	float top_p;
};

struct step_result {
	uint64_t sequence;
	uint32_t row_count;
	uint32_t reserved_zero;
};

struct step_row_result {
	int32_t token_id;
	float logit; /* of the token, as the head computed it */
};

/* Of the verifying session (its DRAFTS row's), on the card that samples: sequence to reserved_zero written by
 * verify_matches_entry, proposals and confidence_logits by draft_confidence_entry (zeros from R on). */
struct step_round {
	uint64_t sequence; /* of the step it answers */
	uint32_t session_slot;
	uint32_t emitted_count; /* e: 1 .. k + 1 */
	int32_t anchor_position; /* C' = P + e: the draft block's first position */
	int32_t anchor_token; /* the target's draw y_e: the block's first token */
	uint32_t proposal_count; /* R */
	uint32_t reserved_zero;
	int32_t proposals[STEP_DRAFT_MOST]; /* for positions C' + 1 .. C' + R */
	float confidence_logits
		[STEP_DRAFT_MOST]; /* raw scores of the confidence head (REF:1100) */
};

#ifdef __cplusplus
#define STEP_RECORD_HAS(record, byte_count) \
	static_assert(sizeof(struct record) == (byte_count), #record)
#else
#define STEP_RECORD_HAS(record, byte_count) \
	_Static_assert(sizeof(struct record) == (byte_count), #record)
#endif
STEP_RECORD_HAS(step_block, STEP_BLOCK_BYTES_VALUE);
STEP_RECORD_HAS(step_row_record, STEP_ROW_BYTES_VALUE);
STEP_RECORD_HAS(step_session, STEP_SESSION_BYTES);
STEP_RECORD_HAS(step_result, STEP_RESULT_BYTES);
STEP_RECORD_HAS(step_row_result, STEP_ROW_RESULT_BYTES);
STEP_RECORD_HAS(step_round, STEP_ROUND_BYTES);

#endif
