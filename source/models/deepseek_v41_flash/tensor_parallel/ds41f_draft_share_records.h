/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_draft_share_records.h (tp4_v3_b, default off): the draft's rows from the result card to the other ranks of a
 * TP group, for the draft's head, Markov bias and sampler cut by outputs (ds41f_tp_stage.h's draft_shards). C11
 * and CUDA.
 *   - draft_share, on the sending card, in a step whose own step area has a DRAFTS row for its verifier (the
 *     verifier'th DRAFTS row; a thin step's verifier is 0): the draft's normed rows, its step area and its views into
 *     every other rank's share area, then that area's word, released.
 *   - draft_share_wait, on every other rank: in a step that has the verifier's DRAFTS row, waits for its word;
 *     in a step that has not, sets the row counts of its copies of the draft's step area and views to zero, so that
 *     the cut, Markov and sampler launches behind it read no row and no sampler waits for a record.
 *   - The word's value for a step and verifier: (uint32_t)(2 (DS41F_DRAFT_SHARE_VERIFIERS sequence + verifier)) +
 *     1, the sequence of the card's own step area; written once a step and verifier, never reset (a fresh area's zero
 *     never matches). A BATCH step's verifiers share one area in turn: the sender rewrites it for the next verifier
 *     behind its last sampler shard of this one, which takes every rank's record of it. Every store it covers is
 * followed by a system fence of     its thread, a barrier and, in the last block of the receiver's group (a ticket
 * counter that block sets back to     zero), one st.release.sys; the waiter polls ld.acquire.sys under a clock64
 * deadline, then traps.   - Reuse: the sender rewrites an area only in a later step, behind its target sampler shard,
 * which takes every     rank's record of that step, which a rank publishes behind its draft launches of the step
 * before. A share area (each receiving rank): the rows, bfloat16 [8][5,120], at 0; the step area's copy (block, 8 rows)
 * at DS41F_DRAFT_SHARE_STEP; the views' copies at DS41F_DRAFT_SHARE_VIEWS; the word at DS41F_DRAFT_SHARE_WORD.
 */
#ifndef DS41F_DRAFT_SHARE_RECORDS_H
#define DS41F_DRAFT_SHARE_RECORDS_H

#include <stdint.h>

#include "engine/library/block_scaled_linear/linear_records.h"
#include "models/deepseek_v41_flash/ds41f_geometry.h"
#include "engine/program/step_records.h"

#define DS41F_MODULE_DRAFT_SHARE "ds41f_draft_share"
#define DS41F_KERNEL_DRAFT_SHARE "draft_share"
#define DS41F_KERNEL_DRAFT_SHARE_WAIT "draft_share_wait"
#define DS41F_DRAFT_SHARE_RANKS 4u
#define DS41F_DRAFT_SHARE_THREADS 256u
#define DS41F_DRAFT_SHARE_BLOCKS_PER_RANK 4u
#define DS41F_DRAFT_SHARE_WAIT_THREADS 32u
#define DS41F_DRAFT_SHARE_PART_BYTES \
	256u /* each part of an area starts on a multiple */
#define DS41F_DRAFT_SHARE_STEP (LINEAR_MAXIMUM_ROWS * DS41F_HIDDEN_ROW_BYTES)
#define DS41F_DRAFT_SHARE_VIEWS \
	(DS41F_DRAFT_SHARE_STEP + DS41F_DRAFT_SHARE_PART_BYTES)
#define DS41F_DRAFT_SHARE_WORD \
	(DS41F_DRAFT_SHARE_VIEWS + LINEAR_MAXIMUM_ROWS * STEP_DRAFT_VIEW_BYTES)
#define DS41F_DRAFT_SHARE_AREA_BYTES \
	(DS41F_DRAFT_SHARE_WORD + DS41F_DRAFT_SHARE_PART_BYTES)
#define DS41F_DRAFT_SHARE_DEADLINE_CYCLES \
	20000000000ull /* about 13 s of a CMP's clock: a draft is ms */
#define DS41F_DRAFT_SHARE_ARGUMENT_BYTES 64u
#define DS41F_DRAFT_SHARE_WAIT_ARGUMENT_BYTES 32u
#define DS41F_DRAFT_SHARE_VERIFIERS \
	8u /* a step's verifiers at most: its sessions */

/* ds41f_draft_share, draft_share: grid DS41F_DRAFT_SHARE_RANKS x DS41F_DRAFT_SHARE_BLOCKS_PER_RANK of
 * DS41F_DRAFT_SHARE_THREADS; block group r copies to rank r (none to the sender's own rank). */
struct ds41f_draft_share_arguments {
	uint64_t rows_offset; /* local: bfloat16 [positions][5,120], the draft's normed rows */
	uint64_t draft_step_offset; /* local: the draft's step area, its block and positions rows */
	uint64_t views_offset; /* local: the draft's views, positions of STEP_DRAFT_VIEW_BYTES */
	uint64_t peer_table_offset; /* local: uint64 [DS41F_DRAFT_SHARE_RANKS], the addresses of the ranks' areas */
	uint64_t counter_offset; /* local: ticket counters [DS41F_DRAFT_SHARE_RANKS] */
	uint64_t step_offset; /* local: its own step area */
	uint32_t positions; /* 1 to STEP_DRAFT_MOST */
	uint32_t rank; /* the sender's */
	uint32_t verifier; /* below DS41F_DRAFT_SHARE_VERIFIERS */
	uint32_t reserved_zero;
};

/* draft_share_wait: grid 1 of DS41F_DRAFT_SHARE_WAIT_THREADS. */
struct ds41f_draft_share_wait_arguments {
	uint64_t area_offset; /* local: its share area */
	uint64_t step_offset; /* local: its own step area */
	uint64_t deadline_cycles;
	uint32_t positions;
	uint32_t verifier;
};

#endif
