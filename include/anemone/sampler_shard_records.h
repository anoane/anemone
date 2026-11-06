/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * sampler_shard_records.h (tp4_v3_claude, default off): the sampler of sampler_records.h over a vocabulary cut into
 * SAMPLER_SHARD_RANKS contiguous shards, one on each card of a TP group, with no gather of logits. C11 and
 * CUDA. Every card answers every row with the token and logit the sampler gives on the whole vocabulary:
 *   - Phase 0: each card's largest key (kernel_sampler.cuh's sampler_key of the logit and the GLOBAL token id) over its
 *     shard, with its logit, goes to every card; the largest of the four is l_max's (a maximum has no order). Greedy
 *     rows (temperature 0, or x_max not finite) answer it.
 *   - Phase 1 (temperature above 0, top_p 1): each card's largest race key over its shard, e_t and the keyed draw
 *     computed by the sampler's own functions from the global x_max and global token ids, with that token's logit,
 *     goes to every card; the largest of the four answers. top_p below 1 is refused (the checked modules trap): its
 *     mass histogram would need an exchange per digit.
 *   - A launch's use: u = SAMPLER_SHARD_CALLS sequence + call, the sequence of its step area (a draft view
 *     holds its step's), call its place in the step (thin: 0 the target's sampler, 1 + j the draft's position j;
 *     BATCH: slab s's s, then verifier v's position j slabs + v P + j); the group's launches of a step run in the
 *     order of their calls on every card.
 *   - A record is 16 bytes, written by one 16-byte store: key (8), logit (4), tag (4); tag = 2 (2 u + phase) + 1 (low
 *     32 bits), never 0, so a slot of a fresh exchange area, all zeros, never matches. A card reads a record once its
 *     tag matches, by one 16-byte volatile load. The slots have two parities (u's low bit). A step's uses are
 *     consecutive integers, so two uses of one parity have a use of the other between them, whose phase-0 records
 *     every card publishes only after it read the earlier one; between steps the group's collectives order every
 *     card's reads of a step before any card's first use of the next (a card publishes nothing of a use before it
 *     reached that step's head).
 *   - Settings (the sampler's session settings): every card reads its own copy, never another card's memory (a
 *     peer read costs ~0.3 ms on the CMPs, sampler02). The card that holds the settings pushes them at the start of
 *     each step that samples (sampler_settings_push, its first launch): every slot's 16 bytes into every
 *     card's copy (its own too) by posted peer stores, a system fence, then the copy's word, the step's sequence, by
 *     st.release.sys. A sampler shard waits (ld.acquire.sys, deadline, trap) until its copy's word is its step's
 *     sequence, then reads its row's slot. The push of step s + 1 cannot overtake a read of step s: the pushing card
 *     reaches step s + 1 only behind its last sampler shard of step s, which takes every card's record of it, and a
 *     card reads the settings before it publishes its record.
 *   - Bounded waits: deadline_cycles of clock64 a record, then trap.
 * Exchange area of every card: [parity 2][phase 2][source rank 4][row 8] records; a card's peer table holds the
 * address of each rank's area (its own too), so the areas may lie at different offsets. Settings copy of every card:
 * the word (uint64) at 0, the slots (sampler_records.h's session settings, 16 bytes each) from
 * SAMPLER_SETTINGS_SLOTS_OFFSET.
 */
#ifndef SAMPLER_SHARD_RECORDS_H
#define SAMPLER_SHARD_RECORDS_H

#include <stdint.h>

#include "anemone/sampler_records.h"

#define KERNEL_FILE_SAMPLER_SHARD "kernel_sampler_shard"
#define KERNEL_SAMPLER_SHARD "sampler_shard"
#define KERNEL_SAMPLER_SETTINGS_PUSH "sampler_settings_push"
#define SAMPLER_SETTINGS_PUSH_ARGUMENT_BYTES 32u
#define SAMPLER_SETTINGS_PUSH_THREADS 32u
#define SAMPLER_SETTINGS_SLOTS_OFFSET \
	16u /* the slots of a settings copy, behind its word */
#define SAMPLER_SETTINGS_SLOT_BYTES 16u
#define SAMPLER_SHARD_RANKS 4u
#define SAMPLER_SHARD_PARITIES 2u
#define SAMPLER_SHARD_PHASES 2u
#define SAMPLER_SHARD_RECORD_BYTES 16u
#define SAMPLER_SHARD_EXCHANGE_BYTES                                           \
	(SAMPLER_SHARD_PARITIES * SAMPLER_SHARD_PHASES * SAMPLER_SHARD_RANKS * \
	 LINEAR_MAXIMUM_ROWS * SAMPLER_SHARD_RECORD_BYTES)
#define SAMPLER_SHARD_ARGUMENT_BYTES 80u
#define SAMPLER_SHARD_CALLS \
	64u /* a step's launches at most: a BATCH step's slabs, then each verifier's
                                           draft positions (a thin step: the target's, then one a position) */
#define SAMPLER_SHARD_DEADLINE_CYCLES \
	20000000000ull /* about 13 s of a CMP's clock: a whole step is ms */

/* kernel_sampler_shard, sampler_shard: grid (1, R, 1) of SAMPLER_THREADS, as the sampler.
 *   reads   step_offset as the sampler's; session_offset: this card's settings copy (its word, then the slots, as
 *           sampler_settings_push writes them); logits_offset:
 *           float32 [R][shard_size], global tokens first_token .. first_token + shard_size - 1; peer_table_offset:
 *           uint64 addresses of the ranks' exchange areas [ranks]; its exchange area (the records, by tag);
 *   writes  result_offset: the result area, its block and R rows (every card the same); every card's exchange area:
 *           this rank's slots. */
struct sampler_shard_arguments {
	uint64_t step_offset;
	uint64_t session_offset; /* this card's settings copy */
	uint64_t logits_offset;
	uint64_t result_offset;
	uint64_t exchange_offset; /* this card's area */
	uint64_t peer_table_offset;
	uint64_t deadline_cycles;
	uint32_t first_token;
	uint32_t shard_size; /* a multiple of SAMPLER_TOKENS_PER_LOAD_VALUE */
	uint32_t vocabulary_size; /* of the whole head: first_token + shard_size at most */
	uint32_t session_capacity;
	uint32_t rank; /* below SAMPLER_SHARD_RANKS */
	uint32_t call; /* below SAMPLER_SHARD_CALLS */
};
LINEAR_RECORD_HAS(sampler_shard_arguments, SAMPLER_SHARD_ARGUMENT_BYTES);

/* kernel_sampler_shard, sampler_settings_push: grid (1, 1, 1) of SAMPLER_SETTINGS_PUSH_THREADS, on the
 * card that holds the settings, at the start of a step that samples.
 *   reads   step_offset: the step block (its sequence); sessions_offset: the settings [session_capacity];
 *           peer_table_offset: uint64 addresses of the ranks' settings copies [ranks] (its own too);
 *   writes  every copy's slots, then its word (the step's sequence). */
struct sampler_settings_push_arguments {
	uint64_t step_offset;
	uint64_t sessions_offset;
	uint64_t peer_table_offset;
	uint32_t session_capacity;
	uint32_t reserved_zero;
};
LINEAR_RECORD_HAS(sampler_settings_push_arguments,
		  SAMPLER_SETTINGS_PUSH_ARGUMENT_BYTES);

#endif
