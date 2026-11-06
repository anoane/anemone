/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Resident-CTA BF16 owner exchange with rank-ordered FP32 sums and one BF16 round. Independent of the old ring.
 * Uses exact_ring_arguments (80bytes), BF16_SUM or BF16_SUM_COLUMNS to describe the input/output layout only.
 * It always adds +0, rank0, rank1, rank2, rank3 in FP32, then rounds to BF16 once. Finite inputs/sums are required
 * for the numerical contract; no claim about NaN payload selection. No intermediate BF16 addition occurs.
 *
 * Four live direct arenas and immutable geometry, one ordered stream per rank; all CTAs must be resident.
 * grid.x<=16, block.x=256, other axes one. Positive packs_per_owner divisible by32. Input/output disjoint or
 * identical; no partial overlap. Bank is disjoint from both and holds128 metadata bytes, two arrays of4*packs
 * uint4 (incoming partials, gathered results), then2*4*16 generation words. Capacity128+128*packs+512 bytes.
 * Existing BF16 ring banks (128+192*packs) suffice. All banks begin zero; all ranks use the same counts/grid.
 * Peer-table region offsets plus peer_bank_delta locate each remote bank; remote output offsets need not match.
 *
 * Each CTA owns the same packet indices on every rank. Peer-bank payload uses the retained ring's volatile
 * vector accessors. Writers system-fence before its leader releases READY.
 * The leader acquires all four READYs before threads read rank-ordered partials. Owners publish results to every
 * bank, fence, and release DONE; acquiring all DONEs precedes local output copies. The next generation cannot
 * overwrite unread results: it needs READY from a rank whose prior result copy precedes that publication on its
 * ordered stream. Likewise DONE n+1 needs every READY n+1, so it cannot overtake a rank still awaiting DONE n.
 * There is no cross-CTA spin barrier. No concurrent flight shares a bank; generations never wrap. Deadline or
 * generation exhaustion poisons/traps the bank. Drain every participant before closing; never reuse failure. */
#ifndef ROUND_ONCE_RING_RECORDS_H
#define ROUND_ONCE_RING_RECORDS_H

#include "anemone/exact_ring_records.h"

#define KERNEL_FILE_ROUND_ONCE_RING "kernel_bf16_round_once_ring"
#define KERNEL_ROUND_ONCE_RING "bf16_round_once_ring"
#define ROUND_ONCE_RING_THREADS 256u
#define ROUND_ONCE_RING_PACKS 32u
#define ROUND_ONCE_RING_PLANES 2u
#define ROUND_ONCE_RING_READY 0u
#define ROUND_ONCE_RING_DONE 1u
#define ROUND_ONCE_RING_THIRD_RANK 2u
#define ROUND_ONCE_RING_FOURTH_RANK 3u
#define ROUND_ONCE_RING_SIGNAL_WORDS \
	(ROUND_ONCE_RING_PLANES * EXACT_RING_RANKS * EXACT_RING_BLOCKS)

#endif
