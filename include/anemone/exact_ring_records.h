/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Exact output-axis transport. Each site and prepared row count owns an immutable two-epoch bank.
 * The peer table is the allocation-qualified ordered-collective table. All four direct mappings must live
 * until the complete flight drains. This record contains arena-relative offsets, never host pointers.
 * mode=0 is exact lossless gather; mode=1 explicitly selects approximate BF16 ring addition.
 * The lossless sentinel codec preserves every payload bit, including FP32 sums and BF16 special values.
 * A failed/deadline-exhausted bank traps and cannot be reused. No concurrent flight shares its scratch.
 */
#ifndef EXACT_RING_RECORDS_H
#define EXACT_RING_RECORDS_H

#include "anemone/ordered_collective_records.h"

#define EXACT_RING_PACK_BYTES 16u
#define EXACT_RING_PACKET_BYTES 20u
#define EXACT_RING_EPOCH_HOPS 6u
/* The BF16 sum's slots a pack: two epochs of its six hops (three summing, three forwarding), a pack each, no bitmap */
#define EXACT_RING_SUM_EPOCH_HOPS 12u
#define EXACT_RING_GRID_GROWTH 2u
#define EXACT_RING_BLOCKS 16u
#define EXACT_RING_RESERVED_WORDS 2u
#define EXACT_RING_METADATA_BYTES 128u
#define EXACT_RING_GATHER 0u
#define EXACT_RING_BF16_SUM 1u
/* The BF16 sum with column owners (--tp_ring_column_owners): the input and output [vectors][row] row-major and owner o
 * the o-th quarter of every row's packs (packs_per_owner / vectors a row), not the o-th quarter of the buffer: each
 * row's chain of additions starts at the same rank whatever the row's index and the vectors, so a row's sum is the same
 * at any row count (mode 1's owners are row quarters: the order follows the row's place in the step). */
#define EXACT_RING_BF16_SUM_COLUMNS 2u
#define EXACT_RING_ARGUMENT_BYTES 80u
#define EXACT_RING_UNEQUAL_ARGUMENT_BYTES 96u
#define EXACT_RING_RANKS 4u
#define EXACT_RING_REQUIRED_MAXIMUM_OWNERS 2u
#define EXACT_RING_MAXIMUM_DEADLINE (UINT64_C(1) << 62u)
#define KERNEL_FILE_EXACT_RING "kernel_exact_ring"
#define KERNEL_EXACT_RING "exact_ring"
#define KERNEL_EXACT_RING_UNEQUAL "exact_ring_unequal"

struct exact_ring_arguments {
	uint64_t input_offset;
	uint64_t output_offset;
	uint64_t bank_offset;
	uint64_t peer_table_offset;
	uint64_t peer_bank_delta;
	uint64_t deadline_cycles;
	uint32_t packs_per_owner;
	uint32_t vectors;
	uint32_t rank;
	uint32_t mode;
	uint64_t reserved_words[EXACT_RING_RESERVED_WORDS];
};

STEP_RECORD_HAS(exact_ring_arguments, EXACT_RING_ARGUMENT_BYTES);

/* Independent lossless gather entry for unequal output-channel owners. All four counts are positive and
 * partition [0, total_packs) without gaps; total_packs fits uint32. No count exceeds maximum_packs, and at least
 * REQUIRED_MAXIMUM_OWNERS counts equal it. Thus every active packet offset has at least two independent
 * producers: their clockwise paths cover the ring, preventing a producer's generation n+2 from overtaking an
 * unread generation n. An owner on either path must consume n before producing or forwarding the other owner's
 * n+1, which the first producer needs before n+2. A unique widest owner has an uncredited tail and is refused.
 * Input is owner-major, and only this rank's first_pack/pack_count range is read. With vectors==0 output is
 * owner-major too; otherwise every count is divisible by vectors and output is [vector][all owner channels].
 * No numerical addition, BF16 conversion or zero canonicalization occurs.
 *
 * Cold preparation proves disjoint complete input/output ranges of total_packs*PACK_BYTES, and local/remote
 * banks of at least METADATA_BYTES + maximum_packs*EPOCH_HOPS*PACKET_BYTES. The metadata holds BLOCKS uint32
 * generation counters, then one uint32 status; the entire metadata and packet bank begin zeroed. Peer table
 * entries name allocation-qualified direct mappings, with matching remote bank offsets and valid lifetimes.
 * This record has no arena sizes and does not itself establish any address, capacity or peer admission.
 * Counts, maximum_packs, vectors and launch grid remain immutable for each bank's lifetime. Every rank submits
 * the same positive-count geometry on its ordered stream; each channel/row count/flight owns separate scratch.
 * All launched CTAs must be resident: grid.x<=BLOCKS, block.x is the cut-through thread count, other axes are one.
 * A missing participant, generation exhaustion or nonzero status fails closed; drain before destroying, never
 * reuse a failed bank. Original equal-owner arguments and their entry remain independent and unchanged. */
struct exact_ring_unequal_arguments {
	uint64_t input_offset;
	uint64_t output_offset;
	uint64_t bank_offset;
	uint64_t peer_table_offset;
	uint64_t peer_bank_delta;
	uint64_t deadline_cycles;
	uint32_t first_pack[EXACT_RING_RANKS];
	uint32_t pack_count[EXACT_RING_RANKS];
	uint32_t maximum_packs;
	uint32_t vectors;
	uint32_t rank;
	uint32_t reserved_zero;
};

STEP_RECORD_HAS(exact_ring_unequal_arguments,
		EXACT_RING_UNEQUAL_ARGUMENT_BYTES);
#endif
