/* SPDX-License-Identifier: AGPL-3.0-or-later */
#ifndef BB_CUT_THROUGH_H
#define BB_CUT_THROUGH_H

#include <stdint.h>

#define BB_RING_RANKS 4u
#define BB_RING_ARGUMENT_BYTES 120u
#define BB_RING_RECORD_VERSION 2u
#define BB_RING_HOPS 6u
#define BB_RING_PACK_WORDS 4u
#define BB_RING_PACK_BYTES 16u
#define BB_RING_CREDIT_PLANES 2u
#define BB_RING_THREADS 256u
#define BB_RING_MAXIMUM_BLOCKS 16u
#define BB_RING_GATHER 0u
#define BB_RING_BF16_SUM 1u
#define BB_RING_LOSSLESS_GATHER 2u
#define BB_RING_GATHER_HOPS (BB_RING_RANKS - 1u)
#define BB_RING_EPOCHS 2u
#define BB_RING_BITMAP_READY 0x100u
#define BB_RING_CHANNEL_BANKS 2u
#define BB_RING_DEADLINE 1u
#define BB_RING_GENERATION_EXHAUSTED 2u

/* C11/CUDA ABI. Pointers are prepared CUDA virtual addresses, never host dereferences.
 * Each rank owns separate input, output, packets, credits, generations and status.
 * All ranks launch the same mode, counts, grid and block on an ordered stream.
 * GATHER copies complete owner chunks, with no arithmetic. BF16_SUM is an explicit
 * approximation: BF16 addition at every ring hop, rotating chunk-dependent order,
 * positive-zero replacement by negative zero, and double-buffered data sentinels.
 * BF16_SUM requires immutable counts and grid for the channel's entire lifetime;
 * different decode widths use different channel banks. Its packets are laid out
 * [two epochs][BB_RING_HOPS][maximum_packs] of uint4, initially zero. It does not
 * use the exact path's credits. No concurrent calls on one channel are admitted.
 * LOSSLESS_GATHER is another exact owner gather: two epochs of three-hop
 * 16-byte data-sentinel packets plus one uint32 positive-zero bitmap per packet.
 * Counts and grid must remain immutable, as for BF16_SUM. Every payload bit is
 * restored, including signed zeros and NaN payloads; there is no arithmetic.
 * Its data plane is [two epochs][three hops][maximum_packs] of uint4, followed
 * by the same number of uint32 bitmap words, all initially zero.
 * The initial zero mode is GATHER. No model arithmetic is qualified by this record.
 * Packet words hold the unmodified 32 payload bits and a 32-bit generation. Credits
 * prevent overwrite, including graph replay and arbitrarily delayed neighbours.
 * A deadline failure poisons the channel: drain every participant before destroy;
 * never reuse a failed channel or consume output from a failed operation.
 * Credits and sent generations start at zero. The sender compares its local
 * last-sent generation with the acknowledgement, then records the new generation.
 * The consumer returns that generation to the previous sender after reading.
 * Generations start at zero, increment once per block per call, and never wrap.
 * Packets: [BB_RING_HOPS][maximum_packs][BB_RING_PACK_WORDS] of uint64.
 * Credits: [acknowledgements, last-sent][BB_RING_HOPS][maximum_packs] of uint32.
 * Only the sender writes last-sent; only its neighbour writes acknowledgements.
 * For exact gathers, output_vectors=0 selects packed owner-major output.
 * A nonzero count selects [vector][all owner channels] output, fused into the
 * gather. Every owner contributes an integral number of 16-byte packs per vector.
 * BF16_SUM requires output_vectors=0. Input/output are complete arrays. GATHER reads only this rank's owned range.
 * Owners partition [0, total_packs), without gaps, with 16-byte alignment.
 * Prepared allocations outlive all graph executions. No allocation on execution.
 */
struct bb_cut_through {
	uint64_t input;
	uint64_t output;
	uint64_t local_packets;
	uint64_t next_packets;
	uint64_t local_credits;
	uint64_t previous_credits;
	uint64_t generations;
	uint64_t status;
	uint64_t deadline_cycles;
	uint32_t first_pack[BB_RING_RANKS];
	uint32_t pack_count[BB_RING_RANKS];
	uint32_t maximum_packs;
	uint32_t rank;
	uint32_t mode;
	uint32_t output_vectors;
	uint32_t sum_row_packs; /* BF16_SUM: 0 owner-major input / output; else each row's packs of an owner (column
				   owners: owner o's pack p at row p / this, its o-th quarter) */
};

#endif
