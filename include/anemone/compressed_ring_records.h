/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Optional lossless transport for the existing BF16 ring arithmetic, with an independent kernel entry.
 * Uses exact_ring_arguments, BF16_SUM or BF16_SUM_COLUMNS only. All four direct mappings must be proved
 * and live until every rank drains. Input/output may coincide. All ranks use immutable geometry and one ordered
 * stream; grid.x <= EXACT_RING_BLOCKS, block.x=256, other axes one. packs_per_owner is positive and divisible by32.
 *
 * A warp transports32 consecutive uint4 packs. Each lane sends three words (eight sign/mantissa bytes plus eight
 * exponent nibbles) if all its exponents lie within15 of the warp's maximum; otherwise four original words.
 * Header: top exponent, raw-lane bitmap, two unused zero words. At most128 payload words follow. The ready and
 * acknowledgement words are separate from payload; eight bytes after those words pad each packet to544 bytes.
 * Each hop/group owns one packet. Payload/READY on rank r are written by its previous neighbour; ACK there is
 * written by its next neighbour to acknowledge rank r's outgoing packet. The two messages use disjoint words.
 * Sender waits for acknowledgement generation n-1 before overwriting; every writer fences system-wide before
 * the warp publishes READY n with release. Receiver acquires READY n before reading, then the warp releases ACK n
 * to the previous sender after every read. Generations never wrap. Metadata matches exact_ring's128-byte prefix.
 * Bank capacity:128 + six hops * (packs_per_owner/32) *544 bytes; zero it before the first call. Old banks/kernels
 * are independent. Deadlines poison the bank and trap: drain all ranks before closing, never reuse a failed bank.
 * The codec preserves every payload bit. The caller's BF16 ring retains its existing positive-zero replacement
 * and hop-by-hop BF16 additions; this transport does not claim FP32 round-once arithmetic or model qualification. */
#ifndef COMPRESSED_RING_RECORDS_H
#define COMPRESSED_RING_RECORDS_H

#include "anemone/exact_ring_records.h"

#define KERNEL_FILE_COMPRESSED_RING "kernel_bf16_compressed_ring"
#define KERNEL_COMPRESSED_RING "bf16_compressed_ring"
#define COMPRESSED_RING_PACKS 32u
#define COMPRESSED_RING_THREADS 256u
#define COMPRESSED_RING_PACKET_WORDS 136u
#define COMPRESSED_RING_PACKET_BYTES 544u
#define COMPRESSED_RING_PAYLOAD_WORD 4u
#define COMPRESSED_RING_READY_WORD 132u
#define COMPRESSED_RING_ACK_WORD 133u
#define COMPRESSED_RING_HOPS 6u
#define COMPRESSED_RING_WARP_MASK 0xFFFFFFFFu
#define COMPRESSED_RING_COMPRESSED_WORDS 3u
#define COMPRESSED_RING_RAW_WORDS 4u
#define COMPRESSED_RING_RAW_MASK_WORD 1u

#endif
