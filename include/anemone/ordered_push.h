/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * Prepared one-shot push all-reduce. Each rank owns two slots with ready/done lines for every source rank,
 * FP32 input/output arrays and rank_count inbox arrays. A source writes its input once into every remote inbox,
 * then each receiver sums its own inboxes in +0, rank 0, rank 1, ... FP32 round-to-nearest order. FP32 wire is
 * bit-exact for finite inputs whose ordered sum is not NaN. BF16 wire rounds every rank's FP32 input, including
 * self, to BF16 round-to-nearest-even before summing; the output remains FP32. This changes arithmetic and needs
 * the caller's relaxed/fast admission. No FP8 path or changed product reduction is implied.
 * BF16 conversion can overflow a finite FP32 value to infinity; callers must qualify ranges and numerical gates.
 * BF16 wire does not inherit the lossless FP32 contract. Four-byte inbox strides include padding for an odd
 * BF16 tail word; its unused upper half is ignored by the receiver.
 *
 * The peer table uses ordered_collective_peer records, indexed by destination. arena_address+region_offset
 * points at the remote GPU slot region for direct routes or its mapped pinned host mirror for host routes.
 * Self stays local. Self.host_address holds the local mirror region base when any incoming edge uses host;
 * region_offset is not added to this address. It names the same region that incoming host peers address.
 * incoming_host_mask names sources whose pushes arrive in that mirror; the caller freezes the entire directed
 * route graph and checks remote/mirror extents at preparation. Direct edges must have runtime peer_connect grants
 * and current-stack allocation load/store qualification. This API never infers access from card names.
 * ready/done lines occupy phases*rank_count*128 bytes at the start of each slot; source is the only writer of
 * its line in every destination. Inputs/outputs/inboxes and signals never overlap. All regions are initially zero.
 * Every rank uses identical positive strictly increasing full uint64 step sequences, counts, formats and offsets.
 * Generation advances per slot use, not by engine sequence; skipped sequences and uint32 generation wrap work.
 * Full uint64 sequence wrap/replay is refused. The all-rank done barrier precedes return and protects reuse of
 * each destination's inbox even when the next producer is much faster. No CPU participates after launch.
 *
 * Only one CTA runs, so there is no inter-CTA/grid residency dependency. Device helpers keep the publish/sum
 * phases separate for a future single-CTA producer epilogue, under the same fences/barriers and ownership.
 * No other operation may use these slots concurrently. On failure/cancellation every participant must drain
 * before reinitializing; deadlines trap rather than reading stale data. Mappings outlive all possible readers.
 * prepare validates local bounds/alignment/disjointness and returns INVALID_ARGUMENT without changing output.
 * No allocation, mutable planning or host progress occurs on the token path.
 */
#ifndef ORDERED_PUSH_H
#define ORDERED_PUSH_H

#include "anemone/ordered_collective.h"
#include "anemone/ordered_push_records.h"

struct ordered_push_description {
	struct ordered_push_arguments arguments;
	uint64_t arena_byte_count;
};

MUST_CHECK result_status
ordered_push_kernel_prepare(const struct backend_operations *backend,
			    struct device_card *card, uint32_t *kernel_out);
MUST_CHECK result_status
ordered_push_prepare(const struct ordered_push_description *description,
		     uint32_t kernel, struct kernel_launch *launch_out);

/* The generic measured route planner provides qualified-direct/denied-host fallback with no old read-collective
 * performance preference. Pass zero measured rows: writer_masks[owner] qualifies accessors writing that owner.
 * host_writer_masks[owner] names the sources whose pushes use the owner's host mirror. A measured push policy
 * must be qualified separately; peer-read collective performance rows do not qualify push traffic.
 */
MUST_CHECK result_status ordered_push_routes_prepare(
	const struct ordered_collective_route_description *description,
	uint32_t *host_writer_masks_out);

#endif
