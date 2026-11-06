/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ordered_collective.h: prepare the peer collectives of ordered_collective_records.h. Host C11 only.
 *
 * The caller prepares every directed table entry at load: direct VRAM when qualified, otherwise a mapped pinned
 * host mirror. An owner with any host reader also has its own mapped mirror address in its local table entry.
 * Thus one prepared interface supports a mixed direct/host group with no CPU progress after launch. Missing host
 * mapping support selects the explicit staged host fallback below; no unavailable address is dereferenced.
 * The caller checks remote/mirror bounds and group/sequence agreement, zeroes phase lines before use, and retains
 * mappings until all possible readers drain. prepare checks the local
 * arena, alignment, disjoint arrays/signals and separate step/table storage. Zero elements/ranks/deadline, more than
 * the stated bounds, an unknown operation or missing output is INVALID_ARGUMENT, leaving outputs untouched.
 * All operations are one CTA, intended for small messages; no allocation or table mutation occurs at launch.
 * host is the no-peer fallback over already staged arrays, with the same +0, rank 0, rank 1, ... FP32 order.
 * It requires round-to-nearest and disjoint input/output storage of element_count values on every rank; pointer
 * tables stay valid through the call. It allocates nothing. Reduce-scatter writes each rank's indexed slice only;
 * gather reads those indexed input slices. Finite inputs whose ordered sum is not NaN have the device's bits.
 */
#ifndef ORDERED_COLLECTIVE_H
#define ORDERED_COLLECTIVE_H

#include "anemone/backend_seam.h"
#include "anemone/ordered_collective_records.h"

struct ordered_collective_kernels {
	uint32_t handles[ORDERED_COLLECTIVE_OPERATIONS];
};

struct ordered_collective_description {
	struct ordered_collective_arguments arguments;
	uint64_t arena_byte_count;
	enum ordered_collective_operation operation;
};

/* Preparation-only default routes. direct_reader_masks[source] contains runtime-qualified readers, including
 * source itself. Each output mask names the readers of that source's host mirror; self always stays local.
 * Missing access always selects host. Recognized common-NUMA/root groups may prefer host despite peer access:
 * four CMP170HX/class80/Gen2x16 with a complete peer clique, or that clique plus one PRO6000/class120/Gen5x16
 * with all eight mixed directions denied. No BDF, ordinal, name of a machine or engine placement is assumed.
 * Unknown family/link/topology, or newly enabled mixed access, retains qualified-direct/denied-host routing.
 * The measured common_numa_root qualification is supplied by the caller; false keeps legacy routing. True
 * asserts a proved common NUMA/root domain, not merely equal NUMA numbers on an otherwise unknown machine.
 * Measured preferences assume cached registered host mirrors on the qualified root/NUMA domain.
 * Mixed host preference starts at 16 FP32 values (64 B); smaller inputs retain legacy routes. Standalone reduce-
 * scatter/gather also retain legacy routes unless paired_scatter_gather=true: the caller then prepares and runs
 * both operations under one frozen route/ownership plan. That flag is valid only for reduce-scatter or gather;
 * all-reduce/two-shot with the flag set is INVALID_ARGUMENT and leaves outputs untouched. A paired plan cannot
 * reset or change routes between its scatter and gather phases.
 * The recognized four-CMP clique prefers host at 32 values (128 B) for all-reduce/two-shot, and at 128 values (512 B)
 * for a paired scatter/gather plan. These are mode-specific measured cutoffs; smaller inputs keep legacy routes.
 * A plan freezes these masks, arrays and signal regions before launch. A route/size change needs an independent
 * prepared plan or complete group drain/reset; this function never changes a live peer table or credit bank.
 * rank_count from 1 through the maximum, valid elements/operation, terminated names, zero reserved fields and
 * masks containing self without out-of-group bits are required. Outputs have rank_count writable words;
 * errors leave all words unchanged. In-place output over direct_reader_masks is permitted.
 * No allocation or backend progress occurs.
 */
struct ordered_collective_route_description {
	const struct card_description_record *cards;
	const uint32_t *direct_reader_masks;
	uint32_t rank_count, element_count;
	enum ordered_collective_operation operation;
	bool common_numa_root;
	bool paired_scatter_gather;
};

MUST_CHECK result_status ordered_collective_routes_prepare(
	const struct ordered_collective_route_description *description,
	uint32_t *host_reader_masks_out);

/* Optional preparation-only measured overrides. Each row describes one complete operation and exact input size,
 * in caller rank order. The caller qualifies the rows for its current driver, IOMMU, topology and mailbox memory
 * type; this function does not infer that performance qualification. expected_direct_reader_masks must match the
 * current runtime-qualified reader masks exactly for the selected row. Per-rank arithmetic class and link fields
 * must match the description; the caller binds and reorders rows with the same cards and qualified cached registered
 * host allocation scope. All denied readers must use host, self
 * stays local, and no out-of-group bits are accepted. Paired scatter/gather needs separate, identical rows for both
 * operation keys; when both are supplied, mismatching masks/class/link bindings are INVALID_ARGUMENT. A single
 * paired row leaves its mate to the caller; the caller retains one frozen plan through both phases. Two-shot uses
 * one table for all internal phases. The table is bounded, immutable, allocation-free and has no live progress.
 * Duplicate keys are invalid.
 * Missing rows select qualified-direct/denied-host, without historical performance preferences. A matching stale
 * access pattern is UNSUPPORTED. Malformed rows are INVALID_ARGUMENT. All errors leave output words untouched.
 */
#define ORDERED_ROUTE_POLICY_MAXIMUM_ROWS 256u
struct ordered_collective_route_policy {
	uint32_t rank_count, element_count;
	enum ordered_collective_operation operation;
	bool paired_scatter_gather;
	uint32_t arithmetic_classes[ORDERED_COLLECTIVE_MAXIMUM_RANKS];
	uint32_t link_generations[ORDERED_COLLECTIVE_MAXIMUM_RANKS];
	uint32_t link_lane_counts[ORDERED_COLLECTIVE_MAXIMUM_RANKS];
	uint32_t expected_direct_reader_masks[ORDERED_COLLECTIVE_MAXIMUM_RANKS];
	uint32_t host_reader_masks[ORDERED_COLLECTIVE_MAXIMUM_RANKS];
};
MUST_CHECK result_status ordered_collective_routes_prepare_measured(
	const struct ordered_collective_route_description *description,
	const struct ordered_collective_route_policy *rows, uint32_t row_count,
	uint32_t *host_reader_masks_out);

MUST_CHECK result_status ordered_collective_kernels_prepare(
	const struct backend_operations *backend, struct device_card *card,
	struct ordered_collective_kernels *kernels_out);
MUST_CHECK result_status ordered_collective_prepare(
	const struct ordered_collective_description *description,
	const struct ordered_collective_kernels *kernels,
	struct kernel_launch *launch_out);
MUST_CHECK result_status ordered_collective_host(
	const float *const *inputs, float *const *outputs, uint32_t rank_count,
	uint32_t element_count, enum ordered_collective_operation operation);

#endif
