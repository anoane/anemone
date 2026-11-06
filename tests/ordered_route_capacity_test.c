/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Metadata-only regression for the TP qualifier's full wide count; no payload or CUDA allocation. */
#include "ordered_push_internal.h"
#include "anemone/tp_exchange_records.h"

#include <stdio.h>
#include <string.h>

#define CAPACITY_RANKS 4u
#define CAPACITY_DENIED_SOURCE 1u
#define CAPACITY_DENIED_READER 3u
#define CAPACITY_CMP_LINK 2u
#define CAPACITY_PCIE_LANES 16u
#define CAPACITY_POISON 0xA5A5A5A5u

static int failures;

static void expect(bool valid, const char *what)
{
	if (!valid) {
		(void)fprintf(stderr, "FAILED: route capacity: %s\n", what);
		failures++;
	}
}

static void
refused(const struct ordered_collective_route_description *description,
	uint32_t maximum_elements)
{
	uint32_t output[CAPACITY_RANKS + 1u];
	for (uint32_t rank = 0; rank <= CAPACITY_RANKS; rank++)
		output[rank] = CAPACITY_POISON;
	expect(ordered_push_routes_prepare_capacity(description,
						    maximum_elements, output) ==
		       STATUS_INVALID_ARGUMENT,
	       "invalid route or bound is refused");
	for (uint32_t rank = 0; rank <= CAPACITY_RANKS; rank++)
		expect(output[rank] == CAPACITY_POISON,
		       "refusal preserves all output words");
}

int main(void)
{
	struct card_description_record cards[CAPACITY_RANKS] = { 0 };
	uint32_t direct[CAPACITY_RANKS], output[CAPACITY_RANKS + 1u];
	const uint32_t all = (1u << CAPACITY_RANKS) - 1u;
	struct ordered_collective_route_description description = {
		.cards = cards,
		.direct_reader_masks = direct,
		.rank_count = CAPACITY_RANKS,
		.element_count = ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS,
		.operation = ORDERED_ALL_REDUCE,
		.common_numa_root = true
	};
	for (uint32_t rank = 0; rank < CAPACITY_RANKS; rank++) {
		(void)snprintf(cards[rank].name, sizeof(cards[rank].name), "%s",
			       "CMP 170HX");
		cards[rank].arithmetic_class = ARITHMETIC_CLASS_SM_80;
		cards[rank].link_generation = CAPACITY_CMP_LINK;
		cards[rank].link_lane_count = CAPACITY_PCIE_LANES;
		direct[rank] = all;
	}
	expect(ordered_collective_routes_prepare(&description, output) ==
		       STATUS_OK,
	       "public collective accepts its unchanged bound");
	for (uint32_t rank = 0; rank < CAPACITY_RANKS; rank++)
		expect(output[rank] == (all & ~(1u << rank)),
		       "public collective retains the common-root host preference");
	expect(ordered_push_routes_prepare(&description, output) == STATUS_OK,
	       "public push accepts its unchanged bound");
	for (uint32_t rank = 0; rank < CAPACITY_RANKS; rank++)
		expect(output[rank] == 0,
		       "public push retains zero-row capability routing");
	description.element_count = TP_EXCHANGE_MAXIMUM_ELEMENTS;
	expect(description.element_count > ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS,
	       "the full TP wide shape exceeds the generic bound");
	for (uint32_t rank = 0; rank <= CAPACITY_RANKS; rank++)
		output[rank] = CAPACITY_POISON;
	expect(ordered_collective_routes_prepare(&description, output) ==
		       STATUS_INVALID_ARGUMENT,
	       "public collective refuses the full TP count");
	expect(ordered_push_routes_prepare(&description, output) ==
		       STATUS_INVALID_ARGUMENT,
	       "public push refuses the full TP count");
	for (uint32_t rank = 0; rank <= CAPACITY_RANKS; rank++)
		expect(output[rank] == CAPACITY_POISON,
		       "public refusals preserve all output words");
	expect(ordered_push_routes_prepare_capacity(
		       &description, TP_EXCHANGE_MAXIMUM_ELEMENTS, output) ==
		       STATUS_OK,
	       "private planner accepts the full 1024-row TP count");
	for (uint32_t rank = 0; rank < CAPACITY_RANKS; rank++)
		expect(output[rank] == 0,
		       "full-count direct clique has no host preference");
	expect(output[CAPACITY_RANKS] == CAPACITY_POISON,
	       "success writes only the rank-count words");
	direct[CAPACITY_DENIED_SOURCE] &= ~(1u << CAPACITY_DENIED_READER);
	expect(ordered_push_routes_prepare_capacity(
		       &description, TP_EXCHANGE_MAXIMUM_ELEMENTS, output) ==
		       STATUS_OK,
	       "directed missing edge falls back to host");
	for (uint32_t rank = 0; rank < CAPACITY_RANKS; rank++)
		expect(output[rank] == (rank == CAPACITY_DENIED_SOURCE ?
						1u << CAPACITY_DENIED_READER :
						0),
		       "only the missing directed edge uses host");
	refused(&description, ORDERED_COLLECTIVE_MAXIMUM_ELEMENTS);
	refused(&description, TP_EXCHANGE_MAXIMUM_ELEMENTS - 1u);
	refused(&description, 0);
	refused(&description, UINT32_MAX - ORDERED_COLLECTIVE_THREADS + 1u);
	refused(&description, UINT32_MAX);
	expect(ordered_push_routes_prepare_capacity(
		       &description, UINT32_MAX - ORDERED_COLLECTIVE_THREADS,
		       output) == STATUS_OK,
	       "largest safe explicit limit accepts the unchanged full count");
	description.element_count++;
	refused(&description, TP_EXCHANGE_MAXIMUM_ELEMENTS);
	description.element_count = TP_EXCHANGE_MAXIMUM_ELEMENTS;
	description.operation = ORDERED_TWO_SHOT;
	refused(&description, TP_EXCHANGE_MAXIMUM_ELEMENTS);
	description.operation = ORDERED_ALL_REDUCE;
	description.paired_scatter_gather = true;
	refused(&description, TP_EXCHANGE_MAXIMUM_ELEMENTS);
	description.paired_scatter_gather = false;
	direct[0] &= ~1u;
	refused(&description, TP_EXCHANGE_MAXIMUM_ELEMENTS);
	direct[0] = all | (1u << CAPACITY_RANKS);
	refused(&description, TP_EXCHANGE_MAXIMUM_ELEMENTS);
	direct[0] = all;
	expect(ordered_push_routes_prepare_capacity(
		       &description, TP_EXCHANGE_MAXIMUM_ELEMENTS, direct) ==
		       STATUS_OK,
	       "full-count output may alias the directed input masks");
	for (uint32_t rank = 0; rank < CAPACITY_RANKS; rank++)
		expect(direct[rank] == output[rank],
		       "in-place output retains directed-edge orientation");
	(void)printf("ordered_route_capacity_test: %s\n",
		     failures == 0 ? "passed" : "FAILED");
	return failures == 0 ? 0 : 1;
}
