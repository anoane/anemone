/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Pure C11 refusal, direction, binding and atomic-commit checks. No CUDA calls. */
#include "anemone/ordered_collective.h"
#include <stdio.h>
#include <string.h>
#define RANKS 5u
#define POLICY_ROWS 2u
#define ELEMENTS 128u
#define LANES 16u
#define CMP_LINK 2u
#define PRO_LINK 5u
#define POISON 0xa5a5a5a5u
static int failures;
static struct card_description_record cards[RANKS];
static uint32_t direct[RANKS], output[RANKS + 1u];
static void expect(bool valid, const char *what)
{
	if (!valid) {
		(void)fprintf(stderr, "FAILED: measured routes: %s\n", what);
		failures++;
	}
}
static void
refused(const struct ordered_collective_route_description *description,
	const struct ordered_collective_route_policy *rows, uint32_t count,
	result_status wanted)
{
	for (uint32_t source = 0; source <= RANKS; source++)
		output[source] = POISON;
	expect(ordered_collective_routes_prepare_measured(
		       description, rows, count, output) == wanted,
	       "refusal status");
	for (uint32_t source = 0; source <= RANKS; source++)
		expect(output[source] == POISON,
		       "refusal leaves every output word untouched");
}
int main(void)
{
	const uint32_t all = (1u << RANKS) - 1u;
	struct ordered_collective_route_description description = {
		.cards = cards,
		.direct_reader_masks = direct,
		.rank_count = RANKS,
		.element_count = ELEMENTS,
		.operation = ORDERED_ALL_REDUCE,
		.common_numa_root = true
	};
	struct ordered_collective_route_policy rows[POLICY_ROWS] = { 0 };
	rows[0].rank_count = RANKS;
	rows[0].element_count = ELEMENTS;
	rows[0].operation = ORDERED_ALL_REDUCE;
	for (uint32_t source = 0; source < RANKS; source++) {
		(void)snprintf(cards[source].name, sizeof(cards[source].name), "%s",
			       source == 0 ? "RTX PRO 6000" : "CMP 170HX");
		cards[source].arithmetic_class =
			source == 0 ? ARITHMETIC_CLASS_SM_120 :
				      ARITHMETIC_CLASS_SM_80;
		cards[source].link_generation = source == 0 ? PRO_LINK :
							      CMP_LINK;
		cards[source].link_lane_count = LANES;
		direct[source] = all;
		rows[0].arithmetic_classes[source] =
			(uint32_t)cards[source].arithmetic_class;
		rows[0].link_generations[source] =
			cards[source].link_generation;
		rows[0].link_lane_counts[source] = LANES;
		rows[0].expected_direct_reader_masks[source] = all;
		rows[0].host_reader_masks[source] = all & ~(1u << source);
	}
	expect(ordered_collective_routes_prepare_measured(&description, rows, 1,
							  output) == STATUS_OK,
	       "measured all-host override");
	for (uint32_t source = 0; source < RANKS; source++)
		expect(output[source] == (all & ~(1u << source)),
		       "override uses source-reader direction");
	description.element_count++;
	expect(ordered_collective_routes_prepare_measured(&description, rows, 1,
							  output) == STATUS_OK,
	       "unmeasured exact size retains direct");
	for (uint32_t source = 0; source < RANKS; source++)
		expect(output[source] == 0, "no interpolation");
	description.element_count = ELEMENTS;
	rows[1] = rows[0];
	refused(&description, rows, POLICY_ROWS, STATUS_INVALID_ARGUMENT);
	rows[0].host_reader_masks[0] |= 1u;
	refused(&description, rows, 1, STATUS_INVALID_ARGUMENT);
	rows[0].host_reader_masks[0] &= ~1u;
	direct[1] &=
		~1u; /* Missing accessor0 -> owner1, the reverse remains granted. */
	refused(&description, rows, 1, STATUS_UNSUPPORTED);
	rows[0].expected_direct_reader_masks[1] = direct[1];
	rows[0].host_reader_masks[1] &= ~1u;
	refused(&description, rows, 1, STATUS_INVALID_ARGUMENT);
	rows[0].host_reader_masks[1] |= 1u;
	expect(ordered_collective_routes_prepare_measured(&description, rows, 1,
							  output) == STATUS_OK,
	       "denied directed edge is safely mirrored");
	cards[0].arithmetic_class = ARITHMETIC_CLASS_SM_80;
	refused(&description, rows, 1, STATUS_UNSUPPORTED);
	cards[0].arithmetic_class = ARITHMETIC_CLASS_SM_120;
	cards[0].link_generation = CMP_LINK;
	refused(&description, rows, 1, STATUS_UNSUPPORTED);
	cards[0].link_generation = PRO_LINK;
	refused(&description, NULL, 1, STATUS_INVALID_ARGUMENT);
	refused(&description, rows, ORDERED_ROUTE_POLICY_MAXIMUM_ROWS + 1u,
		STATUS_INVALID_ARGUMENT);
	rows[0].paired_scatter_gather = true;
	refused(&description, rows, 1, STATUS_INVALID_ARGUMENT);
	rows[0].operation = ORDERED_REDUCE_SCATTER;
	description.operation = ORDERED_REDUCE_SCATTER;
	description.paired_scatter_gather = true;
	rows[1] = rows[0];
	rows[1].operation = ORDERED_GATHER;
	expect(ordered_collective_routes_prepare_measured(
		       &description, rows, POLICY_ROWS, output) == STATUS_OK,
	       "matching paired mate is accepted");
	rows[1].host_reader_masks[0] ^= 1u << 1u;
	refused(&description, rows, POLICY_ROWS, STATUS_INVALID_ARGUMENT);
	rows[1] = rows[0];
	rows[1].operation = ORDERED_GATHER;
	rows[1].link_generations[0] = CMP_LINK;
	refused(&description, rows, POLICY_ROWS, STATUS_INVALID_ARGUMENT);
	rows[1] = rows[0];
	rows[1].operation = ORDERED_GATHER;
	expect(ordered_collective_routes_prepare_measured(
		       &description, rows, POLICY_ROWS, direct) == STATUS_OK,
	       "in-place table commit is safe after validation");
	for (uint32_t source = 0; source < RANKS; source++)
		expect(direct[source] == rows[0].host_reader_masks[source],
		       "in-place output");
	(void)printf("ordered_routes_measured_test: %s\n",
		     failures == 0 ? "passed" : "FAILED");
	return failures == 0 ? 0 : 1;
}
