/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* backend_cuda_bus_order_test: how the CUDA backend numbers the visible cards, without a card (backend_seam.h's
 * backend_cuda_bus_order, which backend_initialise runs on the driver's PCI attributes): PCI address order whatever the
 * driver's (reversed, the fastest card first, as CUDA_DEVICE_ORDER or a CUDA_VISIBLE_DEVICES list gives it), cards of
 * one address in the driver's order, the domain before the bus and the bus before the device (a later bus's low device
 * after an earlier bus's high one); the startup line only when a card moved, the driver's ordinals in the cards' order.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/backend_seam.h"

#define UNWRITTEN_BYTE \
	'x' /* the line before the call: it must come back terminated */

static unsigned failures, cases_run;

/* The driver's places (domain, bus, device, ordinal), its ordinals once sorted, and the line. */
static const struct backend_cuda_bus_place reversed[] = { { 0, 0x05, 0, 0 },
							  { 0, 0x04, 0, 1 },
							  { 0, 0x03, 0, 2 },
							  { 0, 0x02, 0, 3 },
							  { 0, 0x01, 0, 4 } };
static const int reversed_sorted[] = { 4, 3, 2, 1, 0 };
/* the measured machine under FASTEST_FIRST: the RTX PRO 6000, third on the bus, is the driver's ordinal 0 */
static const struct backend_cuda_bus_place fastest_first[] = {
	{ 0, 0x03, 0, 0 },
	{ 0, 0x01, 0, 1 },
	{ 0, 0x02, 0, 2 },
	{ 0, 0x04, 0, 3 },
	{ 0, 0x05, 0, 4 }
};
static const int fastest_first_sorted[] = { 1, 2, 0, 3, 4 };
#define FASTEST_FIRST_LINE                                                                                    \
	"cards: numbered by PCI address, not in the driver's order: cards 0 to 4 are its ordinals 1,2,0,3,4 " \
	"(--card_order orders the cards)\n"
static const struct backend_cuda_bus_place in_order[] = { { 0, 0x01, 0, 0 },
							  { 0, 0x02, 0, 1 },
							  { 0, 0x03, 0, 2 },
							  { 0, 0x04, 0, 3 },
							  { 0, 0x05, 0, 4 } };
static const int in_order_sorted[] = { 0, 1, 2, 3, 4 };
static const struct backend_cuda_bus_place ties[] = { { 0, 0x02, 0, 0 },
						      { 0, 0x01, 0, 1 },
						      { 0, 0x02, 0, 2 },
						      { 0, 0x01, 0, 3 } };
static const int ties_sorted[] = { 1, 3, 0, 2 };
static const struct backend_cuda_bus_place domains[] = { { 1, 0x00, 0x00, 0 },
							 { 0, 0xff, 0x1f, 1 },
							 { 0, 0xff, 0x00, 2 } };
static const int domains_sorted[] = { 2, 1, 0 };
/* a later bus with a low device, an earlier bus with a high one: the bus decides */
static const struct backend_cuda_bus_place bus_before_device[] = {
	{ 0, 0x02, 0x00, 0 },
	{ 0, 0x01, 0x1f, 1 }
};
static const int bus_before_device_sorted[] = { 1, 0 };

/* Sorts a copy of the driver's places and expects the cards' ordinals, whether a card moved and the line. */
static void order_case(const struct backend_cuda_bus_place *driver,
		       uint32_t count, const int *sorted, bool moved,
		       const char *expected_line, const char *what)
{
	struct backend_cuda_bus_place places[BACKEND_MAXIMUM_CARDS] = { 0 };
	char line[BACKEND_CUDA_ORDER_LINE_BYTES];
	memset(line, UNWRITTEN_BYTE, sizeof(line));
	memcpy(places, driver, count * sizeof(places[0]));
	bool held = backend_cuda_bus_order(count, places, line) == moved &&
		    memchr(line, '\0', sizeof(line)) != NULL &&
		    strcmp(line, expected_line) == 0;
	for (uint32_t index = 0; index < count; index++)
		held = held && places[index].ordinal == sorted[index];
	cases_run++;
	if (!held) {
		failures++;
		(void)fprintf(stderr, "FAILED: %s\n", what);
	}
}

int main(void)
{
	order_case(
		reversed, sizeof(reversed) / sizeof(reversed[0]),
		reversed_sorted, true,
		"cards: numbered by PCI address, not in the driver's order: cards 0 to 4 are its ordinals 4,3,2,1,0 "
		"(--card_order orders the cards)\n",
		"cards the driver numbers in reverse are numbered in PCI address order, and the line says so");
	order_case(
		fastest_first, sizeof(fastest_first) / sizeof(fastest_first[0]),
		fastest_first_sorted, true, FASTEST_FIRST_LINE,
		"the fastest card first (the driver's default) is numbered in PCI address order, and the line says so");
	order_case(
		in_order, sizeof(in_order) / sizeof(in_order[0]),
		in_order_sorted, false, "",
		"cards the driver numbers in PCI address order keep their numbers, and no line is printed");
	order_case(
		ties, sizeof(ties) / sizeof(ties[0]), ties_sorted, true,
		"cards: numbered by PCI address, not in the driver's order: cards 0 to 3 are its ordinals 1,3,0,2 "
		"(--card_order orders the cards)\n",
		"cards of one PCI address keep the driver's order");
	order_case(
		domains, sizeof(domains) / sizeof(domains[0]), domains_sorted,
		true,
		"cards: numbered by PCI address, not in the driver's order: cards 0 to 2 are its ordinals 2,1,0 "
		"(--card_order orders the cards)\n",
		"the PCI domain orders before the bus, and the device within a bus");
	order_case(
		bus_before_device,
		sizeof(bus_before_device) / sizeof(bus_before_device[0]),
		bus_before_device_sorted, true,
		"cards: numbered by PCI address, not in the driver's order: cards 0 to 1 are its ordinals 1,0 "
		"(--card_order orders the cards)\n",
		"the bus orders before the device");
	order_case(in_order, 1u, in_order_sorted, false, "",
		   "one card is card 0, and no line is printed");
	order_case(in_order, 0u, in_order_sorted, false, "",
		   "no card: nothing moves, and no line is printed");
	(void)printf("backend_cuda_bus_order: %s (%u cases)\n",
		     failures == 0 ? "PASS" : "FAIL", cases_run);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
