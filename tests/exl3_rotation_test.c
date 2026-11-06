/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * exl3_rotation_test.c: --exl3_expert_rotation's host arithmetic (exl3_tc_records.h): for each rotation mode, rank,
 * target layer and expert, the width the rank's records give the expert (exl3_tc_width at the record's
 * rotation and intermediate) is the width of the chunk the loader takes (exl3_tc_chunk_of: chunks 512, 640,
 * 512, 640); every expert's four chunks lie on four distinct ranks; and the ranks' columns of a layer add up to the
 * expected (four-way: 576 on average on every rank; three-way: 512 on rank 0, 597.3 on ranks 1-3).
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "anemone/exl3_tc_records.h"

#define RANKS 4u
#define LAYERS 43u /* the walk's target layers and more: every rotation phase */
#define EXPERTS EXL3_TCP_EXPERTS
#define NARROW (EXL3_TC_MIXED_WIDTH - EXL3_TC_NARROWER)

static int failures;

static void expect(bool condition, const char *what, uint32_t mode, uint32_t rank, uint32_t layer, uint32_t expert)
{
	if (!condition) {
		failures++;
		if (failures < 10)
			printf("exl3_rotation_test: %s (mode %u rank %u layer %u expert %u)\n", what, mode, rank, layer,
			       expert);
	}
}

/* chunk c of the 2,304 columns: 512, 640, 512, 640 (exl3_layout's 128-column split, rank c's chunk) */
static uint32_t chunk_width(uint32_t chunk)
{
	return chunk % 2u == 0 ? NARROW : EXL3_TC_MIXED_WIDTH;
}

int main(void)
{
	static const uint32_t modes[] = { EXL3_TC_ROTATION_FOUR_WAY,
					  EXL3_TC_ROTATION_THREE_WAY };
	for (uint32_t index = 0; index < sizeof(modes) / sizeof(modes[0]); index++) {
		const uint32_t mode = modes[index];
		for (uint32_t layer = 0; layer < LAYERS; layer++) {
			uint64_t columns[RANKS] = { 0 };
			for (uint32_t expert = 0; expert < EXPERTS; expert++) {
				uint32_t held = 0;
				for (uint32_t rank = 0; rank < RANKS; rank++) {
					const uint32_t chunk = exl3_tc_chunk_of(
						mode, rank, layer, expert);
					const uint32_t rotation =
						exl3_tc_rotation_of(mode, rank,
								    layer);
					/* a record without rotation is the rank's one width: its chunk's for every expert */
					const uint32_t intermediate =
						rotation != 0 ?
							EXL3_TC_MIXED_WIDTH :
							chunk_width(chunk);
					expect(chunk < RANKS, "a chunk of four", mode, rank, layer, expert);
					expect(exl3_tc_width(intermediate,
							     rotation,
							     expert) ==
						       chunk_width(chunk),
					       "the record's width is the loaded chunk's",
					       mode, rank, layer, expert);
					expect(rotation != 0 ||
						       exl3_tc_chunk_of(
							       mode, rank,
							       layer,
							       0) == chunk,
					       "an unrotated rank holds one chunk",
					       mode, rank, layer, expert);
					held |= 1u << chunk;
					columns[rank] += chunk_width(chunk);
				}
				expect(held == (1u << RANKS) - 1u, "an expert's chunks on four distinct ranks", mode,
				       RANKS, layer, expert);
			}
			/* four-way: half the experts at each width on every rank; three-way: rank 0 all narrow, ranks
			 * 1-3 a third of the experts at each of chunks 1, 2 and 3 (640, 512, 640) */
			const uint64_t four_way =
				(uint64_t)EXPERTS / 2u *
				(NARROW + EXL3_TC_MIXED_WIDTH);
			const uint64_t three_way =
				(uint64_t)EXPERTS / EXL3_TC_THREE_WAY_RANKS *
				(NARROW + 2u * EXL3_TC_MIXED_WIDTH);
			for (uint32_t rank = 0; rank < RANKS; rank++) {
				const bool expected =
					mode == EXL3_TC_ROTATION_FOUR_WAY ?
						columns[rank] == four_way :
					rank == 0 ? columns[rank] ==
							    (uint64_t)EXPERTS *
								    NARROW :
						    columns[rank] == three_way;
				expect(expected, "the rank's columns of the layer", mode, rank, layer, EXPERTS);
			}
		}
	}
	printf("exl3_rotation_test %s\n", failures == 0 ? "passed" : "failed");
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
