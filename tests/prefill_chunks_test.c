/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Exercise the real exact planner against prepared row inventories. Changing a whole-chunk cap must leave every
 * tail program, its starting row and its final thin remainder unchanged, including all granule boundary residues. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "engine_internal.h"
#include "step_driver.h"

#define GRANULE MODEL_FULL_CONE_LEAST_ROWS
#define MOST_ROWS 8192u
#define LIMIT_SOURCES 2u
#define MOST_PROGRAMS 32u
#define MOST_TAIL_STEPS 16u
#define TAIL_FROM (GRANULE * 2u)
#define REQUIRE(condition)                                                 \
	do {                                                               \
		if (!(condition)) {                                        \
			fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
				#condition);                               \
			exit(1);                                           \
		}                                                          \
	} while (0)

struct tail_plan {
	uint32_t first, remainder, count, rows[MOST_TAIL_STEPS];
};

static struct tail_plan tail_of(struct inference_engine *engine,
				enum program_kind kind, uint32_t rows,
				uint32_t cap, bool has_tail)
{
	struct tail_plan tail = { 0 };
	const uint32_t end = rows;
	const uint32_t tail_rows = !has_tail	    ? 0 :
				   rows < TAIL_FROM ? rows :
						      GRANULE + rows % GRANULE;
	engine->description.prefill_chunk_rows = cap;
	engine->prefill_chunk_limit =
		engine->description.prefill_decode_rows != 0 ? cap : 0;
	tail.first = UINT32_MAX;
	while (rows != 0) {
		const uint32_t chunk =
			engine_exact_chunk_rows(engine, kind, rows, has_tail);
		if (rows <= tail_rows && tail.first == UINT32_MAX)
			tail.first = end - rows;
		if (chunk == 0)
			break;
		REQUIRE(chunk <= rows);
		if (rows > tail_rows) {
			REQUIRE(chunk % GRANULE == 0 &&
				chunk <= rows - tail_rows);
			REQUIRE(cap == 0 || chunk <= cap);
		} else {
			REQUIRE(tail.count < MOST_TAIL_STEPS);
			tail.rows[tail.count++] = chunk;
		}
		rows -= chunk;
	}
	REQUIRE(rows <= tail_rows);
	if (tail.first == UINT32_MAX)
		tail.first = end - rows;
	tail.remainder = rows;
	return tail;
}

int main(void)
{
	static const uint32_t sizes[] = { 2048u, 1792u, 1536u, 1280u, 1024u,
					  768u,	 512u,	256u,  448u,  320u,
					  128u,	 64u,	56u,   48u,   40u,
					  32u,	 24u,	16u };
	static const uint32_t caps[] = { GRANULE, GRANULE * 2u, GRANULE * 3u,
					 GRANULE * 4u };
	static const uint32_t tails[] = { 128u, 320u, 448u };
	struct execution_program programs[MOST_PROGRAMS] = { 0 };
	const struct execution_program *pointers[MOST_PROGRAMS];
	struct step_driver driver = { .programs = pointers };
	struct inference_engine engine = { .driver = &driver };
	for (uint32_t index = 0; index < sizeof(sizes) / sizeof(sizes[0]);
	     index++) {
		programs[index].kind = PROGRAM_FULL;
		programs[index].row_count = sizes[index];
		pointers[index] = &programs[index];
		driver.program_count++;
	}
	static const enum program_kind kinds[] = { PROGRAM_WIDE, PROGRAM_FULL };
	const uint32_t kind_count = sizeof(kinds) / sizeof(kinds[0]);
	for (uint32_t test_case = 0; test_case < kind_count * LIMIT_SOURCES;
	     test_case++) {
		const uint32_t kind = test_case % kind_count;
		engine.description.prefill_decode_rows =
			test_case < kind_count ? 0u : GRANULE;
		for (uint32_t index = 0; index < driver.program_count; index++)
			programs[index].kind = kinds[kind];
		for (uint32_t tail = 0; tail < sizeof(tails) / sizeof(tails[0]);
		     tail++) {
			engine.description.prefill_tail_rows = tails[tail];
			for (uint32_t rows = 0; rows <= MOST_ROWS; rows++) {
				const struct tail_plan original = tail_of(
					&engine, kinds[kind], rows, 0, true);
				for (uint32_t cap = 0;
				     cap < sizeof(caps) / sizeof(caps[0]);
				     cap++) {
					const struct tail_plan limited =
						tail_of(&engine, kinds[kind],
							rows, caps[cap], true);
					REQUIRE(original.first ==
							limited.first &&
						original.remainder ==
							limited.remainder &&
						original.count ==
							limited.count);
					for (uint32_t step = 0;
					     step < original.count; step++)
						REQUIRE(original.rows[step] ==
							limited.rows[step]);
				}
			}
		}
		for (uint32_t rows = 0; rows <= MOST_ROWS; rows += GRANULE)
			for (uint32_t cap = 0;
			     cap < sizeof(caps) / sizeof(caps[0]); cap++) {
				const struct tail_plan front =
					tail_of(&engine, kinds[kind], rows,
						caps[cap], false);
				REQUIRE(front.first == rows &&
					front.remainder == 0 &&
					front.count == 0);
			}
	}
	puts("prefill_chunks_test: preserved tails for 8193 sizes, 3 tails, 4 caps and both chunk kinds;"
	     " front granules and both static/dynamic limit sources too");
	return 0;
}
