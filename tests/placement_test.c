/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * placement_test: placement_fill_in_order on packs that test_pack_build writes and test_memory_storage
 * serves. Hand-computed placements, every refusal with its status, and a property loop that compares the engine
 * with the rule of placement.h written the long way, over random plans and budgets drawn from a fixed seed. Then
 * placement_from_layout_text on a pack laid out as a model's: hand-computed placements, to the byte of a
 * budget and a byte short, and every refusal of placement.h with its status and its one error line.
 *
 * While the error line of a refused placement is read back, descriptor 2 is a temporary file whose name the test
 * prints first. The test removes it at its end, so the file is still there, with any sanitizer report, only when
 * the test died in such a section.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "log_expectation.h"
#include "anemone/model_seam.h"
#include "anemone/pack.h"
#include "anemone/placement.h"
#include "anemone/status.h"
#include "anemone/storage_seam.h"

#include "memory_storage.h"
#include "pack_builder.h"

#define TEST_TIME_LIMIT_SECONDS 300              /* the alarm ends the test here: a search that never ends fails it */
#define MOST_UNITS 24                            /* of a plan of this test */
#define MOST_CASE_UNITS 8                        /* of a hand-computed case; a unit with no extent ends its list */
#define MOST_CASE_CARDS 4
#define CONTENT_SEED 0x706C6163656D656Eull       /* any value: the placement never reads an extent */
#define UNTOUCHED_OUTPUT_VALUE 0x5A5A5A5Au       /* no card and not host memory */
#define CAPTURE_CAPACITY 4096                    /* more than the one line a refused placement logs */
#define TEXT_CAPACITY 160                        /* more than the longest text the test formats */
#define REFUSAL_BLOCKS_PER_CARD 4                /* holds the two units of the refusal pack */

#define RANDOM_PLAN_COUNT 48
#define BUDGET_SETS_PER_PLAN 24
#define MOST_RANDOM_CARDS 4
#define MOST_EXTENTS_PER_RANDOM_UNIT 2
#define MOST_BLOCKS_PER_RANDOM_EXTENT 3
#define KIND_ROLL_SIDES 10                       /* of 10 rolls: 3 a stage unit, 5 on demand, 2 host-only */
#define STAGE_ROLLS 3
#define ON_DEMAND_ROLLS 5
#define BUDGET_SCALES 8                          /* the cards hold on average from 1/4 of a plan to twice it */
#define BUDGET_SCALE_DIVISOR 2
#define GENERATOR_SEED 0x0123456789ABCDEFull
#define GENERATOR_MULTIPLIER 6364136223846793005ull
#define GENERATOR_INCREMENT 1442695040888963407ull
#define GENERATOR_LOW_HALF 4294967296ull         /* 2^32: the low bits of such a generator repeat early */
#define LAYOUT_CARDS 2u                          /* of the layout cases but where one says otherwise */
#define LAYOUT_ROOMY_BLOCKS 64u                  /* a budget that holds every unit of the layout pack */
#define LAYOUT_CARD_ZERO_BLOCKS 14u              /* what the first hand-computed text puts on each card */
#define LAYOUT_CARD_ONE_BLOCKS 7u
#define LAYOUT_TEXT_LONGEST_BYTES \
	(PLACEMENT_MOST_TEXT_BYTES + 2u) /* one byte too many, and the NUL */
#define LAYOUT_LAST_CARD (BACKEND_MAXIMUM_CARDS - 1u)
#define LAYOUT_LAYER_ONE_UNIT 5u                 /* of the layout pack: layer 1's stage unit, its host-only unit and */
#define LAYOUT_LAYER_ONE_HOST_ONLY_UNIT 7u       /* its second on-demand unit */
#define LAYOUT_LAYER_ONE_SECOND_UNIT 8u

#define IN_HOST PLACEMENT_IN_HOST_MEMORY
#define BLOCKS(block_count) ((uint64_t)(block_count) * PACK_BLOCK_BYTES)
#define STAGE_UNIT(block_count) \
	{ PLACEMENT_UNIT_STAGE, PACK_NO_LAYER, 1, (block_count) }
#define ON_DEMAND_UNIT(block_count) \
	{ PLACEMENT_UNIT_ON_DEMAND, PACK_NO_LAYER, 1, (block_count) }
#define HOST_ONLY_UNIT(block_count) \
	{ PLACEMENT_UNIT_HOST_ONLY, PACK_NO_LAYER, 1, (block_count) }

/* A pack as a test holds it: its bytes, the reader that serves them and the open pack. */
struct opened_pack {
    void *pack_bytes;
    struct storage_reader *reader;
    struct pack_record *pack;
};

/* What one call of the placement gave. The entry after the last unit must stay as it was. */
struct placement_result {
	result_status status;
	uint32_t on_demand_units_per_owner;
	uint32_t card_of_unit[MOST_UNITS + 1];
};

struct placement_case {
    const char *what;
    struct test_pack_unit_plan units[MOST_CASE_UNITS];
    uint64_t budget_bytes_of_card[MOST_CASE_CARDS];
    uint32_t card_count;
    result_status expected_status;
    uint32_t expected_on_demand_units_per_owner;
    uint32_t expected_card_of_unit[MOST_CASE_UNITS];
};

/* The budgets are in bytes, a unit is its kind and its blocks. Every placement was worked out by hand from the rule
 * of placement.h; the comment of a case says what its count has to weigh. */
static const struct placement_case placement_cases[] = {
	/* the count 2 asks for 4 + 4 blocks */
	{ .what = "one card that holds everything to the byte",
	  .units = { STAGE_UNIT(2), ON_DEMAND_UNIT(1), ON_DEMAND_UNIT(1),
		     STAGE_UNIT(3), ON_DEMAND_UNIT(1) },
	  .budget_bytes_of_card = { BLOCKS(8) },
	  .card_count = 1,
	  .expected_status = STATUS_OK,
	  .expected_on_demand_units_per_owner = 2,
	  .expected_card_of_unit = { 0, 0, 0, 0, 0 } },
	/* the count 1 asks for 3 + 4 blocks */
	{ .what = "one card a byte short of everything",
	  .units = { STAGE_UNIT(2), ON_DEMAND_UNIT(1), ON_DEMAND_UNIT(1),
		     STAGE_UNIT(3), ON_DEMAND_UNIT(1) },
	  .budget_bytes_of_card = { BLOCKS(8) - 1 },
	  .card_count = 1,
	  .expected_status = STATUS_OK,
	  .expected_on_demand_units_per_owner = 1,
	  .expected_card_of_unit = { 0, 0, IN_HOST, 0, 0 } },
	/* the count 0 asks for 2 + 3 blocks, the count 1 for 7 */
	{ .what = "one card that holds the stages to the byte and nothing else",
	  .units = { STAGE_UNIT(2), ON_DEMAND_UNIT(1), ON_DEMAND_UNIT(1),
		     STAGE_UNIT(3), ON_DEMAND_UNIT(1) },
	  .budget_bytes_of_card = { BLOCKS(5) },
	  .card_count = 1,
	  .expected_status = STATUS_OK,
	  .expected_on_demand_units_per_owner = 0,
	  .expected_card_of_unit = { 0, IN_HOST, IN_HOST, 0, IN_HOST } },
	{ .what = "one card a byte short of the stages",
	  .units = { STAGE_UNIT(2), ON_DEMAND_UNIT(1), ON_DEMAND_UNIT(1),
		     STAGE_UNIT(3), ON_DEMAND_UNIT(1) },
	  .budget_bytes_of_card = { BLOCKS(5) - 1 },
	  .card_count = 1,
	  .expected_status = STATUS_OUT_OF_MEMORY },
	/* the last stage would fit in the block the first card has left, and must not go back to it */
	{ .what = "two cards: a stage moves to the next card and none returns",
	  .units = { STAGE_UNIT(3), STAGE_UNIT(3), STAGE_UNIT(1) },
	  .budget_bytes_of_card = { BLOCKS(4), BLOCKS(4) },
	  .card_count = 2,
	  .expected_status = STATUS_OK,
	  .expected_on_demand_units_per_owner = 0,
	  .expected_card_of_unit = { 0, 1, 1 } },
	/* the count 1 asks for 4, 4 and 1 blocks: 4 on the first card, 5 on the second; the count 2 asks for 6 twice */
	{ .what = "two cards: the count that fits spreads the stages",
	  .units = { STAGE_UNIT(2), ON_DEMAND_UNIT(2), ON_DEMAND_UNIT(2),
		     STAGE_UNIT(2), ON_DEMAND_UNIT(2), ON_DEMAND_UNIT(2),
		     STAGE_UNIT(1) },
	  .budget_bytes_of_card = { BLOCKS(5), BLOCKS(9) },
	  .card_count = 2,
	  .expected_status = STATUS_OK,
	  .expected_on_demand_units_per_owner = 1,
	  .expected_card_of_unit = { 0, 0, IN_HOST, 1, 1, IN_HOST, 1 } },
	{ .what = "four cards: one without budget and one too small are passed over",
	  .units = { STAGE_UNIT(2), STAGE_UNIT(3) },
	  .budget_bytes_of_card = { 0, BLOCKS(2), BLOCKS(1), BLOCKS(4) },
	  .card_count = 4,
	  .expected_status = STATUS_OK,
	  .expected_on_demand_units_per_owner = 0,
	  .expected_card_of_unit = { 1, 3 } },
	/* the stages own 3, 1 and 0: the count 1 asks for 2 + 2 + 1 blocks, the count 2 for 3 + 2 + 1, the count 3 for 7 */
	{ .what = "stages that own different numbers, count 1",
	  .units = { STAGE_UNIT(1), ON_DEMAND_UNIT(1), ON_DEMAND_UNIT(1),
		     ON_DEMAND_UNIT(1), STAGE_UNIT(1), ON_DEMAND_UNIT(1),
		     STAGE_UNIT(1) },
	  .budget_bytes_of_card = { BLOCKS(5) },
	  .card_count = 1,
	  .expected_status = STATUS_OK,
	  .expected_on_demand_units_per_owner = 1,
	  .expected_card_of_unit = { 0, 0, IN_HOST, IN_HOST, 0, 0, 0 } },
	{ .what = "stages that own different numbers, count 2",
	  .units = { STAGE_UNIT(1), ON_DEMAND_UNIT(1), ON_DEMAND_UNIT(1),
		     ON_DEMAND_UNIT(1), STAGE_UNIT(1), ON_DEMAND_UNIT(1),
		     STAGE_UNIT(1) },
	  .budget_bytes_of_card = { BLOCKS(6) },
	  .card_count = 1,
	  .expected_status = STATUS_OK,
	  .expected_on_demand_units_per_owner = 2,
	  .expected_card_of_unit = { 0, 0, 0, IN_HOST, 0, 0, 0 } },
	{ .what = "stages that own different numbers, the largest count",
	  .units = { STAGE_UNIT(1), ON_DEMAND_UNIT(1), ON_DEMAND_UNIT(1),
		     ON_DEMAND_UNIT(1), STAGE_UNIT(1), ON_DEMAND_UNIT(1),
		     STAGE_UNIT(1) },
	  .budget_bytes_of_card = { BLOCKS(7) },
	  .card_count = 1,
	  .expected_status = STATUS_OK,
	  .expected_on_demand_units_per_owner = 3,
	  .expected_card_of_unit = { 0, 0, 0, 0, 0, 0, 0 } },
	/* the 6 blocks of the host-only units weigh nothing: the count 2 asks for 3 + 1 blocks */
	{ .what = "host-only units before, between and after",
	  .units = { HOST_ONLY_UNIT(2), STAGE_UNIT(1), ON_DEMAND_UNIT(1),
		     HOST_ONLY_UNIT(3), ON_DEMAND_UNIT(1), STAGE_UNIT(1),
		     HOST_ONLY_UNIT(1) },
	  .budget_bytes_of_card = { BLOCKS(4) },
	  .card_count = 1,
	  .expected_status = STATUS_OK,
	  .expected_on_demand_units_per_owner = 2,
	  .expected_card_of_unit = { IN_HOST, 0, 0, IN_HOST, 0, 0, IN_HOST } },
	/* the first on-demand unit of the stage is the one before the host-only unit */
	{ .what = "a host-only unit between the on-demand units of a stage",
	  .units = { HOST_ONLY_UNIT(2), STAGE_UNIT(1), ON_DEMAND_UNIT(1),
		     HOST_ONLY_UNIT(3), ON_DEMAND_UNIT(1), STAGE_UNIT(1),
		     HOST_ONLY_UNIT(1) },
	  .budget_bytes_of_card = { BLOCKS(3) },
	  .card_count = 1,
	  .expected_status = STATUS_OK,
	  .expected_on_demand_units_per_owner = 1,
	  .expected_card_of_unit = { IN_HOST, 0, 0, IN_HOST, IN_HOST, 0,
				     IN_HOST } },
	/* a stage of two extents of two blocks and an on-demand unit of two extents of one */
	{ .what = "units of several extents, to the byte",
	  .units = { { PLACEMENT_UNIT_STAGE, 0, 2, 2 },
		     { PLACEMENT_UNIT_ON_DEMAND, 0, 2, 1 } },
	  .budget_bytes_of_card = { BLOCKS(6) },
	  .card_count = 1,
	  .expected_status = STATUS_OK,
	  .expected_on_demand_units_per_owner = 1,
	  .expected_card_of_unit = { 0, 0 } },
	{ .what = "units of several extents, a byte short",
	  .units = { { PLACEMENT_UNIT_STAGE, 0, 2, 2 },
		     { PLACEMENT_UNIT_ON_DEMAND, 0, 2, 1 } },
	  .budget_bytes_of_card = { BLOCKS(6) - 1 },
	  .card_count = 1,
	  .expected_status = STATUS_OK,
	  .expected_on_demand_units_per_owner = 0,
	  .expected_card_of_unit = { 0, IN_HOST } },
	/* 6 blocks of budget for 6 blocks of stages, but no stage is cut in two */
	{ .what = "two cards that hold the stages only if one were cut",
	  .units = { STAGE_UNIT(2), STAGE_UNIT(2), STAGE_UNIT(2) },
	  .budget_bytes_of_card = { BLOCKS(3), BLOCKS(3) },
	  .card_count = 2,
	  .expected_status = STATUS_OUT_OF_MEMORY },
	/* the search tries 3, which fits, and 4, which does not */
	{ .what = "one stage that owns five and keeps three",
	  .units = { STAGE_UNIT(1), ON_DEMAND_UNIT(1), ON_DEMAND_UNIT(1),
		     ON_DEMAND_UNIT(1), ON_DEMAND_UNIT(1), ON_DEMAND_UNIT(1) },
	  .budget_bytes_of_card = { BLOCKS(4) },
	  .card_count = 1,
	  .expected_status = STATUS_OK,
	  .expected_on_demand_units_per_owner = 3,
	  .expected_card_of_unit = { 0, 0, 0, 0, IN_HOST, IN_HOST } },
};

static const struct test_pack_unit_plan refusal_units[] = {STAGE_UNIT(1), ON_DEMAND_UNIT(1)};

/* A pack laid out as a model's (pack.h: in the order units run), its units' blocks in brackets: vision [1], the
 * embedding [2], layers 0 to 2 [3] of two on-demand units [1] each (a host-only unit [4] between those of layer 1),
 * the head [2], a draft head of two blocks [1], the first with two on-demand units [1], then host-only units: an
 * n-gram table of layer 1 [2] and constants [1]. */
#define LAYOUT_STAGE(layer, block_count) \
	{ PLACEMENT_UNIT_STAGE, (layer), 1, (block_count) }
#define LAYOUT_ON_DEMAND(layer) { PLACEMENT_UNIT_ON_DEMAND, (layer), 1, 1 }
#define LAYOUT_HOST_ONLY(layer, block_count) \
	{ PLACEMENT_UNIT_HOST_ONLY, (layer), 1, (block_count) }
#define NO_LAYER PACK_NO_LAYER
static const struct test_pack_unit_plan layout_units[] = {
    LAYOUT_STAGE(NO_LAYER, 1), LAYOUT_STAGE(NO_LAYER, 2), LAYOUT_STAGE(0, 3), LAYOUT_ON_DEMAND(0), LAYOUT_ON_DEMAND(0),
    LAYOUT_STAGE(1, 3), LAYOUT_ON_DEMAND(1), LAYOUT_HOST_ONLY(1, 4), LAYOUT_ON_DEMAND(1), LAYOUT_STAGE(2, 3),
    LAYOUT_ON_DEMAND(2), LAYOUT_ON_DEMAND(2), LAYOUT_STAGE(NO_LAYER, 2), LAYOUT_STAGE(NO_LAYER, 1),
    LAYOUT_ON_DEMAND(NO_LAYER), LAYOUT_ON_DEMAND(NO_LAYER), LAYOUT_STAGE(NO_LAYER, 1), LAYOUT_HOST_ONLY(1, 2),
    LAYOUT_HOST_ONLY(NO_LAYER, 1)};
#define LAYOUT_UNITS (sizeof(layout_units) / sizeof(layout_units[0]))
/* A pack without vision whose layer 0 has two stage units and layer 1 none, and whose layer 3 runs before layer 2. */
static const struct test_pack_unit_plan twice_units[] = {LAYOUT_STAGE(NO_LAYER, 1), LAYOUT_STAGE(0, 1),
                                                         LAYOUT_STAGE(0, 1), LAYOUT_STAGE(3, 1), LAYOUT_STAGE(2, 1),
                                                         LAYOUT_STAGE(NO_LAYER, 1)};
#define TWICE_UNITS ((uint32_t)(sizeof(twice_units) / sizeof(twice_units[0])))

/* Every name on the cards of a two-card layout, layer 1 keeping its first expert (stated before its card) and the
 * draft head none: 14 blocks on card 0, 7 on card 1. */
static const char *const layout_named_text = "experts kept: layers 1: 1; vision: card 1; embedding: card 0; "
                                             "layers 0, 2: card 0; layers 1: card 1; head: card 1; "
                                             "draft head: card 0; experts kept: draft head: 0";
static const uint32_t layout_named_cards[] = {1, 0, 0, 0, 0, 1, 1, IN_HOST, IN_HOST, 0, 0, 0, 1, 0, IN_HOST,
                                              IN_HOST, 0, IN_HOST, IN_HOST};
/* The units but the layers, named once on card 0: what a refusal case adds behind its layers. */
#define LAYOUT_OTHERS "; vision: card 0; embedding: card 0; head: card 0; draft head: card 0"
#define LAYOUT_NOT_ONCE "names a layer that the stage units do not hold once, in order\n"
#define LAYOUT_FORM "is not \"<units>: card <c>\" or \"experts kept: <units>: <n>\"\n"

/* A text refused, and the end of its one error line. */
static const struct {
    const char *text;
    const char *line_end;
} layout_refusals[] = {
    {"", "statement 1 " LAYOUT_FORM},
    {"layers : card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers 0..: card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers 2..1: card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers 0..1..2: card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers 0, : card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers 0,1: card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers 0 : card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {" layers 0..2: card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"Layers 0..2: card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers -1: card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers +1: card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers 0x1: card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers 4294967295: card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers 18446744073709551616: card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers 0..2:card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers 0..2: card" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers 0..2: card x" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers 0..2: card 1x" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers 0..2: card 0, 1 experts" LAYOUT_OTHERS, "statement 1 " LAYOUT_FORM},
    {"layers 0..2: card 0; experts kept: layers 0..2:  1" LAYOUT_OTHERS, "statement 2 " LAYOUT_FORM},
    {"layers 0..2: card 0; experts kept: layers 0..2: x" LAYOUT_OTHERS, "statement 2 " LAYOUT_FORM},
    {"layers 0..2: card 0; experts kept: layers 0..2: card 0" LAYOUT_OTHERS, "statement 2 " LAYOUT_FORM},
    {"layers 0..2: card 0; experts kept: layers 0..2: 1 experts" LAYOUT_OTHERS, "statement 2 " LAYOUT_FORM},
    {"layers 0..2: card 0; experts kept: layers 0..2" LAYOUT_OTHERS, "statement 2 " LAYOUT_FORM},
    {"layers 0..2: card 0; experts kept:layers 0..2: 1" LAYOUT_OTHERS, "statement 2 " LAYOUT_FORM},
    {"layers 0..2: card 0; experts kept: experts kept: layers 0..2: 1" LAYOUT_OTHERS, "statement 2 " LAYOUT_FORM},
    {"layers 0..2: card 0;vision: card 0", "statement 1 " LAYOUT_FORM},
    {"layers 0..2: card 0" LAYOUT_OTHERS ";", "statement 5 " LAYOUT_FORM},
    {"layers 0..2: card 0" LAYOUT_OTHERS "; ", "statement 5 " LAYOUT_FORM},
    {"layers 0..2: card 0" LAYOUT_OTHERS " ", "statement 5 " LAYOUT_FORM},
    {"vision : card 0", "statement 1 " LAYOUT_FORM},
    {"heads: card 0", "statement 1 " LAYOUT_FORM},
    {"draft: card 0", "statement 1 " LAYOUT_FORM},
    {"layers 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0: card 0", "statement 1 " LAYOUT_FORM},
    {"layers 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0: card 0",
     "statement 1 names a unit that a range or statement names already\n"},
    {"layers 0..1, 1..2: card 0" LAYOUT_OTHERS, "statement 1 names a unit that a range or statement names already\n"},
    {"layers 0..2: card 0; layers 2: card 1", "statement 2 names a unit that a range or statement names already\n"},
    {"head: card 0; head: card 1", "statement 2 names a unit that a range or statement names already\n"},
    {"experts kept: layers 1: 1; experts kept: layers 0..2: 2",
     "statement 2 names a unit that a range or statement names already\n"},
    {"experts kept: layers 0..1, 1..2: 1", "statement 1 names a unit that a range or statement names already\n"},
    {"layers 0..2: card 2" LAYOUT_OTHERS, "statement 1 names a card the load does not have\n"},
    {"layers 0..3: card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_NOT_ONCE},
    {"layers 3: card 0" LAYOUT_OTHERS, "statement 1 " LAYOUT_NOT_ONCE},
    {"experts kept: layers 3: 0", "statement 1 " LAYOUT_NOT_ONCE},
    {"layers 0..2: card 0; experts kept: layers 0..2: 3" LAYOUT_OTHERS,
     "statement 2 keeps more experts than a unit it names owns\n"},
    {"experts kept: embedding: 1", "statement 1 keeps more experts than a unit it names owns\n"},
    {"layers 0..2: card 0", "no statement gives the unit outside the layers numbered 0 a card\n"},
    {"layers 0, 2: card 0" LAYOUT_OTHERS, "no statement gives layer 1 a card\n"},
    {"layers 0, 2: card 0; experts kept: layers 1: 1" LAYOUT_OTHERS, "no statement gives layer 1 a card\n"},
    {"layers 0..2: card 0; vision: card 0; embedding: card 0; head: card 0",
     "no statement gives the unit outside the layers numbered 13 a card\n"},
};

static int failures = 0;
static char capture_path[] = "/tmp/anemone_placement_test_XXXXXX";
static int capture_descriptor = -1;              /* the temporary file that stands in for the error stream */
static int error_stream_descriptor = -1;         /* a duplicate of descriptor 2, kept to put it back */
static uint64_t generator_state = GENERATOR_SEED;
static uint32_t random_placements_refused = 0;   /* what the property loop met: printed at the end */
static uint32_t random_placements_with_no_unit_kept = 0;
static uint32_t random_placements_with_some_units_kept = 0;
static uint32_t random_placements_with_every_unit_kept = 0;

static void expect(bool condition, const char *where, const char *what)
{
    if (!condition) {
        (void)fprintf(stderr, "FAILED: %s: %s\n", where, what);
        failures++;
    }
}

/* For what the test cannot go on without. */
static _Noreturn void give_up(const char *what)
{
    (void)fprintf(stderr, "FAILED: %s\n", what);
    exit(EXIT_FAILURE);
}

/* From here on, what is logged goes to the temporary file, from its start. */
static void capture_begin(void)
{
    if (capture_descriptor < 0 || fflush(stderr) != 0 || ftruncate(capture_descriptor, 0) != 0 ||
        lseek(capture_descriptor, 0, SEEK_SET) != 0 || dup2(capture_descriptor, STDERR_FILENO) != STDERR_FILENO) {
        give_up("the error stream could not be pointed at the temporary file");
    }
}

/* Copies what was logged since capture_begin into text, as a string, and puts descriptor 2 back. */
static void capture_end(char *text, size_t text_capacity)
{
    ssize_t byte_count = -1;
    if (fflush(stderr) == 0) {
        byte_count = pread(capture_descriptor, text, text_capacity - 1, 0);
    }
    if (dup2(error_stream_descriptor, STDERR_FILENO) != STDERR_FILENO || byte_count < 0) {
        give_up("the temporary file could not be read back");
    }
    text[(size_t)byte_count] = '\0';
}

/* True when the text is one line of the log, an error, that ends in the message. */
static bool is_one_error_line(const char *text, const char *message_end)
{
	if (!LOGGING_HOOKS_ENABLED) {
		return test_log_is_expected(text, message_end, true);
	}
    const size_t text_length = strlen(text);
    const size_t end_length = strlen(message_end);
    return text_length > end_length && strchr(text, '\n') == &text[text_length - 1] &&
           strstr(text, " anemone error ") != NULL && strcmp(&text[text_length - end_length], message_end) == 0;
}

/* A generator of the test's own, so that the plans are the same on every run and on every machine. */
static uint64_t next_random(void)
{
    generator_state = generator_state * GENERATOR_MULTIPLIER + GENERATOR_INCREMENT;
    return generator_state / GENERATOR_LOW_HALF;
}

/* A number from 0 to bound - 1; bound is above 0. */
static uint64_t random_below(uint64_t bound)
{
    const uint64_t high_half = next_random();
    return (high_half * GENERATOR_LOW_HALF + next_random()) % bound;
}

/* Builds the pack of a plan and opens it through the memory storage. */
static void open_pack(const struct test_pack_unit_plan *units, uint32_t unit_count, struct opened_pack *opened)
{
    static const char *const copy_paths[] = {"placement_test"};
    const struct test_pack_plan plan = {.model_name = "placement_test", .units = units, .unit_count = unit_count,
                                        .tensors_per_extent = 0, .content_seed = CONTENT_SEED};
    uint64_t pack_byte_count = 0;
    opened->pack_bytes = NULL;
    opened->reader = NULL;
    opened->pack = NULL;
    if (test_pack_build(&plan, &opened->pack_bytes, &pack_byte_count) !=
	STATUS_OK) {
	    give_up("a pack of the test could not be built");
    }
    test_memory_storage_reset();
    test_memory_storage_set_pack(opened->pack_bytes, pack_byte_count);
    if (test_memory_storage.reader_open(copy_paths, 1, &opened->reader) !=
		STATUS_OK ||
	pack_open_entry(&test_memory_storage, opened->reader, &opened->pack) !=
		STATUS_OK) {
	    give_up("a pack of the test could not be opened");
    }
    if (pack_header_of(opened->pack)->unit_count != unit_count) {
	    give_up("a pack of the test does not hold the units of its plan");
    }
}

static void close_pack(struct opened_pack *opened)
{
	pack_close(opened->pack);
	test_memory_storage.reader_close(opened->reader);
	test_memory_storage_reset();
	free(opened->pack_bytes);
}

static uint64_t unit_byte_count(const struct test_pack_unit_plan *unit)
{
	return (uint64_t)unit->extent_count * unit->blocks_per_extent *
	       PACK_BLOCK_BYTES;
}

/* The most on-demand units one stage unit of the plan owns. */
static uint32_t largest_owned_count(const struct test_pack_unit_plan *units, uint32_t unit_count)
{
    uint32_t largest_count = 0;
    uint32_t owned_count = 0;
    for (uint32_t unit_index = 0; unit_index < unit_count; unit_index++) {
	    if (units[unit_index].unit_kind == PLACEMENT_UNIT_STAGE) {
		    owned_count = 0;
	    }
	    if (units[unit_index].unit_kind == PLACEMENT_UNIT_ON_DEMAND) {
		    owned_count++;
	    }
	if (owned_count > largest_count) {
            largest_count = owned_count;
        }
    }
    return largest_count;
}

/* The rule of placement.h written the long way for one candidate count, from the plan and not from the pack: the
 * oracle of this test. True when every stage unit finds a card; card_of_unit_out is then the placement. When not,
 * *bytes_lacking_out is what the last card would have to hold more for the candidate to fit. */
static bool reference_fill(const struct test_pack_unit_plan *units, uint32_t unit_count,
                           const uint64_t *budget_bytes_of_card, uint32_t card_count, uint32_t candidate_count,
                           uint32_t *card_of_unit_out, uint64_t *bytes_lacking_out)
{
	uint64_t bytes_used_of_card[BACKEND_MAXIMUM_CARDS] = { 0 };
	uint64_t bytes_lacking = 0;
	uint32_t present_card = 0;
	uint32_t owner_card = IN_HOST;
	uint32_t owned_units_seen = 0;
	for (uint32_t unit_index = 0; unit_index < unit_count; unit_index++) {
		card_of_unit_out[unit_index] = IN_HOST;
		if (units[unit_index].unit_kind == PLACEMENT_UNIT_ON_DEMAND) {
			if (owned_units_seen < candidate_count) {
				card_of_unit_out[unit_index] = owner_card;
			}
			owned_units_seen++;
		}
		if (units[unit_index].unit_kind == PLACEMENT_UNIT_STAGE) {
			uint64_t bytes_asked =
				unit_byte_count(&units[unit_index]);
			uint32_t units_asked_for = 0;
			for (uint32_t later_index = unit_index + 1;
			     later_index < unit_count &&
			     units[later_index].unit_kind !=
				     PLACEMENT_UNIT_STAGE;
			     later_index++) {
				if (units[later_index].unit_kind ==
					    PLACEMENT_UNIT_ON_DEMAND &&
				    units_asked_for < candidate_count) {
					bytes_asked += unit_byte_count(
						&units[later_index]);
					units_asked_for++;
				}
			}
			while (present_card < card_count &&
			       bytes_asked >
				       budget_bytes_of_card[present_card] -
					       bytes_used_of_card[present_card]) {
				present_card++;
			}
			if (present_card < card_count) {
				bytes_used_of_card[present_card] += bytes_asked;
			} else if (bytes_lacking == 0 && card_count > 0) {
				bytes_lacking =
					bytes_asked -
					(budget_bytes_of_card[card_count - 1] -
					 bytes_used_of_card[card_count - 1]);
			} else {
				bytes_lacking += bytes_asked;
			}
			owner_card = present_card;
			owned_units_seen = 0;
			card_of_unit_out[unit_index] = present_card;
		}
    }
    *bytes_lacking_out = bytes_lacking;
    return bytes_lacking == 0;
}

static void clear_result(struct placement_result *result)
{
	result->status = STATUS_OK;
	result->on_demand_units_per_owner = UNTOUCHED_OUTPUT_VALUE;
	for (uint32_t unit_index = 0; unit_index <= MOST_UNITS; unit_index++) {
		result->card_of_unit[unit_index] = UNTOUCHED_OUTPUT_VALUE;
	}
}

/* True when a call wrote nothing: neither the count nor one entry. */
static bool result_is_untouched(const struct placement_result *result)
{
    bool is_untouched = result->on_demand_units_per_owner == UNTOUCHED_OUTPUT_VALUE;
    for (uint32_t unit_index = 0; unit_index <= MOST_UNITS; unit_index++) {
        is_untouched = is_untouched && result->card_of_unit[unit_index] == UNTOUCHED_OUTPUT_VALUE;
    }
    return is_untouched;
}

static bool results_are_the_same(const struct placement_result *left, const struct placement_result *right)
{
    return left->status == right->status && left->on_demand_units_per_owner == right->on_demand_units_per_owner &&
           memcmp(left->card_of_unit, right->card_of_unit, sizeof(left->card_of_unit)) == 0;
}

/* One call of the placement, into a result that was cleared first. With is_refusal_expected the error stream is
 * the temporary file during the call, and what the call logged comes back in captured. */
static void place(const struct pack_record *pack,
		  const uint64_t *budget_bytes_of_card, uint32_t card_count,
		  uint32_t unit_count, bool is_refusal_expected,
		  struct placement_result *result_out, char *captured)
{
    clear_result(result_out);
    captured[0] = '\0';
    if (is_refusal_expected) {
        capture_begin();
    }
    result_out->status = placement_fill_in_order(
	    pack, budget_bytes_of_card, card_count, unit_count,
	    result_out->card_of_unit, &result_out->on_demand_units_per_owner);
    if (is_refusal_expected) {
        capture_end(captured, CAPTURE_CAPACITY);
    }
}

/* Places the units of a plan and proves what holds for every plan and all budgets: the status, the count and the
 * placement are the oracle's, a refusal writes nothing and logs one line with the bytes lacking, and a placement
 * is valid, within the budgets, the same when asked twice, and tight: it is found again when every budget is cut
 * to what its card took, and not when one of those budgets loses a byte. The result is the caller's to compare
 * with its own expectations. */
static void place_and_check(const char *where, const struct pack_record *pack,
			    const struct test_pack_unit_plan *units,
			    uint32_t unit_count,
			    const uint64_t *budget_bytes_of_card,
			    uint32_t card_count,
			    struct placement_result *result_out)
{
    struct placement_result second_result;
    char captured[CAPTURE_CAPACITY];
    char expected_line_end[TEXT_CAPACITY];
    uint32_t reference_card_of_unit[MOST_UNITS];
    uint64_t bytes_used_of_card[BACKEND_MAXIMUM_CARDS] = { 0 };
    uint64_t bytes_lacking = 0;
    const uint32_t largest_count = largest_owned_count(units, unit_count);
    const bool stages_fit = reference_fill(units, unit_count, budget_bytes_of_card, card_count, 0,
                                           reference_card_of_unit, &bytes_lacking);
    bool every_candidate_agrees = true;
    bool budgets_hold = true;
    uint32_t first_card_with_bytes = card_count;

    place(pack, budget_bytes_of_card, card_count, unit_count, !stages_fit, result_out, captured);
    if (!stages_fit) {
        if (snprintf(expected_line_end, sizeof(expected_line_end), "the last card lacks %" PRIu64 " bytes\n",
                     bytes_lacking) < 0) {
            give_up("a text of the test could not be formatted");
        }
	expect(result_out->status == STATUS_OUT_OF_MEMORY, where,
	       "stage units that do not fit are ANEMONE_STATUS_OUT_OF_MEMORY");
	expect(result_is_untouched(result_out), where, "a refused placement writes nothing");
        expect(is_one_error_line(captured, expected_line_end), where,
               "a refused placement logs one error line with the bytes the last card lacks");
        return;
    }
    expect(result_out->status == STATUS_OK, where,
	   "stage units that fit are placed");
    if (result_out->status != STATUS_OK) {
	    return;
    }
    expect(result_out->card_of_unit[unit_count] == UNTOUCHED_OUTPUT_VALUE, where,
           "nothing is written behind the entry of the last unit");
    expect(pack_placement_is_valid(pack, result_out->card_of_unit, unit_count,
				   card_count),
	   where, "the placement obeys anemone_pack_placement_is_valid");

    for (uint32_t candidate_count = 0; candidate_count <= largest_count; candidate_count++) {
        const bool candidate_fits = reference_fill(units, unit_count, budget_bytes_of_card, card_count,
                                                   candidate_count, reference_card_of_unit, &bytes_lacking);
        every_candidate_agrees = every_candidate_agrees &&
                                 candidate_fits == (candidate_count <= result_out->on_demand_units_per_owner);
    }
    expect(result_out->on_demand_units_per_owner <= largest_count, where,
           "the count is at most the largest number of on-demand units one stage owns");
    expect(every_candidate_agrees, where,
           "every count up to the one found fits and every larger one does not: the count is the largest");
    expect(reference_fill(units, unit_count, budget_bytes_of_card, card_count,
                          result_out->on_demand_units_per_owner, reference_card_of_unit, &bytes_lacking) &&
               memcmp(result_out->card_of_unit, reference_card_of_unit, unit_count * sizeof(uint32_t)) == 0,
           where, "the placement is the one the rule gives for that count");

    for (uint32_t unit_index = 0; unit_index < unit_count; unit_index++) {
        const uint32_t card_of_unit = result_out->card_of_unit[unit_index];
        if (card_of_unit < card_count) {
            bytes_used_of_card[card_of_unit] += unit_byte_count(&units[unit_index]);
        } else {
            budgets_hold = budgets_hold && card_of_unit == IN_HOST;
        }
    }
    for (uint32_t card_index = card_count; card_index > 0; card_index--) {
        budgets_hold = budgets_hold && bytes_used_of_card[card_index - 1] <= budget_bytes_of_card[card_index - 1];
        if (bytes_used_of_card[card_index - 1] != 0) {
            first_card_with_bytes = card_index - 1;
        }
    }
    expect(budgets_hold, where, "every card's bytes are within its budget");

    place(pack, budget_bytes_of_card, card_count, unit_count, false, &second_result, captured);
    expect(results_are_the_same(result_out, &second_result), where, "the same inputs give the same placement twice");
    place(pack, bytes_used_of_card, card_count, unit_count, false, &second_result, captured);
    expect(results_are_the_same(result_out, &second_result), where,
           "budgets cut to what every card took give the same placement: it fits to the byte");
    if (first_card_with_bytes < card_count) {
        bytes_used_of_card[first_card_with_bytes]--;
        place(pack, bytes_used_of_card, card_count, unit_count, true, &second_result, captured);
        expect(!results_are_the_same(result_out, &second_result), where,
               "a budget a byte below what its card took does not give the same placement");
    }
}

/* The cases worked out by hand: the engine against the table, and against the oracle. */
static void test_hand_computed_cases(void)
{
    for (size_t case_index = 0; case_index < sizeof(placement_cases) / sizeof(placement_cases[0]); case_index++) {
        const struct placement_case *one_case = &placement_cases[case_index];
        struct opened_pack opened;
        struct placement_result result;
        uint32_t unit_count = 0;
        while (unit_count < MOST_CASE_UNITS && one_case->units[unit_count].extent_count != 0) {
            unit_count++;
        }
        open_pack(one_case->units, unit_count, &opened);
        place_and_check(one_case->what, opened.pack, one_case->units, unit_count, one_case->budget_bytes_of_card,
                        one_case->card_count, &result);
        expect(result.status == one_case->expected_status, one_case->what, "the status worked out by hand");
	if (one_case->expected_status == STATUS_OK &&
	    result.status == STATUS_OK) {
		expect(result.on_demand_units_per_owner ==
			       one_case->expected_on_demand_units_per_owner,
		       one_case->what, "the count worked out by hand");
		expect(memcmp(result.card_of_unit,
			      one_case->expected_card_of_unit,
			      unit_count * sizeof(uint32_t)) == 0,
		       one_case->what, "the placement worked out by hand");
	}
	close_pack(&opened);
    }
}

/* Every refusal of the contract with its status, and nothing written by any of them. */
static void test_refusals(void)
{
    const uint32_t unit_count = sizeof(refusal_units) / sizeof(refusal_units[0]);
    struct opened_pack opened;
    struct placement_result result;
    uint64_t budget_bytes_of_card[BACKEND_MAXIMUM_CARDS + 1];
    char captured[CAPTURE_CAPACITY];
    uint32_t *const card_of_unit = result.card_of_unit;
    uint32_t *const count = &result.on_demand_units_per_owner;

    for (uint32_t card_index = 0; card_index <= BACKEND_MAXIMUM_CARDS;
	 card_index++) {
	    budget_bytes_of_card[card_index] = BLOCKS(REFUSAL_BLOCKS_PER_CARD);
    }
    open_pack(refusal_units, unit_count, &opened);
    clear_result(&result);
    expect(placement_fill_in_order(NULL, budget_bytes_of_card, 1, unit_count,
				   card_of_unit,
				   count) == STATUS_INVALID_ARGUMENT,
	   "refusals", "a missing pack is ANEMONE_STATUS_INVALID_ARGUMENT");
    expect(placement_fill_in_order(opened.pack, NULL, 1, unit_count,
				   card_of_unit,
				   count) == STATUS_INVALID_ARGUMENT,
	   "refusals", "missing budgets are ANEMONE_STATUS_INVALID_ARGUMENT");
    expect(placement_fill_in_order(opened.pack, budget_bytes_of_card, 1,
				   unit_count, NULL,
				   count) == STATUS_INVALID_ARGUMENT,
	   "refusals",
	   "a missing placement output is ANEMONE_STATUS_INVALID_ARGUMENT");
    expect(placement_fill_in_order(opened.pack, budget_bytes_of_card, 1,
				   unit_count, card_of_unit,
				   NULL) == STATUS_INVALID_ARGUMENT,
	   "refusals",
	   "a missing count output is ANEMONE_STATUS_INVALID_ARGUMENT");
    expect(placement_fill_in_order(opened.pack, budget_bytes_of_card, 1,
				   unit_count + 1, card_of_unit,
				   count) == STATUS_INVALID_ARGUMENT,
	   "refusals",
	   "a unit count above the pack's is ANEMONE_STATUS_INVALID_ARGUMENT");
    expect(placement_fill_in_order(opened.pack, budget_bytes_of_card, 1,
				   unit_count - 1, card_of_unit,
				   count) == STATUS_INVALID_ARGUMENT,
	   "refusals",
	   "a unit count below the pack's is ANEMONE_STATUS_INVALID_ARGUMENT");
    expect(placement_fill_in_order(opened.pack, budget_bytes_of_card, 1, 0,
				   card_of_unit,
				   count) == STATUS_INVALID_ARGUMENT,
	   "refusals",
	   "a unit count of zero is ANEMONE_STATUS_INVALID_ARGUMENT");
    expect(placement_fill_in_order(opened.pack, budget_bytes_of_card, 0,
				   unit_count, card_of_unit,
				   count) == STATUS_INVALID_ARGUMENT,
	   "refusals",
	   "a card count of zero is ANEMONE_STATUS_INVALID_ARGUMENT");
    expect(placement_fill_in_order(
		   opened.pack, budget_bytes_of_card, BACKEND_MAXIMUM_CARDS + 1,
		   unit_count, card_of_unit, count) == STATUS_INVALID_ARGUMENT,
	   "refusals",
	   "a card count above ANEMONE_BACKEND_MAXIMUM_CARDS is ANEMONE_STATUS_INVALID_ARGUMENT");
    expect(result_is_untouched(&result), "refusals", "a refused call writes nothing");

    place(opened.pack, budget_bytes_of_card, BACKEND_MAXIMUM_CARDS, unit_count,
	  false, &result, captured);
    expect(result.status == STATUS_OK &&
		   result.on_demand_units_per_owner == 1 &&
		   result.card_of_unit[0] == 0 && result.card_of_unit[1] == 0,
	   "refusals",
	   "ANEMONE_BACKEND_MAXIMUM_CARDS cards are accepted, and the first takes what it holds");
    close_pack(&opened);
}

/* Draws a plan that test_pack_build accepts: an on-demand unit only after a stage unit. Returns the bytes of its
 * stage and on-demand units together, which is what cards can hold of it. */
static uint64_t draw_plan(struct test_pack_unit_plan *units, uint32_t unit_count)
{
    uint64_t card_byte_count = 0;
    bool stage_was_drawn = false;
    for (uint32_t unit_index = 0; unit_index < unit_count; unit_index++) {
        const uint64_t roll = random_below(KIND_ROLL_SIDES);
	uint32_t unit_kind = PLACEMENT_UNIT_HOST_ONLY;
	if (roll < STAGE_ROLLS) {
		unit_kind = PLACEMENT_UNIT_STAGE;
		stage_was_drawn = true;
        } else if (roll < STAGE_ROLLS + ON_DEMAND_ROLLS && stage_was_drawn) {
		unit_kind = PLACEMENT_UNIT_ON_DEMAND;
	}
        units[unit_index].unit_kind = unit_kind;
        units[unit_index].layer_index = unit_index;
        units[unit_index].extent_count = 1 + (uint32_t)random_below(MOST_EXTENTS_PER_RANDOM_UNIT);
        units[unit_index].blocks_per_extent = 1 + (uint32_t)random_below(MOST_BLOCKS_PER_RANDOM_EXTENT);
	if (unit_kind != PLACEMENT_UNIT_HOST_ONLY) {
		card_byte_count += unit_byte_count(&units[unit_index]);
	}
    }
    return card_byte_count;
}

/* Random plans, each under many random budgets of any number of bytes: place_and_check proves on every one that
 * the placement is valid and within the budgets and that no larger count fits. */
static void test_random_plans(void)
{
    for (uint32_t plan_index = 0; plan_index < RANDOM_PLAN_COUNT; plan_index++) {
        struct test_pack_unit_plan units[MOST_UNITS];
        struct opened_pack opened;
        const uint32_t unit_count = 1 + (uint32_t)random_below(MOST_UNITS);
        const uint64_t card_byte_count = draw_plan(units, unit_count);
        const uint32_t largest_count = largest_owned_count(units, unit_count);
        open_pack(units, unit_count, &opened);
        for (uint32_t budget_set_index = 0; budget_set_index < BUDGET_SETS_PER_PLAN; budget_set_index++) {
            struct placement_result result;
            char where[TEXT_CAPACITY];
            uint64_t budget_bytes_of_card[MOST_RANDOM_CARDS];
            const uint32_t card_count = 1 + (uint32_t)random_below(MOST_RANDOM_CARDS);
            const uint64_t most_budget_bytes = card_byte_count * (1 + random_below(BUDGET_SCALES)) /
                                               ((uint64_t)card_count * BUDGET_SCALE_DIVISOR);
            for (uint32_t card_index = 0; card_index < card_count; card_index++) {
                budget_bytes_of_card[card_index] = random_below(most_budget_bytes + 1);
            }
            if (snprintf(where, sizeof(where), "random plan %" PRIu32 " under budget set %" PRIu32, plan_index,
                         budget_set_index) < 0) {
                give_up("a text of the test could not be formatted");
            }
            place_and_check(where, opened.pack, units, unit_count, budget_bytes_of_card, card_count, &result);
	    if (result.status != STATUS_OK) {
		    random_placements_refused++;
	    } else if (result.on_demand_units_per_owner == largest_count) {
		    random_placements_with_every_unit_kept++;
	    } else if (result.on_demand_units_per_owner == 0) {
		    random_placements_with_no_unit_kept++;
	    } else {
		    random_placements_with_some_units_kept++;
	    }
	}
        close_pack(&opened);
    }
    /* a loop that met only refusals, or only budgets that hold everything, would prove nothing about the count */
    expect(random_placements_refused != 0 && random_placements_with_no_unit_kept != 0 &&
               random_placements_with_some_units_kept != 0 && random_placements_with_every_unit_kept != 0,
           "random plans", "the plans and budgets drawn meet every outcome: refused, no unit kept, some, all");
}

/* One call of the layout text, its error stream read back into captured; the entry behind the last unit must stay. */
MUST_CHECK static result_status
layout_place(const struct pack_record *pack, const char *text,
	     const uint64_t *budgets, uint32_t card_count, uint32_t unit_count,
	     uint32_t *card_of_unit, char *captured)
{
    card_of_unit[unit_count] = UNTOUCHED_OUTPUT_VALUE;
    capture_begin();
    const result_status status = placement_from_layout_text(
	    pack, text, budgets, card_count, unit_count, card_of_unit);
    capture_end(captured, CAPTURE_CAPACITY);
    expect(card_of_unit[unit_count] == UNTOUCHED_OUTPUT_VALUE, text, "nothing is written behind the last unit");
    expect(status != STATUS_OK ||
		   (captured[0] == '\0' &&
		    pack_placement_is_valid(pack, card_of_unit, unit_count,
					    card_count)),
	   text, "an accepted text logs nothing and gives a valid placement");
    return status;
}

/* A text of exactly length bytes that places every unit of the layout pack on card 0: zeros lead its first number.
 * True when it was formatted. */
static bool layout_text_of_length(char *text, size_t length)
{
    static const char *const tail = ", 1..2: card 0" LAYOUT_OTHERS;
    const int zeros = (int)(length - strlen("layers ") - strlen(tail));
    return snprintf(text, length + 1u, "layers %0*u%s", zeros, 0u, tail) == (int)length;
}

/* The hand-computed texts, to the byte of the budgets and a byte short; every refusal of placement.h. */
static void test_layout_text(void)
{
    const uint32_t unit_count = (uint32_t)LAYOUT_UNITS;
    struct opened_pack opened;
    uint32_t card_of_unit[LAYOUT_UNITS + 1];
    uint64_t budgets[BACKEND_MAXIMUM_CARDS + 1];
    char captured[CAPTURE_CAPACITY];
    char text[LAYOUT_TEXT_LONGEST_BYTES];
    char line_end[TEXT_CAPACITY];
    for (uint32_t card = 0; card <= BACKEND_MAXIMUM_CARDS; card++) {
	    budgets[card] = BLOCKS(LAYOUT_ROOMY_BLOCKS);
    }
    open_pack(layout_units, unit_count, &opened);
    const char *const where = layout_named_text;
    expect(layout_place(opened.pack, layout_named_text, budgets, LAYOUT_CARDS,
			unit_count, card_of_unit, captured) == STATUS_OK &&
		   memcmp(card_of_unit, layout_named_cards,
			  sizeof(layout_named_cards)) == 0,
	   where, "every unit where the text puts it, worked out by hand");
    budgets[0] = BLOCKS(LAYOUT_CARD_ZERO_BLOCKS);
    budgets[1] = BLOCKS(LAYOUT_CARD_ONE_BLOCKS);
    expect(layout_place(opened.pack, layout_named_text, budgets, LAYOUT_CARDS,
			unit_count, card_of_unit, captured) == STATUS_OK,
	   where, "budgets that hold each card's units to the byte");
    budgets[1]--;
    expect(snprintf(line_end, sizeof(line_end), "card 1 holds %" PRIu64 " bytes of weights, 1 above its budget\n",
                    budgets[1] + 1u) > 0, where, "the text of the test is formatted");
    expect(layout_place(opened.pack, layout_named_text, budgets, LAYOUT_CARDS,
			unit_count, card_of_unit,
			captured) == STATUS_OUT_OF_MEMORY &&
		   is_one_error_line(captured, line_end),
	   where,
	   "a byte short is ANEMONE_STATUS_OUT_OF_MEMORY with one error line");
    budgets[0] = BLOCKS(LAYOUT_ROOMY_BLOCKS);
    budgets[1] = BLOCKS(LAYOUT_ROOMY_BLOCKS);
    expect(layout_place(
		   opened.pack,
		   "layers 0..2: card 1; experts kept: layers 0..2: 2" LAYOUT_OTHERS,
		   budgets, LAYOUT_CARDS, unit_count, card_of_unit,
		   captured) == STATUS_OK &&
		   card_of_unit[LAYOUT_LAYER_ONE_SECOND_UNIT] == 1 &&
		   card_of_unit[LAYOUT_LAYER_ONE_HOST_ONLY_UNIT] == IN_HOST,
	   "two experts",
	   "a layer keeps two on-demand units on its card past a host-only unit between them");
    for (size_t index = 0; index < sizeof(layout_refusals) / sizeof(layout_refusals[0]); index++) {
	    expect(layout_place(opened.pack, layout_refusals[index].text,
				budgets, LAYOUT_CARDS, unit_count, card_of_unit,
				captured) == STATUS_INVALID_ARGUMENT &&
			   is_one_error_line(captured,
					     layout_refusals[index].line_end),
		   layout_refusals[index].text,
		   "refused with ANEMONE_STATUS_INVALID_ARGUMENT and its one error line");
    }
    expect(layout_text_of_length(text, PLACEMENT_MOST_TEXT_BYTES),
	   "the longest text", "it is formatted");
    expect(layout_place(opened.pack, text, budgets, LAYOUT_CARDS, unit_count,
			card_of_unit, captured) == STATUS_OK,
	   "the longest text",
	   "a text of ANEMONE_PLACEMENT_MOST_TEXT_BYTES is placed");
    expect(snprintf(line_end, sizeof(line_end), "longer than %u bytes\n",
		    PLACEMENT_MOST_TEXT_BYTES) > 0,
	   "a text a byte too long", "the text of the test is formatted");
    expect(layout_text_of_length(text, PLACEMENT_MOST_TEXT_BYTES + 1u),
	   "a text a byte too long", "it is formatted");
    expect(layout_place(opened.pack, text, budgets, LAYOUT_CARDS, unit_count,
			card_of_unit, captured) == STATUS_INVALID_ARGUMENT &&
		   is_one_error_line(captured, line_end),
	   "a text a byte too long", "refused with its one error line");
    expect(snprintf(text, sizeof(text), "layers 1: card %u; layers 0, 2: card 0" LAYOUT_OTHERS, LAYOUT_LAST_CARD) > 0,
           "the last card", "the text of the test is formatted");
    expect(layout_place(opened.pack, text, budgets, BACKEND_MAXIMUM_CARDS,
			unit_count, card_of_unit, captured) == STATUS_OK &&
		   card_of_unit[LAYOUT_LAYER_ONE_UNIT] == LAYOUT_LAST_CARD,
	   "the last card",
	   "ANEMONE_BACKEND_MAXIMUM_CARDS cards are accepted, the last one named");
    const char *const text_of_all = "layers 0..2: card 0" LAYOUT_OTHERS;
    expect(placement_from_layout_text(NULL, text_of_all, budgets, 1, unit_count,
				      card_of_unit) ==
			   STATUS_INVALID_ARGUMENT &&
		   placement_from_layout_text(opened.pack, NULL, budgets, 1,
					      unit_count, card_of_unit) ==
			   STATUS_INVALID_ARGUMENT &&
		   placement_from_layout_text(opened.pack, text_of_all, NULL, 1,
					      unit_count, card_of_unit) ==
			   STATUS_INVALID_ARGUMENT &&
		   placement_from_layout_text(opened.pack, text_of_all, budgets,
					      1, unit_count, NULL) ==
			   STATUS_INVALID_ARGUMENT &&
		   placement_from_layout_text(opened.pack, text_of_all, budgets,
					      0, unit_count, card_of_unit) ==
			   STATUS_INVALID_ARGUMENT &&
		   placement_from_layout_text(opened.pack, text_of_all, budgets,
					      BACKEND_MAXIMUM_CARDS + 1u,
					      unit_count, card_of_unit) ==
			   STATUS_INVALID_ARGUMENT &&
		   placement_from_layout_text(opened.pack, text_of_all, budgets,
					      1, unit_count - 1u,
					      card_of_unit) ==
			   STATUS_INVALID_ARGUMENT &&
		   placement_from_layout_text(opened.pack, text_of_all, budgets,
					      1, unit_count + 1u,
					      card_of_unit) ==
			   STATUS_INVALID_ARGUMENT,
	   "layout refusals",
	   "a missing argument, card counts of 0 and above the most, a wrong unit count");
    close_pack(&opened);

    open_pack(twice_units, TWICE_UNITS, &opened);
    expect(layout_place(opened.pack, "vision: card 0", budgets, LAYOUT_CARDS,
			TWICE_UNITS, card_of_unit,
			captured) == STATUS_INVALID_ARGUMENT &&
		   is_one_error_line(captured,
				     "statement 1 names no unit of the pack\n"),
	   "vision on a pack without", "a name that names no unit is refused");
    expect(layout_place(opened.pack, "embedding: card 0; layers 0: card 0",
			budgets, LAYOUT_CARDS, TWICE_UNITS, card_of_unit,
			captured) == STATUS_INVALID_ARGUMENT &&
		   is_one_error_line(captured, "statement 2 " LAYOUT_NOT_ONCE),
	   "a layer of two stage units",
	   "a layer the stage units hold twice is refused");
    expect(layout_place(opened.pack, "embedding: card 0; layers 0..1: card 0",
			budgets, LAYOUT_CARDS, TWICE_UNITS, card_of_unit,
			captured) == STATUS_INVALID_ARGUMENT &&
		   is_one_error_line(captured, "statement 2 " LAYOUT_NOT_ONCE),
	   "a layer twice, the next none",
	   "a range of as many stage units as layers, one layer twice, is refused");
    expect(layout_place(opened.pack, "layers 2..3: card 0", budgets,
			LAYOUT_CARDS, TWICE_UNITS, card_of_unit,
			captured) == STATUS_INVALID_ARGUMENT &&
		   is_one_error_line(captured, "statement 1 " LAYOUT_NOT_ONCE),
	   "layers out of order",
	   "a range whose stage units hold its layers out of order is refused");
    close_pack(&opened);
}

int main(void)
{
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    error_stream_descriptor = dup(STDERR_FILENO);
    capture_descriptor = mkstemp(capture_path);
    if (error_stream_descriptor < 0 || capture_descriptor < 0) {
        give_up("the temporary file that stands in for the error stream could not be made");
    }
    (void)printf("placement_test: while a refusal is read back, the error stream is %s\n", capture_path);

    test_hand_computed_cases();
    test_refusals();
    test_random_plans();
    test_layout_text();

    expect(close(capture_descriptor) == 0 && unlink(capture_path) == 0 && close(error_stream_descriptor) == 0,
           "the end of the test", "the temporary file is removed");
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("placement_test passed (random placements: %" PRIu32 " refused, %" PRIu32 " with no on-demand unit "
                 "on a card, %" PRIu32 " with some, %" PRIu32 " with all)\n",
                 random_placements_refused, random_placements_with_no_unit_kept,
                 random_placements_with_some_units_kept, random_placements_with_every_unit_kept);
    return EXIT_SUCCESS;
}
