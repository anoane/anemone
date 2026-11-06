/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Four-CMP lossless transport qualification against the retained BF16 ring. No model or engine default claim.
 * Both rings use their own banks, equal inputs, the same owner geometry, and the same per-hop BF16 arithmetic.
 * Repeated graph execution exercises credited packet reuse while rank submission order/skew changes. */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "anemone/compressed_ring_records.h"
#include "anemone/round_once_ring_records.h"
#include "linear_card_fixture.h"
#include "prepare_common.h"

#define RANKS EXACT_RING_RANKS
#define FORMS 2u
#define OLD_FORM 0u
#define COMPRESSED_FORM 1u
#define WORK_BYTES (16ull << 20u)
#define WEIGHT_BYTES (1ull << 20u)
#define MAXIMUM_PACKS 7680u
#define MAXIMUM_BYTES ((uint64_t)RANKS * MAXIMUM_PACKS * EXACT_RING_PACK_BYTES)
#define BANK_BYTES                   \
	(EXACT_RING_METADATA_BYTES + \
	 MAXIMUM_PACKS * EXACT_RING_SUM_EPOCH_HOPS * EXACT_RING_PACK_BYTES)
#define TEST_SECONDS 900u
#define DEADLINE_CYCLES 2000000000ull
#define REPEATS 16u
#define PATTERNS 4u
#define PLAIN_EXPONENT 120u
#define EXPONENT_SPREAD 8u
#define WIDE_EXPONENT_SPREAD 80u
#define SIGN_MASK 0x8000u
#define MANTISSA_MASK 0x7fu
#define EXPONENT_SHIFT 7u
#define HASH_MULTIPLIER 2654435761u
#define HASH_INCREMENT 1013904223u
#define SPECIAL_PERIOD 17u
#define SPECIAL_NEGATIVE_ZERO 1u
#define SPECIAL_INFINITY 2u
#define SPECIAL_NAN 3u
#define POSITIVE_INFINITY 0x7f80u
#define QUIET_NAN 0x7fc5u
#define WIDE_PATTERN 1u
#define SPECIAL_PATTERN 2u
#define RANDOM_PATTERN 3u
#define CMP_BUS_COUNT 4u
#define LAUNCH_SKEW_US 3000u
#define TIMED_REPEATS 128u
#define TIMING_SAMPLES 7u
#define INPUT_PHASES 2u
#define FINITE_EXPONENTS 200u
#define BF16_ONE 0x3f80u
#define BF16_HALF_ULP 0x3b80u
#define BF16_TIE_EPSILON 0x3780u
#define BF16_LARGE 0x4780u
#define BF16_NEXT_ONE 0x3f81u
#define BF16_TWO 0x4000u
#define BF16_FOUR_NEXT 0x4081u
#define BF16_EIGHT 0x4100u
#define ORACLE_VALUES 4u

static bool timing, round_once;
static struct test_linear_card cards[RANKS];
static struct linear_range tables[RANKS], inputs[RANKS][FORMS],
	outputs[RANKS][FORMS], banks[RANKS][FORMS];
static struct linear_range alternates[RANKS][FORMS];
static uint32_t kernels[RANKS][FORMS];
static float reference_sums[MAXIMUM_BYTES / sizeof(uint16_t)];
static uint16_t expected[MAXIMUM_BYTES / sizeof(uint16_t)];
static uint16_t values[MAXIMUM_BYTES / sizeof(uint16_t)];
static const unsigned char empty_bank[BANK_BYTES] = { 0 };
static const struct backend_operations *const backend = &backend_cuda_value;
static const char *const cmp_buses[CMP_BUS_COUNT] = {
	"0000:01:00.0", "0000:41:00.0", "0000:82:00.0", "0000:c1:00.0"
};

struct geometry {
	uint32_t packs, blocks, rows;
};
static const struct geometry geometries[] = { { 32u, 1u, 1u },
					      { 160u, 1u, 1u },
					      { 960u, 4u, 6u },
					      { 1280u, 8u, 8u },
					      { 7680u, 16u, 48u } };
#define GEOMETRIES ((uint32_t)(sizeof(geometries) / sizeof(geometries[0])))

static bool okay(result_status status, const char *what)
{
	return test_linear_expect_status(status, STATUS_OK, what);
}

static bool opened(void)
{
	uint32_t count = 0, selected[RANKS], found = 0;
	if (!okay(backend->card_count(&count), "card inventory"))
		return false;
	for (uint32_t index = 0; index < count; index++) {
		struct card_description_record description;
		if (!okay(backend->card_describe(index, &description),
			  "physical card described"))
			return false;
		if (strcmp(description.name, "NVIDIA CMP 170HX") != 0)
			continue;
		if (!test_linear_expect(found < RANKS &&
						description.arithmetic_class ==
							ARITHMETIC_CLASS_SM_80,
					"exactly four SM80 CMPs"))
			return false;
		bool known_bus = false;
		for (uint32_t bus = 0; bus < CMP_BUS_COUNT; bus++)
			known_bus = known_bus ||
				    strcasecmp(description.bus_address,
					       cmp_buses[bus]) == 0;
		if (!test_linear_expect(
			    known_bus,
			    "CMP PCI identity checked before context creation"))
			return false;
		selected[found++] = index;
	}
	if (!test_linear_expect(found == RANKS,
				"four CMPs visible; PRO is never opened"))
		return false;
	for (uint32_t rank = 0; rank < RANKS; rank++) {
		if (!test_linear_card_open(&cards[rank], selected[rank],
					   WORK_BYTES, WEIGHT_BYTES))
			return false;
		tables[rank] = test_linear_card_work(
			&cards[rank],
			RANKS * sizeof(struct ordered_collective_peer));
		for (uint32_t form = 0; form < FORMS; form++) {
			inputs[rank][form] = test_linear_card_work(
				&cards[rank], MAXIMUM_BYTES);
			outputs[rank][form] = test_linear_card_work(
				&cards[rank], MAXIMUM_BYTES);
			alternates[rank][form] = test_linear_card_work(
				&cards[rank], MAXIMUM_BYTES);
			banks[rank][form] =
				test_linear_card_work(&cards[rank], BANK_BYTES);
			if (!okay(backend->kernel_prepare(
					  cards[rank].card,
					  form == OLD_FORM ?
						  KERNEL_FILE_EXACT_RING :
						  (round_once ?
							   KERNEL_FILE_ROUND_ONCE_RING :
							   KERNEL_FILE_COMPRESSED_RING),
					  form == OLD_FORM ?
						  KERNEL_EXACT_RING :
						  (round_once ?
							   KERNEL_ROUND_ONCE_RING :
							   KERNEL_COMPRESSED_RING),
					  EXACT_RING_ARGUMENT_BYTES,
					  &kernels[rank][form]),
				  "independent ring kernel prepared"))
				return false;
		}
		if (!test_linear_card_has_room(&cards[rank]))
			return false;
	}
	return true;
}

static void pattern(uint32_t rank, uint32_t kind, uint32_t bytes)
{
	uint32_t mixed = HASH_INCREMENT + rank;
	for (uint32_t index = 0; index < bytes / sizeof(uint16_t); index++) {
		mixed = mixed * HASH_MULTIPLIER + HASH_INCREMENT;
		uint32_t value =
			(mixed & (SIGN_MASK | MANTISSA_MASK)) |
			((PLAIN_EXPONENT -
			  mixed % (kind == WIDE_PATTERN ? WIDE_EXPONENT_SPREAD :
							  EXPONENT_SPREAD))
			 << EXPONENT_SHIFT);
		if (kind == RANDOM_PATTERN)
			value = round_once ?
					(mixed & (SIGN_MASK | MANTISSA_MASK)) |
						(((mixed >> EXPONENT_SHIFT) %
						  FINITE_EXPONENTS)
						 << EXPONENT_SHIFT) :
					mixed;
		if (kind == SPECIAL_PATTERN) {
			switch (index % SPECIAL_PERIOD) {
			case 0:
				value = 0;
				break;
			case SPECIAL_NEGATIVE_ZERO:
				value = SIGN_MASK;
				break;
			case SPECIAL_INFINITY:
				value = !round_once	   ? POSITIVE_INFINITY :
					rank % RANKS == 0  ? BF16_ONE :
					rank % RANKS == 1u ? BF16_HALF_ULP :
					rank % RANKS == ROUND_ONCE_RING_THIRD_RANK ?
							     BF16_TIE_EPSILON :
							     0u;
				break;
			case SPECIAL_NAN:
				value = !round_once ? QUIET_NAN :
					rank % RANKS == 0 ?
						      BF16_LARGE :
					rank % RANKS == ROUND_ONCE_RING_THIRD_RANK ?
						      (BF16_LARGE | SIGN_MASK) :
						      BF16_ONE;
				break;
			default:
				break;
			}
		}
		values[index] = (uint16_t)value;
	}
}

/* Independent CPU oracle: preserve FP32 rank order explicitly, then use the existing reference RNE encoder. */
static void reference_build(uint32_t bytes, uint32_t kind, bool alternate,
			    bool following_in_place)
{
	if (!following_in_place)
		memset(reference_sums, 0,
		       bytes / sizeof(uint16_t) * sizeof(float));
	for (uint32_t rank = 0; rank < RANKS; rank++) {
		if (!following_in_place)
			pattern(rank + (alternate ? RANKS : 0u), kind, bytes);
		for (uint32_t index = 0; index < bytes / sizeof(uint16_t);
		     index++) {
			const float value = (float)test_linear_bfloat16_value(
				following_in_place ? expected[index] :
						     values[index]);
			reference_sums[index] =
				(following_in_place && rank == 0 ?
					 0.0f :
					 reference_sums[index]) +
				value;
		}
	}
	for (uint32_t index = 0; index < bytes / sizeof(uint16_t); index++)
		expected[index] = test_linear_bfloat16_nearest(
			(double)reference_sums[index]);
}

static bool output_matches(uint32_t rank, uint32_t bytes, bool in_place)
{
	const struct linear_range old = in_place ? inputs[rank][OLD_FORM] :
						   outputs[rank][OLD_FORM];
	const struct linear_range candidate =
		in_place ? inputs[rank][COMPRESSED_FORM] :
			   outputs[rank][COMPRESSED_FORM];
	return round_once ?
		       test_linear_card_expect_bytes(
			       &cards[rank], candidate, 0, expected, bytes,
			       "round-once bytes match independent rank-ordered FP32 oracle") :
		       test_linear_card_expect_same(
			       &cards[rank], old, 0, candidate, 0, bytes,
			       "compressed transport preserves retained BF16 ring bytes");
}

static bool prepared(const struct geometry *geometry, uint32_t kind,
		     uint32_t mode, bool in_place,
		     struct kernel_launch launches[RANKS][FORMS])
{
	const uint32_t bytes = RANKS * geometry->packs * EXACT_RING_PACK_BYTES;
	for (uint32_t rank = 0; rank < RANKS; rank++) {
		pattern(rank, kind, bytes);
		for (uint32_t form = 0; form < FORMS; form++) {
			/* writable() is for the weights region; the guarded work bank uses the checked put API. */
			if (!test_linear_card_put(
				    &cards[rank], banks[rank][form], 0,
				    empty_bank, sizeof(empty_bank)))
				return false;
			if (!test_linear_card_put(&cards[rank],
						  inputs[rank][form], 0, values,
						  bytes))
				return false;
			const struct exact_ring_arguments arguments = {
				.input_offset = inputs[rank][form].byte_offset,
				.output_offset =
					in_place ?
						inputs[rank][form].byte_offset :
						outputs[rank][form].byte_offset,
				.bank_offset = banks[rank][form].byte_offset,
				.peer_table_offset = tables[rank].byte_offset,
				.peer_bank_delta =
					banks[rank][form].byte_offset,
				.deadline_cycles = DEADLINE_CYCLES,
				.packs_per_owner = geometry->packs,
				.vectors = geometry->rows,
				.rank = rank,
				.mode = mode
			};
			prepare_launch_fill(&launches[rank][form],
					    kernels[rank][form],
					    geometry->blocks, 1,
					    COMPRESSED_RING_THREADS, 0,
					    &arguments, sizeof(arguments));
		}
		pattern(rank + RANKS, kind, bytes);
		for (uint32_t form = 0; form < FORMS; form++)
			if (!test_linear_card_put(&cards[rank],
						  alternates[rank][form], 0,
						  values, bytes))
				return false;
		if (!test_linear_card_load(&cards[rank]))
			return false;
	}
	/* A current-allocation transport pilot: complete compared ranges and guarded arenas qualify these direct
	 * routes here. No prepared address or proof is exported to a model or reused after these arenas close. */
	for (uint32_t rank = 0; rank < RANKS; rank++) {
		for (uint32_t peer = 0; peer < RANKS; peer++) {
			if (!okay(backend->peer_connect(
					  cards[rank].card, cards[peer].card, 0,
					  tables[rank].byte_offset +
						  peer * sizeof(struct ordered_collective_peer)),
				  "direct current arena connected"))
				return false;
		}
	}
	return true;
}

/* Capture the full repeat interval, then enqueue every rank before waiting for any duration. Each sample is
 * the maximum rank's own device interval divided by repeats; no clocks from different cards are subtracted. */
static bool timed(const struct geometry *geometry,
		  struct kernel_launch launches[RANKS][FORMS])
{
	uint32_t graphs[RANKS][FORMS];
	struct kernel_launch repeated[TIMED_REPEATS];
	double samples[FORMS][TIMING_SAMPLES];
	if (!test_linear_expect(backend->measure_begin != NULL &&
					backend->measure_end != NULL &&
					backend->measure_elapsed_us != NULL,
				"timing requires a profile/debug build"))
		return false;
	for (uint32_t rank = 0; rank < RANKS; rank++) {
		for (uint32_t form = 0; form < FORMS; form++) {
			for (uint32_t repeat = 0; repeat < TIMED_REPEATS;
			     repeat++) {
				repeated[repeat] = launches[rank][form];
				if (repeat % INPUT_PHASES != 0) {
					struct exact_ring_arguments arguments;
					memcpy(&arguments,
					       repeated[repeat].argument,
					       sizeof(arguments));
					arguments.input_offset =
						alternates[rank][form]
							.byte_offset;
					memcpy(repeated[repeat].argument,
					       &arguments, sizeof(arguments));
				}
			}
			if (!okay(backend->graph_capture(
					  cards[rank].card, repeated,
					  TIMED_REPEATS, &graphs[rank][form]),
				  "complete repeated interval captured"))
				return false;
		}
	}
	for (uint32_t sample = 0; sample < TIMING_SAMPLES; sample++) {
		for (uint32_t place = 0; place < FORMS; place++) {
			const uint32_t form = (sample + place) % FORMS;
			for (uint32_t rank = 0; rank < RANKS; rank++)
				if (!okay(backend->measure_begin(
						  cards[rank].card),
					  "rank device interval begins"))
					return false;
			for (uint32_t rank = 0; rank < RANKS; rank++)
				if (!okay(backend->graph_run(cards[rank].card,
							     graphs[rank][form]),
					  "all communicating graphs submitted"))
					return false;
			for (uint32_t rank = 0; rank < RANKS; rank++)
				if (!okay(backend->measure_end(cards[rank].card),
					  "rank device interval ends"))
					return false;
			double longest = 0.0;
			for (uint32_t rank = 0; rank < RANKS; rank++) {
				double microseconds = 0.0;
				if (!okay(backend->measure_elapsed_us(
						  cards[rank].card,
						  TEST_LINEAR_WAIT_LIMIT_NANOSECONDS,
						  &microseconds),
					  "all rank intervals retired"))
					return false;
				longest = microseconds > longest ?
						  microseconds :
						  longest;
			}
			samples[form][sample] = longest / (double)TIMED_REPEATS;
		}
	}
	for (uint32_t form = 0; form < FORMS; form++) {
		for (uint32_t sample = 1; sample < TIMING_SAMPLES; sample++) {
			const double value = samples[form][sample];
			uint32_t place = sample;
			while (place != 0 &&
			       samples[form][place - 1u] > value) {
				samples[form][place] =
					samples[form][place - 1u];
				place--;
			}
			samples[form][place] = value;
		}
	}
	const uint32_t bytes = RANKS * geometry->packs * EXACT_RING_PACK_BYTES;
	if (round_once)
		reference_build(bytes, 0, true, false);
	for (uint32_t rank = 0; rank < RANKS; rank++) {
		if (!test_linear_card_fetch(&cards[rank]) ||
		    !output_matches(rank, bytes, false))
			return false;
		for (uint32_t form = 0; form < FORMS; form++) {
			for (uint32_t block = 0; block < geometry->blocks;
			     block++) {
				uint32_t generation = 0;
				memcpy(&generation,
				       cards[rank].received +
					       banks[rank][form].byte_offset +
					       (uint64_t)block *
						       sizeof(generation),
				       sizeof(generation));
				if (!test_linear_expect(
					    generation ==
						    REPEATS +
							    TIMING_SAMPLES *
								    TIMED_REPEATS,
					    "every graph generation advances exactly once"))
					return false;
			}
		}
	}
	(void)printf("ring timing: packs=%" PRIu32 " grid=%" PRIu32
		     " old_us=%.3f %s_us=%.3f repeats=%u samples=%u\n",
		     geometry->packs, geometry->blocks,
		     samples[OLD_FORM][TIMING_SAMPLES / FORMS],
		     round_once ? "round_once" : "compressed",
		     samples[COMPRESSED_FORM][TIMING_SAMPLES / FORMS],
		     TIMED_REPEATS, TIMING_SAMPLES);
	return true;
}

static bool case_run(const struct geometry *geometry, uint32_t kind,
		     uint32_t mode, bool in_place)
{
	struct kernel_launch launches[RANKS][FORMS];
	if (!prepared(geometry, kind, mode, in_place, launches))
		return false;
	uint32_t graphs[RANKS][FORMS];
	for (uint32_t rank = 0; rank < RANKS; rank++) {
		for (uint32_t form = 0; form < FORMS; form++) {
			if (!okay(backend->graph_capture(cards[rank].card,
							 &launches[rank][form],
							 1,
							 &graphs[rank][form]),
				  "each immutable ring captured separately"))
				return false;
		}
	}
	const uint32_t bytes = RANKS * geometry->packs * EXACT_RING_PACK_BYTES;
	for (uint32_t repeat = 0; repeat < REPEATS; repeat++) {
		for (uint32_t form = 0; form < FORMS; form++) {
			for (uint32_t place = 0; place < RANKS; place++) {
				const uint32_t rank = (place + repeat) % RANKS;
				if (place + 1u == RANKS && repeat == 0)
					(void)usleep(LAUNCH_SKEW_US);
				if (!okay(backend->graph_run(cards[rank].card,
							     graphs[rank][form]),
					  "ring submitted before waits"))
					return false;
			}
			for (uint32_t rank = 0; rank < RANKS; rank++) {
				if (!okay(backend->launch_wait(
						  cards[rank].card,
						  TEST_LINEAR_WAIT_LIMIT_NANOSECONDS),
					  "ring retired"))
					return false;
			}
		}
		if (round_once)
			reference_build(bytes, kind, false,
					in_place && repeat != 0);
		for (uint32_t rank = 0; rank < RANKS; rank++) {
			if (!test_linear_card_fetch(&cards[rank]) ||
			    !output_matches(rank, bytes, in_place))
				return false;
		}
	}
	if (timing && !timed(geometry, launches))
		return false;
	(void)printf("%s ring: packs=%" PRIu32 " grid=%" PRIu32
		     " pattern=%" PRIu32 " mode=%" PRIu32
		     " in_place=%u repeats=%u passed\n",
		     round_once ? "round-once" : "compressed", geometry->packs,
		     geometry->blocks, kind, mode, (unsigned int)in_place,
		     REPEATS);
	return true;
}

static bool oracle_test(void)
{
	round_once = true;
	reference_build(ORACLE_VALUES * sizeof(uint16_t), SPECIAL_PATTERN,
			false, false);
	bool passed = test_linear_expect(
		expected[0] == 0 && expected[SPECIAL_NEGATIVE_ZERO] == 0 &&
			expected[SPECIAL_INFINITY] == BF16_NEXT_ONE &&
			expected[SPECIAL_NAN] == BF16_TWO,
		"independent oracle keeps a BF16 tie's small excess and a cancelled large term's residual");
	reference_build(ORACLE_VALUES * sizeof(uint16_t), SPECIAL_PATTERN,
			false, true);
	passed =
		test_linear_expect(
			expected[SPECIAL_INFINITY] == BF16_FOUR_NEXT &&
				expected[SPECIAL_NAN] == BF16_EIGHT,
			"in-place oracle advances from the previous rounded result") &&
		passed;
	return passed;
}

int main(int argument_count, char **arguments)
{
	if (argument_count == INPUT_PHASES &&
	    strcmp(arguments[1], "--oracle") == 0)
		return oracle_test() ? EXIT_SUCCESS : EXIT_FAILURE;
	for (int argument = 1; argument < argument_count; argument++) {
		if (!timing && strcmp(arguments[argument], "--timing") == 0)
			timing = true;
		else if (!round_once &&
			 strcmp(arguments[argument], "--round-once") == 0)
			round_once = true;
		else
			return EXIT_FAILURE;
	}
	(void)alarm(TEST_SECONDS);
	bool passed = opened();
	for (uint32_t geometry = 0; passed && geometry < GEOMETRIES; geometry++)
		for (uint32_t kind = 0;
		     passed && kind < (timing ? 1u : PATTERNS); kind++)
			for (uint32_t mode = EXACT_RING_BF16_SUM;
			     passed &&
			     mode <= (timing ? EXACT_RING_BF16_SUM :
					       EXACT_RING_BF16_SUM_COLUMNS);
			     mode++)
				for (uint32_t in_place = 0;
				     passed && in_place < (timing ? 1u : FORMS);
				     in_place++)
					passed = case_run(&geometries[geometry],
							  kind, mode,
							  in_place != 0);
	/* Attempt every rank's retirement before closing any peer arena, including behind a failed case. */
	for (uint32_t rank = 0; rank < RANKS; rank++) {
		if (cards[rank].card != NULL) {
			const result_status drained = backend->launch_wait(
				cards[rank].card,
				TEST_LINEAR_WAIT_LIMIT_NANOSECONDS);
			passed =
				okay(drained,
				     "every participating stream drained before any arena closes") &&
				passed;
		}
	}
	for (uint32_t rank = 0; rank < RANKS; rank++)
		test_linear_card_close(&cards[rank]);
	return passed && test_linear_failures() == 0 ? EXIT_SUCCESS :
						       EXIT_FAILURE;
}
