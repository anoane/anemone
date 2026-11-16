/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* EXL3 widths on the host: the stored-width table and the supported set, S(i) of exl3_records.h against the oracle's
 * bit walk, the lane windows in two words, known-answer trellis states per width (a transcription of exllamav3
 * v1.6.5's dq<bits> and frac_s/frac_window, with MASK 0xAAAA for a half K, over splitmix64 words of a 32 x 32 matrix:
 * FNV-1a 64 of the states, input-major, and the first eight), the byte counts and refusals of the format, the TP
 * shard plans, and slices that keep every weight's state. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/exl3.h"
#include "anemone/exl3_layout.h"
#include "anemone/exl3_tc.h"
#include "exl3_reference.h"

#define KAT_SIDE 32u
#define KAT_SEED 0x5EED0000u
#define KAT_FIRST_STATES 8u
#define FNV_OFFSET UINT64_C(0xCBF29CE484222325)
#define FNV_PRIME UINT64_C(0x100000001B3)
#define SPLITMIX_INCREMENT UINT64_C(0x9E3779B97F4A7C15)
#define SPLITMIX_FIRST UINT64_C(0xBF58476D1CE4E5B9)
#define SPLITMIX_SECOND UINT64_C(0x94D049BB133111EB)
#define LOW_WORD UINT64_C(0xFFFFFFFF)
#define BYTE_MASK 0xFFu
#define BYTE_BITS 8u
#define STORED_WIDTH_SCAN 300u
#define EXPERT_HIDDEN 5120u
#define EXPERT_INNER 2304u
#define SHARD_RANKS 4u
#define SLICE_SIDE 256u
#define SLICE_HALF 128u
#define SENTINEL UINT64_C(0xBA5E)
#define TILE_WORDS_MOST 64u
#define LANES 32u
#define LANE_SYMBOLS 8u
#define LANE_LOAD_BITS 64u
#define STATE_BITS 16u
#define WORD_BITS 32u
#define SPLITMIX_FIRST_SHIFT 30u
#define SPLITMIX_SECOND_SHIFT 27u
#define SPLITMIX_LAST_SHIFT 31u
#define SLICE_FORMS 2u /* an output slice, an input slice */
#define VECTOR_COUNT 2u /* suh and svh */
#define NO_WIDTH_ZERO 0u
#define WIDTH_K1 1u
#define WIDTH_K5 5u
#define WIDTH_K8 8u
#define NO_WIDTH_NINE 9u
#define WIDTH_K1_5 24u
#define NO_WIDTH_ODD_SIXTEENTHS 41u
#define WIDTH_K3_5 56u
#define WIDTH_K7_5 120u
#define NO_WIDTH_K8_SIXTEENTHS 128u
#define TC_BEYOND_WIDTHS 17u /* bits_x2 past K8 */

struct width_case {
	uint32_t stored;
	uint32_t bits_x2;
	uint64_t expert_bytes; /* 2,304 x 5,120 */
	uint64_t digest;
	uint16_t first[KAT_FIRST_STATES];
};

static const struct width_case widths[] = {
	{ EXL3_K2,
	  EXL3_BITS_X2_K2,
	  2963968u,
	  UINT64_C(0x9aa468cda0d05865),
	  { 0x2bb0, 0x2d6e, 0x03cf, 0x2bf4, 0xa18f, 0x4acb, 0x3f70, 0xd4bc } },
	{ EXL3_K2_5,
	  EXL3_BITS_X2_K2_5,
	  3701248u,
	  UINT64_C(0x74df956ae53ff003),
	  { 0x57ce, 0xd7a4, 0xa2c3, 0xc4ab, 0xe9a3, 0x46e6, 0xab67, 0x004a } },
	{ EXL3_K3,
	  EXL3_BITS_X2_K3,
	  4438528u,
	  UINT64_C(0x287d3f9960cc2c14),
	  { 0x3cc0, 0x1747, 0x7329, 0xd56e, 0x8ee0, 0x878f, 0x1e84, 0x3771 } },
	{ EXL3_K4,
	  EXL3_BITS_X2_K4,
	  5913088u,
	  UINT64_C(0x37e393a1b989b40f),
	  { 0xb0f6, 0x29b6, 0x5e8e, 0xc4c9, 0x0ab9, 0x0ede, 0xacff, 0xfa94 } },
};
#define WIDTH_COUNT (sizeof(widths) / sizeof(widths[0]))

static int failures;

static void expect(bool condition, const char *what, uint32_t stored)
{
	if (!condition) {
		(void)fprintf(stderr,
			      "EXL3 widths failure (stored width %u): %s\n",
			      stored, what);
		failures++;
	}
}

static uint32_t splitmix_next(uint64_t *state)
{
	*state += SPLITMIX_INCREMENT;
	uint64_t mixed = *state;
	mixed = (mixed ^ (mixed >> SPLITMIX_FIRST_SHIFT)) * SPLITMIX_FIRST;
	mixed = (mixed ^ (mixed >> SPLITMIX_SECOND_SHIFT)) * SPLITMIX_SECOND;
	return (uint32_t)((mixed ^ (mixed >> SPLITMIX_LAST_SHIFT)) & LOW_WORD);
}

/* The oracle's S(i): its own bit walk, a whole K's K steps or a half K's KA and KA + 1 */
static uint32_t oracle_window_end(uint32_t stored, uint32_t position)
{
	uint32_t end = 0;
	for (uint32_t step = 0; step <= position; step++)
		end += stored <= EXL3_WHOLE_WIDTH_MOST ?
			       stored :
			       stored / EXL3_HALF_WIDTH_SCALE + (step & 1u);
	return end;
}

static void table_check(void)
{
	for (uint32_t stored = 0; stored < STORED_WIDTH_SCAN; stored++) {
		const bool whole = stored >= 1u &&
				   stored <= EXL3_WHOLE_WIDTH_MOST;
		const bool half =
			stored % EXL3_HALF_WIDTH_SCALE ==
				EXL3_HALF_WIDTH_RESIDUE &&
			stored < EXL3_WHOLE_WIDTH_MOST * EXL3_HALF_WIDTH_SCALE;
		const uint32_t bits_x2 =
			whole ? EXL3_HALF_BITS_PER_BIT * stored :
			half  ? stored / EXL3_HALF_WIDTH_RESIDUE :
				0u;
		bool listed = false;
		for (uint32_t index = 0; index < WIDTH_COUNT; index++)
			listed = listed || widths[index].stored == stored;
		expect(exl3_bits_x2(stored) == bits_x2,
		       "bits_x2 of a stored width", stored);
		expect(exl3_width_supported(stored) == listed,
		       "the supported set is K2, K2.5, K3, K4", stored);
		expect((exl3_width_name(bits_x2) != NULL) == listed,
		       "every supported width names its entries", stored);
	}
}

static void ring_check(const struct width_case *width)
{
	const uint32_t ring_bits =
		oracle_window_end(width->stored, EXL3_TILE_VALUES - 1u);
	expect(ring_bits == EXL3_TILE_VALUES * width->bits_x2 /
				    EXL3_HALF_BITS_PER_BIT,
	       "a tile is 128 bits_x2 bits", width->stored);
	bool ends_equal = true, two_words = true;
	for (uint32_t position = 0; position < EXL3_TILE_VALUES; position++)
		ends_equal = ends_equal &&
			     exl3_window_end(position, width->bits_x2) ==
				     oracle_window_end(width->stored, position);
	for (uint32_t lane = 0; lane < LANES; lane++) {
		const uint32_t first =
			oracle_window_end(width->stored, lane * LANE_SYMBOLS);
		const uint32_t begin =
			(first + ring_bits - STATE_BITS) % ring_bits;
		const uint32_t span =
			STATE_BITS +
			oracle_window_end(width->stored, lane * LANE_SYMBOLS +
								 LANE_SYMBOLS -
								 1u) -
			first;
		two_words = two_words &&
			    begin % WORD_BITS + span <= LANE_LOAD_BITS;
	}
	expect(ends_equal,
	       "exl3_window_end is the bit walk's S(i) at every position",
	       width->stored);
	expect(two_words,
	       "every lane's eight windows lie in two words from its start",
	       width->stored);
}

static void known_answer_check(const struct width_case *width)
{
	const uint32_t tile_words = EXL3_TILE_VALUES * width->bits_x2 /
				    EXL3_HALF_BITS_PER_BIT / WORD_BITS;
	const uint32_t tiles = (KAT_SIDE / EXL3_TILE) * (KAT_SIDE / EXL3_TILE);
	uint32_t words[(KAT_SIDE / EXL3_TILE) * (KAT_SIDE / EXL3_TILE) *
		       TILE_WORDS_MOST];
	uint64_t state = KAT_SEED + width->stored;
	for (uint32_t word = 0; word < tiles * tile_words; word++)
		words[word] = splitmix_next(&state);
	uint64_t digest = FNV_OFFSET;
	bool first_equal = true;
	for (uint32_t input = 0; input < KAT_SIDE; input++)
		for (uint32_t output = 0; output < KAT_SIDE; output++) {
			const uint16_t value = test_exl3_state(
				(const unsigned char *)words, KAT_SIDE,
				KAT_SIDE, width->stored, input, output);
			const uint32_t index = input * KAT_SIDE + output;
			if (index < KAT_FIRST_STATES)
				first_equal = first_equal &&
					      value == width->first[index];
			digest = (digest ^ (value & BYTE_MASK)) * FNV_PRIME;
			digest = (digest ^ (uint32_t)(value >> BYTE_BITS)) *
				 FNV_PRIME;
		}
	expect(first_equal, "the first eight known states", width->stored);
	expect(digest == width->digest, "the known states' digest",
	       width->stored);
}

static void format_check(const struct width_case *width)
{
	struct matrix_shape_record shape = { EXPERT_INNER, EXPERT_HIDDEN,
					     width->stored, 0 };
	uint64_t bytes = SENTINEL;
	expect(exl3_weight_operations.stored_byte_count(&shape, &bytes) ==
			       STATUS_OK &&
		       bytes == width->expert_bytes,
	       "an expert matrix's bytes", width->stored);
	expect(exl3_code_bytes(EXPERT_HIDDEN, EXPERT_INNER, width->bits_x2) ==
		       width->expert_bytes -
			       (uint64_t)EXL3_HALF_BYTES *
				       (EXPERT_HIDDEN + EXPERT_INNER),
	       "an expert matrix's code bytes", width->stored);
	uint32_t inner_begin = 0;
	for (uint32_t rank = 0; rank < SHARD_RANKS; rank++) {
		struct bb_exl3_output_shard_plan plan;
		const bool planned = bb_exl3_output_shard_prepare(
					     EXPERT_HIDDEN, EXPERT_INNER,
					     width->stored, SHARD_RANKS, rank,
					     BB_EXL3_REFERENCE_ORDER,
					     &plan) == STATUS_OK;
		struct bb_exl3_slice_sizes sizes = { 0, 0 };
		expect(planned && plan.gate_up.output_begin == inner_begin &&
			       bb_exl3_matrix_slice_prepare(
				       &plan.gate_up, &sizes) == STATUS_OK &&
			       sizes.source_bytes == width->expert_bytes &&
			       sizes.packed_bytes ==
				       exl3_code_bytes(EXPERT_HIDDEN,
						       plan.gate_up.output_count,
						       width->bits_x2) +
					       (uint64_t)EXL3_HALF_BYTES *
						       (EXPERT_HIDDEN +
							plan.gate_up
								.output_count),
		       "a rank's gate/up shard: whole tiles, its bytes",
		       width->stored);
		inner_begin += planned ? plan.gate_up.output_count : 0u;
	}
	expect(inner_begin == EXPERT_INNER,
	       "the ranks' shards cover the inner width", width->stored);
}

/* A 256 x 256 matrix sliced to outputs [128, 256) and to inputs [128, 256): every kept weight keeps its state */
static void slice_check(const struct width_case *width)
{
	const uint64_t code_bytes =
		exl3_code_bytes(SLICE_SIDE, SLICE_SIDE, width->bits_x2);
	const size_t matrix_bytes =
		(size_t)code_bytes +
		(size_t)VECTOR_COUNT * EXL3_HALF_BYTES * SLICE_SIDE;
	unsigned char *const source = malloc(matrix_bytes),
			     *const sliced = malloc(matrix_bytes);
	if (source == NULL || sliced == NULL) {
		free(source);
		free(sliced);
		expect(false, "slice buffers", width->stored);
		return;
	}
	uint64_t state = KAT_SEED ^ width->stored;
	for (size_t byte = 0; byte < matrix_bytes; byte += sizeof(uint32_t)) {
		const uint32_t word = splitmix_next(&state);
		memcpy(source + byte, &word, sizeof(word));
	}
	for (uint32_t form = 0; form < SLICE_FORMS; form++) {
		const bool outputs = form == 0;
		const struct bb_exl3_matrix_slice slice = {
			.input_size = SLICE_SIDE,
			.output_size = SLICE_SIDE,
			.input_begin = outputs ? 0u : SLICE_HALF,
			.input_count = outputs ? SLICE_SIDE : SLICE_HALF,
			.output_begin = outputs ? SLICE_HALF : 0u,
			.output_count = outputs ? SLICE_HALF : SLICE_SIDE,
			.bits_per_weight = width->stored
		};
		bool kept = bb_exl3_matrix_slice_copy(&slice, source,
						      matrix_bytes, sliced,
						      matrix_bytes) ==
			    STATUS_OK;
		for (uint32_t input = 0; kept && input < slice.input_count;
		     input++)
			for (uint32_t output = 0; output < slice.output_count;
			     output++)
				kept = kept &&
				       test_exl3_state(sliced,
						       slice.input_count,
						       slice.output_count,
						       width->stored, input,
						       output) ==
					       test_exl3_state(
						       source, SLICE_SIDE,
						       SLICE_SIDE,
						       width->stored,
						       slice.input_begin +
							       input,
						       slice.output_begin +
							       output);
		expect(kept,
		       outputs ? "an output slice keeps every state" :
				 "an input slice keeps every state",
		       width->stored);
	}
	free(source);
	free(sliced);
}

static void refusal_check(void)
{
	static const struct {
		uint32_t stored;
		result_status status;
	} refusals[] = { { NO_WIDTH_ZERO, STATUS_INVALID_ARGUMENT },
			 { WIDTH_K1, STATUS_UNSUPPORTED },
			 { WIDTH_K5, STATUS_UNSUPPORTED },
			 { WIDTH_K8, STATUS_UNSUPPORTED },
			 { NO_WIDTH_NINE, STATUS_INVALID_ARGUMENT },
			 { WIDTH_K1_5, STATUS_UNSUPPORTED },
			 { NO_WIDTH_ODD_SIXTEENTHS, STATUS_INVALID_ARGUMENT },
			 { WIDTH_K3_5, STATUS_UNSUPPORTED },
			 { WIDTH_K7_5, STATUS_UNSUPPORTED },
			 { NO_WIDTH_K8_SIXTEENTHS, STATUS_INVALID_ARGUMENT } };
	for (uint32_t index = 0; index < sizeof(refusals) / sizeof(refusals[0]);
	     index++) {
		const struct matrix_shape_record shape = {
			EXPERT_INNER, EXPERT_HIDDEN, refusals[index].stored, 0
		};
		uint64_t bytes = SENTINEL;
		struct bb_exl3_output_shard_plan plan;
		memset(&plan, 0, sizeof(plan));
		expect(exl3_weight_operations.stored_byte_count(
			       &shape, &bytes) == refusals[index].status &&
			       bytes == SENTINEL,
		       "a width this build does not decode is refused, the count unchanged",
		       refusals[index].stored);
		expect(bb_exl3_output_shard_prepare(
			       EXPERT_HIDDEN, EXPERT_INNER,
			       refusals[index].stored, SHARD_RANKS, 0,
			       BB_EXL3_REFERENCE_ORDER,
			       &plan) == STATUS_INVALID_ARGUMENT,
		       "no shard plan of a width this build does not decode",
		       refusals[index].stored);
	}
}

/* The backend keeps this opaque; the fake only needs an address */
struct device_card {
	uint32_t unused;
};

/* A kernel request means a refusal came too late */
static uint32_t kernel_requests;
MUST_CHECK static result_status
no_kernel(struct device_card *card, const char *module, const char *kernel,
	  uint32_t argument_bytes, uint32_t *kernel_out)
{
	(void)card;
	(void)module;
	(void)kernel;
	(void)argument_bytes;
	(void)kernel_out;
	kernel_requests++;
	return STATUS_INVALID_ARGUMENT;
}

/* The tensor-core MoE (decode and prefill forms) takes K2, K2.5 and K3; K4 refuses, unsupported, before any kernel */
static void tensor_core_check(void)
{
	for (uint32_t index = 0; index < WIDTH_COUNT; index++)
		expect(exl3_tc_width_supported(widths[index].bits_x2) ==
			       (widths[index].stored != EXL3_K4),
		       "the tensor-core widths are K2, K2.5, K3",
		       widths[index].stored);
	expect(!exl3_tc_width_supported(0u) &&
		       !exl3_tc_width_supported(TC_BEYOND_WIDTHS),
	       "no tensor-core width outside the format", 0u);
	const struct backend_operations backend = { .kernel_prepare =
							    no_kernel };
	struct device_card card_place = { 0 };
	struct device_card *const card = &card_place;
	struct kernel_launch launches[EXL3_TCP_MOST_LAUNCHES];
	uint32_t launch_count = 0;
	const struct exl3_tc_record decode = {
		.row_count = 1u,
		.job_capacity = EXL3_TC_PAIRS_PER_ROW,
		.intermediate = EXL3_TC_MIXED_WIDTH,
		.bits_x2 = EXL3_BITS_X2_K4
	};
	const struct exl3_tcp_record prefill = { .row_count = 1u,
						 .intermediate =
							 EXL3_TC_MIXED_WIDTH,
						 .bits_x2 = EXL3_BITS_X2_K4 };
	expect(exl3_tc_prepare(&decode, SENTINEL, &backend, card, launches) ==
		       STATUS_UNSUPPORTED,
	       "the decode tensor-core MoE refuses K4, unsupported", EXL3_K4);
	expect(exl3_tcp_prepare(&prefill, SENTINEL, &backend, card, launches,
				&launch_count) == STATUS_UNSUPPORTED,
	       "the prefill tensor-core MoE refuses K4, unsupported", EXL3_K4);
	expect(kernel_requests == 0, "before any kernel", EXL3_K4);
}

int main(void)
{
	table_check();
	for (uint32_t index = 0; index < WIDTH_COUNT; index++) {
		ring_check(&widths[index]);
		known_answer_check(&widths[index]);
		format_check(&widths[index]);
		slice_check(&widths[index]);
	}
	refusal_check();
	tensor_core_check();
	(void)printf(
		"EXL3 widths %s: stored widths 0-%u, S(i) and lane windows, known states, bytes, shards, slices "
		"and refusals of K2, K2.5, K3, K4\n",
		failures == 0 ? "PASS" : "FAIL", STORED_WIDTH_SCAN - 1u);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
