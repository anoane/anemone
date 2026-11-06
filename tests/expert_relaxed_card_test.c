/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * expert_relaxed_card_test: on the real cards of class 80, the routed experts in the relaxed order (expert_relaxed.h,
 * expert_relaxed_records.h) against the reference order's chain (linear_routed_experts_prepare with class 80's
 * one slice) and against the host reference in double precision (linear_reference.h), at the model's shapes: 5,120 to
 * 2,304 and back, six experts a row of 384.
 *
 *     expert_relaxed_card_test [--time]
 *     expert_relaxed_card_test --profiled --rows=4096
 * --profiled is timing-only with compact buffers; an external profiler must flush caches for cold timings.
 * --compact keeps all correctness cases with six physical weight sets and one device output set: the reference
 * bytes are saved on the host before distinct output sentinels and the selected chain overwrite that set.
 *
 * The forms: exact and ordered (the reference K32 order), and the relaxed kernels' two: narrow (the job table: the
 * class forms' order
 * of 8 slices, whose bits the expert file built with LINEAR_CLASS_FORMS gave on 26 October,
 * receipts_experts/run2/card_test_v10_correctness.log) and wide (the tiles of a wide select, its plain and small
 * kernels: one chain of folded weights).
 * Other cards check that both PRO arithmetic classes refuse the CMP-only kernel table without changing it.
 *
 * Cases (each on its own synthetic numbers, linear_card_fixture.h, CORRECT_PHYSICAL weight sets that the 384 experts
 * share): one row; eight rows of plain and edge values whose experts repeat (jobs of several columns), with an empty
 * job filled with rubbish first and last; 200 rows with one expert in every row (a run of 200 pairs: four tiles, the
 * last ragged); 96 rows over 32 experts (runs of about 18 pairs: tiles for both wide kernels, of every group count
 * from about 1 to 4); 64 rows of edge values whose gate and up products cancel (pairs of inputs of opposite values meet
 * pairs of equal weights in seven of eight blocks); 64 rows whose down products cancel (gate's and up's output rows
 * come in pairs, up's negated, so the hidden vector holds pairs of opposite values; down's weights pair in seven of
 * eight words);
 * eight rows whose down scale codes are 95 to 97, 207 to 209 and 229 to 231 by output (the ends of the relaxed
 * range, and beyond the carried form's, where both orders cut the power alike: those outputs are held to the exact
 * form's alone).
 *
 * FAILED lines (the test fails on any):
 *   - every hidden value and pair output of the sampled pairs lies within its form's tolerance of the double products:
 *     linear_reference.h's for the carried chains (exact: 1 slice; narrow: 8), for the wide chain its bound
 *     with 2 K units of 2^-24 m for K inputs in place of its blocks (each product cut once to the grid of a sum no
 *     larger than m, twice over);
 *   - the relaxed forms give the same bits when run again, and a row launched alone (its six experts each a job of
 *     one column and a tile of one pair, arrays of its own) the bits it has in company;
 *   - no launch writes beyond its ranges (the guards), the ranges only read and the matrices come back untouched.
 * Measured (lines "measure ..." on the standard output, key=value): each relaxed form's hidden values against the
 * exact form's (the same inputs), its pair outputs against the exact form's on the exact form's hidden vectors (down
 * alone), its row sums against the exact chain's; each form against the double reference: bit-equal share, largest
 * distance in bfloat16 last places, relative RMS, largest relative error of an output, largest error over m.
 * Joined cases at 64, 256, 1,024 and 4,096 rows use the real GPU router's jobs, tiles and pairs. All six stored
 * intermediates of the ordered expert chain must equal exact, including the hidden quantization's codes,
 * exponents and carried operands. --chunk-only skips the edge/reference cases; --rows=N selects one joined size.
 * --time (after the cases) or --time-only: for 1, 8, 64, 256, 1,024 and 4,096 rows of uniform routing (or the one
 * count --rows=N; TIMED_PHYSICAL weight sets: rows of up to 8 read distinct ones), each form's gate with up, down, and
 * whole chain (the two, the hidden quantization, the row sum), each run behind the card's read of FLUSH_BYTES (its
 * digest: clean lines through the L2); the median host time of TIMED_RUNS runs (lines "time ...": launch and wait
 * included; the card's own times are a trace's, nsys), with TFLOP/s and the weights' GB/s (each expert's matrices
 * once: the floor of any form). Lock the clocks first (clocks_lock.sh).
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/expert_relaxed.h"
#include "anemone/linear.h"
#include "anemone/linear_records.h"
#include "anemone/model_layout.h"
#include "anemone/router.h"
#include "anemone/status.h"
#include "anemone/wide_linear_records.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"

#define TEST_TIME_LIMIT_SECONDS 3600
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define HIDDEN_SIZE MODEL_HIDDEN
#define INNER_SIZE 2304u /* the model's moe_inter_dim (model_v41.h) */
#define PAIRS_PER_ROW MODEL_EXPERTS_PER_ROW
#define LOGICAL_EXPERTS MODEL_EXPERTS
#define MATRICES MODEL_EXPERT_MATRICES
#define JOB_COLUMNS LINEAR_MAXIMUM_ROWS
#define TILE_PAIRS WIDE_TILE_PAIRS
#define CORRECT_PHYSICAL 16u /* weight sets of the cases */
#define TIMED_PHYSICAL \
	48u /* of the timings: eight rows touch at most 48 experts */
#define TIMED_POOLED_PHYSICAL \
	16u /* large batches still expose 384 logical experts; DRAM traffic is measured */
#define PROFILED_PHYSICAL \
	6u /* fits beside a full production arena; report measured DRAM aliasing */
#define MOST_TIMED_ROWS MODEL_WIDE_MOST_ROWS
#define E2M1 WEIGHT_FORMAT_E2M1_SCALED
#define E2M1_BITS_PER_WEIGHT 4u
#define E2M1_CODES_PER_BYTE 2u
#define ACTIVATION_LIMIT 10.0f /* the model's */
#define VALUE_BYTES ((uint64_t)LINEAR_BFLOAT16_BYTES)
#define SUM_BYTES ((uint64_t)LINEAR_SUM_BYTES)
#define VALUES_PER_BLOCK LINEAR_VALUES_PER_BLOCK
#define VALUES_PER_LOAD LINEAR_VALUES_PER_LOAD
#define WORK_CAPACITY \
	(1536ull      \
	 << 20u) /* two joined chains at 4,096 rows, with guards and routing */
#define COMPACT_WORK_CAPACITY \
	(720ull               \
	 << 20u) /* one joined chain; its reference bytes are saved on the host */
#define TIMED_WORK_CAPACITY \
	(800ull << 20u) /* one chain, rows, router scratch and L2 flush */
#define TIMED_THIN_WORK_CAPACITY (80ull << 20u)
#define PROFILED_WORK_CAPACITY \
	(680ull << 20u) /* reuse source rows; the profiler flushes L2 */
#define MATRIX_BYTES                               \
	((uint64_t)INNER_SIZE * HIDDEN_SIZE / 2u + \
	 (uint64_t)INNER_SIZE * HIDDEN_SIZE / VALUES_PER_BLOCK)
#define WEIGHTS_CAPACITY ((uint64_t)TIMED_PHYSICAL * MATRICES * MATRIX_BYTES)
#define FLUSH_BYTES (64ull << 20u) /* twice the 32 MiB L2 of a CMP 170HX */
#define EDGE_EXPERT_POOL \
	12u /* the eight-row case's experts: repeated among its rows */
#define POOLED_EXPERT_POOL \
	32u /* the pooled case's: runs of 18 pairs on average */
#define SKEW_EXPERT 7u /* in every row of the skewed case */
#define EDGE_FIRST_ROW 4u /* rows from here hold the edges */
#define GATE_UP_SCALE_BASE \
	119u /* gate and up reach beyond the activation limit */
#define DOWN_SCALE_BASE 118u
#define SCALE_SPREAD 4u
#define SMALL_SCALE_BASE 95u /* EXPERT_RELAXED_SCALE_SMALLEST */
#define EXTREME_SPREAD 3u
#define LARGE_SCALE_BASE                     \
	(EXPERT_RELAXED_SCALE_LARGEST + 1u - \
	 EXTREME_SPREAD) /* up to the largest */
#define BEYOND_SCALE_BASE               \
	(EXPERT_CARRIED_SCALE_LARGEST + \
	 1u) /* both orders cut the power alike */
#define SCALE_CLASSES 3u /* small, large, beyond: by output */
#define BEYOND_CLASS 2u
#define CANCEL_SHARE 8u /* seven of eight blocks or words cancel */
#define WORD_BITS 32u
#define HALF_WORD_MASK 0xFFFFu
#define BOTH_HALVES 0x00010001u /* a half times this: the same in both halves */
#define CODE_SIGNS 0x88888888u /* the sign bit of each of eight E2M1 codes */
#define BFLOAT16_SIGN 0x8000u
#define BYTE_BITS 8u
#define BYTE_MASK 0xFFu
#define SCALE_BYTES_A_WORD 4u
#define OUTPUT_HALVES 2u
#define OUTPUTS_PER_HALF_TILE 8u
#define LANES 32u
#define LANES_PER_GROUP 4u
#define LANE_GROUPS 8u
#define WORDS_PER_LANE_LOAD 4u
#define WORDS_PER_GROUP_LOAD ((uint64_t)LANES_PER_GROUP * WORDS_PER_LANE_LOAD)
#define ROUTE_WEIGHT_STEPS 12u /* a route weight is 1 to 12 eighths */
#define ROUTE_WEIGHT_STEP 0.125f
#define RUBBISH_BYTE 0xA7
#define SAMPLED_PAIRS \
	6u /* checked against the double reference, the last pair besides */
#define CHAIN_STEP_UNITS \
	2u /* units of 2^-24 m an input of a relaxed chain may cost */
#define EMPTY_END_JOBS 2u /* an empty job first and last */
#define SCALE_STREAM       \
	(STREAM_PER_CASE / \
	 2u) /* a matrix's scale codes, beside its words' stream */
#define CANCEL_STREAM      \
	(STREAM_PER_CASE / \
	 2u) /* which blocks of a row cancel, beside its values' stream */
#define MEDIAN_RUN (TIMED_RUNS / 2u)
#define ROWS_OPTION "--rows="
#define DECIMAL 10
#define INPUT_PAIR 2u /* inputs 2 k and 2 k + 1: opposite values */
#define TIMED_ROW_COUNTS 7u
#define MOST_LAUNCHES EXPERT_RELAXED_WIDE_LAUNCH_COUNT
#define WIDE_PRODUCT_LAUNCHES \
	(EXPERT_RELAXED_WIDE_SMALL_GATE_UP - EXPERT_RELAXED_WIDE_GATE_UP + 1u)
#define TIMED_RUNS 9u
#define NANOSECONDS_PER_SECOND 1000000000ull
#define NANOSECONDS_PER_MICROSECOND 1000.0
#define FLOP_PER_PRODUCT 2.0
#define READ_RATE_GIGABYTES \
	1440.0 /* a CMP 170HX's measured read rate (research/speed/LEDGER.md) */
#define TERAFLOP_PER_SECOND_IN_FLOP_PER_MICROSECOND 1.0e6
#define GIGABYTE_PER_SECOND_IN_BYTES_PER_MICROSECOND 1.0e3
#define READ_RATE_BYTES_PER_MICROSECOND \
	(READ_RATE_GIGABYTES * GIGABYTE_PER_SECOND_IN_BYTES_PER_MICROSECOND)
#define PERCENT 100.0
#define STREAM_PER_CASE \
	0x1000000ull /* the streams of the hashes, far from those of the other tests */
#define RELAXED_STREAM 0x13000000ull
#define ROUTE_STREAM 0x10000ull
#define WEIGHT_STREAM 0x20000ull
#define VALUE_STREAM 0x400000ull
#define CHOICE_STREAM 0x30000ull
#define SCORE_STEPS 4096u
#define SCORE_CENTER 2048
#define SCORE_UNIT 0.00390625f
#define OUTPUT_SENTINEL 0xA5
#define SELECTED_SENTINEL 0x5A
#define HIDDEN_RMS_LIMIT \
	0.00015 /* the supplied final report's largest ordinary operator envelope */
#define DOWN_RMS_LIMIT \
	0.000062 /* 6.1e-5 reported, allowing its printed rounding */
#define ROW_SUM_RMS_LIMIT \
	0.00019 /* 1.8e-4 reported for the cancelling joined row sum */
#define COMPOUND_PAIR_RMS_LIMIT \
	0.000062 /* separate joined-chain gate; the report's down-only measurements hold the hidden input fixed */
#define JOINED_FIRST_TIMED 2u /* 64, 256, 1,024 and 4,096 rows */
#define CARRIED_ONE 0x7180u /* 2^100, the carried representation of one */
#define CARRIED_POSITIVE_INFINITY 0x7F80u
#define CARRIED_NEGATIVE_INFINITY 0xFF80u
#define CARRIED_QUIET_NAN 0x7FC1u
#define CARRIED_NEGATIVE_ZERO 0x8000u
#define SPECIAL_INPUTS_PER_CASE 8u
#define SPECIAL_ROWS \
	64u /* one expert has a full large tile, the other experts have small tiles */

enum case_kind {
	KIND_PLAIN,
	KIND_EDGES,
	KIND_SKEW,
	KIND_POOLED,
	KIND_CANCEL_GATE_UP,
	KIND_CANCEL_DOWN,
	KIND_EXTREME_SCALES
};
enum form { FORM_EXACT, FORM_NARROW, FORM_WIDE, FORM_ORDERED, FORM_COUNT };
enum matrix_kind { MATRIX_GATE, MATRIX_UP, MATRIX_DOWN };
static const char *const form_names[FORM_COUNT] = { "exact", "narrow", "wide",
						    "ordered" };
static const uint32_t form_slices[FORM_COUNT] = { 1,
						  EXPERT_RELAXED_NARROW_SLICES,
						  1, 1 };
static const uint32_t timed_row_counts[TIMED_ROW_COUNTS] = {
	1, 8, 64, 256, 512, 1024, MOST_TIMED_ROWS
};

struct relaxed_case {
	const char *name;
	uint32_t row_count;
	enum case_kind kind;
};
static const struct relaxed_case relaxed_cases[] = {
	{ "one row", 1, KIND_PLAIN },
	{ "eight rows, edges, experts repeated", JOB_COLUMNS, KIND_EDGES },
	{ "200 rows, one expert in every row", 200, KIND_SKEW },
	{ "96 rows over 32 experts", 96, KIND_POOLED },
	{ "64 rows, gate and up cancel", 64, KIND_CANCEL_GATE_UP },
	{ "64 rows, down cancels", 64, KIND_CANCEL_DOWN },
	{ "eight rows, down scales at the ends", JOB_COLUMNS,
	  KIND_EXTREME_SCALES },
};

/* Where a form's chain holds its launches: the wide form runs two kernels a product (expert_relaxed.h). */
struct chain_layout {
	uint32_t launch_count, gate_up_first, down_first, product_launches;
};

/* What a card runs beside the linear's kernels. */
struct card_kernels {
	struct expert_relaxed_kernels relaxed;
	struct router_kernels_record router;
};

/* A batch's routing: pair row * 6 + slot holds the slot-th of the row's experts in ascending order. */
struct routing {
	uint32_t row_count, pair_count;
	uint32_t *experts; /* [pair] */
	float *route_weights; /* [pair] */
	struct wide_pair *sorted; /* [pair]: by expert, then row */
	uint32_t run_first[LOGICAL_EXPERTS], run_count[LOGICAL_EXPERTS];
	uint32_t distinct_count, job_count,
		tile_count; /* jobs and tiles that hold pairs */
};

/* The ranges every form of a case reads. */
struct case_inputs {
	struct linear_range jobs, tiles, pairs, codes, exponents, operands,
		weights;
	struct linear_range matrices[TIMED_PHYSICAL][MATRICES];
	uint32_t physical_count, job_entries,
		tile_entries; /* entries of the tables, the empty ones too */
};

/* The arrays one chain writes. */
struct chain_arrays {
	struct linear_range hidden, hidden_codes, hidden_exponents,
		hidden_operands, pair_output, row_output;
};

/* What a comparison of outputs measured. */
struct difference {
	uint64_t count, equal_count, outside_count, zero_reference_count;
	uint32_t largest_distance;
	double squared_difference, squared_reference, largest_relative,
		largest_over_magnitude;
};

static const struct backend_operations *const backend = &backend_cuda_value;

static uint64_t now_nanoseconds(void)
{
	struct timespec now;
	return clock_gettime(CLOCK_MONOTONIC, &now) == 0 ?
		       (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND +
			       (uint64_t)now.tv_nsec :
		       0;
}

static uint32_t hash32(uint64_t stream, uint64_t position)
{
	return (uint32_t)test_linear_hash(stream, position);
}

/* ---- routing ---- */

static void routing_free(struct routing *routing)
{
	free(routing->experts);
	free(routing->route_weights);
	free(routing->sorted);
	memset(routing, 0, sizeof(*routing));
}

/* Six distinct experts a row from the first `pool` (SKEW_EXPERT in every row of a skewed case), weights of 1 to 12
 * eighths, and the pairs sorted by expert, then row, as the router's wide select sorts them. */
static bool routing_make(struct routing *routing, uint32_t row_count,
			 uint32_t pool, bool is_skewed, uint64_t stream)
{
	memset(routing, 0, sizeof(*routing));
	routing->row_count = row_count;
	routing->pair_count = row_count * PAIRS_PER_ROW;
	routing->experts =
		malloc((size_t)routing->pair_count * sizeof(uint32_t));
	routing->route_weights =
		malloc((size_t)routing->pair_count * sizeof(float));
	routing->sorted =
		malloc((size_t)routing->pair_count * sizeof(struct wide_pair));
	if (!test_linear_expect(routing->experts != NULL &&
					routing->route_weights != NULL &&
					routing->sorted != NULL,
				"the routing is allocated")) {
		routing_free(routing);
		return false;
	}
	uint32_t placed[LOGICAL_EXPERTS] = { 0 };
	for (uint32_t row = 0; row < row_count; row++) {
		bool chosen[LOGICAL_EXPERTS] = { false };
		uint32_t count = 0;
		if (is_skewed) {
			chosen[SKEW_EXPERT] = true;
			count++;
		}
		for (uint64_t draw = 0; count < PAIRS_PER_ROW; draw++) {
			const uint32_t expert =
				hash32(stream + ROUTE_STREAM + row, draw) %
				pool;
			count += chosen[expert] ? 0u : 1u;
			chosen[expert] = true;
		}
		for (uint32_t expert = 0, slot = 0; expert < LOGICAL_EXPERTS;
		     expert++) {
			if (chosen[expert]) {
				const uint32_t pair =
					row * PAIRS_PER_ROW + slot++;
				routing->experts[pair] = expert;
				routing->route_weights[pair] =
					(float)(1u +
						hash32(stream + CHOICE_STREAM,
						       pair) %
							ROUTE_WEIGHT_STEPS) *
					ROUTE_WEIGHT_STEP;
				routing->run_count[expert]++;
			}
		}
	}
	for (uint32_t expert = 0, first = 0; expert < LOGICAL_EXPERTS;
	     expert++) {
		routing->run_first[expert] = first;
		first += routing->run_count[expert];
		routing->distinct_count +=
			routing->run_count[expert] != 0 ? 1u : 0u;
		routing->job_count +=
			(routing->run_count[expert] + JOB_COLUMNS - 1u) /
			JOB_COLUMNS;
		routing->tile_count +=
			(routing->run_count[expert] + TILE_PAIRS - 1u) /
			TILE_PAIRS;
	}
	for (uint32_t pair = 0; pair < routing->pair_count;
	     pair++) { /* pairs ascend: so do the rows of a run */
		const uint32_t expert = routing->experts[pair];
		routing->sorted[routing->run_first[expert] + placed[expert]++] =
			(struct wide_pair){ pair,
					    routing->route_weights[pair] };
	}
	return true;
}

/* ---- the image of a case ---- */

/* Word index of a matrix's bytes in the image, little-endian as every card reads it. */
static uint32_t word_at(const unsigned char *bytes, uint64_t index)
{
	uint32_t word = 0;
	memcpy(&word, bytes + index * sizeof(word), sizeof(word));
	return word;
}
static void word_put(unsigned char *bytes, uint64_t index, uint32_t word)
{
	memcpy(bytes + index * sizeof(word), &word, sizeof(word));
}

/* The matrices of physical_count weight sets, a case's kind shaping them (see the top). */
static void weights_fill(struct test_linear_card *card,
			 const struct case_inputs *inputs, enum case_kind kind,
			 uint64_t stream)
{
	for (uint32_t physical = 0; physical < inputs->physical_count;
	     physical++) {
		for (uint32_t matrix = 0; matrix < MATRICES; matrix++) {
			unsigned char *const bytes = test_linear_card_writable(
				card, inputs->matrices[physical][matrix]);
			const bool is_down = matrix == MATRIX_DOWN;
			const uint32_t output_size = is_down ? HIDDEN_SIZE :
							       INNER_SIZE;
			const uint32_t load_count =
				(is_down ? INNER_SIZE : HIDDEN_SIZE) /
				VALUES_PER_LOAD;
			const uint64_t tile_loads = (uint64_t)output_size /
						    LINEAR_OUTPUTS_PER_TILE *
						    load_count;
			const uint64_t word_count =
				tile_loads * LANES * WORDS_PER_LANE_LOAD;
			const uint64_t scale_word_count =
				tile_loads * LANE_GROUPS;
			const uint64_t matrix_stream =
				stream + WEIGHT_STREAM +
				((uint64_t)physical * MATRICES + matrix) *
					STREAM_PER_CASE;
			if (bytes == NULL) {
				test_linear_fail(
					"a matrix does not lie in the weights region");
				return;
			}
			unsigned char *const scale_bytes =
				bytes + word_count * sizeof(uint32_t);
			const bool pairs_rows =
				kind == KIND_CANCEL_DOWN &&
				!is_down; /* lane group 2 k + 1 copies 2 k */
			const bool pairs_inputs =
				(kind == KIND_CANCEL_GATE_UP && !is_down) ||
				(kind == KIND_CANCEL_DOWN && is_down);
			for (uint64_t word = 0; word < word_count; word++) {
				const uint64_t hash =
					test_linear_hash(matrix_stream, word);
				const uint32_t lane_group =
					(uint32_t)(word / WORDS_PER_LANE_LOAD %
						   LANES / LANES_PER_GROUP);
				uint32_t bits = (uint32_t)hash;
				if (pairs_inputs &&
				    (hash >> WORD_BITS) % CANCEL_SHARE != 0) {
					bits = (bits & HALF_WORD_MASK) *
					       BOTH_HALVES; /* inputs 2 t + h, h = 0 and 1: one weight */
				}
				if (pairs_rows &&
				    lane_group % OUTPUT_HALVES != 0) {
					bits = word_at(bytes,
						       word - WORDS_PER_GROUP_LOAD) ^
					       (matrix == MATRIX_UP ?
							CODE_SIGNS :
							0u);
				}
				word_put(bytes, word, bits);
			}
			for (uint64_t scale_word = 0;
			     scale_word < scale_word_count; scale_word++) {
				const uint32_t lane_group =
					(uint32_t)(scale_word % LANE_GROUPS);
				const uint32_t tile =
					(uint32_t)(scale_word / LANE_GROUPS /
						   load_count);
				uint32_t bits = 0;
				for (uint32_t byte = 0;
				     byte < SCALE_BYTES_A_WORD; byte++) {
					const uint32_t hash = hash32(
						matrix_stream + SCALE_STREAM,
						scale_word * SCALE_BYTES_A_WORD +
							byte);
					const uint32_t output =
						tile * LINEAR_OUTPUTS_PER_TILE +
						lane_group +
						byte % OUTPUT_HALVES *
							OUTPUTS_PER_HALF_TILE;
					static const uint32_t
						extreme_bases[SCALE_CLASSES] = {
							SMALL_SCALE_BASE,
							LARGE_SCALE_BASE,
							BEYOND_SCALE_BASE
						};
					const uint32_t code =
						kind == KIND_EXTREME_SCALES &&
								is_down ?
							extreme_bases[output %
								      SCALE_CLASSES] +
								hash % EXTREME_SPREAD :
							(is_down ?
								 DOWN_SCALE_BASE :
								 GATE_UP_SCALE_BASE) +
								hash % SCALE_SPREAD;
					bits |= code << (byte * BYTE_BITS);
				}
				word_put(
					scale_bytes, scale_word,
					pairs_rows && lane_group % OUTPUT_HALVES !=
								0 ?
						word_at(scale_bytes,
							scale_word - 1u) :
						bits);
			}
		}
	}
}

/* A row's values: plain, the edges from EDGE_FIRST_ROW on in a case of edges; where gate and up cancel, the edges
 * (magnitudes far apart in a block: sums a float32 cannot hold exactly) with opposite pairs of inputs in seven of eight
 * blocks. */
static void row_values(uint16_t *values, uint32_t row, enum case_kind kind,
		       uint64_t stream)
{
	const bool is_edges = (kind == KIND_EDGES && row >= EDGE_FIRST_ROW) ||
			      kind == KIND_CANCEL_GATE_UP;
	test_linear_values(values, HIDDEN_SIZE, stream + VALUE_STREAM + row,
			   is_edges ? TEST_LINEAR_VALUES_EDGES :
				      TEST_LINEAR_VALUES_PLAIN);
	for (uint32_t value = 0;
	     kind == KIND_CANCEL_GATE_UP && value < HIDDEN_SIZE;
	     value += INPUT_PAIR) {
		if (hash32(stream + VALUE_STREAM + CANCEL_STREAM + row,
			   value / VALUES_PER_BLOCK) %
			    CANCEL_SHARE !=
		    0) {
			values[value + 1u] =
				(uint16_t)(values[value] ^ BFLOAT16_SIGN);
		}
	}
}

/* The job table (runs cut in 8s, ascending experts; empty_ends: an empty job of rubbish first and last), the tiles
 * (runs cut in 64s; an empty tile last) and the sorted pairs of a routing, into the image. */
static void tables_put(struct test_linear_card *card,
		       const struct case_inputs *inputs,
		       const struct routing *routing, bool empty_ends)
{
	struct expert_job_record job;
	uint32_t entry = 0, tile_entry = 0;
	memset(&job, RUBBISH_BYTE, sizeof(job));
	job.row_count = 0;
	if (empty_ends) {
		(void)test_linear_card_put(card, inputs->jobs,
					   (uint64_t)entry++ * sizeof(job),
					   &job, sizeof(job));
	}
	for (uint32_t expert = 0; expert < LOGICAL_EXPERTS; expert++) {
		const uint32_t physical = expert % inputs->physical_count;
		const uint64_t offsets[MATRICES] = {
			inputs->matrices[physical][MATRIX_GATE].byte_offset,
			inputs->matrices[physical][MATRIX_UP].byte_offset,
			inputs->matrices[physical][MATRIX_DOWN].byte_offset
		};
		for (uint32_t first = 0; first < routing->run_count[expert];
		     first += JOB_COLUMNS) {
			memset(&job, RUBBISH_BYTE,
			       sizeof(job)); /* the columns not in use */
			job.gate_offset = offsets[MATRIX_GATE];
			job.up_offset = offsets[MATRIX_UP];
			job.down_offset = offsets[MATRIX_DOWN];
			job.expert = 0;
			job.row_count = 0;
			for (uint32_t column = 0;
			     column < JOB_COLUMNS &&
			     first + column < routing->run_count[expert];
			     column++) {
				const struct wide_pair pair =
					routing->sorted
						[routing->run_first[expert] +
						 first + column];
				job.pair_index[column] = pair.pair_index;
				job.route_weight[column] = pair.route_weight;
				job.row_count++;
			}
			(void)test_linear_card_put(card, inputs->jobs,
						   (uint64_t)entry++ *
							   sizeof(job),
						   &job, sizeof(job));
		}
		for (uint32_t first = 0; first < routing->run_count[expert];
		     first += TILE_PAIRS) {
			const uint32_t left =
				routing->run_count[expert] - first;
			const struct wide_expert_tile tile = {
				offsets[MATRIX_GATE], offsets[MATRIX_UP],
				offsets[MATRIX_DOWN],
				routing->run_first[expert] + first,
				left < TILE_PAIRS ? left : TILE_PAIRS
			};
			(void)test_linear_card_put(card, inputs->tiles,
						   (uint64_t)tile_entry++ *
							   sizeof(tile),
						   &tile, sizeof(tile));
		}
	}
	memset(&job, RUBBISH_BYTE, sizeof(job));
	job.row_count = 0;
	if (empty_ends) {
		(void)test_linear_card_put(card, inputs->jobs,
					   (uint64_t)entry * sizeof(job), &job,
					   sizeof(job));
	}
	(void)test_linear_card_put(card, inputs->pairs, 0, routing->sorted,
				   (uint64_t)routing->pair_count *
					   sizeof(struct wide_pair));
}

/* The ranges of the inputs of a routing (the rows' arrays, the tables, physical_count weight sets). */
static void inputs_set_aside(struct test_linear_card *card,
			     struct case_inputs *inputs,
			     const struct routing *routing,
			     uint32_t physical_count, bool empty_ends)
{
	memset(inputs, 0, sizeof(*inputs));
	inputs->physical_count = physical_count;
	inputs->job_entries =
		routing->job_count + (empty_ends ? EMPTY_END_JOBS : 0u);
	inputs->tile_entries = routing->tile_count + 1u;
	inputs->jobs = test_linear_card_work(
		card, linear_jobs_byte_count(inputs->job_entries));
	inputs->tiles = test_linear_card_work(
		card, (uint64_t)inputs->tile_entries *
			      sizeof(struct wide_expert_tile));
	inputs->pairs = test_linear_card_work(
		card, (uint64_t)routing->pair_count * sizeof(struct wide_pair));
	inputs->codes = test_linear_card_work(
		card, linear_codes_byte_count(routing->row_count, HIDDEN_SIZE));
	inputs->exponents = test_linear_card_work(
		card,
		linear_exponents_byte_count(routing->row_count, HIDDEN_SIZE));
	inputs->operands = test_linear_card_work(
		card,
		linear_values_byte_count(routing->row_count, HIDDEN_SIZE));
	for (uint32_t physical = 0; physical < physical_count; physical++) {
		for (uint32_t matrix = 0; matrix < MATRICES; matrix++) {
			inputs->matrices[physical][matrix] =
				test_linear_card_weights(card, MATRIX_BYTES);
		}
	}
	inputs->weights = test_linear_card_all_weights(card);
}

static void chain_set_aside(struct test_linear_card *card,
			    struct chain_arrays *arrays, uint32_t row_count)
{
	const uint64_t pairs = (uint64_t)row_count * PAIRS_PER_ROW;
	arrays->hidden = test_linear_card_work(
		card, linear_values_byte_count(pairs, INNER_SIZE));
	arrays->hidden_codes = test_linear_card_work(
		card, linear_codes_byte_count(pairs, INNER_SIZE));
	arrays->hidden_exponents = test_linear_card_work(
		card, linear_exponents_byte_count(pairs, INNER_SIZE));
	arrays->hidden_operands = test_linear_card_work(
		card, linear_values_byte_count(pairs, INNER_SIZE));
	arrays->pair_output = test_linear_card_work(
		card, linear_values_byte_count(pairs, HIDDEN_SIZE));
	arrays->row_output = test_linear_card_work(
		card, linear_sums_byte_count(row_count, HIDDEN_SIZE));
}

/* ---- the launches of a form ---- */

static struct chain_layout layout_of(enum form form)
{
	return form >= FORM_WIDE ?
		       (struct chain_layout){ EXPERT_RELAXED_WIDE_LAUNCH_COUNT,
					      EXPERT_RELAXED_WIDE_GATE_UP,
					      EXPERT_RELAXED_WIDE_DOWN,
					      WIDE_PRODUCT_LAUNCHES } :
		       (struct chain_layout){ LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT,
					      LINEAR_ROUTED_EXPERTS_GATE_UP,
					      LINEAR_ROUTED_EXPERTS_DOWN, 1u };
}

static struct linear_routed_experts_description
experts_of(const struct case_inputs *inputs, const struct chain_arrays *arrays,
	   uint32_t row_count, uint32_t slice_count)
{
	struct linear_routed_experts_description experts;
	memset(&experts, 0, sizeof(experts));
	experts.jobs = inputs->jobs;
	experts.expert_weights = inputs->weights;
	experts.codes = inputs->codes;
	experts.exponents = inputs->exponents;
	experts.operands = inputs->operands;
	experts.hidden = arrays->hidden;
	experts.hidden_codes = arrays->hidden_codes;
	experts.hidden_exponents = arrays->hidden_exponents;
	experts.hidden_operands = arrays->hidden_operands;
	experts.pair_output = arrays->pair_output;
	experts.row_output = arrays->row_output;
	experts.gate_up_shape =
		(struct matrix_shape_record){ INNER_SIZE, HIDDEN_SIZE,
					      E2M1_BITS_PER_WEIGHT, 0 };
	experts.down_shape =
		(struct matrix_shape_record){ HIDDEN_SIZE, INNER_SIZE,
					      E2M1_BITS_PER_WEIGHT, 0 };
	experts.weight_format = E2M1;
	experts.job_count = inputs->job_entries;
	experts.row_count = row_count;
	experts.pairs_per_row = PAIRS_PER_ROW;
	experts.gate_up_slice_count = slice_count;
	experts.down_slice_count = slice_count;
	experts.activation_limit = ACTIVATION_LIMIT;
	return experts;
}

/* The four launches of a form's chain over a case's inputs into arrays. */
static bool form_prepare(const struct test_linear_card *card,
			 const struct card_kernels *kernels, enum form form,
			 const struct case_inputs *inputs,
			 const struct chain_arrays *arrays, uint32_t row_count,
			 uint32_t tile_count, struct kernel_launch *launches)
{
	const struct linear_routed_experts_description experts =
		experts_of(inputs, arrays, row_count, 1u);
	if (form == FORM_EXACT) {
		return test_linear_expect_status(
			linear_routed_experts_prepare(&experts,
						      card->arena_byte_count,
						      &card->kernels, launches),
			STATUS_OK, "the reference order's chain is prepared");
	}
	const struct expert_relaxed_description description = {
		experts, inputs->tiles, inputs->pairs,
		form >= FORM_WIDE ? tile_count : 0u, 0
	};
	const result_status status =
		form == FORM_ORDERED ?
			expert_ordered_prepare(&description,
					       card->arena_byte_count,
					       &card->kernels,
					       &kernels->relaxed, launches) :
			expert_relaxed_prepare(&description,
					       card->arena_byte_count,
					       &card->kernels,
					       &kernels->relaxed, launches);
	return test_linear_expect_status(
		status, STATUS_OK, "the selected expert chain is prepared");
}

/* ---- comparisons ---- */

/* The value of a float32 held as its bits. */
static double float_of_bits(uint32_t bits)
{
	float value = 0.0f;
	memcpy(&value, &bits, sizeof(value));
	return (double)value;
}

static void difference_add(struct difference *self, double value,
			   double reference, bool is_equal, uint32_t distance,
			   double magnitude, bool is_outside)
{
	const double error = value - reference;
	self->count++;
	self->equal_count += is_equal ? 1u : 0u;
	self->outside_count += is_outside ? 1u : 0u;
	self->largest_distance = distance > self->largest_distance ?
					 distance :
					 self->largest_distance;
	self->squared_difference += error * error;
	self->squared_reference += reference * reference;
	if (reference != 0.0) {
		const double relative = fabs(error) / fabs(reference);
		self->largest_relative = relative > self->largest_relative ?
						 relative :
						 self->largest_relative;
	} else {
		self->zero_reference_count += value != 0.0 ? 1u : 0u;
	}
	if (magnitude > 0.0) {
		const double over = fabs(error) / magnitude;
		self->largest_over_magnitude =
			over > self->largest_over_magnitude ?
				over :
				self->largest_over_magnitude;
	}
}

/* Two bfloat16 outputs, both finite (an output not finite in either is counted outside unless both are). */
static void difference_add_bfloat16(struct difference *self, uint16_t value,
				    uint16_t reference)
{
	if (!test_linear_bfloat16_is_finite(value) ||
	    !test_linear_bfloat16_is_finite(reference)) {
		self->outside_count +=
			test_linear_bfloat16_is_finite(value) !=
					test_linear_bfloat16_is_finite(
						reference) ?
				1u :
				0u;
		return;
	}
	const int32_t distance = test_linear_bfloat16_ordinal(value) -
				 test_linear_bfloat16_ordinal(reference);
	difference_add(self, test_linear_bfloat16_value(value),
		       test_linear_bfloat16_value(reference),
		       value == reference,
		       (uint32_t)(distance < 0 ? -distance : distance), 0.0,
		       false);
}

static void difference_print(const char *case_name, const char *form,
			     const char *quantity, const char *against,
			     const struct difference *self)
{
	(void)printf(
		"measure case=\"%s\" form=%s quantity=%s against=%s outputs=%" PRIu64
		" bit_equal=%.4f%% "
		"largest_ulp=%" PRIu32
		" rel_rms=%.3e largest_relative=%.3e largest_over_m=%.3e outside=%" PRIu64
		" zero_reference_differ=%" PRIu64 "\n",
		case_name, form, quantity, against, self->count,
		test_linear_percent(self->equal_count, self->count),
		self->largest_distance,
		self->squared_reference > 0.0 ? sqrt(self->squared_difference /
						     self->squared_reference) :
						0.0,
		self->largest_relative, self->largest_over_magnitude,
		self->outside_count, self->zero_reference_count);
}

/* The slice count linear_reference.h's tolerance takes for a form: the carried chains (exact, narrow) theirs,
 * its bound T + B + S units of 2^-24 m; the wide chain of K inputs T + 2 K + 1 (a product cut once to the grid of a
 * sum no larger than m, twice over), as the slice count that gives it. */
static uint32_t tolerance_slices(enum form form, uint32_t input_size)
{
	return form == FORM_WIDE ? CHAIN_STEP_UNITS * input_size + 1u -
					   input_size / VALUES_PER_BLOCK :
				   form_slices[form];
}

/* The fed values of a quantized vector held in lane order (codes) with its exponents. */
static bool fed_of_lane_codes(const uint8_t *lane_codes, uint32_t width,
			      double *fed_values)
{
	uint8_t *const codes = malloc(width);
	bool is_fed = codes != NULL;
	for (uint32_t value = 0; is_fed && value < width; value++) {
		codes[value] = lane_codes[test_linear_code_place(value)];
	}
	is_fed = is_fed &&
		 test_linear_fed_values_of_codes(codes, width, fed_values) ==
			 STATUS_OK;
	free(codes);
	return is_fed;
}

/* ---- a case ---- */

struct case_results {
	uint16_t *hidden[FORM_COUNT], *pair_output[FORM_COUNT],
		*down_alone[FORM_COUNT];
	uint32_t *row_output[FORM_COUNT]; /* float32 sums, as their bits */
	uint8_t *exact_hidden_codes, *exact_hidden_exponents;
};

static void case_results_free(struct case_results *results)
{
	for (uint32_t form = 0; form < FORM_COUNT; form++) {
		free(results->hidden[form]);
		free(results->pair_output[form]);
		free(results->down_alone[form]);
		free(results->row_output[form]);
	}
	free(results->exact_hidden_codes);
	free(results->exact_hidden_exponents);
	memset(results, 0, sizeof(*results));
}

static bool case_results_get(const struct test_linear_card *card,
			     struct case_results *results,
			     const struct chain_arrays forms[FORM_COUNT],
			     const struct chain_arrays downs[FORM_COUNT],
			     uint32_t row_count)
{
	const uint64_t pairs = (uint64_t)row_count * PAIRS_PER_ROW;
	const uint64_t hidden_bytes = pairs * INNER_SIZE * VALUE_BYTES,
		       output_bytes = pairs * HIDDEN_SIZE * VALUE_BYTES;
	const uint64_t row_bytes =
		(uint64_t)row_count * HIDDEN_SIZE * SUM_BYTES;
	bool is_got = true;
	for (uint32_t form = 0; form < FORM_COUNT; form++) {
		results->hidden[form] = calloc((size_t)hidden_bytes, 1);
		results->pair_output[form] = calloc((size_t)output_bytes, 1);
		results->down_alone[form] = calloc((size_t)output_bytes, 1);
		results->row_output[form] = calloc((size_t)row_bytes, 1);
		is_got = is_got && results->hidden[form] != NULL &&
			 results->pair_output[form] != NULL &&
			 results->down_alone[form] != NULL &&
			 results->row_output[form] != NULL &&
			 test_linear_card_get(card, forms[form].hidden, 0,
					      results->hidden[form],
					      hidden_bytes) &&
			 test_linear_card_get(card, forms[form].pair_output, 0,
					      results->pair_output[form],
					      output_bytes) &&
			 test_linear_card_get(card, forms[form].row_output, 0,
					      results->row_output[form],
					      row_bytes) &&
			 (form == FORM_EXACT ||
			  test_linear_card_get(card, downs[form].pair_output, 0,
					       results->down_alone[form],
					       output_bytes));
	}
	results->exact_hidden_codes = calloc((size_t)(pairs * INNER_SIZE), 1);
	results->exact_hidden_exponents =
		calloc((size_t)(pairs * INNER_SIZE / VALUES_PER_BLOCK), 1);
	return test_linear_expect(
		is_got && results->exact_hidden_codes != NULL &&
			results->exact_hidden_exponents != NULL &&
			test_linear_card_get(card,
					     forms[FORM_EXACT].hidden_codes, 0,
					     results->exact_hidden_codes,
					     pairs * INNER_SIZE) &&
			test_linear_card_get(
				card, forms[FORM_EXACT].hidden_exponents, 0,
				results->exact_hidden_exponents,
				pairs * INNER_SIZE / VALUES_PER_BLOCK),
		"the outputs of the forms come back");
}

/* Every relaxed form against the exact one on the same inputs: hidden values, down alone, row sums. */
static void case_compare_forms(const char *case_name,
			       const struct case_results *results,
			       uint32_t row_count, bool is_extreme)
{
	const uint64_t pairs = (uint64_t)row_count * PAIRS_PER_ROW;
	for (uint32_t form = FORM_NARROW; form < FORM_COUNT; form++) {
		struct difference hidden = { 0 }, down = { 0 }, rows = { 0 };
		for (uint64_t index = 0; index < pairs * INNER_SIZE; index++) {
			difference_add_bfloat16(
				&hidden, results->hidden[form][index],
				results->hidden[FORM_EXACT][index]);
		}
		for (uint64_t index = 0; index < pairs * HIDDEN_SIZE; index++) {
			difference_add_bfloat16(
				&down, results->down_alone[form][index],
				results->pair_output[FORM_EXACT][index]);
		}
		for (uint64_t index = 0;
		     index < (uint64_t)row_count * HIDDEN_SIZE; index++) {
			difference_add(
				&rows,
				float_of_bits(results->row_output[form][index]),
				float_of_bits(
					results->row_output[FORM_EXACT][index]),
				results->row_output[form][index] ==
					results->row_output[FORM_EXACT][index],
				0, 0.0, false);
		}
		difference_print(case_name, form_names[form], "hidden", "exact",
				 &hidden);
		difference_print(case_name, form_names[form], "down_alone",
				 "exact", &down);
		difference_print(case_name, form_names[form], "row_sum_chain",
				 "exact", &rows);
		if (form != FORM_ORDERED && !is_extreme) {
			test_linear_expect(
				hidden.outside_count == 0 &&
					down.outside_count == 0 &&
					hidden.squared_difference <=
						HIDDEN_RMS_LIMIT *
							HIDDEN_RMS_LIMIT *
							hidden.squared_reference &&
					down.squared_difference <=
						DOWN_RMS_LIMIT *
							DOWN_RMS_LIMIT *
							down.squared_reference &&
					rows.squared_difference <=
						ROW_SUM_RMS_LIMIT *
							ROW_SUM_RMS_LIMIT *
							rows.squared_reference,
				"the relaxed operator errors stay within the supplied report's bounds");
		}
		if (form == FORM_ORDERED) {
			test_linear_expect(
				memcmp(results->hidden[form],
				       results->hidden[FORM_EXACT],
				       pairs * INNER_SIZE * VALUE_BYTES) == 0 &&
					memcmp(results->pair_output[form],
					       results->pair_output[FORM_EXACT],
					       pairs * HIDDEN_SIZE *
						       VALUE_BYTES) == 0 &&
					memcmp(results->down_alone[form],
					       results->pair_output[FORM_EXACT],
					       pairs * HIDDEN_SIZE *
						       VALUE_BYTES) == 0 &&
					memcmp(results->row_output[form],
					       results->row_output[FORM_EXACT],
					       (uint64_t)row_count *
						       HIDDEN_SIZE *
						       SUM_BYTES) == 0,
				"the ordered chain has every reference byte");
		}
	}
}

/* The sampled pairs against the double reference: hidden values (held to the hidden range of each form's tolerance)
 * and the down products of the exact hidden vector. */
static void case_compare_reference(struct test_linear_card *card,
				   const char *case_name, bool is_extreme,
				   const struct case_inputs *inputs,
				   const struct routing *routing,
				   const struct case_results *results,
				   uint8_t *const row_codes[],
				   uint8_t *const row_exponents[])
{
	struct difference hidden[FORM_COUNT] = { { 0 } },
			  down[FORM_COUNT] = { { 0 } };
	const uint32_t stride = routing->pair_count > SAMPLED_PAIRS ?
					routing->pair_count / SAMPLED_PAIRS :
					1u;
	double *const fed = malloc(HIDDEN_SIZE * sizeof(double));
	struct test_linear_product *const gate =
		malloc(INNER_SIZE * sizeof(*gate));
	struct test_linear_product *const up = malloc(INNER_SIZE * sizeof(*up));
	struct test_linear_product *const outputs =
		malloc(HIDDEN_SIZE * sizeof(*outputs));
	const struct matrix_shape_record gate_up_shape = {
		INNER_SIZE, HIDDEN_SIZE, E2M1_BITS_PER_WEIGHT, 0
	};
	const struct matrix_shape_record down_shape = { HIDDEN_SIZE, INNER_SIZE,
							E2M1_BITS_PER_WEIGHT,
							0 };
	if (fed == NULL || gate == NULL || up == NULL || outputs == NULL) {
		test_linear_fail("the reference's memory is not obtained");
		free(fed);
		free(gate);
		free(up);
		free(outputs);
		return;
	}
	bool is_computed = true;
	for (uint32_t sample = 0, previous = UINT32_MAX;
	     is_computed && sample <= SAMPLED_PAIRS; sample++) {
		const uint32_t spaced = sample * stride;
		const uint32_t pair = spaced < routing->pair_count ?
					      spaced :
					      routing->pair_count - 1u;
		if (pair == previous) {
			continue; /* the last pair, sampled already */
		}
		previous = pair;
		const uint32_t row = pair / PAIRS_PER_ROW,
			       physical = routing->experts[pair] %
					  inputs->physical_count;
		const struct linear_range *const matrices =
			inputs->matrices[physical];
		const double route_weight =
			(double)routing->route_weights[pair];
		const result_status fed_status =
			test_linear_fed_values_of_codes(row_codes[row],
							HIDDEN_SIZE, fed);
		const result_status gate_status = test_linear_products(
			E2M1, &gate_up_shape,
			test_linear_card_stored(card, matrices[MATRIX_GATE]),
			MATRIX_BYTES, 0, INNER_SIZE, fed, row_exponents[row], 1,
			1, gate);
		const result_status up_status = test_linear_products(
			E2M1, &gate_up_shape,
			test_linear_card_stored(card, matrices[MATRIX_UP]),
			MATRIX_BYTES, 0, INNER_SIZE, fed, row_exponents[row], 1,
			1, up);
		is_computed =
			test_linear_expect_status(fed_status, STATUS_OK,
						  "the row's codes are fed") &&
			test_linear_expect_status(
				gate_status, STATUS_OK,
				"the reference computes gate") &&
			test_linear_expect_status(up_status, STATUS_OK,
						  "the reference computes up");
		for (uint32_t output = 0; is_computed && output < INNER_SIZE;
		     output++) {
			const double central = test_linear_hidden_value(
				gate[output].sum, up[output].sum,
				(double)ACTIVATION_LIMIT, route_weight);
			for (uint32_t form = 0; form < FORM_COUNT; form++) {
				struct test_linear_hidden_range range = { 0 };
				const uint32_t slices = tolerance_slices(
					(enum form)form, HIDDEN_SIZE);
				test_linear_hidden_range(
					&gate[output], &up[output], HIDDEN_SIZE,
					slices, (double)ACTIVATION_LIMIT,
					route_weight, &range);
				const uint16_t bits =
					results->hidden[form][(uint64_t)pair *
								      INNER_SIZE +
							      output];
				const int32_t ordinal =
					test_linear_bfloat16_ordinal(bits);
				difference_add(
					&hidden[form],
					test_linear_bfloat16_value(bits),
					central, bits == range.central_bits, 0,
					0.0,
					!test_linear_bfloat16_is_finite(bits) ||
						ordinal <
							range.lowest_ordinal ||
						ordinal >
							range.highest_ordinal);
			}
		}
		/* down of the exact hidden vector, which every form's down alone read */
		is_computed =
			is_computed &&
			fed_of_lane_codes(results->exact_hidden_codes +
						  (uint64_t)pair * INNER_SIZE,
					  INNER_SIZE, fed) &&
			test_linear_products(
				E2M1, &down_shape,
				test_linear_card_stored(card,
							matrices[MATRIX_DOWN]),
				MATRIX_BYTES, 0, HIDDEN_SIZE, fed,
				results->exact_hidden_exponents +
					(uint64_t)pair * INNER_SIZE /
						VALUES_PER_BLOCK,
				1, 1, outputs) == STATUS_OK;
		for (uint32_t output = 0; is_computed && output < HIDDEN_SIZE;
		     output++) {
			if (is_extreme &&
			    output % SCALE_CLASSES == BEYOND_CLASS) {
				continue; /* both orders cut the power alike: no double product holds */
			}
			for (uint32_t form = 0; form < FORM_COUNT; form++) {
				const uint16_t *const values =
					form == FORM_EXACT ?
						results->pair_output[FORM_EXACT] :
						results->down_alone[form];
				const uint16_t bits =
					values[(uint64_t)pair * HIDDEN_SIZE +
					       output];
				const uint32_t slices = tolerance_slices(
					(enum form)form, INNER_SIZE);
				difference_add(
					&down[form],
					test_linear_bfloat16_value(bits),
					outputs[output].sum,
					bits == test_linear_bfloat16_nearest(
							outputs[output].sum),
					0, outputs[output].magnitude_sum,
					!test_linear_output_is_within_tolerance(
						bits, &outputs[output],
						INNER_SIZE, slices));
			}
		}
		test_linear_expect(is_computed,
				   "the reference computes a sampled pair");
	}
	for (uint32_t form = 0; form < FORM_COUNT; form++) {
		difference_print(case_name, form_names[form], "hidden",
				 "double", &hidden[form]);
		difference_print(case_name, form_names[form],
				 form == FORM_EXACT ? "down_chain" :
						      "down_alone",
				 "double", &down[form]);
		if (hidden[form].outside_count != 0 ||
		    down[form].outside_count != 0) {
			test_linear_fail(
				"%s: %" PRIu64 " hidden values and %" PRIu64
				" pair outputs of the sampled pairs lie "
				"outside the form's tolerance of the double products",
				form_names[form], hidden[form].outside_count,
				down[form].outside_count);
		}
	}
	free(fed);
	free(gate);
	free(up);
	free(outputs);
}

/* A row launched alone in the narrow and wide forms: its six experts each a job of one column and a tile of one pair,
 * arrays of its own. Its hidden values, pair outputs and row sum must be the bits it has in company. */
static void case_row_alone(struct test_linear_card *card,
			   const struct card_kernels *kernels,
			   const struct case_inputs *inputs,
			   const struct routing *routing,
			   const struct case_results *results, uint32_t row)
{
	struct routing alone = { 0 };
	struct case_inputs alone_inputs = *inputs;
	struct chain_arrays arrays[FORM_COUNT];
	struct kernel_launch launches[FORM_COUNT][MOST_LAUNCHES];
	uint8_t row_bytes[HIDDEN_SIZE * VALUE_BYTES];
	uint16_t hidden[PAIRS_PER_ROW * INNER_SIZE],
		outputs[PAIRS_PER_ROW * HIDDEN_SIZE];
	uint32_t sums[HIDDEN_SIZE];
	alone.row_count = 1;
	alone.pair_count = PAIRS_PER_ROW;
	alone.experts = routing->experts + (size_t)row * PAIRS_PER_ROW;
	alone.route_weights =
		routing->route_weights + (size_t)row * PAIRS_PER_ROW;
	struct wide_pair sorted[PAIRS_PER_ROW];
	alone.sorted = sorted;
	for (uint32_t slot = 0; slot < PAIRS_PER_ROW; slot++) {
		const uint32_t expert = alone.experts[slot];
		sorted[slot] =
			(struct wide_pair){ slot, alone.route_weights[slot] };
		alone.run_first[expert] = slot;
		alone.run_count[expert] = 1;
	}
	alone.job_count = alone.tile_count = PAIRS_PER_ROW;
	alone_inputs.job_entries = PAIRS_PER_ROW;
	alone_inputs.tile_entries = PAIRS_PER_ROW;
	alone_inputs.jobs = test_linear_card_work(
		card, linear_jobs_byte_count(PAIRS_PER_ROW));
	alone_inputs.tiles = test_linear_card_work(
		card, PAIRS_PER_ROW * sizeof(struct wide_expert_tile));
	alone_inputs.pairs = test_linear_card_work(
		card, PAIRS_PER_ROW * sizeof(struct wide_pair));
	alone_inputs.codes = test_linear_card_work(card, HIDDEN_SIZE);
	alone_inputs.exponents =
		test_linear_card_work(card, HIDDEN_SIZE / VALUES_PER_BLOCK);
	alone_inputs.operands =
		test_linear_card_work(card, HIDDEN_SIZE * VALUE_BYTES);
	for (uint32_t form = FORM_NARROW; form < FORM_COUNT; form++) {
		chain_set_aside(card, &arrays[form], 1);
	}
	if (!test_linear_card_has_room(card)) {
		return;
	}
	tables_put(card, &alone_inputs, &alone, false);
	/* the row's arrays: its bytes of the case's, from the image */
	(void)test_linear_card_put(card, alone_inputs.codes, 0,
				   test_linear_card_stored(card,
							   inputs->codes) +
					   (size_t)row * HIDDEN_SIZE,
				   HIDDEN_SIZE);
	(void)test_linear_card_put(
		card, alone_inputs.exponents, 0,
		test_linear_card_stored(card, inputs->exponents) +
			(size_t)row * HIDDEN_SIZE / VALUES_PER_BLOCK,
		HIDDEN_SIZE / VALUES_PER_BLOCK);
	memcpy(row_bytes,
	       test_linear_card_stored(card, inputs->operands) +
		       (size_t)row * HIDDEN_SIZE * VALUE_BYTES,
	       sizeof(row_bytes));
	(void)test_linear_card_put(card, alone_inputs.operands, 0, row_bytes,
				   sizeof(row_bytes));
	bool is_run = test_linear_card_load(card);
	for (uint32_t form = FORM_NARROW; is_run && form < FORM_COUNT; form++) {
		is_run = form_prepare(card, kernels, (enum form)form,
				      &alone_inputs, &arrays[form], 1,
				      PAIRS_PER_ROW, launches[form]) &&
			 test_linear_card_run(
				 card, launches[form],
				 layout_of((enum form)form).launch_count);
	}
	is_run = is_run && test_linear_card_fetch(card);
	for (uint32_t form = FORM_NARROW; is_run && form < FORM_COUNT; form++) {
		const uint64_t first_pair = (uint64_t)row * PAIRS_PER_ROW;
		is_run = test_linear_card_get(card, arrays[form].hidden, 0,
					      hidden, sizeof(hidden)) &&
			 test_linear_card_get(card, arrays[form].pair_output, 0,
					      outputs, sizeof(outputs)) &&
			 test_linear_card_get(card, arrays[form].row_output, 0,
					      sums, sizeof(sums));
		if (is_run &&
		    (memcmp(hidden,
			    results->hidden[form] + first_pair * INNER_SIZE,
			    sizeof(hidden)) != 0 ||
		     memcmp(outputs,
			    results->pair_output[form] +
				    first_pair * HIDDEN_SIZE,
			    sizeof(outputs)) != 0 ||
		     memcmp(sums,
			    results->row_output[form] +
				    (uint64_t)row * HIDDEN_SIZE,
			    sizeof(sums)) != 0)) {
			test_linear_fail(
				"%s: row %" PRIu32
				" launched alone has other bits than in company",
				form_names[form], row);
		}
	}
	test_linear_expect(is_run, "the row runs alone");
}

static void case_run(struct test_linear_card *card,
		     const struct card_kernels *kernels,
		     const struct relaxed_case *test_case, uint64_t stream,
		     uint32_t physical_count)
{
	struct routing routing;
	struct case_inputs inputs;
	struct chain_arrays forms[FORM_COUNT], downs[FORM_COUNT];
	struct kernel_launch launches[FORM_COUNT][MOST_LAUNCHES];
	struct kernel_launch down_launches[FORM_COUNT][MOST_LAUNCHES];
	struct case_results results, again;
	const uint32_t row_count = test_case->row_count;
	const bool is_job_case = row_count <= JOB_COLUMNS;
	uint8_t **const row_codes =
		(uint8_t **)calloc(row_count, sizeof(uint8_t *));
	uint8_t **const row_exponents =
		(uint8_t **)calloc(row_count, sizeof(uint8_t *));
	uint16_t values[HIDDEN_SIZE];
	memset(&results, 0, sizeof(results));
	memset(&again, 0, sizeof(again));
	test_linear_context("%s", test_case->name);
	if (row_codes == NULL || row_exponents == NULL) {
		test_linear_fail("the rows' codes are not obtained");
		free((void *)row_codes);
		free((void *)row_exponents);
		return;
	}
	if (!routing_make(&routing, row_count,
			  test_case->kind == KIND_EDGES	 ? EDGE_EXPERT_POOL :
			  test_case->kind == KIND_POOLED ? POOLED_EXPERT_POOL :
							   LOGICAL_EXPERTS,
			  test_case->kind == KIND_SKEW, stream)) {
		free((void *)row_codes);
		free((void *)row_exponents);
		return;
	}
	test_linear_card_clear(card);
	inputs_set_aside(card, &inputs, &routing, physical_count, is_job_case);
	for (uint32_t form = 0; form < FORM_COUNT; form++) {
		chain_set_aside(card, &forms[form], row_count);
		downs[form] = forms
			[FORM_EXACT]; /* down alone reads the exact chain's hidden vectors */
		if (form != FORM_EXACT) {
			downs[form].pair_output = test_linear_card_work(
				card, linear_values_byte_count(
					      routing.pair_count, HIDDEN_SIZE));
			downs[form].row_output = test_linear_card_work(
				card,
				linear_sums_byte_count(row_count, HIDDEN_SIZE));
		}
	}
	bool is_run = test_linear_card_has_room(card);
	for (uint32_t row = 0; is_run && row < row_count; row++) {
		row_codes[row] = malloc(HIDDEN_SIZE);
		row_exponents[row] = malloc(HIDDEN_SIZE / VALUES_PER_BLOCK);
		row_values(values, row, test_case->kind, stream);
		is_run = row_codes[row] != NULL && row_exponents[row] != NULL &&
			 test_linear_quantize(values, HIDDEN_SIZE,
					      row_codes[row],
					      row_exponents[row]) ==
				 STATUS_OK &&
			 test_linear_card_put_codes(card, inputs.codes,
						    (uint64_t)row * HIDDEN_SIZE,
						    row_codes[row],
						    HIDDEN_SIZE) &&
			 test_linear_card_put(card, inputs.exponents,
					      (uint64_t)row * HIDDEN_SIZE /
						      VALUES_PER_BLOCK,
					      row_exponents[row],
					      HIDDEN_SIZE / VALUES_PER_BLOCK) &&
			 test_linear_card_put_operands(
				 card, inputs.operands,
				 (uint64_t)row * HIDDEN_SIZE * VALUE_BYTES,
				 row_codes[row], row_exponents[row],
				 HIDDEN_SIZE);
	}
	if (is_run) {
		weights_fill(card, &inputs, test_case->kind, stream);
		tables_put(card, &inputs, &routing, is_job_case);
		is_run = test_linear_card_load(card);
	}
	for (uint32_t form = 0; is_run && form < FORM_COUNT; form++) {
		const struct chain_layout layout = layout_of((enum form)form);
		is_run = form_prepare(card, kernels, (enum form)form, &inputs,
				      &forms[form], row_count,
				      inputs.tile_entries, launches[form]) &&
			 test_linear_card_run(card, launches[form],
					      layout.launch_count);
		if (is_run &&
		    form != FORM_EXACT) { /* down alone and its row sum, on the exact hidden vectors */
			is_run =
				form_prepare(card, kernels, (enum form)form,
					     &inputs, &downs[form], row_count,
					     inputs.tile_entries,
					     down_launches[form]) &&
				test_linear_card_run(
					card,
					&down_launches[form][layout.down_first],
					layout.launch_count -
						layout.down_first);
		}
	}
	is_run = is_run && test_linear_card_fetch(card) &&
		 case_results_get(card, &results, forms, downs, row_count);
	if (is_run) {
		case_compare_forms(test_case->name, &results, row_count,
				   test_case->kind == KIND_EXTREME_SCALES);
		case_compare_reference(card, test_case->name,
				       test_case->kind == KIND_EXTREME_SCALES,
				       &inputs, &routing, &results, row_codes,
				       row_exponents);
		test_linear_card_expect_untouched(card, inputs.jobs,
						  "the job table");
		test_linear_card_expect_untouched(card, inputs.tiles,
						  "the tiles");
		test_linear_card_expect_untouched(card, inputs.pairs,
						  "the sorted pairs");
		test_linear_card_expect_untouched(card, inputs.codes,
						  "the rows' codes");
		test_linear_card_expect_untouched(card, inputs.exponents,
						  "the rows' exponents");
		test_linear_card_expect_untouched(card, inputs.operands,
						  "the rows' operands");
		test_linear_card_expect_weights_untouched(card);
		/* again: the same bits */
		for (uint32_t form = FORM_NARROW; is_run && form < FORM_COUNT;
		     form++) {
			const struct chain_layout layout =
				layout_of((enum form)form);
			is_run =
				test_linear_card_run(card, launches[form],
						     layout.launch_count) &&
				test_linear_card_run(
					card,
					&down_launches[form][layout.down_first],
					layout.launch_count -
						layout.down_first);
		}
		is_run =
			is_run && test_linear_card_fetch(card) &&
			case_results_get(card, &again, forms, downs, row_count);
		for (uint32_t form = FORM_NARROW; is_run && form < FORM_COUNT;
		     form++) {
			const uint64_t pairs = routing.pair_count;
			if (memcmp(again.hidden[form], results.hidden[form],
				   pairs * INNER_SIZE * VALUE_BYTES) != 0 ||
			    memcmp(again.pair_output[form],
				   results.pair_output[form],
				   pairs * HIDDEN_SIZE * VALUE_BYTES) != 0 ||
			    memcmp(again.down_alone[form],
				   results.down_alone[form],
				   pairs * HIDDEN_SIZE * VALUE_BYTES) != 0 ||
			    memcmp(again.row_output[form],
				   results.row_output[form],
				   (uint64_t)row_count * HIDDEN_SIZE *
					   SUM_BYTES) != 0) {
				test_linear_fail(
					"%s: run again, the form gives other bits",
					form_names[form]);
			}
		}
		if (is_run) {
			case_row_alone(card, kernels, &inputs, &routing,
				       &results, row_count - 1u);
		}
	}
	test_linear_expect(is_run, "the case runs");
	case_results_free(&results);
	case_results_free(&again);
	for (uint32_t row = 0; row < row_count; row++) {
		free(row_codes[row]);
		free(row_exponents[row]);
	}
	free((void *)row_codes);
	free((void *)row_exponents);
	routing_free(&routing);
}

/* ---- timings ---- */

/* The actual GPU router writes the job table, sorted pairs and tiles consumed by both expert chains. Scores force
 * the six synthetic choices, while the router computes their route weights with its production arithmetic. */
static bool router_tables_run(struct test_linear_card *card,
			      const struct card_kernels *kernels,
			      struct case_inputs *inputs,
			      const struct routing *routing)
{
	uint64_t offsets[LOGICAL_EXPERTS * MATRICES];
	float bias[LOGICAL_EXPERTS] = { 0 };
	const uint32_t row_count = routing->row_count;
	const struct step_block step = { 1, row_count, 0 };
	const struct linear_range step_range =
		test_linear_card_work(card, sizeof(step));
	const struct linear_range scores_range =
		test_linear_card_work(card, (uint64_t)row_count * sizeof(bias));
	const struct linear_range bias_range =
		test_linear_card_work(card, sizeof(bias));
	const struct linear_range offsets_range =
		test_linear_card_work(card, sizeof(offsets));
	const struct linear_range affinities =
		test_linear_card_work(card, scores_range.byte_count);
	const struct linear_range picks =
		test_linear_card_work(card, (uint64_t)routing->pair_count *
						    sizeof(struct router_pick));
	const struct linear_range counts = test_linear_card_work(
		card, (uint64_t)((row_count + ROUTER_SORT_ROWS - 1u) /
				 ROUTER_SORT_ROWS) *
			      sizeof(bias));
	const struct linear_range runs = test_linear_card_work(
		card, LOGICAL_EXPERTS * sizeof(struct wide_run));
	inputs->job_entries = ROUTER_WIDE_JOBS(row_count);
	inputs->tile_entries = ROUTER_WIDE_TILES(row_count);
	inputs->jobs = test_linear_card_work(
		card, linear_jobs_byte_count(inputs->job_entries));
	inputs->tiles = test_linear_card_work(
		card, (uint64_t)inputs->tile_entries *
			      sizeof(struct wide_expert_tile));
	float *const scores = malloc((size_t)scores_range.byte_count);
	struct kernel_launch launches[ROUTER_WIDE_LAUNCH_COUNT];
	bool is_run = test_linear_card_has_room(card) && scores != NULL;
	for (uint32_t expert = 0; expert < LOGICAL_EXPERTS; expert++) {
		for (uint32_t matrix = 0; matrix < MATRICES; matrix++) {
			offsets[expert * MATRICES + matrix] =
				inputs->matrices[expert %
						 inputs->physical_count][matrix]
					.byte_offset;
		}
	}
	for (uint32_t row = 0; is_run && row < row_count; row++) {
		for (uint32_t expert = 0; expert < LOGICAL_EXPERTS; expert++) {
			bool is_chosen = false;
			for (uint32_t slot = 0; slot < PAIRS_PER_ROW; slot++) {
				is_chosen =
					is_chosen ||
					routing->experts[row * PAIRS_PER_ROW +
							 slot] == expert;
			}
			const int32_t centered =
				(int32_t)(hash32(ROUTE_STREAM + row, expert) %
					  SCORE_STEPS) -
				SCORE_CENTER;
			const float score = (float)centered * SCORE_UNIT;
			scores[(size_t)row * LOGICAL_EXPERTS + expert] =
				is_chosen ? fabsf(score) + 1.0f :
					    -fabsf(score) - 1.0f;
		}
	}
	const struct router_wide_description description = {
		.expert_offsets = offsets,
		.step = step_range,
		.scores = scores_range,
		.bias = bias_range,
		.offsets = offsets_range,
		.expert_weights = inputs->weights,
		.affinities = affinities,
		.picks = picks,
		.counts = counts,
		.pairs = inputs->pairs,
		.runs = runs,
		.tiles = inputs->tiles,
		.jobs = inputs->jobs,
		.gate_up_shape = { INNER_SIZE, HIDDEN_SIZE,
				   E2M1_BITS_PER_WEIGHT, 0 },
		.down_shape = { HIDDEN_SIZE, INNER_SIZE, E2M1_BITS_PER_WEIGHT,
				0 },
		.weight_format = E2M1,
		.order = LINEAR_ORDER_REFERENCE,
		.row_count = row_count,
	};
	is_run =
		is_run &&
		test_linear_card_put(card, step_range, 0, &step,
				     sizeof(step)) &&
		test_linear_card_put(card, scores_range, 0, scores,
				     scores_range.byte_count) &&
		test_linear_card_put(card, bias_range, 0, bias, sizeof(bias)) &&
		test_linear_card_put(card, offsets_range, 0, offsets,
				     sizeof(offsets)) &&
		test_linear_expect_status(
			router_wide_prepare(&description,
					    card->arena_byte_count,
					    &kernels->router, launches),
			STATUS_OK, "the joined router is prepared") &&
		test_linear_card_load(card) &&
		test_linear_card_run(card, launches, COUNT_OF(launches)) &&
		test_linear_card_fetch(card);
	const struct linear_range written[] = { affinities,  picks,
						counts,	     inputs->pairs,
						runs,	     inputs->tiles,
						inputs->jobs };
	for (uint32_t index = 0; is_run && index < COUNT_OF(written); index++) {
		is_run = test_linear_card_put(
			card, written[index], 0,
			card->received + written[index].byte_offset,
			written[index].byte_count);
	}
	/* A second production router walk must reproduce every intermediate byte before either expert form reads it. */
	is_run = is_run &&
		 test_linear_card_run(card, launches, COUNT_OF(launches)) &&
		 test_linear_card_fetch(card);
	for (uint32_t index = 0; is_run && index < COUNT_OF(written); index++) {
		is_run = test_linear_card_expect_untouched(
			card, written[index],
			"the repeated router intermediate");
	}
	test_linear_card_expect_untouched(card, step_range, "the router step");
	test_linear_card_expect_untouched(card, scores_range,
					  "the router scores");
	test_linear_card_expect_untouched(card, bias_range, "the router bias");
	test_linear_card_expect_untouched(card, offsets_range,
					  "the router expert table");
	free(scores);
	return test_linear_expect(
		is_run,
		"the production router writes reproducible expert tables");
}

static void chain_sentinel(struct test_linear_card *card,
			   const struct chain_arrays *arrays, int sentinel)
{
	const struct linear_range ranges[] = {
		arrays->hidden,		  arrays->hidden_codes,
		arrays->hidden_exponents, arrays->hidden_operands,
		arrays->pair_output,	  arrays->row_output
	};
	for (uint32_t index = 0; index < COUNT_OF(ranges); index++) {
		memset(card->image + ranges[index].byte_offset, sentinel,
		       (size_t)ranges[index].byte_count);
	}
}

static bool joined_compare(struct test_linear_card *card,
			   const struct chain_arrays *reference,
			   const struct chain_arrays *ordered,
			   const unsigned char *reference_bytes)
{
	const struct linear_range first[] = {
		reference->hidden,	     reference->hidden_codes,
		reference->hidden_exponents, reference->hidden_operands,
		reference->pair_output,	     reference->row_output
	};
	const struct linear_range second[] = {
		ordered->hidden,	   ordered->hidden_codes,
		ordered->hidden_exponents, ordered->hidden_operands,
		ordered->pair_output,	   ordered->row_output
	};
	static const char *const names[] = {
		"joined hidden BF16",	   "joined hidden codes",
		"joined hidden exponents", "joined hidden carried operands",
		"joined pair output BF16", "joined six-slot row sum FP32"
	};
	bool is_equal = true;
	for (uint32_t index = 0; index < COUNT_OF(first); index++) {
		is_equal = test_linear_card_expect_bytes(
				   card, second[index], 0,
				   reference_bytes + first[index].byte_offset,
				   first[index].byte_count, names[index]) &&
			   is_equal;
	}
	return is_equal;
}

static bool joined_prepare(const struct test_linear_card *card,
			   const struct card_kernels *kernels,
			   const struct case_inputs *inputs,
			   const struct chain_arrays *arrays,
			   uint32_t row_count, enum cmp_order_code order,
			   struct kernel_launch *launches)
{
	const struct expert_relaxed_description description = {
		experts_of(inputs, arrays, row_count, 1u), inputs->tiles,
		inputs->pairs, inputs->tile_entries, 0
	};
	return test_linear_expect_status(
		order == CMP_ORDER_REFERENCE ?
			expert_ordered_prepare(
				&description, card->arena_byte_count,
				&card->kernels, &kernels->relaxed, launches) :
			expert_relaxed_prepare(
				&description, card->arena_byte_count,
				&card->kernels, &kernels->relaxed, launches),
		STATUS_OK,
		"the production wide expert prepare builds the joined chain");
}

/* Direct carried operands also cover cases outside quantization's ordinary finite inputs. Only reference order
 * is tested here: these scales intentionally include values refused by relaxed scale-fold admission. */
static uint16_t special_carried_value(uint32_t row, uint32_t input)
{
	enum special_kind {
		SPECIAL_POSITIVE_ZERO,
		SPECIAL_NEGATIVE_ZERO,
		SPECIAL_ONE,
		SPECIAL_POSITIVE_INFINITY,
		SPECIAL_NEGATIVE_INFINITY,
		SPECIAL_QUIET_NAN,
		SPECIAL_OPPOSITE_INFINITIES,
		SPECIAL_SIGNED_ZEROS
	};
	switch (row % SPECIAL_INPUTS_PER_CASE) {
	case SPECIAL_NEGATIVE_ZERO:
		return CARRIED_NEGATIVE_ZERO;
	case SPECIAL_ONE:
		return CARRIED_ONE;
	case SPECIAL_POSITIVE_INFINITY:
		return input == 0 ? CARRIED_POSITIVE_INFINITY : 0;
	case SPECIAL_NEGATIVE_INFINITY:
		return input == 0 ? CARRIED_NEGATIVE_INFINITY : 0;
	case SPECIAL_QUIET_NAN:
		return input == 0 ? CARRIED_QUIET_NAN : 0;
	case SPECIAL_OPPOSITE_INFINITIES:
		return input % INPUT_PAIR == 0 ? CARRIED_POSITIVE_INFINITY :
						 CARRIED_NEGATIVE_INFINITY;
	case SPECIAL_SIGNED_ZEROS:
		return input % INPUT_PAIR == 0 ? 0 : CARRIED_NEGATIVE_ZERO;
	default:
		return 0;
	}
}

static void special_scales_fill(struct test_linear_card *card,
				const struct case_inputs *inputs)
{
	static const uint32_t codes[] = { 0,   94,  95,	 209, 210,
					  228, 229, 254, 255 };
	for (uint32_t physical = 0; physical < inputs->physical_count;
	     physical++) {
		for (uint32_t matrix = 0; matrix < MATRICES; matrix++) {
			unsigned char *const bytes = test_linear_card_writable(
				card, inputs->matrices[physical][matrix]);
			if (bytes == NULL) {
				test_linear_fail(
					"the special matrix is inside the weights range");
				return;
			}
			const uint32_t output_size = matrix == MATRIX_DOWN ?
							     HIDDEN_SIZE :
							     INNER_SIZE;
			const uint32_t input_size = matrix == MATRIX_DOWN ?
							    INNER_SIZE :
							    HIDDEN_SIZE;
			const uint32_t load_count =
				input_size / VALUES_PER_LOAD;
			const uint64_t words = (uint64_t)output_size *
					       input_size / VALUES_PER_BLOCK /
					       sizeof(uint32_t);
			unsigned char *const scales =
				bytes + (uint64_t)output_size * input_size /
						E2M1_CODES_PER_BYTE;
			for (uint64_t word = 0; word < words; word++) {
				const uint32_t lane_group =
					(uint32_t)(word % LANE_GROUPS);
				const uint32_t tile =
					(uint32_t)(word / LANE_GROUPS /
						   load_count);
				uint32_t packed = 0;
				for (uint32_t byte = 0;
				     byte < SCALE_BYTES_A_WORD; byte++) {
					const uint32_t output =
						tile * LINEAR_OUTPUTS_PER_TILE +
						lane_group +
						byte % OUTPUT_HALVES *
							OUTPUTS_PER_HALF_TILE;
					packed |= codes[(output +
							 byte / OUTPUT_HALVES) %
							COUNT_OF(codes)]
						  << (byte * BYTE_BITS);
				}
				word_put(scales, word, packed);
			}
		}
	}
}

static void special_ordered_run(struct test_linear_card *card,
				const struct card_kernels *kernels,
				uint32_t physical_count)
{
	struct routing routing;
	struct case_inputs inputs;
	struct chain_arrays reference, ordered;
	struct kernel_launch base[MOST_LAUNCHES], selected[MOST_LAUNCHES];
	uint16_t source_values[HIDDEN_SIZE], pair_values[INNER_SIZE];
	test_linear_context(
		"ordered carried zeros, infinities, NaN and scale boundaries");
	if (!routing_make(&routing, SPECIAL_ROWS, LOGICAL_EXPERTS, true,
			  RELAXED_STREAM + SCALE_STREAM)) {
		return;
	}
	test_linear_card_clear(card);
	inputs_set_aside(card, &inputs, &routing, physical_count, true);
	chain_set_aside(card, &reference, SPECIAL_ROWS);
	chain_set_aside(card, &ordered, SPECIAL_ROWS);
	chain_sentinel(card, &reference, OUTPUT_SENTINEL);
	chain_sentinel(card, &ordered, SELECTED_SENTINEL);
	bool is_run = test_linear_card_has_room(card);
	for (uint32_t row = 0; is_run && row < SPECIAL_ROWS; row++) {
		for (uint32_t input = 0; input < HIDDEN_SIZE; input++) {
			source_values[input] =
				special_carried_value(row, input);
		}
		is_run = test_linear_card_put(
			card, inputs.operands,
			(uint64_t)row * sizeof(source_values), source_values,
			sizeof(source_values));
	}
	for (uint32_t pair = 0; is_run && pair < routing.pair_count; pair++) {
		for (uint32_t input = 0; input < INNER_SIZE; input++) {
			pair_values[input] = special_carried_value(pair, input);
		}
		is_run = test_linear_card_put(
				 card, reference.hidden_operands,
				 (uint64_t)pair * sizeof(pair_values),
				 pair_values, sizeof(pair_values)) &&
			 test_linear_card_put(card, ordered.hidden_operands,
					      (uint64_t)pair *
						      sizeof(pair_values),
					      pair_values, sizeof(pair_values));
	}
	if (is_run) {
		weights_fill(card, &inputs, KIND_PLAIN,
			     RELAXED_STREAM + SCALE_STREAM);
		special_scales_fill(card, &inputs);
		tables_put(card, &inputs, &routing, true);
		is_run = test_linear_card_load(card) &&
			 form_prepare(card, kernels, FORM_EXACT, &inputs,
				      &reference, SPECIAL_ROWS,
				      inputs.tile_entries, base) &&
			 form_prepare(card, kernels, FORM_ORDERED, &inputs,
				      &ordered, SPECIAL_ROWS,
				      inputs.tile_entries, selected);
	}
	const struct chain_layout exact_layout = layout_of(FORM_EXACT),
				  ordered_layout = layout_of(FORM_ORDERED);
	is_run = is_run &&
		 test_linear_card_run(card, &base[exact_layout.gate_up_first],
				      exact_layout.product_launches) &&
		 test_linear_card_run(card,
				      &selected[ordered_layout.gate_up_first],
				      ordered_layout.product_launches) &&
		 test_linear_card_run(card, &base[exact_layout.down_first],
				      exact_layout.launch_count -
					      exact_layout.down_first) &&
		 test_linear_card_run(card,
				      &selected[ordered_layout.down_first],
				      ordered_layout.launch_count -
					      ordered_layout.down_first) &&
		 test_linear_card_fetch(card) &&
		 test_linear_card_expect_same(
			 card, reference.hidden, 0, ordered.hidden, 0,
			 reference.hidden.byte_count,
			 "special carried gate/up hidden bytes") &&
		 test_linear_card_expect_same(
			 card, reference.pair_output, 0, ordered.pair_output, 0,
			 reference.pair_output.byte_count,
			 "special carried down output bytes") &&
		 test_linear_card_expect_same(
			 card, reference.row_output, 0, ordered.row_output, 0,
			 reference.row_output.byte_count,
			 "special carried six-slot sum bytes");
	if (is_run) {
		test_linear_card_expect_untouched(
			card, inputs.operands,
			"the special carried source operands");
		test_linear_card_expect_untouched(
			card, reference.hidden_operands,
			"the special carried down operands");
		test_linear_card_expect_untouched(
			card, ordered.hidden_operands,
			"the ordered special down operands");
		test_linear_card_expect_weights_untouched(card);
		(void)printf(
			"special ordered=byte-equal scales=0,94,95,209,210,228,229,254,255 "
			"operands=positive-zero,negative-zero,finite,infinity,negative-infinity,NaN\n");
	}
	test_linear_expect(is_run, "the ordered special carried inputs run");
	routing_free(&routing);
}

/* Full routed chunk: production router -> gate/up and hidden -> quantization -> down -> ordered six-slot sum.
 * Compact mode saves the first output set on the host before overwriting it; both modes compare every byte. */
static void joined_run(struct test_linear_card *card,
		       const struct card_kernels *kernels, uint32_t row_count,
		       uint32_t physical_count, bool compact)
{
	struct routing routing;
	struct case_inputs inputs;
	struct chain_arrays reference, selected;
	struct kernel_launch base[MOST_LAUNCHES], ordered[MOST_LAUNCHES],
		relaxed[MOST_LAUNCHES], quantize;
	uint16_t values[HIDDEN_SIZE];
	unsigned char *saved_reference = NULL;
	test_linear_context("joined router/expert chunk, %" PRIu32 " rows",
			    row_count);
	if (!routing_make(&routing, row_count, LOGICAL_EXPERTS, false,
			  RELAXED_STREAM + row_count)) {
		return;
	}
	test_linear_card_clear(card);
	inputs_set_aside(card, &inputs, &routing, physical_count, false);
	chain_set_aside(card, &reference, row_count);
	if (compact) {
		selected = reference;
	} else {
		chain_set_aside(card, &selected, row_count);
	}
	const struct linear_range source = test_linear_card_work(
		card, linear_values_byte_count(row_count, HIDDEN_SIZE));
	bool is_run = test_linear_card_has_room(card);
	for (uint32_t row = 0; is_run && row < row_count; row++) {
		row_values(values, row, KIND_PLAIN, RELAXED_STREAM + row_count);
		is_run = test_linear_card_put(card, source,
					      (uint64_t)row * sizeof(values),
					      values, sizeof(values));
	}
	if (is_run) {
		weights_fill(card, &inputs, KIND_PLAIN, RELAXED_STREAM);
		chain_sentinel(card, &reference, OUTPUT_SENTINEL);
		if (!compact) {
			chain_sentinel(card, &selected, SELECTED_SENTINEL);
		}
		is_run = router_tables_run(card, kernels, &inputs, &routing);
	}
	const struct linear_quantize_description rows = {
		source,		 inputs.codes, inputs.exponents,
		inputs.operands, HIDDEN_SIZE,  row_count
	};
	is_run = is_run &&
		 test_linear_expect_status(
			 linear_quantize_prepare(&rows, card->arena_byte_count,
						 &card->kernels, &quantize),
			 STATUS_OK,
			 "the joined source quantization is prepared") &&
		 test_linear_card_run(card, &quantize, 1) &&
		 form_prepare(card, kernels, FORM_EXACT, &inputs, &reference,
			      row_count, inputs.tile_entries, base) &&
		 joined_prepare(card, kernels, &inputs, &selected, row_count,
				CMP_ORDER_REFERENCE, ordered) &&
		 joined_prepare(card, kernels, &inputs, &selected, row_count,
				CMP_ORDER_RELAXED, relaxed) &&
		 test_linear_card_run(card, base,
				      layout_of(FORM_EXACT).launch_count);
	if (is_run && compact) {
		is_run = test_linear_card_fetch(card);
		saved_reference =
			is_run ? malloc((size_t)card->work_byte_count) : NULL;
		is_run = is_run &&
			 test_linear_expect(
				 saved_reference != NULL,
				 "the reference bytes are saved on the host");
		if (is_run) {
			memcpy(saved_reference, card->received,
			       (size_t)card->work_byte_count);
			chain_sentinel(card, &selected, SELECTED_SENTINEL);
			const struct linear_range ranges[] = {
				selected.hidden,
				selected.hidden_codes,
				selected.hidden_exponents,
				selected.hidden_operands,
				selected.pair_output,
				selected.row_output
			};
			for (uint32_t index = 0;
			     is_run && index < COUNT_OF(ranges); index++) {
				is_run = test_linear_card_upload(
					card, ranges[index], 0,
					ranges[index].byte_count);
			}
		}
	}
	const unsigned char *const reference_bytes = compact ? saved_reference :
							       card->received;
	is_run = is_run &&
		 test_linear_card_run(card, ordered,
				      layout_of(FORM_ORDERED).launch_count) &&
		 test_linear_card_fetch(card) &&
		 joined_compare(card, &reference, &selected, reference_bytes);
	if (is_run) {
		(void)printf(
			"joined rows=%" PRIu32
			" ordered=byte-equal hidden,codes,exponents,operands,"
			"pairs,row_sum\n",
			row_count);
		/* Compare the relaxed joined chain's accumulated BF16-rounding error to the reference. */
		is_run = test_linear_card_run(
				 card, relaxed,
				 layout_of(FORM_WIDE).launch_count) &&
			 test_linear_card_fetch(card);
		struct difference hidden = { 0 }, pairs = { 0 }, sums = { 0 };
		for (uint64_t index = 0;
		     is_run &&
		     index < (uint64_t)routing.pair_count * INNER_SIZE;
		     index++) {
			uint16_t expected = 0, actual = 0;
			memcpy(&expected,
			       reference_bytes + reference.hidden.byte_offset +
				       index * VALUE_BYTES,
			       sizeof(expected));
			memcpy(&actual,
			       card->received + selected.hidden.byte_offset +
				       index * VALUE_BYTES,
			       sizeof(actual));
			difference_add_bfloat16(&hidden, actual, expected);
		}
		for (uint64_t index = 0;
		     is_run &&
		     index < (uint64_t)routing.pair_count * HIDDEN_SIZE;
		     index++) {
			uint16_t expected = 0, actual = 0;
			memcpy(&expected,
			       reference_bytes +
				       reference.pair_output.byte_offset +
				       index * VALUE_BYTES,
			       sizeof(expected));
			memcpy(&actual,
			       card->received +
				       selected.pair_output.byte_offset +
				       index * VALUE_BYTES,
			       sizeof(actual));
			difference_add_bfloat16(&pairs, actual, expected);
		}
		for (uint64_t index = 0;
		     is_run && index < (uint64_t)row_count * HIDDEN_SIZE;
		     index++) {
			uint32_t expected = 0, actual = 0;
			memcpy(&expected,
			       reference_bytes +
				       reference.row_output.byte_offset +
				       index * SUM_BYTES,
			       sizeof(expected));
			memcpy(&actual,
			       card->received +
				       selected.row_output.byte_offset +
				       index * SUM_BYTES,
			       sizeof(actual));
			difference_add(&sums, float_of_bits(actual),
				       float_of_bits(expected),
				       actual == expected, 0, 0.0,
				       !isfinite(float_of_bits(actual)));
		}
		difference_print("joined router chunk", "wide", "hidden",
				 "exact", &hidden);
		difference_print("joined router chunk", "wide", "pair_output",
				 "exact", &pairs);
		difference_print("joined router chunk", "wide", "row_sum",
				 "exact", &sums);
		test_linear_expect(
			hidden.outside_count == 0 && pairs.outside_count == 0 &&
				sums.outside_count == 0 &&
				hidden.squared_difference <=
					HIDDEN_RMS_LIMIT * HIDDEN_RMS_LIMIT *
						hidden.squared_reference &&
				pairs.squared_difference <=
					COMPOUND_PAIR_RMS_LIMIT *
						COMPOUND_PAIR_RMS_LIMIT *
						pairs.squared_reference &&
				sums.squared_difference <=
					ROW_SUM_RMS_LIMIT * ROW_SUM_RMS_LIMIT *
						sums.squared_reference,
			"the relaxed joined quantities pass their stated RMS gates");
		test_linear_card_expect_untouched(card, inputs.jobs,
						  "the router's real jobs");
		test_linear_card_expect_untouched(card, inputs.tiles,
						  "the router's real tiles");
		test_linear_card_expect_untouched(card, inputs.pairs,
						  "the router's real pairs");
		test_linear_card_expect_untouched(
			card, source, "the joined source BF16 rows");
		test_linear_card_expect_weights_untouched(card);
	}
	test_linear_expect(is_run, "the complete routed chunk runs");
	free(saved_reference);
	routing_free(&routing);
}

static int microseconds_order(const void *first, const void *second)
{
	const double left = *(const double *)first,
		     right = *(const double *)second;
	return left < right ? -1 : (left > right ? 1 : 0);
}

/* The median wall microseconds of TIMED_RUNS runs of items, each queued and waited for behind a read of the flush
 * range (the card's digest of it: clean lines through the whole L2). An empty range delegates the flush to the
 * external profiler. Report GPU durations from the profiler; these wall times include its overhead. */
static double run_microseconds(struct test_linear_card *card,
			       struct linear_range flush,
			       const struct kernel_launch *items,
			       uint32_t item_count)
{
	double runs[TIMED_RUNS];
	for (uint32_t run = 0; run < TIMED_RUNS; run++) {
		uint64_t digest = 0;
		const bool is_flushed =
			flush.byte_count == 0 ||
			test_linear_expect_status(
				backend->digest(
					card->card, flush.byte_offset,
					flush.byte_count,
					TEST_LINEAR_WAIT_LIMIT_NANOSECONDS,
					&digest),
				STATUS_OK, "the card reads the flush range");
		const uint64_t started = now_nanoseconds();
		const bool is_run =
			is_flushed &&
			test_linear_card_run(card, items, item_count);
		runs[run] = is_run ? (double)(now_nanoseconds() - started) /
					     NANOSECONDS_PER_MICROSECOND :
				     0.0;
	}
	qsort(runs, TIMED_RUNS, sizeof(runs[0]), microseconds_order);
	return runs[MEDIAN_RUN];
}

/* only_rows: the one row count to time, or 0 for every count of timed_row_counts. */
static void timing_run(struct test_linear_card *card,
		       const struct card_kernels *kernels, uint64_t stream,
		       uint32_t only_rows, bool profiler_flush)
{
	uint16_t values[HIDDEN_SIZE];
	for (uint32_t timed = 0;
	     timed < (only_rows != 0 ? 1u : TIMED_ROW_COUNTS); timed++) {
		const uint32_t row_count =
			only_rows != 0 ? only_rows : timed_row_counts[timed];
		struct routing routing;
		struct case_inputs inputs;
		struct chain_arrays arrays;
		struct kernel_launch launches[FORM_COUNT][MOST_LAUNCHES];
		struct kernel_launch quantize;
		test_linear_context("timed, %" PRIu32 " rows", row_count);
		if (!routing_make(&routing, row_count,
				  row_count <= JOB_COLUMNS ? TIMED_PHYSICAL :
							     LOGICAL_EXPERTS,
				  false, stream + timed * STREAM_PER_CASE)) {
			return;
		}
		test_linear_card_clear(card);
		const uint32_t physical_count =
			row_count > JOB_COLUMNS ?
				(profiler_flush ? PROFILED_PHYSICAL :
						  TIMED_POOLED_PHYSICAL) :
				TIMED_PHYSICAL;
		inputs_set_aside(card, &inputs, &routing, physical_count,
				 false);
		chain_set_aside(card, &arrays, row_count);
		/* Source BF16 rows are dead after initial quantization, before the first gate/up writes hidden. */
		const uint64_t source_bytes =
			linear_values_byte_count(row_count, HIDDEN_SIZE);
		const struct linear_range row_values_range =
			profiler_flush ?
				(struct linear_range){
					arrays.hidden.byte_offset, source_bytes
				} :
				test_linear_card_work(card, source_bytes);
		const struct linear_range flush_range =
			profiler_flush ?
				(struct linear_range){ 0, 0 } :
				test_linear_card_work(card, FLUSH_BYTES);
		bool is_run = test_linear_card_has_room(card);
		for (uint32_t row = 0; is_run && row < row_count; row++) {
			row_values(values, row, KIND_PLAIN,
				   stream + timed * STREAM_PER_CASE);
			is_run = test_linear_card_put(
				card, row_values_range,
				(uint64_t)row * HIDDEN_SIZE * VALUE_BYTES,
				values, sizeof(values));
		}
		if (is_run) {
			weights_fill(card, &inputs, KIND_PLAIN, stream);
			is_run = router_tables_run(card, kernels, &inputs,
						   &routing);
			const struct linear_quantize_description rows = {
				row_values_range, inputs.codes,
				inputs.exponents, inputs.operands,
				HIDDEN_SIZE,	  row_count
			};
			is_run =
				is_run &&
				test_linear_expect_status(
					linear_quantize_prepare(
						&rows, card->arena_byte_count,
						&card->kernels, &quantize),
					STATUS_OK,
					"the rows' quantization is prepared") &&
				test_linear_card_run(card, &quantize, 1);
		}
		const uint64_t expert_matrix_bytes =
			(uint64_t)routing.distinct_count * MATRIX_BYTES;
		const double gate_up_bytes =
			(double)(expert_matrix_bytes * OUTPUT_HALVES);
		const double down_bytes = (double)expert_matrix_bytes;
		const double gate_up_flop = (double)routing.pair_count *
					    OUTPUT_HALVES * INNER_SIZE *
					    HIDDEN_SIZE * FLOP_PER_PRODUCT;
		const double down_flop = (double)routing.pair_count *
					 HIDDEN_SIZE * INNER_SIZE *
					 FLOP_PER_PRODUCT;
		(void)printf(
			"time rows=%" PRIu32 " pairs=%" PRIu32
			" experts=%" PRIu32 " jobs=%" PRIu32 " tiles=%" PRIu32
			" floor_us=%.1f gate_up_floor_us=%.1f down_floor_us=%.1f read_rate_gbs=%.0f\n",
			row_count, routing.pair_count, routing.distinct_count,
			routing.job_count, routing.tile_count,
			(gate_up_bytes + down_bytes) /
				READ_RATE_BYTES_PER_MICROSECOND,
			gate_up_bytes / READ_RATE_BYTES_PER_MICROSECOND,
			down_bytes / READ_RATE_BYTES_PER_MICROSECOND,
			READ_RATE_GIGABYTES);
		for (uint32_t form = 0; is_run && form < FORM_COUNT; form++) {
			struct kernel_launch *const chain = launches[form];
			const struct chain_layout layout =
				layout_of((enum form)form);
			is_run = form_prepare(card, kernels, (enum form)form,
					      &inputs, &arrays, row_count,
					      inputs.tile_entries, chain) &&
				 test_linear_card_run(card, chain,
						      layout.launch_count);
			if (!is_run) {
				break;
			}
			const double gate_up = run_microseconds(
				card, flush_range, &chain[layout.gate_up_first],
				layout.product_launches);
			const double down = run_microseconds(
				card, flush_range, &chain[layout.down_first],
				layout.product_launches);
			const double whole = run_microseconds(
				card, flush_range, chain, layout.launch_count);
			(void)printf(
				"time rows=%" PRIu32
				" form=%s gate_up_us=%.1f down_us=%.1f chain_us=%.1f "
				"gate_up_tflops=%.2f down_tflops=%.2f gate_up_gbs=%.0f down_gbs=%.0f\n",
				row_count, form_names[form], gate_up, down,
				whole,
				gate_up_flop / gate_up /
					TERAFLOP_PER_SECOND_IN_FLOP_PER_MICROSECOND,
				down_flop / down /
					TERAFLOP_PER_SECOND_IN_FLOP_PER_MICROSECOND,
				gate_up_bytes / gate_up /
					GIGABYTE_PER_SECOND_IN_BYTES_PER_MICROSECOND,
				down_bytes / down /
					GIGABYTE_PER_SECOND_IN_BYTES_PER_MICROSECOND);
			(void)fflush(stdout);
		}
		test_linear_expect(is_run, "the timed launches run");
		routing_free(&routing);
		if (!is_run) {
			return;
		}
	}
}

/* ---- the cards ---- */

static bool card_kernels_prepare(struct test_linear_card *card,
				 struct card_kernels *kernels)
{
	return test_linear_expect_status(
		       expert_relaxed_kernels_prepare(backend, card->card,
						      card->arithmetic_class,
						      &kernels->relaxed),
		       STATUS_OK, "the relaxed kernels are prepared") &&
	       test_linear_expect_status(
		       router_kernels_prepare(backend, card->card,
					      &kernels->router),
		       STATUS_OK, "the production router kernels are prepared");
}

static void other_card_refusals(uint32_t card_index)
{
	static const uint32_t classes[] = { ARITHMETIC_CLASS_SM_120,
					    ARITHMETIC_CLASS_SM_120_NATIVE };
	struct device_card *card = NULL;
	if (!test_linear_expect_status(
		    backend->card_open(card_index,
				       BACKEND_STAGING_ALIGNMENT_BYTES, &card),
		    STATUS_OK,
		    "the other card opens for CMP-kernel refusal checks")) {
		return;
	}
	for (uint32_t index = 0; index < COUNT_OF(classes); index++) {
		struct expert_relaxed_kernels kernels, untouched;
		memset(&kernels, OUTPUT_SENTINEL, sizeof(kernels));
		untouched = kernels;
		test_linear_expect_status(
			expert_relaxed_kernels_prepare(
				backend, card, classes[index], &kernels),
			STATUS_UNSUPPORTED,
			"the PRO class refuses CMP-only kernels");
		test_linear_expect(
			memcmp(&kernels, &untouched, sizeof(kernels)) == 0,
			"a refused PRO kernel table remains untouched");
	}
	backend->card_close(card, TEST_LINEAR_WAIT_LIMIT_NANOSECONDS);
}

int main(int argument_count, char **arguments)
{
	uint32_t card_count = 0, only_rows = 0;
	bool is_timed = false, profiler_flush = false, is_checked = true,
	     is_joined = true, is_special = true, is_understood = true,
	     compact = false;
	for (int argument = 1; argument < argument_count; argument++) {
		unsigned long rows = 0;
		char *end = NULL;
		if (strcmp(arguments[argument], "--compact") == 0) {
			compact = true;
		} else if (strcmp(arguments[argument], "--profiled") == 0) {
			profiler_flush = true;
			is_timed = true;
			is_checked = false;
			is_joined = false;
			is_special = false;
		} else if (strcmp(arguments[argument], "--time") == 0 ||
			   strcmp(arguments[argument], "--time-only") == 0) {
			is_timed = true;
			is_checked = strcmp(arguments[argument], "--time") == 0;
			is_joined = is_checked;
			is_special = is_checked;
		} else if (strcmp(arguments[argument], "--chunk-only") == 0) {
			is_checked = false;
			is_joined = true;
			is_special = false;
		} else if (strcmp(arguments[argument], "--special-only") == 0) {
			is_checked = false;
			is_joined = false;
			is_special = true;
		} else if (strncmp(arguments[argument], ROWS_OPTION,
				   sizeof(ROWS_OPTION) - 1u) == 0) {
			rows = strtoul(arguments[argument] +
					       sizeof(ROWS_OPTION) - 1u,
				       &end, DECIMAL);
			is_understood = is_understood && rows != 0 &&
					*end == '\0' && rows <= MOST_TIMED_ROWS;
			only_rows = is_understood ? (uint32_t)rows : 0u;
		} else {
			is_understood = false;
		}
	}
	(void)alarm(TEST_TIME_LIMIT_SECONDS);
	if (!is_understood ||
	    (profiler_flush && (is_checked || is_joined || is_special))) {
		(void)fprintf(
			stderr,
			"usage: expert_relaxed_card_test [--time | --time-only | --profiled | "
			"--chunk-only | --special-only] "
			"[--rows=1..4096] [--compact]\n");
		return EXIT_FAILURE;
	}
	if (!test_linear_expect_status(backend->card_count(&card_count),
				       STATUS_OK, "the cards are counted") ||
	    !test_linear_expect(card_count != 0, "a card is there")) {
		return EXIT_FAILURE;
	}
	for (uint32_t card_index = 0; card_index < card_count; card_index++) {
		struct card_description_record description;
		struct test_linear_card card;
		struct card_kernels kernels;
		if (!test_linear_expect_status(
			    backend->card_describe(card_index, &description),
			    STATUS_OK, "the card is described")) {
			continue;
		}
		if (description.arithmetic_class != ARITHMETIC_CLASS_SM_80) {
			(void)printf("card %" PRIu32 ": %s, class %" PRIu32
				     ": checking CMP-only kernel refusals\n",
				     card_index, description.name,
				     description.arithmetic_class);
			other_card_refusals(card_index);
			continue;
		}
		const bool timing_only = is_timed && !is_checked &&
					 !is_joined && !is_special;
		const uint32_t allocated_rows =
			only_rows == 0 ? MOST_TIMED_ROWS : only_rows;
		const uint64_t most_work = profiler_flush ?
						   PROFILED_WORK_CAPACITY :
						   TIMED_WORK_CAPACITY;
		const uint64_t work_needed =
			timing_only ?
				TIMED_THIN_WORK_CAPACITY +
					(most_work - TIMED_THIN_WORK_CAPACITY) *
						allocated_rows /
						MOST_TIMED_ROWS :
				(compact ? COMPACT_WORK_CAPACITY :
					   WORK_CAPACITY);
		const uint64_t work_capacity =
			(work_needed + BACKEND_STAGING_ALIGNMENT_BYTES - 1u) /
			BACKEND_STAGING_ALIGNMENT_BYTES *
			BACKEND_STAGING_ALIGNMENT_BYTES;
		const uint32_t physical_count =
			timing_only && only_rows > JOB_COLUMNS ?
				(profiler_flush ? PROFILED_PHYSICAL :
						  TIMED_POOLED_PHYSICAL) :
				(compact && !is_timed ? PROFILED_PHYSICAL :
							TIMED_PHYSICAL);
		const uint64_t weights_capacity =
			(uint64_t)physical_count * MATRICES * MATRIX_BYTES;
		{
			(void)printf("allocation work_bytes=%" PRIu64
				     " weight_bytes=%" PRIu64
				     " physical_experts=%" PRIu32
				     " free_bytes=%" PRIu64
				     " backend_reserve_bytes=%" PRIu64 "\n",
				     work_capacity, weights_capacity,
				     physical_count,
				     description.memory_free_bytes,
				     description.memory_reserved_bytes);
		}
		if (!test_linear_card_open(&card, card_index, work_capacity,
					   weights_capacity)) {
			continue;
		}
		if (card_kernels_prepare(&card, &kernels)) {
			if (is_special) {
				special_ordered_run(&card, &kernels,
						    compact ?
							    PROFILED_PHYSICAL :
							    CORRECT_PHYSICAL);
				(void)fflush(stdout);
			}
			for (uint32_t case_index = 0;
			     is_checked && case_index < COUNT_OF(relaxed_cases);
			     case_index++) {
				case_run(&card, &kernels,
					 &relaxed_cases[case_index],
					 RELAXED_STREAM +
						 (uint64_t)case_index *
							 STREAM_PER_CASE *
							 MATRICES *
							 TIMED_PHYSICAL,
					 compact ? PROFILED_PHYSICAL :
						   CORRECT_PHYSICAL);
				(void)fflush(stdout);
			}
			for (uint32_t index = JOINED_FIRST_TIMED;
			     is_joined && index < TIMED_ROW_COUNTS; index++) {
				if (only_rows == 0 ||
				    only_rows == timed_row_counts[index]) {
					joined_run(&card, &kernels,
						   timed_row_counts[index],
						   compact ? PROFILED_PHYSICAL :
							     TIMED_PHYSICAL,
						   compact);
					(void)fflush(stdout);
				}
			}
			if (is_timed) {
				timing_run(&card, &kernels, RELAXED_STREAM,
					   only_rows, profiler_flush);
			}
		}
		test_linear_card_close(&card);
	}
	(void)printf("expert_relaxed_card_test: %d failures\n",
		     test_linear_failures());
	return test_linear_failures() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
