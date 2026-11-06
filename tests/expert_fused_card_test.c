/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * expert_fused_card_test: on the real cards, the routed experts' phase in fewer launches (expert_fused.h) against
 * the launches the engine runs today, on the same inputs: every byte of what they leave must be the same.
 *
 *     expert_fused_card_test [--time]
 *
 * On every card, in each class it runs (class 80; a card of class 120 natively, ARITHMETIC_CLASS_SM_120_NATIVE,
 * the PRO's default, then in the exact profile's emulation, class 120: the card opened again for each), with the
 * model's experts (5,120 to 2,304, six to a row; 48 of them, E2M1 weights of random codes; 24 on a card of class 120,
 * which serves production beside the test, jobs naming expert e % 24), the model's shared down (E4M3, 2,304 to 5,120)
 * and rows quantized on the card by today's quantizer (the first PLAIN_ROWS of plain values, the last of the
 * extremes, the others of the edges); scale codes up to 2^15 apart, so that the chains round:
 *   - THE PRODUCER against gate with up and the quantization of the hidden vectors (launches GATE_UP and QUANTIZE of
 *     linear_routed_experts_prepare), for 1, 3 and 8 rows whose experts no row shares, all rows share, and
 *     half of them share, at every slice count the class runs: the hidden values, codes, exponents and operands of
 *     every pair, every byte, and the arrival counters back at zero. Each chain writes arrays of its own that start
 *     as rubbish of a byte of its own, so that a byte one chain leaves unwritten shows. In every other block of 32
 *     outputs of an expert the first tile's scales are 2^-10 of the usual and the second's 2^4 above them: a block's
 *     exponent comes from the larger tile. Job tables hold an empty job of rubbish first and last, rubbish in the
 *     columns not in use, and the rows of every other expert in descending order (a row takes any column). Then rows
 *     of which only the first stand in a job (2 of 3, 5 and 1 of 8, as the router leaves rows that are not live): the
 *     hidden values of the other pairs are the same stale numbers in both chains, and their codes, exponents and
 *     operands must be today's too.
 *   - THE JOIN against the row sum and the FFN site's post with the combine (ROW_SUM of the routed experts' launches,
 *     HYPER_SITE_POST of hyper_site_prepare), and THE SHARED JOIN against the shared expert's down
 *     (linear_trunk_quantized_prepare), the row sum and the post, for 1, 3 and 8 rows at the shared expert's
 *     slice count, and at 1 and 8 rows at every other count the class runs: the four streams of every row and the
 *     shared expert's output, every byte; each given no array for its row sums (that array untouched) and given one
 *     (today's row sums, every byte). Pair outputs of the edges (one slot of row 1 of the extremes), and in every row
 *     coordinates whose six slots hold 2^30, 1, -2^30, 1, 0 and 0, rotated by the row: their float32 sum depends on
 *     the order of addition.
 *   - SHOWN FAILING in every class: (1) the producer once more with one route weight half as large again, in a job
 *     table of its own: that pair's bytes must differ, no other pair's; (2) the producer at another slice count than
 *     today's launch: class 120 native cuts the chains by S, so some bytes must differ; class 80 and the emulation
 *     keep the reference order whatever S (linear_ordered_chains), so none may; (3) the join and the shared join over
 *     pair outputs whose slots 0 and 3 of row 0 are swapped: the streams must differ from today's somewhere, only in
 *     row 0, and only at coordinates where the host's float32 sums of the two orders differ.
 *   - No launch writes outside its ranges (the guards), the ranges only read come back untouched, and the card's
 *     digest of the matrices is the host's.
 * --time: per class, at 1 and 8 rows (the producer's whose experts no row shares), today's launches and the fused
 * one, each repeated back to back in one graph, sampled in pairs on the card's clock (card_timing.h): each chain's
 * percentiles and those of the saving, today's sample less the fused one beside it, a repetition. The producer at 1
 * row also with each repetition on another of eight sets of six experts (weights from memory on a card whose L2 holds
 * six); each join given no row sums against itself storing them (the saving: less what the stores cost). L2 is never
 * flushed: decode keeps activations hot. TIME_ROUNDS lines TIME a part.
 * It FAILS on any difference and when a card has no room.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "anemone/expert_fused.h"
#include "anemone/hyper_connection.h"
#include "anemone/linear.h"
#include "anemone/linear_records.h"
#include "anemone/model_layout.h"
#include "anemone/status.h"
#include "card_timing.h"
#include "linear_card_fixture.h"

#define TEST_TIME_LIMIT_SECONDS 3600            /* the alarm ends the test here, whatever the driver does */
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define ROWS LINEAR_MAXIMUM_ROWS
#define PICKS MODEL_EXPERTS_PER_ROW
#define HIDDEN MODEL_HIDDEN
#define INNER 2304u                             /* the model's expert inner size (model_v41.h) */
#define EXPERTS (ROWS * PICKS)                  /* eight rows of six experts no row shares */
#define STORED_EXPERTS_CLASS_120 24u            /* a PRO serves production beside the test: half the matrices */
#define MOST_PAIRS ((uint64_t)ROWS * PICKS)
#define MOST_JOBS (EXPERTS + 2u)                /* an empty job before the experts' and one behind */
#define HALF_SHARED 3u                          /* of a row's six experts, these are every row's */
#define PLAIN_ROWS 4u
#define EXTREME_INPUT_ROW (ROWS - 1u)           /* the last row of the extremes: infinite operands and sums */
#define E2M1_BITS 4u
#define E4M3_BITS 8u
#define VALUE_BYTES ((uint64_t)LINEAR_BFLOAT16_BYTES)
#define BLOCK_VALUES LINEAR_VALUES_PER_BLOCK
#define BLOCKS_PER_PAIR (INNER / BLOCK_VALUES)
#define MATRIX_CODE_BYTES ((uint64_t)INNER * HIDDEN / 2u)
#define MATRIX_BYTES (MATRIX_CODE_BYTES + (uint64_t)INNER * HIDDEN / BLOCK_VALUES)
#define DOWN_CODE_BYTES ((uint64_t)HIDDEN * INNER)
#define DOWN_BYTES (DOWN_CODE_BYTES + (uint64_t)HIDDEN / BLOCK_VALUES * (INNER / BLOCK_VALUES))
#define MATRICES_PER_EXPERT 2u                  /* gate and up: the producer reads no down */
#define WORK_CAPACITY (16ull << 20u)
#define STORED_DOWNS 8u                         /* the routed experts' down matrices, of the tail alone: e % 8 */
#define SHARED_MATRICES 3u                      /* the shared expert's gate, up and down, of one size */
#define WEIGHTS_CAPACITY(stored)                                                                                  \
    (((uint64_t)(stored) * MATRICES_PER_EXPERT + STORED_DOWNS) * MATRIX_BYTES + SHARED_MATRICES * DOWN_BYTES)
#define TILE_LOADS (HIDDEN / LINEAR_VALUES_PER_LOAD)
#define SCALE_WORD_BYTES 4u                     /* E2M1 scales: a word of four codes a lane group and load */
#define LANE_GROUPS 8u
#define SCALE_BASE 110u                         /* so that gate and up reach beyond the activation limit, and */
#define SCALE_SPREAD 15u                        /* blocks' terms 2^15 apart round a chain's partial sums */
#define SMALL_TILE_SCALE_DROP 10u               /* every other block: its first tile 2^-10, its second 2^4 */
#define LARGE_TILE_SCALE_RISE 4u
#define E4M3_NO_NUMBER_MAGNITUDE 0x7Fu          /* 0x7F and 0xFF stand for no number: none in a matrix */
#define ACTIVATION_LIMIT 10.0f
#define ROUTE_WEIGHT_STEPS 12u                  /* a route weight is 1 to 12 eighths */
#define ROUTE_WEIGHT_STEP 0.125f
#define CHANGED_WEIGHT_FACTOR 1.5f
#define SUM_KINDS 2u                            /* a join given no array for its row sums, then one given it */
#define SLICE_CHANGE 2u                         /* the control's other slice count: twice one, else half */
#define ORDER_KINDS 2u                          /* experts' rows ascending, descending, in turn */
#define UNEVEN_PERIOD 2u                        /* every other block of 32 outputs has uneven tiles */
#define CONTROL_ROWS 3u
#define ORPHAN_LIVE_ROWS 5u                     /* of 8 rows, the live ones of an orphan case */
#define RUBBISH_BYTE 0xA7
#define TODAY_RUBBISH 0x3C                      /* what each chain's outputs start as */
#define FUSED_RUBBISH 0x5D
#define HASH_BYTES 8u
#define HASH_FRACTION_SHIFT 40u                 /* 24 bits of a hash: a fraction exact in float32 */
#define HASH_FRACTION_UNIT (1.0f / 16777216.0f)
#define COMB_FRACTION 0.5f
#define WEIGHTS_STREAM 0x0E100000ull            /* the streams of the hashes, far from those of the other tests */
#define ROWS_STREAM 0x0E200000ull
#define ROUTE_STREAM 0x0E300000ull
#define JOIN_STREAM 0x0E400000ull
#define DOWN_STREAM 0x0E500000ull
#define SHARED_HIDDEN_STREAM 0x0E600000ull
#define ROUTED_DOWN_STREAM 0x0E700000ull
#define STALE_STREAM 0x0E800000ull              /* the hidden values of the pairs of no job */
#define PAIR_STREAMS 64ull
#define COMB_STREAM 2u                          /* of the join's streams: post 1, comb 2, the streams' 2 + s */
#define STREAMS_STREAM 2u
#define ORDER_PERIOD 97u                        /* coordinates d % 97 == 0 of a row hold the order pattern */
#define ORDER_PATTERN_SLOTS 6u
#define EXTREME_ROW 1u                          /* this row's slot 3 holds values of the extremes */
#define EXTREME_SLOT 3u
#define SWAPPED_SLOT 3u                         /* the joins' control swaps slots 0 and 3 of row 0 */
#define BFLOAT16_SHIFT 16u                      /* a bfloat16 is the upper half of its float32 */
#define SITE_SCALE_BYTES ((uint64_t)HYPER_SCALES * sizeof(float))
#define SUMS_ROW_BYTES ((uint64_t)HIDDEN * sizeof(float))
#define ARRIVALS_BYTES ((uint64_t)MOST_JOBS * BLOCKS_PER_PAIR * sizeof(uint32_t))
#define ROTATION_SETS 8u                        /* sets of six experts, for the producer's time at 1 row */
#define PRODUCER_REPETITIONS_ONE_ROW 200u
#define PRODUCER_REPETITIONS_EIGHT_ROWS 24u
/* A graph's launches hold card memory until the card closes, and the PRO, which serves production beside the test,
 * has little: 2,000 and 1,000 left no room to open it again for class 120 (26 October). */
#define JOIN_REPETITIONS 500u
#define SHARED_JOIN_REPETITIONS 250u
#define TODAY_JOIN_LAUNCHES 2u                  /* today's join: the row sum, the post (the shared join: down first) */
#define TODAY_MOST_LAUNCHES (TODAY_JOIN_LAUNCHES + 1u)
#define TAIL_TODAY_LAUNCHES 9u                  /* the routed experts' 4, the shared expert's 4, the post */
#define TAIL_FUSED_LAUNCHES 6u                  /* the producer, down, the shared expert's first 3, the shared join */
#define TAIL_JOINED_LAUNCHES 7u                 /* the same, gate with up and the quantization for the producer */
#define TAIL_VARIANTS 2u                        /* fused tails: with the producer; with the shared join alone */
#define TAIL_REPETITIONS_ONE_ROW 64u
#define TAIL_REPETITIONS_EIGHT_ROWS 8u
#define CHAIN_MOST_LAUNCHES TAIL_TODAY_LAUNCHES
#define MOST_REPETITION_LAUNCHES (TODAY_MOST_LAUNCHES * JOIN_REPETITIONS)
#define TIME_SAMPLES 101u
#define TIME_ROUNDS 3u
#define WARM_RUNS 2u                            /* of each graph before its samples */
#define CLASS_COUNT_MOST 2u
#define MEBIBYTE_SHIFT 20u
#define EXPONENT_CODES 256u                     /* of an E8M0 byte */
#define E4M3_MAGNITUDE_MASK 0x7Fu

/* The join's blocks against its prepared HYPER_PIECE_THREADS (--time): each divides a row's 640 threads. */
static const uint32_t join_geometry_threads[] = {64u, 32u};
static const char *const join_geometry_names[] = {"join_64_threads_against_128", "join_32_threads_against_128"};
/* bfloat16 bits of the joins' order pattern: 2^30, 1, -2^30, 1, 0, 0 */
static const uint16_t order_pattern[ORDER_PATTERN_SLOTS] = {0x4E80u, 0x3F80u, 0xCE80u, 0x3F80u, 0x0000u, 0x0000u};

enum routing { ROUTING_DISJOINT, ROUTING_SHARED, ROUTING_HALF, ROUTING_COUNT };
static const char *const routing_names[ROUTING_COUNT] = {"no row shares an expert", "every row shares all six",
                                                         "every row shares three"};
enum chain { TODAY, FUSED, CHAINS };
static const char *const sum_kind_names[SUM_KINDS] = {"no row sums", "its row sums stored"};

/* Every range of the arena, set aside once a card (the matrices first). */
struct arena_ranges {
	struct linear_range gate[EXPERTS], up[EXPERTS], down[STORED_DOWNS],
		shared_down, shared_gate, shared_up;
	struct linear_range shared_gate_output, shared_up_output;
	struct linear_range tail_hidden, tail_codes, tail_exponents,
		tail_operands; /* the tail's shared hidden */
	struct linear_range row_values, codes, exponents, operands;
	struct linear_range shared_values, shared_codes, shared_exponents,
		shared_operands;
	struct linear_range jobs[CHAINS], rotation_jobs[ROTATION_SETS],
		arrivals;
	struct linear_range hidden[CHAINS], hidden_codes[CHAINS],
		hidden_exponents[CHAINS];
	struct linear_range hidden_operands[CHAINS];
	struct linear_range pair_output, permuted_output, shared_output[CHAINS],
		streams[CHAINS], sums[CHAINS];
	struct linear_range post;
	struct linear_range comb, carried_pre, mixes, scale, base, norm_weight,
		site_hidden;
};

/* A chain of launches that is timed: its variants (repetition r takes variant r % variants), each of launch_count. */
struct timed_chain {
	struct kernel_launch launches[ROTATION_SETS][CHAIN_MOST_LAUNCHES];
	uint32_t launch_count;
};

/* One class of a card under test: its kernels and its slice counts. */
struct class_under_test {
    uint32_t arithmetic_class;
    struct linear_kernels_record linear;
    struct hyper_kernels_record hyper;
    struct expert_fused_kernels fused;
    uint32_t gate_up_slices, down_slices, shared_down_slices, shared_gate_up_slices;
};

static const struct backend_operations *const backend = &backend_cuda_value;
static const struct matrix_shape_record gate_up_shape = { INNER, HIDDEN,
							  E2M1_BITS, 0 };
static const struct matrix_shape_record down_shape = { HIDDEN, INNER, E2M1_BITS,
						       0 };
static const struct matrix_shape_record shared_down_shape = { HIDDEN, INNER,
							      E4M3_BITS, 0 };
static const struct matrix_shape_record shared_gate_up_shape = { INNER, HIDDEN,
								 E4M3_BITS, 0 };
static struct arena_ranges ranges;
static struct expert_job_record job_table[MOST_JOBS];
static struct kernel_launch repetition_launches[CHAINS]
					       [MOST_REPETITION_LAUNCHES];
static uint32_t stored_experts = EXPERTS;      /* expert e's matrices are those stored for e % stored_experts */
static uint16_t vector_values[HIDDEN];

static float hash_fraction(uint64_t stream, uint64_t position)
{
    return (float)(test_linear_hash(stream, position) >> HASH_FRACTION_SHIFT) * HASH_FRACTION_UNIT;
}

/* The expert of a row's slot, ascending in the slot as the router leaves them. */
static uint32_t routed_expert(enum routing routing, uint32_t row, uint32_t slot)
{
    return routing == ROUTING_DISJOINT ? row * PICKS + slot
         : routing == ROUTING_SHARED   ? slot
         : slot < HALF_SHARED          ? slot
                                       : HALF_SHARED + row * (PICKS - HALF_SHARED) + (slot - HALF_SHARED);
}

static float route_weight_of(uint32_t pair)
{
    return (float)(test_linear_hash(ROUTE_STREAM, pair) % ROUTE_WEIGHT_STEPS + 1u) * ROUTE_WEIGHT_STEP;
}

/* The job table of rows routed so, experts from expert_base on, into job_table: an empty job of rubbish, one job a
 * routed expert in ascending order (the rows of every other expert in descending order), an empty job of rubbish.
 * Gives the number of entries. */
static uint32_t jobs_make(enum routing routing, uint32_t row_count, uint32_t expert_base)
{
    uint32_t job_count = 0;
    memset(job_table, RUBBISH_BYTE, sizeof(job_table));
    job_table[job_count++].row_count = 0;
    for (uint32_t expert = 0; expert < EXPERTS; expert++) {
	    struct expert_job_record job;
	    memset(&job, RUBBISH_BYTE, sizeof(job));
	    job.row_count = 0;
	    job.expert = 0;
	    for (uint32_t order = 0; order < row_count; order++) {
		    const uint32_t row = expert % ORDER_KINDS == 0 ?
						 order :
						 row_count - 1u - order;
		    for (uint32_t slot = 0; slot < PICKS; slot++) {
			    if (routed_expert(routing, row, slot) == expert) {
				    job.pair_index[job.row_count] =
					    row * PICKS + slot;
				    job.route_weight[job.row_count] =
					    route_weight_of(row * PICKS + slot);
				    job.row_count++;
			    }
		    }
        }
        if (job.row_count != 0) {
            job.gate_offset = ranges.gate[(expert_base + expert) % stored_experts].byte_offset;
            job.up_offset = ranges.up[(expert_base + expert) % stored_experts].byte_offset;
            job.down_offset = ranges.down[(expert_base + expert) % STORED_DOWNS].byte_offset;  /* the tail's */
            job_table[job_count++] = job;
        }
    }
    job_table[job_count++].row_count = 0;
    return job_count;
}

/* The routed experts' description of row_count rows over a job table: chain's hidden arrays and row sums; the join's
 * pair outputs, which only the row sum and down touch. */
static struct linear_routed_experts_description
routed_of(const struct test_linear_card *card,
	  const struct class_under_test *tested, enum chain chain,
	  struct linear_range jobs, uint32_t job_count, uint32_t row_count,
	  uint32_t gate_up_slices)
{
	return (struct linear_routed_experts_description){
		.jobs = jobs,
		.expert_weights = test_linear_card_all_weights(card),
		.codes = ranges.codes,
		.exponents = ranges.exponents,
		.operands = ranges.operands,
		.hidden = ranges.hidden[chain],
		.hidden_codes = ranges.hidden_codes[chain],
		.hidden_exponents = ranges.hidden_exponents[chain],
		.hidden_operands = ranges.hidden_operands[chain],
		.pair_output = ranges.pair_output,
		.row_output = ranges.sums[chain],
		.gate_up_shape = gate_up_shape,
		.down_shape = down_shape,
		.weight_format = WEIGHT_FORMAT_E2M1_SCALED,
		.job_count = job_count,
		.row_count = row_count,
		.pairs_per_row = PICKS,
		.gate_up_slice_count = gate_up_slices,
		.down_slice_count = tested->down_slices,
		.activation_limit = ACTIVATION_LIMIT,
		.reserved_zero = 0
	};
}

/* Random code bytes of a matrix straight into the image: E2M1 codes are any nibble; an E4M3 byte that stands for no
 * number becomes its neighbour. */
static unsigned char *codes_write(struct test_linear_card *card,
				  struct linear_range stored,
				  uint64_t byte_count, uint64_t code_byte_count,
				  uint64_t stream, bool is_e4m3)
{
    unsigned char *const bytes = test_linear_card_writable(card, stored);
    if (bytes == NULL || stored.byte_count != byte_count) {
        test_linear_fail("a matrix of %" PRIu64 " bytes does not lie in the image", byte_count);
        return NULL;
    }
    for (uint64_t word = 0; word < code_byte_count / HASH_BYTES; word++) {
        const uint64_t hash = test_linear_hash(stream, word);
        memcpy(bytes + word * HASH_BYTES, &hash, HASH_BYTES);
    }
    for (uint64_t code = 0; is_e4m3 && code < code_byte_count; code++) {
        bytes[code] = (unsigned char)((bytes[code] & E4M3_MAGNITUDE_MASK) == E4M3_NO_NUMBER_MAGNITUDE ? bytes[code] - 1u
                                                                                                    : bytes[code]);
    }
    return bytes;
}

/* An expert's E2M1 matrix: random codes, scales from the base (every other block of 32 outputs with a small first
 * tile and a large second one). */
static void expert_matrix_write(struct test_linear_card *card,
				struct linear_range stored, uint64_t stream)
{
    unsigned char *const bytes = codes_write(card, stored, MATRIX_BYTES, MATRIX_CODE_BYTES, stream, false);
    for (uint64_t scale = 0; bytes != NULL && scale < MATRIX_BYTES - MATRIX_CODE_BYTES; scale++) {
        const uint64_t tile = scale / ((uint64_t)TILE_LOADS * LANE_GROUPS * SCALE_WORD_BYTES);
	const bool is_uneven = tile / EXPERT_PRODUCER_TILES % UNEVEN_PERIOD !=
			       0;
	const uint32_t base = !is_uneven ? SCALE_BASE :
			      tile % EXPERT_PRODUCER_TILES == 0 ?
					   SCALE_BASE - SMALL_TILE_SCALE_DROP :
					   SCALE_BASE + LARGE_TILE_SCALE_RISE;
	bytes[MATRIX_CODE_BYTES + scale] =
            (unsigned char)(base + test_linear_hash(stream + 1u, scale) % (SCALE_SPREAD + 1u));
    }
}

/* A matrix of the shared expert (E4M3, 2,304 by 5,120 or the other way: one size): random codes, scales from the
 * base. */
static void down_matrix_write(struct test_linear_card *card,
			      struct linear_range stored, uint64_t stream)
{
    unsigned char *const bytes = codes_write(card, stored, DOWN_BYTES, DOWN_CODE_BYTES, stream, true);
    for (uint64_t scale = 0; bytes != NULL && scale < DOWN_BYTES - DOWN_CODE_BYTES; scale++) {
        bytes[DOWN_CODE_BYTES + scale] =
            (unsigned char)(SCALE_BASE + test_linear_hash(stream + SHARED_MATRICES, scale) % (SCALE_SPREAD + 1u));
    }
}

static struct linear_range work(struct test_linear_card *card,
				uint64_t byte_count)
{
    return test_linear_card_work(card, byte_count);
}

/* Every range of a card, the matrices laid out: false when the card has no room. */
static bool arena_lay_out(struct test_linear_card *card)
{
    for (uint32_t expert = 0; expert < stored_experts; expert++) {
        ranges.gate[expert] = test_linear_card_weights(card, MATRIX_BYTES);
        ranges.up[expert] = test_linear_card_weights(card, MATRIX_BYTES);
        expert_matrix_write(card, ranges.gate[expert], WEIGHTS_STREAM + (uint64_t)MATRICES_PER_EXPERT * expert);
        expert_matrix_write(card, ranges.up[expert], WEIGHTS_STREAM + (uint64_t)MATRICES_PER_EXPERT * expert + 1u);
    }
    for (uint32_t expert = 0; expert < STORED_DOWNS; expert++) {
        ranges.down[expert] = test_linear_card_weights(card, MATRIX_BYTES);
        expert_matrix_write(card, ranges.down[expert], ROUTED_DOWN_STREAM + (uint64_t)MATRICES_PER_EXPERT * expert);
    }
    ranges.shared_down = test_linear_card_weights(card, DOWN_BYTES);
    ranges.shared_gate = test_linear_card_weights(card, DOWN_BYTES);
    ranges.shared_up = test_linear_card_weights(card, DOWN_BYTES);
    down_matrix_write(card, ranges.shared_down, DOWN_STREAM);
    down_matrix_write(card, ranges.shared_gate, DOWN_STREAM + 1u);
    down_matrix_write(card, ranges.shared_up, DOWN_STREAM + SHARED_MATRICES - 1u);
    ranges.shared_gate_output =
	    work(card, linear_values_byte_count(ROWS, INNER));
    ranges.shared_up_output = work(card, linear_values_byte_count(ROWS, INNER));
    ranges.tail_hidden = work(card, linear_values_byte_count(ROWS, INNER));
    ranges.tail_codes = work(card, linear_codes_byte_count(ROWS, INNER));
    ranges.tail_exponents =
	    work(card, linear_exponents_byte_count(ROWS, INNER));
    ranges.tail_operands = work(card, linear_values_byte_count(ROWS, INNER));
    ranges.row_values = work(card, linear_values_byte_count(ROWS, HIDDEN));
    ranges.codes = work(card, linear_codes_byte_count(ROWS, HIDDEN));
    ranges.exponents = work(card, linear_exponents_byte_count(ROWS, HIDDEN));
    ranges.operands = work(card, linear_values_byte_count(ROWS, HIDDEN));
    ranges.shared_values = work(card, linear_values_byte_count(ROWS, INNER));
    ranges.shared_codes = work(card, linear_codes_byte_count(ROWS, INNER));
    ranges.shared_exponents =
	    work(card, linear_exponents_byte_count(ROWS, INNER));
    ranges.shared_operands = work(card, linear_values_byte_count(ROWS, INNER));
    for (uint32_t chain = 0; chain < CHAINS; chain++) {
	    ranges.jobs[chain] = work(card, linear_jobs_byte_count(MOST_JOBS));
	    ranges.hidden[chain] =
		    work(card, linear_values_byte_count(MOST_PAIRS, INNER));
	    ranges.hidden_codes[chain] =
		    work(card, linear_codes_byte_count(MOST_PAIRS, INNER));
	    ranges.hidden_exponents[chain] =
		    work(card, linear_exponents_byte_count(MOST_PAIRS, INNER));
	    ranges.hidden_operands[chain] =
		    work(card, linear_values_byte_count(MOST_PAIRS, INNER));
	    ranges.shared_output[chain] =
		    work(card, (uint64_t)ROWS * MODEL_HIDDEN_ROW_BYTES);
	    ranges.streams[chain] =
		    work(card, (uint64_t)ROWS * MODEL_STREAMS_ROW_BYTES);
	    ranges.sums[chain] = work(card, (uint64_t)ROWS * SUMS_ROW_BYTES);
    }
    for (uint32_t set = 0; set < ROTATION_SETS; set++) {
	    ranges.rotation_jobs[set] =
		    work(card, linear_jobs_byte_count(MOST_JOBS));
    }
    ranges.arrivals = work(card, ARRIVALS_BYTES);
    ranges.pair_output =
	    work(card, linear_values_byte_count(MOST_PAIRS, HIDDEN));
    ranges.permuted_output =
	    work(card, linear_values_byte_count(MOST_PAIRS, HIDDEN));
    ranges.post = work(card, (uint64_t)ROWS * MODEL_SITE_ROW_BYTES);
    ranges.comb = work(card, (uint64_t)ROWS * MODEL_COMB_ROW_BYTES);
    ranges.carried_pre = work(card, (uint64_t)ROWS * MODEL_SITE_ROW_BYTES);
    ranges.mixes = work(card, (uint64_t)ROWS * MODEL_MIXES_ROW_BYTES);
    ranges.scale = work(card, SITE_SCALE_BYTES);
    ranges.base = work(card, MODEL_MIXES_ROW_BYTES);
    ranges.norm_weight = work(card, MODEL_HIDDEN_ROW_BYTES);
    ranges.site_hidden = work(card, (uint64_t)ROWS * MODEL_HIDDEN_ROW_BYTES);
    return test_linear_card_has_room(card);
}

/* ROWS vectors of width values from stream on into values, quantized on the card by today's quantizer into the three
 * arrays; the image then holds what the card holds. */
static bool vectors_quantize(struct test_linear_card *card,
			     const struct class_under_test *tested,
			     uint32_t width, uint64_t stream,
			     struct linear_range values,
			     struct linear_range codes,
			     struct linear_range exponents,
			     struct linear_range operands)
{
	struct kernel_launch launch;
	for (uint32_t row = 0; row < ROWS; row++) {
		test_linear_values(
			vector_values, width, stream + row,
			row < PLAIN_ROWS	 ? TEST_LINEAR_VALUES_PLAIN :
			row == EXTREME_INPUT_ROW ? TEST_LINEAR_VALUES_EXTREMES :
						   TEST_LINEAR_VALUES_EDGES);
		(void)test_linear_card_put(card, values,
					   (uint64_t)row * width * VALUE_BYTES,
					   vector_values, width * VALUE_BYTES);
	}
	const struct linear_quantize_description description = {
		values, codes, exponents, operands, width, ROWS
	};
	const struct linear_range quantized[] = { codes, exponents, operands };
	bool is_done =
		test_linear_card_upload(card, values, 0, values.byte_count) &&
		test_linear_expect_status(
			linear_quantize_prepare(&description,
						card->arena_byte_count,
						&tested->linear, &launch),
			STATUS_OK, "a quantization is prepared") &&
		test_linear_card_run(card, &launch, 1) &&
		test_linear_card_fetch(card);
	for (uint32_t range = 0; is_done && range < COUNT_OF(quantized);
	     range++) {
		unsigned char *const image_bytes =
			card->image + quantized[range].byte_offset;
		is_done = test_linear_card_get(card, quantized[range], 0,
					       image_bytes,
					       quantized[range].byte_count);
	}
    return is_done;
}

/* A range of the image filled with one byte and uploaded. */
static bool fill_upload(struct test_linear_card *card,
			struct linear_range range, int byte)
{
    memset(card->image + range.byte_offset, byte, range.byte_count);
    return test_linear_card_upload(card, range, 0, range.byte_count);
}

/* ---- the producer ---- */

/* Both chains' outputs set to their rubbish, and both copies of the job table, in the image and on the card. */
static bool producer_reset(struct test_linear_card *card, uint32_t job_count)
{
    bool is_reset = true;
    for (uint32_t chain = 0; chain < CHAINS && is_reset; chain++) {
        const int byte = chain == TODAY ? TODAY_RUBBISH : FUSED_RUBBISH;
	is_reset = fill_upload(card, ranges.hidden[chain], byte) &&
		   fill_upload(card, ranges.hidden_codes[chain], byte) &&
		   fill_upload(card, ranges.hidden_exponents[chain], byte) &&
		   fill_upload(card, ranges.hidden_operands[chain], byte) &&
		   test_linear_card_put(card, ranges.jobs[chain], 0, job_table,
					linear_jobs_byte_count(job_count)) &&
		   test_linear_card_upload(card, ranges.jobs[chain], 0,
					   ranges.jobs[chain].byte_count);
    }
    return is_reset;
}

/* Today's two launches at today_slices and the producer at fused_slices over the chains' job tables, run, fetched. */
static bool producer_run(struct test_linear_card *card, const struct class_under_test *tested, uint32_t job_count,
                         uint32_t row_count, uint32_t today_slices, uint32_t fused_slices)
{
	struct kernel_launch today[LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT];
	struct kernel_launch fused;
	const struct linear_routed_experts_description today_description =
		routed_of(card, tested, TODAY, ranges.jobs[TODAY], job_count,
			  row_count, today_slices);
	const struct linear_routed_experts_description fused_description =
		routed_of(card, tested, FUSED, ranges.jobs[FUSED], job_count,
			  row_count, fused_slices);
	return test_linear_expect_status(
		       linear_routed_experts_prepare(&today_description,
						     card->arena_byte_count,
						     &tested->linear, today),
		       STATUS_OK, "today's routed experts are prepared") &&
	       test_linear_expect_status(
		       expert_producer_prepare(
			       &fused_description, ranges.arrivals,
			       card->arena_byte_count, &tested->linear,
			       &tested->fused, &fused),
		       STATUS_OK, "the producer is prepared") &&
	       test_linear_card_run(card, today,
				    LINEAR_ROUTED_EXPERTS_QUANTIZE + 1u) &&
	       test_linear_card_run(card, &fused, 1) &&
	       test_linear_card_fetch(card);
}

/* Bytes in which the chains' hidden values, codes, exponents and operands differ: of pairs 0 .. pair_count - 1, and
 * of those outside pair `allowed_pair` (none: UINT32_MAX). Into *hidden_out, the hidden values' alone. */
static uint64_t producer_differing(const struct test_linear_card *card, uint32_t pair_count, uint32_t allowed_pair,
                                   uint64_t *outside_out, uint64_t *hidden_out)
{
    const uint64_t pair_bytes[] = {INNER * VALUE_BYTES, INNER, BLOCKS_PER_PAIR, INNER * VALUE_BYTES};
    const struct linear_range *const arrays[] = { ranges.hidden,
						  ranges.hidden_codes,
						  ranges.hidden_exponents,
						  ranges.hidden_operands };
    uint64_t differing = 0, outside = 0;
    *hidden_out = 0;
    for (uint32_t array = 0; array < COUNT_OF(arrays); array++) {
        for (uint32_t pair = 0; pair < pair_count; pair++) {
            const uint64_t count = test_linear_card_differing(card, arrays[array][TODAY], pair * pair_bytes[array],
                                                              arrays[array][FUSED], pair * pair_bytes[array],
                                                              pair_bytes[array], NULL);
            differing += count;
            outside += pair == allowed_pair ? 0u : count;
            *hidden_out += array == 0 ? count : 0u;
        }
    }
    *outside_out = outside;
    return differing;
}

/* What the chains only read came back as the image holds it, the arrival counters at zero, and both chains left the
 * pairs beyond the case's as they were. */
static void producer_expect_untouched(const struct test_linear_card *card, uint32_t pair_count)
{
	const struct linear_range read_only[] = {
		ranges.codes,	    ranges.exponents,	ranges.operands,
		ranges.jobs[TODAY], ranges.jobs[FUSED], ranges.arrivals
	};
	const uint64_t pair_bytes[] = { INNER * VALUE_BYTES, INNER,
					BLOCKS_PER_PAIR, INNER * VALUE_BYTES };
	const struct linear_range *const arrays[] = { ranges.hidden,
						      ranges.hidden_codes,
						      ranges.hidden_exponents,
						      ranges.hidden_operands };
	for (uint32_t range = 0; range < COUNT_OF(read_only); range++) {
		(void)test_linear_card_expect_untouched(
			card, read_only[range],
			"the rows' arrays, the job tables and the arrival counters");
	}
    for (uint32_t array = 0; array < COUNT_OF(arrays); array++) {
        for (uint32_t chain = 0; chain < CHAINS; chain++) {
            const uint64_t written = pair_count * pair_bytes[array];
	    const struct linear_range beyond = {
		    arrays[array][chain].byte_offset + written,
		    arrays[array][chain].byte_count - written
	    };
	    if (beyond.byte_count != 0) {
                (void)test_linear_card_expect_untouched(card, beyond, "the arrays of pairs beyond the case's");
            }
        }
    }
}

/* The producer against today's launches at every slice count the class runs. */
static void producer_case(struct test_linear_card *card, const struct class_under_test *tested, enum routing routing,
                          uint32_t row_count)
{
    const uint32_t job_count = jobs_make(routing, row_count, 0);
    const uint32_t pair_count = row_count * PICKS;
    uint64_t outside = 0, hidden = 0;
#define LINEAR_SLICE_STAMP(slices) (slices),
    static const uint32_t stamped[] = { LINEAR_EVERY_SLICE_COUNT };
#undef LINEAR_SLICE_STAMP
    for (uint32_t instance = 0; instance < COUNT_OF(stamped); instance++) {
        const uint32_t slices = stamped[instance];
	if (tested->arithmetic_class == ARITHMETIC_CLASS_SM_120 &&
	    slices > LINEAR_CLASS_120_EXACT_MOST_GATE_UP_SLICES) {
		continue;
	}
	test_linear_context("card %" PRIu32 ", class %" PRIu32 ", the producer, %" PRIu32 " rows, %s, %" PRIu32
                            " slices", card->card_index, tested->arithmetic_class, row_count, routing_names[routing],
                            slices);
        if (!producer_reset(card, job_count) || !producer_run(card, tested, job_count, row_count, slices, slices)) {
            continue;
        }
        const uint64_t differing = producer_differing(card, pair_count, UINT32_MAX, &outside, &hidden);
        if (differing != 0) {
            test_linear_fail("%" PRIu64 " bytes of the hidden values, codes, exponents and operands differ", differing);
        }
        producer_expect_untouched(card, pair_count);
        uint32_t exponent_seen[EXPONENT_CODES] = {0}, distinct = 0;
        uint64_t zero_codes = 0;
        const unsigned char *const exponents = card->received + ranges.hidden_exponents[TODAY].byte_offset;
        const unsigned char *const codes = card->received + ranges.hidden_codes[TODAY].byte_offset;
        for (uint32_t index = 0; index < pair_count * BLOCKS_PER_PAIR; index++) {
            distinct += exponent_seen[exponents[index]]++ == 0 ? 1u : 0u;
        }
        for (uint32_t index = 0; index < pair_count * INNER; index++) {
            zero_codes += (codes[index] & E4M3_MAGNITUDE_MASK) == 0 ? 1u : 0u;
        }
        (void)printf("card %" PRIu32 " class %" PRIu32 " producer rows=%" PRIu32 " routing=%u slices=%" PRIu32
                     " jobs=%" PRIu32 " pairs=%" PRIu32 " (exponents: %" PRIu32 " distinct; zero codes %" PRIu64
                     " of %" PRIu32 "): %s\n", card->card_index, tested->arithmetic_class, row_count,
                     (unsigned int)routing, slices, job_count, pair_count, distinct, zero_codes, pair_count * INNER,
                     differing == 0 ? "every byte the same" : "DIFFERENT");
    }
}

/* Shown failing (1) and (2): a route weight half as large again in the producer's job table, and the producer at
 * another slice count. */
static void producer_controls(struct test_linear_card *card, const struct class_under_test *tested)
{
    const uint32_t job_count = jobs_make(ROUTING_HALF, CONTROL_ROWS, 0);
    const uint32_t changed_pair = job_table[1].pair_index[0];
    const uint32_t other_slices = tested->gate_up_slices == 1u ? SLICE_CHANGE : tested->gate_up_slices / SLICE_CHANGE;
    uint64_t outside = 0, hidden = 0;
    test_linear_context("card %" PRIu32 ", class %" PRIu32 ", shown failing: a route weight half as large again",
                        card->card_index, tested->arithmetic_class);
    if (producer_reset(card, job_count)) {
	    struct expert_job_record changed = job_table[1];
	    changed.route_weight[0] *= CHANGED_WEIGHT_FACTOR;
	    if (test_linear_card_put(card, ranges.jobs[FUSED], sizeof(changed),
				     &changed, sizeof(changed)) &&
		test_linear_card_upload(card, ranges.jobs[FUSED],
					sizeof(changed), sizeof(changed)) &&
		producer_run(card, tested, job_count, CONTROL_ROWS,
			     tested->gate_up_slices, tested->gate_up_slices)) {
		    const uint64_t differing =
			    producer_differing(card, CONTROL_ROWS * PICKS,
					       changed_pair, &outside, &hidden);
		    if (differing == 0 || outside != 0) {
			    test_linear_fail(
				    "%" PRIu64 " bytes differ, %" PRIu64
				    " outside the changed pair %" PRIu32
				    ": the change must show in that pair alone",
				    differing, outside, changed_pair);
		    }
		    (void)printf("card %" PRIu32 " class %" PRIu32
				 " shown failing: route weight of pair %" PRIu32
				 " half as large again: %" PRIu64
				 " bytes differ, %" PRIu64 " outside it\n",
				 card->card_index, tested->arithmetic_class,
				 changed_pair, differing, outside);
	    }
    }
    test_linear_context("card %" PRIu32 ", class %" PRIu32 ", the producer at %" PRIu32 " slices against today's at %"
                        PRIu32, card->card_index, tested->arithmetic_class, other_slices, tested->gate_up_slices);
    if (producer_reset(card, job_count) &&
        producer_run(card, tested, job_count, CONTROL_ROWS, tested->gate_up_slices, other_slices)) {
	    const bool must_differ = tested->arithmetic_class ==
				     ARITHMETIC_CLASS_SM_120_NATIVE;
	    const uint64_t differing = producer_differing(
		    card, CONTROL_ROWS * PICKS, UINT32_MAX, &outside, &hidden);
	    if ((differing != 0) != must_differ) {
		    test_linear_fail(
			    "%" PRIu64 " bytes differ: the slices %s",
			    differing,
			    must_differ ?
				    "cut the native chains, so bytes must differ" :
				    "keep the reference order here, so no byte may differ");
	    }
        (void)printf("card %" PRIu32 " class %" PRIu32 " slices %" PRIu32 " against %" PRIu32 ": %" PRIu64
                     " bytes differ, %" PRIu64 " of them hidden values (%s)\n", card->card_index,
                     tested->arithmetic_class, other_slices, tested->gate_up_slices, differing, hidden,
                     must_differ ? "must differ" : "must be the same");
    }
}

/* The producer over rows of which only the first live_rows stand in a job, as the router leaves a row that is not
 * live: the pairs of the others stand in none, and their hidden values, which neither chain writes, are the same stale
 * values (the edges) in both chains' arrays. Today's quantizer quantizes them too, and the producer must as well: every
 * byte of every pair the same. */
static void producer_orphan_case(struct test_linear_card *card, const struct class_under_test *tested,
                                 uint32_t row_count, uint32_t live_rows)
{
    const uint32_t job_count = jobs_make(ROUTING_HALF, live_rows, 0);
    const uint32_t pair_count = row_count * PICKS;
    const uint64_t pair_bytes = (uint64_t)INNER * VALUE_BYTES;
    uint64_t outside = 0, hidden = 0;
    test_linear_context("card %" PRIu32 ", class %" PRIu32 ", the producer, %" PRIu32 " rows, %" PRIu32 " in a job",
                        card->card_index, tested->arithmetic_class, row_count, live_rows);
    bool is_run = producer_reset(card, job_count);
    for (uint32_t pair = live_rows * PICKS; pair < pair_count && is_run; pair++) {
        test_linear_values(vector_values, INNER, STALE_STREAM + pair, TEST_LINEAR_VALUES_EDGES);
        for (uint32_t chain = 0; chain < CHAINS && is_run; chain++) {
            is_run = test_linear_card_put(card, ranges.hidden[chain], pair * pair_bytes, vector_values, pair_bytes) &&
                     test_linear_card_upload(card, ranges.hidden[chain], pair * pair_bytes, pair_bytes);
        }
    }
    if (!is_run || !producer_run(card, tested, job_count, row_count, tested->gate_up_slices, tested->gate_up_slices)) {
        return;
    }
    const uint64_t differing = producer_differing(card, pair_count, UINT32_MAX, &outside, &hidden);
    if (differing != 0) {
        test_linear_fail("%" PRIu64 " bytes of the hidden values, codes, exponents and operands differ", differing);
    }
    producer_expect_untouched(card, pair_count);
    (void)printf("card %" PRIu32 " class %" PRIu32 " producer rows=%" PRIu32 " of which %" PRIu32 " in a job (pairs %"
                 PRIu32 " to %" PRIu32 " in none): %s\n", card->card_index, tested->arithmetic_class, row_count,
                 live_rows, live_rows * PICKS, pair_count - 1u, differing == 0 ? "every byte the same" : "DIFFERENT");
}

/* ---- the joins ---- */

/* The joins' inputs of row_count rows into the image and the card: pair outputs (and a copy whose slots 0 and 3 of
 * row 0 are swapped), the shared expert's output (in today's array; the fused one rubbish), both chains' streams,
 * post and comb, and each chain's row sums rubbish of its own. */
static bool join_inputs(struct test_linear_card *card, uint32_t row_count)
{
    bool is_put = true;
    for (uint32_t pair = 0; pair < row_count * PICKS && is_put; pair++) {
        const uint32_t row = pair / PICKS, slot = pair % PICKS;
        test_linear_values(vector_values, HIDDEN, JOIN_STREAM + pair,
                           row == EXTREME_ROW && slot == EXTREME_SLOT ? TEST_LINEAR_VALUES_EXTREMES
                                                                      : TEST_LINEAR_VALUES_EDGES);
        for (uint32_t coordinate = 0; coordinate < HIDDEN; coordinate += ORDER_PERIOD) {
            vector_values[coordinate] = order_pattern[(slot + row) % ORDER_PATTERN_SLOTS];
        }
        const uint32_t permuted_slot = row != 0 ? slot : slot == 0 ? SWAPPED_SLOT : slot == SWAPPED_SLOT ? 0 : slot;
        is_put = test_linear_card_put(card, ranges.pair_output, (uint64_t)pair * HIDDEN * VALUE_BYTES, vector_values,
                                      HIDDEN * VALUE_BYTES) &&
                 test_linear_card_put(card, ranges.permuted_output,
                                      ((uint64_t)row * PICKS + permuted_slot) * HIDDEN * VALUE_BYTES, vector_values,
                                      HIDDEN * VALUE_BYTES);
    }
    for (uint32_t row = 0; row < row_count && is_put; row++) {
	    float post[MODEL_STREAMS], comb[MODEL_STREAMS * MODEL_STREAMS];
	    test_linear_values(vector_values, HIDDEN,
			       JOIN_STREAM + PAIR_STREAMS + row,
			       TEST_LINEAR_VALUES_PLAIN);
	    is_put =
		    test_linear_card_put(card, ranges.shared_output[TODAY],
					 (uint64_t)row * MODEL_HIDDEN_ROW_BYTES,
					 vector_values, MODEL_HIDDEN_ROW_BYTES);
	    for (uint32_t stream = 0; stream < MODEL_STREAMS && is_put;
		 stream++) {
		    const uint64_t stream_offset =
			    row * MODEL_STREAMS_ROW_BYTES +
			    stream * MODEL_HIDDEN_ROW_BYTES;
		    test_linear_values(
			    vector_values, HIDDEN,
			    JOIN_STREAM +
				    PAIR_STREAMS * (stream + STREAMS_STREAM) +
				    row,
			    TEST_LINEAR_VALUES_PLAIN);
		    is_put = test_linear_card_put(card, ranges.streams[TODAY],
						  stream_offset, vector_values,
						  MODEL_HIDDEN_ROW_BYTES) &&
			     test_linear_card_put(card, ranges.streams[FUSED],
						  stream_offset, vector_values,
						  MODEL_HIDDEN_ROW_BYTES);
	    }
	for (uint32_t value = 0; value < COUNT_OF(comb); value++) {
		post[value % MODEL_STREAMS] =
			1.0f + hash_fraction(JOIN_STREAM + 1u,
					     row * COUNT_OF(comb) + value);
		comb[value] = COMB_FRACTION *
			      hash_fraction(JOIN_STREAM + COMB_STREAM,
					    row * COUNT_OF(comb) + value);
        }
	is_put = is_put &&
		 test_linear_card_put(card, ranges.post,
				      (uint64_t)row * MODEL_SITE_ROW_BYTES,
				      post, sizeof(post)) &&
		 test_linear_card_put(card, ranges.comb,
				      (uint64_t)row * MODEL_COMB_ROW_BYTES,
				      comb, sizeof(comb));
    }
    memset(card->image + ranges.sums[TODAY].byte_offset, TODAY_RUBBISH, ranges.sums[TODAY].byte_count);
    memset(card->image + ranges.sums[FUSED].byte_offset, FUSED_RUBBISH, ranges.sums[FUSED].byte_count);
    memset(card->image + ranges.shared_output[FUSED].byte_offset, FUSED_RUBBISH,
           ranges.shared_output[FUSED].byte_count);
    const struct linear_range uploaded[] = { ranges.pair_output,
					     ranges.permuted_output,
					     ranges.shared_output[TODAY],
					     ranges.shared_output[FUSED],
					     ranges.streams[TODAY],
					     ranges.streams[FUSED],
					     ranges.post,
					     ranges.comb,
					     ranges.sums[TODAY],
					     ranges.sums[FUSED] };
    for (uint32_t range = 0; range < COUNT_OF(uploaded) && is_put; range++) {
        is_put = test_linear_card_upload(card, uploaded[range], 0, uploaded[range].byte_count);
    }
    return is_put;
}

/* Today's row sum and post of the site (reading the shared expert's output from today's array) into today_out. */
static bool today_join_prepare(const struct test_linear_card *card,
			       const struct class_under_test *tested,
			       uint32_t row_count,
			       struct kernel_launch *today_out)
{
	struct kernel_launch routed[LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT];
	struct kernel_launch site[HYPER_SITE_LAUNCH_COUNT];
	const uint32_t job_count = jobs_make(ROUTING_DISJOINT, row_count, 0);
	const struct linear_routed_experts_description experts =
		routed_of(card, tested, TODAY, ranges.jobs[TODAY], job_count,
			  row_count, tested->gate_up_slices);
	const struct hyper_site_description post = { ranges.streams[TODAY],
						     ranges.carried_pre,
						     ranges.mixes,
						     ranges.scale,
						     ranges.base,
						     ranges.norm_weight,
						     ranges.post,
						     ranges.comb,
						     ranges.site_hidden,
						     ranges.shared_output[TODAY],
						     ranges.sums[TODAY],
						     row_count,
						     0 };
	const bool is_prepared =
		test_linear_expect_status(
			linear_routed_experts_prepare(&experts,
						      card->arena_byte_count,
						      &tested->linear, routed),
			STATUS_OK, "today's row sum is prepared") &&
		test_linear_expect_status(
			hyper_site_prepare(&post, card->arena_byte_count,
					   &tested->hyper, site),
			STATUS_OK, "today's post is prepared");
	if (is_prepared) {
		today_out[0] = routed[LINEAR_ROUTED_EXPERTS_ROW_SUM];
		today_out[1] = site[HYPER_SITE_POST];
	}
    return is_prepared;
}

/* The join over pair_output, of the shared expert's output in today's array; with is_summed its row sums into the
 * fused chain's array, else none. */
static bool join_prepare(const struct test_linear_card *card,
			 const struct class_under_test *tested,
			 uint32_t row_count, struct linear_range pair_output,
			 bool is_summed, struct kernel_launch *fused_out)
{
	const struct linear_range no_sums = { 0, 0 };
	const struct expert_join_description join = {
		pair_output,
		ranges.shared_output[TODAY],
		ranges.streams[FUSED],
		ranges.post,
		ranges.comb,
		is_summed ? ranges.sums[FUSED] : no_sums,
		PICKS,
		row_count
	};
	return test_linear_expect_status(
		expert_join_prepare(&join, card->arena_byte_count,
				    &tested->fused, fused_out),
		STATUS_OK, "the join is prepared");
}

/* Today's shared down of slice_count slices (into today's output array) before its row sum and post, and the shared
 * join over pair_output at the same count (into the fused output array; its row sums as join_prepare's). */
static bool shared_join_prepare(const struct test_linear_card *card,
				const struct class_under_test *tested,
				uint32_t row_count,
				struct linear_range pair_output,
				uint32_t slice_count, bool is_summed,
				struct kernel_launch *today_out,
				struct kernel_launch *fused_out)
{
	const struct linear_range no_sums = { 0, 0 };
	const struct linear_matrix down = {
		ranges.shared_down,	   shared_down_shape,
		WEIGHT_FORMAT_E4M3_SCALED, slice_count,
		LINEAR_WIDE_GROUP_BLOCKS,  0
	};
	const struct linear_trunk_quantized_description today_down = {
		.matrices = { down },
		.codes = ranges.shared_codes,
		.exponents = ranges.shared_exponents,
		.operands = ranges.shared_operands,
		.outputs = { ranges.shared_output[TODAY] },
		.matrix_count = 1,
		.row_count = row_count
	};
	const struct expert_shared_join_description fused = {
		down,
		ranges.shared_codes,
		ranges.shared_exponents,
		ranges.shared_operands,
		{ pair_output, ranges.shared_output[FUSED],
		  ranges.streams[FUSED], ranges.post, ranges.comb,
		  is_summed ? ranges.sums[FUSED] : no_sums, PICKS, row_count }
	};
	return test_linear_expect_status(
		       linear_trunk_quantized_prepare(
			       &today_down, card->arena_byte_count,
			       &tested->linear, &today_out[0]),
		       STATUS_OK, "today's shared down is prepared") &&
	       today_join_prepare(card, tested, row_count, &today_out[1]) &&
	       test_linear_expect_status(
		       expert_shared_join_prepare(
			       &fused, card->arena_byte_count, &tested->linear,
			       &tested->fused, fused_out),
		       STATUS_OK, "the shared join is prepared");
}

/* The float32 sum of row 0's pair outputs at a coordinate, from +0 in slot order, its slots 0 and 3 swapped or not:
 * the host's float32 is the card's (-ffp-contract=off). */
static float row_sum_of(const struct test_linear_card *card, uint32_t coordinate, bool is_swapped)
{
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < PICKS; slot++) {
        const uint32_t read_slot = !is_swapped ? slot : slot == 0 ? SWAPPED_SLOT : slot == SWAPPED_SLOT ? 0 : slot;
        uint16_t bits = 0;
        float value = 0.0f;
        memcpy(&bits, card->image + ranges.pair_output.byte_offset + (read_slot * HIDDEN + coordinate) * VALUE_BYTES,
               sizeof(bits));
        const uint32_t widened = (uint32_t)bits << BFLOAT16_SHIFT;
        memcpy(&value, &widened, sizeof(value));
        sum += value;
    }
    return sum;
}

/* Shown failing (3), behind a run over the swapped pair outputs: the streams differ from today's somewhere, in row 0
 * alone, only where the order changes a sum. */
static void swap_expect(const struct test_linear_card *card, const struct class_under_test *tested, const char *who)
{
	const uint64_t stream_bytes =
		(uint64_t)CONTROL_ROWS * MODEL_STREAMS_ROW_BYTES;
	uint64_t differing = 0, outside = 0, order_changed = 0;
	for (uint32_t coordinate = 0; coordinate < HIDDEN; coordinate++) {
		const bool may_differ = row_sum_of(card, coordinate, false) !=
					row_sum_of(card, coordinate, true);
		order_changed += may_differ ? 1u : 0u;
		for (uint32_t stream = 0; stream < MODEL_STREAMS; stream++) {
			const uint64_t offset =
				(stream * HIDDEN + coordinate) * VALUE_BYTES;
			const uint64_t count = test_linear_card_differing(
				card, ranges.streams[TODAY], offset,
				ranges.streams[FUSED], offset, VALUE_BYTES,
				NULL);
			differing += count;
			outside += may_differ ? 0u : count;
		}
    }
    const uint64_t later_rows = test_linear_card_differing(
	    card, ranges.streams[TODAY], MODEL_STREAMS_ROW_BYTES,
	    ranges.streams[FUSED], MODEL_STREAMS_ROW_BYTES,
	    stream_bytes - MODEL_STREAMS_ROW_BYTES, NULL);
    differing += later_rows;
    outside += later_rows;
    if (differing == 0 || outside != 0) {
        test_linear_fail("%" PRIu64 " bytes differ, %" PRIu64 " where the order changes no sum: the swap must show "
                         "where it changes a sum alone", differing, outside);
    }
    (void)printf("card %" PRIu32 " class %" PRIu32 " shown failing: %s with slots 0 and 3 of row 0 swapped: %" PRIu64
                 " coordinates whose sum the order changes, %" PRIu64 " bytes of the streams differ, %" PRIu64
                 " elsewhere\n", card->card_index, tested->arithmetic_class, who, order_changed, differing, outside);
}

/* The fused chain's row sums after a join: with is_summed today's of row_count rows, every byte; else untouched, the
 * rubbish they were given. */
static bool sums_expect(const struct test_linear_card *card, uint32_t row_count, bool is_summed)
{
    return is_summed ? test_linear_card_expect_same(card, ranges.sums[TODAY], 0, ranges.sums[FUSED], 0,
                                                    (uint64_t)row_count * SUMS_ROW_BYTES, "the row sums")
                     : test_linear_card_expect_untouched(card, ranges.sums[FUSED], "the row sums of a join given none");
}

/* The join against today's row sum and post, given the row sums' array and given none (each from the inputs again);
 * with CONTROL_ROWS rows shown failing (3) too. */
static void join_case(struct test_linear_card *card, const struct class_under_test *tested, uint32_t row_count)
{
	struct kernel_launch today[TODAY_MOST_LAUNCHES], fused, swapped;
	const uint64_t stream_bytes =
		(uint64_t)row_count * MODEL_STREAMS_ROW_BYTES;
	for (uint32_t summed = 0; summed < SUM_KINDS; summed++) {
		test_linear_context("card %" PRIu32 ", class %" PRIu32
				    ", the join, %" PRIu32 " rows, %s",
				    card->card_index, tested->arithmetic_class,
				    row_count, sum_kind_names[summed]);
		if (!join_inputs(card, row_count) ||
		    !today_join_prepare(card, tested, row_count, today) ||
		    !join_prepare(card, tested, row_count, ranges.pair_output,
				  summed != 0, &fused) ||
		    !test_linear_card_run(card, today,
					  TODAY_MOST_LAUNCHES - 1u) ||
		    !test_linear_card_run(card, &fused, 1) ||
		    !test_linear_card_fetch(card)) {
			return;
		}
		const bool is_same = test_linear_card_expect_same(
					     card, ranges.streams[TODAY], 0,
					     ranges.streams[FUSED], 0,
					     stream_bytes,
					     "the streams after the join") &&
				     sums_expect(card, row_count, summed != 0);
		const struct linear_range read_only[] = {
			ranges.pair_output, ranges.shared_output[TODAY],
			ranges.post, ranges.comb, ranges.shared_output[FUSED]
		};
		for (uint32_t range = 0; range < COUNT_OF(read_only); range++) {
			(void)test_linear_card_expect_untouched(
				card, read_only[range],
				"what the join only reads");
		}
		(void)printf("card %" PRIu32 " class %" PRIu32
			     " join rows=%" PRIu32 " (%s): %s\n",
			     card->card_index, tested->arithmetic_class,
			     row_count, sum_kind_names[summed],
			     is_same ? "every byte the same" : "DIFFERENT");
	}
    test_linear_context("card %" PRIu32 ", class %" PRIu32 ", shown failing: the join, slots swapped",
                        card->card_index, tested->arithmetic_class);
    if (row_count == CONTROL_ROWS &&
        test_linear_card_upload(card, ranges.streams[FUSED], 0, ranges.streams[FUSED].byte_count) &&
        join_prepare(card, tested, row_count, ranges.permuted_output, false, &swapped) &&
        test_linear_card_run(card, &swapped, 1) && test_linear_card_fetch(card)) {
        swap_expect(card, tested, "the join");
    }
}

/* The shared join of slice_count slices against today's shared down of as many, row sum and post, given the row sums'
 * array and given none; with CONTROL_ROWS rows and the class's own count shown failing (3) too. */
static void shared_join_case(struct test_linear_card *card, const struct class_under_test *tested,
                             uint32_t row_count, uint32_t slice_count)
{
	struct kernel_launch today[TODAY_MOST_LAUNCHES], fused, swapped;
	const uint64_t stream_bytes =
		(uint64_t)row_count * MODEL_STREAMS_ROW_BYTES;
	const uint64_t output_bytes =
		(uint64_t)row_count * MODEL_HIDDEN_ROW_BYTES;
	for (uint32_t summed = 0; summed < SUM_KINDS; summed++) {
		test_linear_context("card %" PRIu32 ", class %" PRIu32
				    ", the shared join, %" PRIu32
				    " rows, %" PRIu32 " slices, %s",
				    card->card_index, tested->arithmetic_class,
				    row_count, slice_count,
				    sum_kind_names[summed]);
		if (!join_inputs(card, row_count) ||
		    !shared_join_prepare(card, tested, row_count,
					 ranges.pair_output, slice_count,
					 summed != 0, today, &fused) ||
		    !test_linear_card_run(card, today, TODAY_MOST_LAUNCHES) ||
		    !test_linear_card_run(card, &fused, 1) ||
		    !test_linear_card_fetch(card)) {
			return;
		}
		const bool are_streams_same = test_linear_card_expect_same(
			card, ranges.streams[TODAY], 0, ranges.streams[FUSED],
			0, stream_bytes, "the streams after the shared join");
		const bool is_same =
			test_linear_card_expect_same(
				card, ranges.shared_output[TODAY], 0,
				ranges.shared_output[FUSED], 0, output_bytes,
				"the shared expert's output") &&
			sums_expect(card, row_count, summed != 0) &&
			are_streams_same;
		const struct linear_range read_only[] = {
			ranges.pair_output,
			ranges.post,
			ranges.comb,
			ranges.shared_codes,
			ranges.shared_exponents,
			ranges.shared_operands
		};
		for (uint32_t range = 0; range < COUNT_OF(read_only); range++) {
			(void)test_linear_card_expect_untouched(
				card, read_only[range],
				"what the shared join only reads");
		}
		(void)printf("card %" PRIu32 " class %" PRIu32
			     " shared join rows=%" PRIu32 " slices=%" PRIu32
			     " (%s): %s\n",
			     card->card_index, tested->arithmetic_class,
			     row_count, slice_count, sum_kind_names[summed],
			     is_same ? "every byte the same" : "DIFFERENT");
	}
    test_linear_context("card %" PRIu32 ", class %" PRIu32 ", shown failing: the shared join, slots swapped",
                        card->card_index, tested->arithmetic_class);
    if (row_count == CONTROL_ROWS && slice_count == tested->shared_down_slices &&
        test_linear_card_upload(card, ranges.streams[FUSED], 0, ranges.streams[FUSED].byte_count) &&
        shared_join_prepare(card, tested, row_count, ranges.permuted_output, slice_count, false, today, &swapped) &&
        test_linear_card_run(card, &swapped, 1) && test_linear_card_fetch(card)) {
        swap_expect(card, tested, "the shared join");
        (void)test_linear_card_expect_same(card, ranges.shared_output[TODAY], 0, ranges.shared_output[FUSED], 0,
                                           output_bytes, "the shared expert's output, slots swapped");
    }
}

/* The shared join at every count it is stamped for that the class runs (class 120's emulation: up to
 * LINEAR_CLASS_120_EXACT_MOST_SLICES) but the class's own, run above, against today's down at that count, at 1
 * and 8 rows. */
static void shared_join_slices(struct test_linear_card *card, const struct class_under_test *tested)
{
#define LINEAR_SLICE_STAMP(slices) (slices),
	static const uint32_t stamped[] = { LINEAR_EVERY_SLICE_COUNT };
#undef LINEAR_SLICE_STAMP
	static const uint32_t swept_rows[] = { 1u, ROWS };
	for (uint32_t instance = 0; instance < COUNT_OF(stamped); instance++) {
		const uint32_t slices = stamped[instance];
		if (slices == tested->shared_down_slices ||
		    (tested->arithmetic_class == ARITHMETIC_CLASS_SM_120 &&
		     slices > LINEAR_CLASS_120_EXACT_MOST_SLICES)) {
			continue;
		}
		for (uint32_t index = 0; index < COUNT_OF(swept_rows);
		     index++) {
			shared_join_case(card, tested, swept_rows[index],
					 slices);
		}
	}
}

/* ---- the FFN tail: everything behind the router and the rows' quantization ---- */

/* Today's tail (gate with up, the quantization of the hidden vectors, down, the row sum; the shared expert's gate
 * with up, hidden step, quantization and down; the post) and the fused one (the producer, or with is_joined_only
 * today's gate with up and quantization; down; the shared expert's first three; the shared join) for rows whose
 * experts no row shares. The chains' arrays are their own where they write what the other reads later; where both
 * write the same values (the pair outputs, the shared expert's hidden arrays: the tail's own, so that the shared
 * join's cases keep theirs) they share them. */
static bool tail_prepare(struct test_linear_card *card, const struct class_under_test *tested, uint32_t row_count,
                         bool is_joined_only, struct timed_chain *chains_out)
{
	struct kernel_launch routed[CHAINS][LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT];
	struct kernel_launch shared[LINEAR_SHARED_EXPERT_LAUNCH_COUNT];
	struct kernel_launch producer, shared_join, site[TODAY_JOIN_LAUNCHES];
	const uint32_t job_count = jobs_make(ROUTING_DISJOINT, row_count, 0);
	const struct linear_matrix down = {
		ranges.shared_down,	   shared_down_shape,
		WEIGHT_FORMAT_E4M3_SCALED, tested->shared_down_slices,
		LINEAR_WIDE_GROUP_BLOCKS,  0
	};
	const struct linear_shared_expert_description shared_expert = {
		.gate = { ranges.shared_gate, shared_gate_up_shape,
			  WEIGHT_FORMAT_E4M3_SCALED,
			  tested->shared_gate_up_slices,
			  LINEAR_WIDE_GROUP_BLOCKS, 0 },
		.up = { ranges.shared_up, shared_gate_up_shape,
			WEIGHT_FORMAT_E4M3_SCALED,
			tested->shared_gate_up_slices, LINEAR_WIDE_GROUP_BLOCKS,
			0 },
		.down = down,
		.codes = ranges.codes,
		.exponents = ranges.exponents,
		.operands = ranges.operands,
		.gate_output = ranges.shared_gate_output,
		.up_output = ranges.shared_up_output,
		.hidden = ranges.tail_hidden,
		.hidden_codes = ranges.tail_codes,
		.hidden_exponents = ranges.tail_exponents,
		.hidden_operands = ranges.tail_operands,
		.output = ranges.shared_output[TODAY],
		.row_count = row_count,
		.activation_limit = ACTIVATION_LIMIT
	};
	const struct expert_shared_join_description fused_join = {
		down,
		ranges.tail_codes,
		ranges.tail_exponents,
		ranges.tail_operands,
		{ ranges.pair_output,
		  ranges.shared_output[FUSED],
		  ranges.streams[FUSED],
		  ranges.post,
		  ranges.comb,
		  { 0, 0 },
		  PICKS,
		  row_count }
	}; /* as a thin step runs it: no row sums */
	bool is_prepared =
		test_linear_card_put(card, ranges.jobs[TODAY], 0, job_table,
				     linear_jobs_byte_count(job_count)) &&
		test_linear_card_upload(card, ranges.jobs[TODAY], 0,
					ranges.jobs[TODAY].byte_count);
	for (uint32_t chain = 0; chain < CHAINS && is_prepared; chain++) {
		const struct linear_routed_experts_description experts =
			routed_of(card, tested, (enum chain)chain,
				  ranges.jobs[TODAY], job_count, row_count,
				  tested->gate_up_slices);
		is_prepared =
			test_linear_expect_status(
				linear_routed_experts_prepare(
					&experts, card->arena_byte_count,
					&tested->linear, routed[chain]),
				STATUS_OK,
				"the tail's routed experts are prepared") &&
			(chain == TODAY ||
			 test_linear_expect_status(
				 expert_producer_prepare(
					 &experts, ranges.arrivals,
					 card->arena_byte_count,
					 &tested->linear, &tested->fused,
					 &producer),
				 STATUS_OK, "the tail's producer is prepared"));
	}
	is_prepared =
		is_prepared &&
		test_linear_expect_status(
			linear_shared_expert_prepare(&shared_expert,
						     card->arena_byte_count,
						     &tested->linear, shared),
			STATUS_OK, "the tail's shared expert is prepared") &&
		today_join_prepare(card, tested, row_count, site) &&
		test_linear_expect_status(
			expert_shared_join_prepare(
				&fused_join, card->arena_byte_count,
				&tested->linear, &tested->fused, &shared_join),
			STATUS_OK, "the tail's shared join is prepared");
	if (is_prepared) {
		const struct kernel_launch today[TAIL_TODAY_LAUNCHES] = {
			routed[TODAY][LINEAR_ROUTED_EXPERTS_GATE_UP],
			routed[TODAY][LINEAR_ROUTED_EXPERTS_QUANTIZE],
			routed[TODAY][LINEAR_ROUTED_EXPERTS_DOWN],
			site[0],
			shared[LINEAR_SHARED_EXPERT_GATE_UP],
			shared[LINEAR_SHARED_EXPERT_HIDDEN],
			shared[LINEAR_SHARED_EXPERT_QUANTIZE],
			shared[LINEAR_SHARED_EXPERT_DOWN],
			site[1]
		};
		const struct kernel_launch fused[TAIL_JOINED_LAUNCHES] = {
			routed[FUSED][LINEAR_ROUTED_EXPERTS_GATE_UP],
			routed[FUSED][LINEAR_ROUTED_EXPERTS_QUANTIZE],
			routed[FUSED][LINEAR_ROUTED_EXPERTS_DOWN],
			shared[LINEAR_SHARED_EXPERT_GATE_UP],
			shared[LINEAR_SHARED_EXPERT_HIDDEN],
			shared[LINEAR_SHARED_EXPERT_QUANTIZE],
			shared_join
		};
		const uint32_t first =
			is_joined_only ?
				0u :
				TAIL_JOINED_LAUNCHES - TAIL_FUSED_LAUNCHES;
		memcpy(chains_out[TODAY].launches[0], today, sizeof(today));
		memcpy(chains_out[FUSED].launches[0], fused + first,
		       (TAIL_JOINED_LAUNCHES - first) * sizeof(fused[0]));
		chains_out[FUSED].launches[0][0] = is_joined_only ? fused[0] :
								    producer;
		chains_out[TODAY].launch_count = TAIL_TODAY_LAUNCHES;
		chains_out[FUSED].launch_count = TAIL_JOINED_LAUNCHES - first;
	}
    return is_prepared;
}

/* Today's tail and a fused one (with the producer, or the shared join alone), in turn, from the same streams: the
 * streams and the shared expert's output after them, every byte. */
static void tail_case(struct test_linear_card *card, const struct class_under_test *tested, uint32_t row_count,
                      bool is_joined_only)
{
    static struct timed_chain chains[CHAINS];
    test_linear_context("card %" PRIu32 ", class %" PRIu32 ", the FFN tail%s, %" PRIu32 " rows", card->card_index,
                        tested->arithmetic_class, is_joined_only ? " of the shared join alone" : "", row_count);
    if (!join_inputs(card, row_count) || !producer_reset(card, MOST_JOBS) ||
        !tail_prepare(card, tested, row_count, is_joined_only, chains) ||
        !test_linear_card_run(card, chains[TODAY].launches[0], chains[TODAY].launch_count) ||
        !test_linear_card_run(card, chains[FUSED].launches[0], chains[FUSED].launch_count) ||
        !test_linear_card_fetch(card)) {
        return;
    }
    const bool are_streams_same = test_linear_card_expect_same(
	    card, ranges.streams[TODAY], 0, ranges.streams[FUSED], 0,
	    (uint64_t)row_count * MODEL_STREAMS_ROW_BYTES,
	    "the streams after the tail");
    const bool is_same = test_linear_card_expect_same(
				 card, ranges.shared_output[TODAY], 0,
				 ranges.shared_output[FUSED], 0,
				 (uint64_t)row_count * MODEL_HIDDEN_ROW_BYTES,
				 "the shared expert's output after the tail") &&
			 are_streams_same;
    (void)test_linear_card_expect_untouched(card, ranges.arrivals, "the arrival counters after the tail");
    (void)printf("card %" PRIu32 " class %" PRIu32 " FFN tail rows=%" PRIu32 " (%u launches against %" PRIu32 "): %s\n",
                 card->card_index, tested->arithmetic_class, row_count, TAIL_TODAY_LAUNCHES,
                 chains[FUSED].launch_count, is_same ? "every byte the same" : "DIFFERENT");
}

/* ---- the time ---- */

/* The two chains, repetitions times each in a graph of its own, repetition r of variant r % variant_count, timed in
 * pairs (card_timing.h: TIME_SAMPLES samples of one run each after WARM_RUNS); TIME_ROUNDS lines TIME: each chain's
 * percentiles and those of the paired saving, in microseconds a repetition. */
static void part_time(const struct test_linear_card *card, const struct class_under_test *tested, const char *part,
                      uint32_t row_count, const char *weights, const struct timed_chain chains[CHAINS],
                      uint32_t variant_count, uint32_t repetitions)
{
    static double samples[CHAINS * TIME_SAMPLES], savings[TIME_SAMPLES];
    uint32_t graphs[CHAINS] = {0, 0};
    test_linear_context("card %" PRIu32 ", class %" PRIu32 ", the time of %s at %" PRIu32 " rows", card->card_index,
                        tested->arithmetic_class, part, row_count);
    for (uint32_t chain = 0; chain < CHAINS; chain++) {
        const uint32_t count = chains[chain].launch_count;
        if (!test_linear_expect(count != 0 && repetitions * count <= MOST_REPETITION_LAUNCHES,
                                "the repetitions of a chain fit the table")) {
            return;
        }
        for (uint32_t repetition = 0; repetition < repetitions; repetition++) {
		memcpy(&repetition_launches[chain][(size_t)repetition * count],
		       chains[chain].launches[repetition % variant_count],
		       count * sizeof(struct kernel_launch));
	}
	if (!test_linear_expect_status(
		    backend->graph_capture(card->card,
					   repetition_launches[chain],
					   repetitions * count, &graphs[chain]),
		    STATUS_OK, "a timed graph is made")) {
		return;
	}
    }
    const struct test_card_timing_form forms[CHAINS] = {{&graphs[TODAY], 1}, {&graphs[FUSED], 1}};
    for (uint32_t round = 0;
         round < TIME_ROUNDS && test_card_timing_pair(card->card, forms, WARM_RUNS, 1, TIME_SAMPLES, samples, savings);
         round++) {
        (void)printf("TIME card=%" PRIu32 " class=%" PRIu32 " part=%s rows=%" PRIu32 " weights=%s slices=%" PRIu32
                     " launches=%" PRIu32 ":%" PRIu32 " repetitions=%" PRIu32 " samples=%u round=%" PRIu32,
                     card->card_index, tested->arithmetic_class, part, row_count, weights, tested->gate_up_slices,
                     chains[TODAY].launch_count, chains[FUSED].launch_count, repetitions, TIME_SAMPLES, round);
        test_card_timing_print(samples, TIME_SAMPLES, repetitions, "today");
        test_card_timing_print(samples + TIME_SAMPLES, TIME_SAMPLES, repetitions, "fused");
        test_card_timing_print(savings, TIME_SAMPLES, repetitions, "saving");
        (void)printf("\n");
    }
}

/* The producer's chains over one job table range for rows whose experts no row shares, from expert_base on, as
 * variant `variant`. */
static bool producer_launches(struct test_linear_card *card,
			      const struct class_under_test *tested,
			      uint32_t row_count, struct linear_range jobs,
			      uint32_t expert_base, struct timed_chain *chains,
			      uint32_t variant)
{
	struct kernel_launch today[LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT];
	const uint32_t job_count =
		jobs_make(ROUTING_DISJOINT, row_count, expert_base);
	const struct linear_routed_experts_description today_description =
		routed_of(card, tested, TODAY, jobs, job_count, row_count,
			  tested->gate_up_slices);
	const struct linear_routed_experts_description fused_description =
		routed_of(card, tested, FUSED, jobs, job_count, row_count,
			  tested->gate_up_slices);
	const bool is_prepared =
		test_linear_card_put(card, jobs, 0, job_table,
				     linear_jobs_byte_count(job_count)) &&
		test_linear_card_upload(card, jobs, 0, jobs.byte_count) &&
		test_linear_expect_status(
			linear_routed_experts_prepare(&today_description,
						      card->arena_byte_count,
						      &tested->linear, today),
			STATUS_OK, "today's routed experts are prepared") &&
		test_linear_expect_status(
			expert_producer_prepare(
				&fused_description, ranges.arrivals,
				card->arena_byte_count, &tested->linear,
				&tested->fused,
				&chains[FUSED].launches[variant][0]),
			STATUS_OK, "the producer is prepared");
	if (is_prepared) {
		chains[TODAY].launches[variant][0] =
			today[LINEAR_ROUTED_EXPERTS_GATE_UP];
		chains[TODAY].launches[variant][1] =
			today[LINEAR_ROUTED_EXPERTS_QUANTIZE];
		chains[TODAY].launch_count =
			LINEAR_ROUTED_EXPERTS_QUANTIZE + 1u;
		chains[FUSED].launch_count = 1u;
	}
    return is_prepared;
}

/* --time: the producer at 1 row (the same six experts, and eight sets of six in turn) and at 8 rows; the join (and
 * its geometry), the shared join and the whole tail at 1 and 8 rows. */
static void class_time(struct test_linear_card *card, const struct class_under_test *tested)
{
    static struct timed_chain chains[CHAINS];
    const uint32_t part_rows[] = {1u, ROWS};
    if (producer_launches(card, tested, 1u, ranges.jobs[TODAY], 0, chains, 0)) {
        part_time(card, tested, "producer", 1u, "same", chains, 1u, PRODUCER_REPETITIONS_ONE_ROW);
    }
    bool is_prepared = true;
    for (uint32_t set = 0; set < ROTATION_SETS && is_prepared; set++) {
        is_prepared = producer_launches(card, tested, 1u, ranges.rotation_jobs[set], set * PICKS, chains, set);
    }
    if (is_prepared) {
        part_time(card, tested, "producer", 1u, "rotating", chains, ROTATION_SETS, PRODUCER_REPETITIONS_ONE_ROW);
    }
    if (producer_launches(card, tested, ROWS, ranges.jobs[TODAY], 0, chains, 0)) {
        part_time(card, tested, "producer", ROWS, "same", chains, 1u, PRODUCER_REPETITIONS_EIGHT_ROWS);
    }
    for (uint32_t index = 0; index < COUNT_OF(part_rows); index++) {
        const uint32_t row_count = part_rows[index];
        if (join_inputs(card, row_count) && today_join_prepare(card, tested, row_count, chains[TODAY].launches[0]) &&
            join_prepare(card, tested, row_count, ranges.pair_output, false, &chains[FUSED].launches[0][0])) {
            chains[TODAY].launch_count = TODAY_JOIN_LAUNCHES;
            chains[FUSED].launch_count = 1u;
            part_time(card, tested, "join", row_count, "none", chains, 1u, JOIN_REPETITIONS);
            /* the join's geometry: any block that divides a row's threads gives the same bits (a thread's eight
             * values are its own); the prepared 128 threads against smaller blocks on more multiprocessors */
            chains[TODAY].launches[0][0] = chains[FUSED].launches[0][0];
            chains[TODAY].launch_count = 1u;
            for (uint32_t geometry = 0; geometry < COUNT_OF(join_geometry_threads); geometry++) {
                chains[FUSED].launches[0][0].block[0] = join_geometry_threads[geometry];
		chains[FUSED].launches[0][0].grid[0] =
			HYPER_ROW_THREADS / join_geometry_threads[geometry];
		part_time(card, tested, join_geometry_names[geometry], row_count, "none", chains, 1u,
                          JOIN_REPETITIONS);
            }
        }
        if (join_inputs(card, row_count) &&
            shared_join_prepare(card, tested, row_count, ranges.pair_output, tested->shared_down_slices, false,
                                chains[TODAY].launches[0],
                                &chains[FUSED].launches[0][0])) {
            chains[TODAY].launch_count = TODAY_JOIN_LAUNCHES + 1u;
            chains[FUSED].launch_count = 1u;
            part_time(card, tested, "shared_join", row_count, "same", chains, 1u, SHARED_JOIN_REPETITIONS);
        }
        /* what a capture's row sums cost: each join given none (as a thin step runs it) against itself given them */
	struct kernel_launch unused[TODAY_MOST_LAUNCHES];
	if (join_inputs(card, row_count) &&
            join_prepare(card, tested, row_count, ranges.pair_output, false, &chains[TODAY].launches[0][0]) &&
            join_prepare(card, tested, row_count, ranges.pair_output, true, &chains[FUSED].launches[0][0])) {
            chains[TODAY].launch_count = 1u;
            chains[FUSED].launch_count = 1u;
            part_time(card, tested, "join_row_sums_stored", row_count, "none", chains, 1u, JOIN_REPETITIONS);
        }
        if (join_inputs(card, row_count) &&
            shared_join_prepare(card, tested, row_count, ranges.pair_output, tested->shared_down_slices, false, unused,
                                &chains[TODAY].launches[0][0]) &&
            shared_join_prepare(card, tested, row_count, ranges.pair_output, tested->shared_down_slices, true, unused,
                                &chains[FUSED].launches[0][0])) {
            chains[TODAY].launch_count = 1u;
            chains[FUSED].launch_count = 1u;
            part_time(card, tested, "shared_join_row_sums_stored", row_count, "same", chains, 1u,
                      SHARED_JOIN_REPETITIONS);
        }
        for (uint32_t joined_only = 0; joined_only < TAIL_VARIANTS; joined_only++) {
            if (join_inputs(card, row_count) && tail_prepare(card, tested, row_count, joined_only != 0, chains)) {
                part_time(card, tested, joined_only != 0 ? "ffn_tail_shared_join_alone" : "ffn_tail", row_count,
                          "same", chains, 1u, row_count == 1u ? TAIL_REPETITIONS_ONE_ROW : TAIL_REPETITIONS_EIGHT_ROWS);
            }
        }
    }
}

/* A class of the open card: its kernels and its default slice counts. */
static bool class_prepare(const struct test_linear_card *card, uint32_t arithmetic_class,
                          struct class_under_test *tested)
{
    const struct {
	    enum linear_family_code family;
	    const struct matrix_shape_record *shape;
	    uint32_t *slices;
    } defaults[] = { { LINEAR_FAMILY_EXPERT_GATE_UP, &gate_up_shape,
		       &tested->gate_up_slices },
		     { LINEAR_FAMILY_EXPERT_DOWN, &down_shape,
		       &tested->down_slices },
		     { LINEAR_FAMILY_TRUNK_QUANTIZED, &shared_down_shape,
		       &tested->shared_down_slices },
		     { LINEAR_FAMILY_TRUNK_QUANTIZED, &shared_gate_up_shape,
		       &tested->shared_gate_up_slices } };
    tested->arithmetic_class = arithmetic_class;
    bool is_prepared =
	    test_linear_expect_status(
		    linear_kernels_prepare(backend, card->card,
					   arithmetic_class, &tested->linear),
		    STATUS_OK, "the kernels of the linear are prepared") &&
	    test_linear_expect_status(
		    hyper_kernels_prepare(backend, card->card, &tested->hyper),
		    STATUS_OK, "the kernels of the sites are prepared") &&
	    test_linear_expect_status(expert_fused_kernels_prepare(
					      backend, card->card,
					      arithmetic_class, &tested->fused),
				      STATUS_OK,
				      "the fused kernels are prepared");
    for (uint32_t index = 0; index < COUNT_OF(defaults) && is_prepared; index++) {
	    is_prepared = test_linear_expect_status(
		    linear_default_slice_count(
			    LINEAR_PROFILE_SPEED, LINEAR_ORDER_FIXED,
			    defaults[index].family, defaults[index].shape,
			    arithmetic_class, defaults[index].slices),
		    STATUS_OK, "a default slice count is given");
    }
    return is_prepared;
}

/* Every check of one class on the card at card_index, opened for it alone; with is_timed the time too. */
static void class_test(uint32_t card_index, uint32_t arithmetic_class, bool is_timed)
{
    static struct test_linear_card card;
    static const uint32_t case_rows[] = {1u, CONTROL_ROWS, ROWS};
    static const struct {
        uint32_t rows, live_rows;
    } orphan_cases[] = {{CONTROL_ROWS, CONTROL_ROWS - 1u}, {ROWS, ORPHAN_LIVE_ROWS}, {ROWS, 1u}};
    struct class_under_test tested;
    if (!test_linear_card_open(&card, card_index, WORK_CAPACITY, WEIGHTS_CAPACITY(stored_experts))) {
        return;
    }
    test_linear_context("card %" PRIu32 ", class %" PRIu32, card_index, arithmetic_class);
    if (arena_lay_out(&card) && test_linear_card_load(&card) && class_prepare(&card, arithmetic_class, &tested) &&
        vectors_quantize(&card, &tested, HIDDEN, ROWS_STREAM, ranges.row_values, ranges.codes, ranges.exponents,
                         ranges.operands) &&
        vectors_quantize(&card, &tested, INNER, SHARED_HIDDEN_STREAM, ranges.shared_values, ranges.shared_codes,
                         ranges.shared_exponents, ranges.shared_operands)) {
        for (uint32_t index = 0; index < COUNT_OF(case_rows); index++) {
            for (uint32_t routing = 0; routing < (case_rows[index] == 1u ? 1u : ROUTING_COUNT); routing++) {
                producer_case(&card, &tested, (enum routing)routing, case_rows[index]);
            }
        }
        producer_controls(&card, &tested);
        for (uint32_t index = 0; index < COUNT_OF(orphan_cases); index++) {
            producer_orphan_case(&card, &tested, orphan_cases[index].rows, orphan_cases[index].live_rows);
        }
        for (uint32_t index = 0; index < COUNT_OF(case_rows); index++) {
            join_case(&card, &tested, case_rows[index]);
            shared_join_case(&card, &tested, case_rows[index], tested.shared_down_slices);
            tail_case(&card, &tested, case_rows[index], false);
            tail_case(&card, &tested, case_rows[index], true);
        }
        shared_join_slices(&card, &tested);
        if (is_timed) {
            class_time(&card, &tested);
        }
        test_linear_context("card %" PRIu32 ", class %" PRIu32, card_index, arithmetic_class);
        (void)test_linear_card_expect_weights_untouched(&card);
    }
    test_linear_card_close(&card);
}

int main(int argument_count, char **arguments)
{
    const bool is_timed = argument_count > 1 && strcmp(arguments[1], "--time") == 0;
    uint32_t card_count = 0;
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    if (!test_linear_expect_status(backend->card_count(&card_count), STATUS_OK,
				   "the cards are counted") ||
	!test_linear_expect(card_count > 0 &&
				    card_count <= BACKEND_MAXIMUM_CARDS,
			    "a card is there")) {
	    return EXIT_FAILURE;
    }
    for (uint32_t card_index = 0; card_index < card_count; card_index++) {
	    struct card_description_record description;
	    memset(&description, 0, sizeof(description));
	    if (!test_linear_expect_status(
			backend->card_describe(card_index, &description),
			STATUS_OK, "the card is described")) {
		    continue;
	    }
	    const bool is_class_120 = description.arithmetic_class ==
				      ARITHMETIC_CLASS_SM_120;
	    const uint32_t classes[CLASS_COUNT_MOST] = {
		    is_class_120 ? ARITHMETIC_CLASS_SM_120_NATIVE :
				   description.arithmetic_class,
		    ARITHMETIC_CLASS_SM_120
	    };
	    stored_experts = is_class_120 ? STORED_EXPERTS_CLASS_120 : EXPERTS;
	    (void)printf("card %" PRIu32 ": %" PRIu64 " MiB free, %" PRIu64
			 " reserved; %" PRIu32
			 " experts stored, an arena of %" PRIu64 " MiB\n",
			 card_index,
			 description.memory_free_bytes >> MEBIBYTE_SHIFT,
			 description.memory_reserved_bytes >> MEBIBYTE_SHIFT,
			 stored_experts,
			 (uint64_t)((WORK_CAPACITY +
				     WEIGHTS_CAPACITY(stored_experts)) >>
				    MEBIBYTE_SHIFT));
	    for (uint32_t class_index = 0;
		 class_index < (is_class_120 ? CLASS_COUNT_MOST : 1u);
		 class_index++) {
		    class_test(card_index, classes[class_index], is_timed);
	    }
    }
    (void)printf("expert_fused_card_test: %d failures\n", test_linear_failures());
    return test_linear_failures() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
