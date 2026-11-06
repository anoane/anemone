/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * linear_test: source/linear.c against linear.h and linear_records.h. No card: a prepare function is arithmetic on a
 * description, the arena is a number, and the kernel handles are made-up numbers.
 *
 *   - The bytes of the arrays against figures computed by hand, and their stop at 64 bits.
 *   - The matrices of the real model (its inference configuration: 5120 wide, experts 2304 inside, 6 of 384 experts
 *     to a row, a limit of 10): every launch has the kernel, the grid, the block, the shared memory and the record
 *     that linear_records.h states, held against constants computed by hand below.
 *   - Every launch of every kind, for 1 to 8 rows, is walked: the ranges its record names are sized as
 *     linear_records.h says, from the record and the grid alone. A case gives each range of its description exactly
 *     the bytes so named and ends its arena with the last range: it is accepted; one byte less in any range, or at
 *     the end of the arena, is refused; what a launch writes lies on nothing else it touches.
 *   - Every refusal of linear.h, each alone in a description that is otherwise accepted: the status, one error line
 *     that holds the words of its rule, and the launches left as they were. (The preparation of the kernels, the
 *     slice counts, the trunk launch of two matrices and the wide trunk launches: linear_launch_test.c.)
 * No byte of the arena exists, so a layer's 7 GB of experts are laid out and every offset is beyond 32 bits. A
 * failure is one line on the standard output: the case, and what was expected of it.
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/backend_seam.h"
#include "log_expectation.h"
#include "anemone/linear.h"
#include "anemone/linear_records.h"
#include "anemone/model_layout.h"
#include "anemone/status.h"
#include "anemone/weight_format_seam.h"
#include "host_memory_backend.h"
#include "weight_format_reference.h"

#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
#define RECORD(record) &(record), sizeof(record)
#define E2M1 \
	WEIGHT_FORMAT_E2M1_SCALED /* short names for the tables of this test */
#define E4M3 WEIGHT_FORMAT_E4M3_SCALED
#define INVALID STATUS_INVALID_ARGUMENT
#define E2M1_BITS_PER_WEIGHT 4u
#define E4M3_BITS_PER_WEIGHT 8u

/* The model, from its inference configuration. */
#define HIDDEN_SIZE 5120u
#define INNER_SIZE 2304u
#define PAIRS_PER_ROW 6u
#define EXPERT_COUNT 384u
#define ACTIVATION_LIMIT 10.0f
#define ROWS LINEAR_MAXIMUM_ROWS /* 8 */
#define PAIRS_OF_8_ROWS 48u                     /* 8 * 6 */
#define MOST_PAIRS_PER_ROW 0x1FFFFFFFu          /* of 8 rows: one more and their pairs pass 32 bits */
#define A_HUGE_INNER_SIZE 2147483712u           /* 2^31 + 64: of 2^32 - 1 pairs, over 2^64 bytes of hidden values */
#define BYTES_OF_AN_EXPERT_MATRIX 6266880ull    /* 4 bits: 2304 * 5120 / 2 + 2304 * 5120 / 32 */
#define BYTES_OF_EVERY_EXPERT 7219445760ull     /* 384 experts of three such matrices */
#define FEW_SLICES 4u
#define MANY_SLICES 16u
#define INSTANCE_OF_4_SLICES \
	2u /* the place of a count in LINEAR_SLICE_COUNTS */
#define INSTANCE_OF_16_SLICES 4u

/* The launches of the experts, by hand from the table of linear_records.h. A product has a block for each tile of
 * 16 outputs, of 32 threads to a slice (gate with up: 64), with 64 bytes of shared memory to a thread. */
#define TILES_OF_2304_OUTPUTS 144u
#define TILES_OF_5120_OUTPUTS 320u
#define TILES_OF_GATE_WITH_UP 288u              /* 144 of each */
#define THREADS_OF_4_SLICES 128u
#define SHARED_BYTES_OF_4_SLICES 8192u
#define THREADS_OF_16_SLICES 512u
#define SHARED_BYTES_OF_16_SLICES 32768u
#define GATE_UP_THREADS_OF_4_SLICES 256u
#define GATE_UP_SHARED_BYTES_OF_4_SLICES 16384u
#define GATE_UP_THREADS_OF_16_SLICES 1024u
#define GATE_UP_SHARED_BYTES_OF_16_SLICES 65536u
/* Where a thread takes one value: blocks of 256 threads that cover the width, and no shared memory. */
#define ELEMENT_THREADS 256u
#define ELEMENT_BLOCKS_OF_5120 20u              /* 20 * 256 */
#define ELEMENT_BLOCKS_OF_2304 9u               /* 9 * 256 */
#define ELEMENT_BLOCKS_OF_320 2u                /* 256 + 64 */
#define A_BLOCK_AND_A_LOAD 320u

/* The bytes of the arrays, by hand. */
#define VALUES_OF_8_ROWS 81920ull               /* 2 * 8 * 5120 */
#define CODES_OF_8_ROWS 40960ull                /* 8 * 5120 */
#define EXPONENTS_OF_8_ROWS 1280ull             /* 8 * (5120 / 32) */
#define SUMS_OF_8_ROWS 163840ull                /* 4 * 8 * 5120 */
#define HIDDEN_VALUES_OF_48_PAIRS 221184ull     /* 2 * 48 * 2304 */
#define HIDDEN_EXPONENTS_OF_48_PAIRS 3456ull    /* 48 * (2304 / 32) */
#define JOBS_OF_384 36864ull                    /* 96 * 384 */
#define TWO_TO_THE_32 4294967296ull
#define TWO_TO_THE_63 9223372036854775808ull

/* The arena of the test, and what else it needs. */
#define FIRST_OFFSET 8589934592ull              /* 2^33: every offset of the test is beyond 32 bits */
#define WIDE_RANGE_BYTES 1099511627776ull       /* 2^40: more than any array of the test has */
#define MOST_RANGES 13u                         /* of one description: the shared expert's */
#define MOST_LAUNCHES 4u                        /* of one chain: the routed experts' and the shared expert's */
#define MOST_TOUCHED 7u                         /* the ranges one record names */
#define VALUES_PER_EXPONENT LINEAR_VALUES_PER_BLOCK
#define UNTOUCHED_BYTE 0xA5                     /* what an output holds before a call that must leave it alone */
#define HALF_AN_ALIGNMENT 8u
#define CAPTURE_CAPACITY 2048u                  /* more than the test logs between two readings */
#define KERNEL_COUNT 34u                        /* four kernels, and six stamped for five slice counts each */
#define FIRST_MADE_UP_HANDLE 11u
#define VALUES_PER_REFUSAL 2u

/* Words of the one error line of each rule of linear.h. */
#define RULE_MISSING "a missing argument"
#define RULE_KERNEL "is not prepared"
#define RULE_RESERVED "a reserved field of the description"
#define RULE_SHAPE_RESERVED "a reserved field of the shape"
#define RULE_WIDE_FORM "the forms are linear.h's"
#define RULE_FORMAT "its kernel reads format"
#define RULE_SHAPE "stores no matrix"
#define RULE_LOADS "a product reads whole loads"
#define RULE_SLICES "slices: a product kernel is stamped for"
#define RULE_MATRICES "a launch reads 1 or"
#define RULE_PAIR "the two matrices of one launch differ"
#define RULE_SEGMENTS "segments: at least one"
#define RULE_CHAIN "down reads"
#define RULE_ONE_SHAPE "of one shape"
#define RULE_WIDTH "values wide"
#define RULE_ROWS "rows: a trunk launch takes"
#define RULE_CHUNK_ROWS "take 1 to"
#define RULE_PAIRS "pairs per row: at least one"
#define RULE_JOBS "a job table of no entry"
#define RULE_LIMIT "is not above zero"
#define RULE_ALIGNED "an offset is a multiple of"
#define RULE_ARENA "leaves the arena"
#define RULE_TOUCHED "the launches touch"
#define RULE_STORED "its format stores"
#define RULE_SHARED "share a byte"
#define RULE_HIDDEN_VALUES "the range of the hidden values holds"
#define RULE_EXPERTS "the range of the experts' matrices holds"

/* A member of a description of a kind: the kind, where the member lies and how many bytes it has. */
#define RANGE(member) offsetof(union any_description, member)
#define MEMBER(kind, member) RANGE(member), sizeof(((const union any_description *)NULL)->member), kind
#define IN_QUANTIZE(member) MEMBER(QUANTIZE, quantize.member)
#define IN_TRUNK_QUANTIZED(member) MEMBER(TRUNK_QUANTIZED, trunk_quantized.member)
#define IN_TRUNK_BFLOAT16(member) MEMBER(TRUNK_BFLOAT16, trunk_bfloat16.member)
#define IN_ROUTED(member) MEMBER(ROUTED_EXPERTS, routed_experts.member)
#define IN_SHARED(member) MEMBER(SHARED_EXPERT, shared_expert.member)

_Static_assert(
	sizeof(struct linear_kernels_record) ==
			KERNEL_COUNT * sizeof(uint32_t) &&
		__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
	"the test reads the handles as 34 numbers, and sets a member from the low bytes of a number");

enum described { QUANTIZE, TRUNK_QUANTIZED, TRUNK_BFLOAT16, ROUTED_EXPERTS, SHARED_EXPERT, DESCRIBED_COUNT };
enum launch_dimension { DIMENSION_X, DIMENSION_Y, DIMENSION_Z };

union any_description {
	struct linear_quantize_description quantize;
	struct linear_trunk_quantized_description trunk_quantized;
	struct linear_trunk_bfloat16_description trunk_bfloat16;
	struct linear_routed_experts_description routed_experts;
	struct linear_shared_expert_description shared_expert;
};

/* A description with its ranges laid out one behind another, and the arena that ends with the last of them. */
struct test_case {
    enum described kind;
    union any_description description;
    uint64_t arena_byte_count;
    uint64_t touched_byte_count[MOST_RANGES];   /* in each range, as the records of the launches name it */
};

/* A range a record names. */
struct touched_range {
    uint64_t byte_offset;
    uint64_t byte_count;
    bool is_written;
};

/* Where the ranges of each description lie in it, in the order of the test's arena: its matrices first, each a
 * struct linear_matrix that begins with its range; then, of the routed experts, the matrices of a layer. */
static const size_t range_offsets[DESCRIBED_COUNT][MOST_RANGES] = {
    [QUANTIZE] = {RANGE(quantize.values), RANGE(quantize.codes), RANGE(quantize.exponents),
                  RANGE(quantize.operands)},
    [TRUNK_QUANTIZED] = {RANGE(trunk_quantized.matrices[0]), RANGE(trunk_quantized.codes),
                         RANGE(trunk_quantized.exponents), RANGE(trunk_quantized.operands),
                         RANGE(trunk_quantized.outputs[0])},
    [TRUNK_BFLOAT16] = {RANGE(trunk_bfloat16.matrix), RANGE(trunk_bfloat16.values), RANGE(trunk_bfloat16.output)},
    [ROUTED_EXPERTS] = {RANGE(routed_experts.expert_weights), RANGE(routed_experts.jobs),
                        RANGE(routed_experts.codes), RANGE(routed_experts.exponents),
                        RANGE(routed_experts.operands), RANGE(routed_experts.hidden),
                        RANGE(routed_experts.hidden_codes), RANGE(routed_experts.hidden_exponents),
                        RANGE(routed_experts.hidden_operands), RANGE(routed_experts.pair_output),
                        RANGE(routed_experts.row_output)},
    [SHARED_EXPERT] = {RANGE(shared_expert.gate), RANGE(shared_expert.up), RANGE(shared_expert.down),
                       RANGE(shared_expert.codes), RANGE(shared_expert.exponents), RANGE(shared_expert.operands),
                       RANGE(shared_expert.gate_output), RANGE(shared_expert.up_output), RANGE(shared_expert.hidden),
                       RANGE(shared_expert.hidden_codes), RANGE(shared_expert.hidden_exponents),
                       RANGE(shared_expert.hidden_operands), RANGE(shared_expert.output)},
};
static const uint32_t range_counts[DESCRIBED_COUNT] = {4, 5, 3, 11, 13};    /* these three: in the order of the kinds */
static const uint32_t matrix_counts[DESCRIBED_COUNT] = {0, 1, 1, 0, 3};
static const uint32_t launch_counts[DESCRIBED_COUNT] = {
	1, 1, 1, LINEAR_ROUTED_EXPERTS_LAUNCH_COUNT,
	LINEAR_SHARED_EXPERT_LAUNCH_COUNT
};
/* Made-up handles, in the order of the members: FIRST_MADE_UP_HANDLE and on (made_up_kernels_make). */
static struct linear_kernels_record made_up_kernels;

/* The 8-bit matrices of the model with the slices and segments this test gives each, and its launch by hand: tiles
 * are outputs / 16, threads 32 * slices, shared bytes 16 * threads. The first is the matrix of the cases with
 * quantized rows; the second, block-diagonal over 8 groups of 1024 outputs, that of the cases with bfloat16 rows.
 * The last is no matrix of the model: it has as many loads as its slices. */
struct trunk_shape {
    const char *what;
    uint32_t output_size;
    uint32_t input_size;
    uint32_t slice_count;
    uint32_t instance;                          /* of the slice count */
    uint32_t segment_count;                     /* of the launch with bfloat16 rows */
    uint32_t tile_count;
    uint32_t block_threads;
    uint32_t shared_byte_count;
};
static const struct trunk_shape trunk_shapes[] = {
    {"[512, 5120]", 512, 5120, 8, 3, 1, 32, 256, 16384},
    {"[8192, 4096]", 8192, 4096, 8, 3, 8, 512, 256, 16384},
    {"[1280, 5120]", 1280, 5120, 4, 2, 1, 80, 128, 8192},
    {"[32768, 1280]", 32768, 1280, 16, 4, 1, 2048, 512, 32768},
    {"[5120, 8192]", 5120, 8192, 16, 4, 1, 320, 512, 32768},
    {"[2304, 5120]", 2304, 5120, 1, 0, 1, 144, 32, 2048},
    {"[5120, 2304]", 5120, 2304, 2, 1, 1, 320, 64, 4096},
    {"[16, 128]", 16, 128, 2, 1, 1, 1, 64, 4096},
};

static int failures = 0;
static const char *context_text = "";           /* the case a failure line names */
static char capture_path[] = "/tmp/anemone_linear_test_XXXXXX";
static int capture_descriptor = -1;             /* the temporary file that stands in for the error stream */
static int error_stream_descriptor = -1;        /* a duplicate of descriptor 2, kept to put it back */

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)printf("FAILED: %s: %s\n", context_text, what);
        failures++;
    }
}

/* From here on, what is logged goes to the temporary file, from its start. */
static void capture_begin(void)
{
    expect(capture_descriptor >= 0 && fflush(stderr) == 0 && ftruncate(capture_descriptor, 0) == 0 &&
               lseek(capture_descriptor, 0, SEEK_SET) == 0 && dup2(capture_descriptor, STDERR_FILENO) == STDERR_FILENO,
           "the error stream is pointed at the temporary file");
}

/* Copies what was logged since capture_begin into text, as a string, and puts descriptor 2 back. */
static void capture_end(char *text, size_t text_capacity)
{
    const bool is_open = capture_descriptor >= 0 && error_stream_descriptor >= 0 && fflush(stderr) == 0;
    const ssize_t byte_count = is_open ? pread(capture_descriptor, text, text_capacity - 1, 0) : -1;
    expect(is_open && dup2(error_stream_descriptor, STDERR_FILENO) == STDERR_FILENO && byte_count >= 0,
           "the temporary file is read back and the error stream restored");
    text[byte_count >= 0 ? (size_t)byte_count : 0] = '\0';
}

/* The text is one line, of an error, that holds the words. */
static void expect_one_error_line_with(const char *text, const char *words, const char *what)
{
    if (!test_log_is_expected(text, words, true)) {
        (void)printf("FAILED: %s: %s: no single error line with \"%s\": \"%s\"\n", context_text, what, words, text);
        failures++;
    }
}

/* The bytes of a matrix as the reference of the formats counts them; 0 for a shape its format does not store. */
static uint64_t matrix_bytes(enum weight_format_code format,
			     const struct matrix_shape_record *shape)
{
    struct test_weight_format_sizes sizes = {0};
    return test_weight_format_sizes_of(format, shape, &sizes) == STATUS_OK ?
		   sizes.stored_byte_count :
		   0;
}

static void name_range(struct touched_range *touched, uint32_t *count, uint64_t byte_offset, uint64_t byte_count,
                       bool is_written)
{
    touched[*count] = (struct touched_range){byte_offset, byte_count, is_written};
    *count += 1;
}

/* The slices of a kernel stamped per slice count whose handles are instances, if it is one of them; else 0. */
static uint32_t slices_of_instance(uint32_t kernel, const uint32_t *instances)
{
	static const uint32_t slice_counts[LINEAR_SLICE_INSTANCES] = {
		1, 2, 4, 8, 16
	}; /* by hand */
	uint32_t slices = 0;
	for (uint32_t instance = 0; instance < LINEAR_SLICE_INSTANCES;
	     instance++) {
		slices = instances[instance] == kernel ?
				 slice_counts[instance] :
				 slices;
	}
    return slices;
}

/* The ranges the record of a launch names, sized as linear_records.h says from the record and the grid alone (the
 * rows of the routed experts, which their records do not hold, are the case's; the matrices of a job are named by
 * no record), and what every launch has: a grid of two dimensions, a block of one, the record of its kernel and
 * nothing behind it, and of a product the block and shared memory of its instance. The two records of the jobs
 * begin with the same seven members. */
static uint32_t walk_launch(const struct test_case *test_case,
			    const struct kernel_launch *launch,
			    struct touched_range *touched)
{
	static const unsigned char zero_bytes[LAUNCH_ARGUMENT_BYTES] = { 0 };
	const bool is_routed = test_case->kind == ROUTED_EXPERTS;
	const uint64_t rows =
		is_routed ? test_case->description.routed_experts.row_count : 0;
	const uint64_t vectors = launch->grid[DIMENSION_Y];
	const uint32_t kernel = launch->kernel;
	const uint32_t gate_up_slices =
		slices_of_instance(kernel, made_up_kernels.expert_gate_up);
	const uint32_t product_slices =
		gate_up_slices +
		slices_of_instance(kernel, made_up_kernels.expert_down) +
		slices_of_instance(kernel,
				   made_up_kernels.trunk_linear_quantized) +
		slices_of_instance(kernel,
				   made_up_kernels.trunk_linear_bfloat16);
	const uint32_t product_threads =
		product_slices * LINEAR_THREADS_PER_WARP *
		(gate_up_slices != 0 ? EXPERT_GATE_UP_MATRICES : 1);
	size_t record_bytes = 0;
	uint32_t count = 0;
	if (kernel == made_up_kernels.activation_quantize) {
		struct activation_quantize_arguments record;
		record_bytes = sizeof(record);
		memcpy(&record, launch->argument, sizeof(record));
		name_range(touched, &count, record.values_offset,
			   vectors * record.width * sizeof(uint16_t), false);
		name_range(touched, &count, record.codes_offset,
			   vectors * record.width, true);
		name_range(touched, &count, record.exponents_offset,
			   vectors * (record.width / VALUES_PER_EXPONENT),
			   true);
		name_range(touched, &count, record.operands_offset,
			   vectors * record.width * sizeof(uint16_t), true);
	} else if (gate_up_slices != 0 ||
		   slices_of_instance(kernel, made_up_kernels.expert_down) !=
			   0) {
		struct expert_down_arguments record;
		record_bytes = gate_up_slices != 0 ?
				       sizeof(struct expert_gate_up_arguments) :
				       sizeof(record);
		memcpy(&record, launch->argument, sizeof(record));
		const uint64_t pairs = record.pair_count;
		const uint64_t vectors_fed = gate_up_slices != 0 ? rows : pairs;
		name_range(touched, &count, record.jobs_offset,
			   vectors * sizeof(struct expert_job_record), false);
		name_range(touched, &count, record.codes_offset,
			   vectors_fed * record.input_size, false);
		name_range(touched, &count, record.exponents_offset,
			   vectors_fed *
				   (record.input_size / VALUES_PER_EXPONENT),
			   false);
		name_range(touched, &count, record.operands_offset,
			   vectors_fed * record.input_size * sizeof(uint16_t),
			   false);
		name_range(touched, &count, record.output_offset,
			   pairs * record.output_size * sizeof(uint16_t), true);
	} else if (kernel == made_up_kernels.expert_row_sum) {
		struct expert_row_sum_arguments record;
		record_bytes = sizeof(record);
		memcpy(&record, launch->argument, sizeof(record));
		name_range(touched, &count, record.pair_output_offset,
			   vectors * record.pairs_per_row * record.width *
				   sizeof(uint16_t),
			   false);
		name_range(touched, &count, record.row_output_offset,
			   vectors * record.width * sizeof(float), true);
	} else if (slices_of_instance(kernel,
				      made_up_kernels.trunk_linear_quantized) !=
		   0) {
		struct trunk_linear_quantized_arguments record;
		record_bytes = sizeof(record);
		memcpy(&record, launch->argument, sizeof(record));
		const uint64_t rows_fed = record.row_count;
		name_range(touched, &count, record.codes_offset,
			   rows_fed * record.input_size, false);
		name_range(touched, &count, record.exponents_offset,
			   rows_fed * (record.input_size / VALUES_PER_EXPONENT),
			   false);
		name_range(touched, &count, record.operands_offset,
			   rows_fed * record.input_size * sizeof(uint16_t),
			   false);
		for (uint32_t matrix = 0; matrix < LINEAR_MERGED_MATRICES &&
					  record.output_size[matrix] != 0;
		     matrix++) {
			const struct matrix_shape_record shape = {
				record.output_size[matrix], record.input_size,
				E4M3_BITS_PER_WEIGHT, 0
			};
			name_range(touched, &count,
				   record.weights_offset[matrix],
				   matrix_bytes(E4M3, &shape), false);
			name_range(touched, &count,
				   record.output_offset[matrix],
				   rows_fed * record.output_size[matrix] *
					   sizeof(uint16_t),
				   true);
		}
		expect(launch->grid[DIMENSION_X] ==
			       (record.output_size[0] + record.output_size[1]) /
				       LINEAR_OUTPUTS_PER_TILE,
		       "a trunk launch has a block for every tile of its matrices");
	} else if (slices_of_instance(kernel,
				      made_up_kernels.trunk_linear_bfloat16) !=
		   0) {
		struct trunk_linear_bfloat16_arguments record;
		record_bytes = sizeof(record);
		memcpy(&record, launch->argument, sizeof(record));
		const struct matrix_shape_record shape = { record.output_size,
							   record.input_size,
							   E4M3_BITS_PER_WEIGHT,
							   0 };
		const uint64_t rows_fed = record.row_count;
		name_range(touched, &count, record.weights_offset,
			   matrix_bytes(E4M3, &shape), false);
		name_range(touched, &count, record.values_offset,
			   rows_fed * record.segment_count * record.input_size *
				   sizeof(uint16_t),
			   false);
		name_range(touched, &count, record.output_offset,
			   rows_fed * record.output_size * sizeof(uint16_t),
			   true);
	} else if (kernel == made_up_kernels.trunk_hidden) {
		struct trunk_hidden_arguments record;
		record_bytes = sizeof(record);
		memcpy(&record, launch->argument, sizeof(record));
		name_range(touched, &count, record.gate_offset,
			   vectors * record.width * sizeof(uint16_t), false);
		name_range(touched, &count, record.up_offset,
			   vectors * record.width * sizeof(uint16_t), false);
		name_range(touched, &count, record.hidden_offset,
			   vectors * record.width * sizeof(uint16_t), true);
	}
	expect(count != 0 && launch->argument_byte_count == record_bytes &&
		       launch->branch == LAUNCH_CHAIN &&
		       memcmp(launch->argument + record_bytes, zero_bytes,
			      sizeof(launch->argument) - record_bytes) == 0 &&
		       launch->grid[DIMENSION_X] != 0 &&
		       launch->grid[DIMENSION_Y] != 0 &&
		       launch->grid[DIMENSION_Z] == 1 &&
		       launch->block[DIMENSION_X] != 0 &&
		       launch->block[DIMENSION_Y] == 1 &&
		       launch->block[DIMENSION_Z] == 1,
	       "a launch names a kernel of the linear, holds its record with zero bytes behind it, and has a grid of "
	       "two dimensions and a block of one");
	expect(product_slices == 0 ||
		       (launch->block[DIMENSION_X] == product_threads &&
			launch->dynamic_shared_byte_count ==
				product_threads *
					LINEAR_SHARED_BYTES_PER_THREAD),
	       "a product has the block of its instance and its shared memory");
	return count;
}

/* The prepare function of a kind. */
MUST_CHECK static result_status
prepare(enum described kind, const void *description, uint64_t arena_bytes,
	const struct linear_kernels_record *kernels, struct kernel_launch *out)
{
	return kind == QUANTIZE ?
		       linear_quantize_prepare(description, arena_bytes,
					       kernels, out) :
	       kind == TRUNK_QUANTIZED ?
		       linear_trunk_quantized_prepare(description, arena_bytes,
						      kernels, out) :
	       kind == TRUNK_BFLOAT16 ?
		       linear_trunk_bfloat16_prepare(description, arena_bytes,
						     kernels, out) :
	       kind == ROUTED_EXPERTS ?
		       linear_routed_experts_prepare(description, arena_bytes,
						     kernels, out) :
		       linear_shared_expert_prepare(description, arena_bytes,
						    kernels, out);
}

/* A range of a case, in the order of the test's arena. */
static struct linear_range *range_of(struct test_case *test_case,
				     uint32_t range_index)
{
    unsigned char *const description_bytes = (unsigned char *)&test_case->description;
    void *const range = description_bytes + range_offsets[test_case->kind][range_index];   /* a member of it */
    return range;
}

/* The range of the routed experts that holds the matrices of a layer: no record names it. */
static bool is_every_expert(const struct test_case *test_case, uint32_t range_index)
{
    return test_case->kind == ROUTED_EXPERTS && range_index == 0;
}

/* Lays the ranges of a case out from FIRST_OFFSET on, each on the next multiple of the alignment behind the one
 * before, and ends the arena with the last. A matrix gets the bytes its format stores and the experts those of a
 * layer; any other range gets what the launches touch in it, or WIDE_RANGE_BYTES as long as that is not known. */
static void lay_out(struct test_case *test_case, bool is_wide)
{
    uint64_t cursor = FIRST_OFFSET;
    for (uint32_t range_index = 0; range_index < range_counts[test_case->kind]; range_index++) {
	    struct linear_range *const range = range_of(test_case, range_index);
	    const struct linear_matrix *const matrix = (const void *)range;
	    const bool is_matrix = range_index < matrix_counts[test_case->kind];
	    range->byte_offset = cursor;
	    range->byte_count =
		    is_matrix ? matrix_bytes(matrix->weight_format,
					     &matrix->shape) :
		    is_every_expert(test_case, range_index) ?
				BYTES_OF_EVERY_EXPERT :
		    is_wide ? WIDE_RANGE_BYTES :
			      test_case->touched_byte_count[range_index];
	    test_case->arena_byte_count = cursor + range->byte_count;
	    cursor = (test_case->arena_byte_count +
		      LINEAR_OFFSET_ALIGNMENT_BYTES - 1) /
		     LINEAR_OFFSET_ALIGNMENT_BYTES *
		     LINEAR_OFFSET_ALIGNMENT_BYTES;
    }
}

/* Walks the launches of a case that was prepared. While the case is sized (is_learning), a range of the description
 * takes the most bytes a record names in it. Afterwards the walk is the check: what a record names begins with a
 * range of the description and ends inside it and the arena, and what a launch writes lies on nothing else of it. */
static void walk_case(struct test_case *test_case,
		      const struct kernel_launch *launches, bool is_learning)
{
    const uint32_t range_count = range_counts[test_case->kind];
    for (uint32_t launch_index = 0; launch_index < launch_counts[test_case->kind]; launch_index++) {
        struct touched_range touched[MOST_TOUCHED];
        const uint32_t touched_count = walk_launch(test_case, &launches[launch_index], touched);
        for (uint32_t pair_index = 0; pair_index < touched_count * touched_count; pair_index++) {
            const struct touched_range *const named = &touched[pair_index / touched_count];
            const struct touched_range *const other = &touched[pair_index % touched_count];
            uint32_t range_index = 0;
            while (range_index < range_count && range_of(test_case, range_index)->byte_offset != named->byte_offset) {
                range_index++;
            }
            if (is_learning && range_index < range_count &&
                test_case->touched_byte_count[range_index] < named->byte_count) {
                test_case->touched_byte_count[range_index] = named->byte_count;
            }
            expect(is_learning || (range_index < range_count && named->byte_count != 0 &&
                                   named->byte_count <= range_of(test_case, range_index)->byte_count &&
                                   named->byte_offset <= test_case->arena_byte_count &&
                                   named->byte_count <= test_case->arena_byte_count - named->byte_offset),
                   "what a record names begins with a range of the description and ends inside it and the arena");
            expect(named == other || !named->is_written ||
                       named->byte_offset >= other->byte_offset + other->byte_count ||
                       other->byte_offset >= named->byte_offset + named->byte_count,
                   "what a launch writes lies on nothing else it touches");
        }
    }
}

/* Gives every range of a case exactly the bytes the records of its launches name in it: the case is laid out wide,
 * prepared and walked, and laid out again. */
static void lay_out_tight(struct test_case *test_case)
{
	struct kernel_launch launches[MOST_LAUNCHES];
	memset(test_case->touched_byte_count, 0,
	       sizeof(test_case->touched_byte_count));
	lay_out(test_case, true);
	if (prepare(test_case->kind, &test_case->description,
		    test_case->arena_byte_count, &made_up_kernels,
		    launches) == STATUS_OK) {
		walk_case(test_case, launches, true);
	}
    lay_out(test_case, false);
}

/* An 8-bit matrix; where it lies is the layout's to say. */
static struct linear_matrix matrix_of(uint32_t output_size, uint32_t input_size,
				      uint32_t slice_count)
{
	return (struct linear_matrix){ .shape = { output_size, input_size,
						  E4M3_BITS_PER_WEIGHT, 0 },
				       .weight_format = E4M3,
				       .slice_count = slice_count };
}

/* The case of a kind with the sizes of the model: rows of 5120 values; a trunk matrix (where none is given, the
 * first of the table for quantized rows and the second for bfloat16 rows); the routed experts of a layer with a job
 * for each of its 384 experts; the shared expert. slice_count is that of every matrix. */
static struct test_case case_of(enum described kind, const struct trunk_shape *shape, uint32_t row_count,
                                uint32_t slice_count)
{
    const struct trunk_shape *const trunk = shape != NULL ? shape : &trunk_shapes[kind == TRUNK_QUANTIZED ? 0 : 1];
    struct test_case test_case;
    memset(&test_case, 0, sizeof(test_case));
    test_case.kind = kind;
    if (kind == QUANTIZE) {
        test_case.description.quantize.width = HIDDEN_SIZE;
        test_case.description.quantize.row_count = row_count;
    } else if (kind == TRUNK_QUANTIZED) {
        test_case.description.trunk_quantized.matrices[0] =
            matrix_of(trunk->output_size, trunk->input_size, slice_count);
        test_case.description.trunk_quantized.matrix_count = 1;
        test_case.description.trunk_quantized.row_count = row_count;
    } else if (kind == TRUNK_BFLOAT16) {
        test_case.description.trunk_bfloat16.matrix = matrix_of(trunk->output_size, trunk->input_size, slice_count);
        test_case.description.trunk_bfloat16.row_count = row_count;
        test_case.description.trunk_bfloat16.segment_count = trunk->segment_count;
    } else if (kind == ROUTED_EXPERTS) {
	    test_case.description.routed_experts =
		    (struct linear_routed_experts_description){
			    .gate_up_shape = { INNER_SIZE, HIDDEN_SIZE,
					       E2M1_BITS_PER_WEIGHT, 0 },
			    .down_shape = { HIDDEN_SIZE, INNER_SIZE,
					    E2M1_BITS_PER_WEIGHT, 0 },
			    .weight_format = E2M1,
			    .job_count = EXPERT_COUNT,
			    .row_count = row_count,
			    .pairs_per_row = PAIRS_PER_ROW,
			    .gate_up_slice_count = slice_count,
			    .down_slice_count = slice_count,
			    .activation_limit = ACTIVATION_LIMIT
		    };
    } else {
	    test_case.description
		    .shared_expert = (struct linear_shared_expert_description){
		    .gate = matrix_of(INNER_SIZE, HIDDEN_SIZE, slice_count),
		    .up = matrix_of(INNER_SIZE, HIDDEN_SIZE, slice_count),
		    .down = matrix_of(HIDDEN_SIZE, INNER_SIZE, slice_count),
		    .row_count = row_count,
		    .activation_limit = ACTIVATION_LIMIT
	    };
    }
    lay_out_tight(&test_case);
    return test_case;
}

/* Prepares a case that must be accepted, without a word on the error stream, and walks its launches. */
static void accepted(struct test_case *test_case,
		     struct kernel_launch *launches, const char *what)
{
    char logged[CAPTURE_CAPACITY];
    context_text = what;
    capture_begin();
    const result_status status =
	    prepare(test_case->kind, &test_case->description,
		    test_case->arena_byte_count, &made_up_kernels, launches);
    capture_end(logged, sizeof(logged));
    expect(status == STATUS_OK, "the description is accepted");
    expect(logged[0] == '\0', "an accepted description logs nothing");
    if (status == STATUS_OK) {
	    walk_case(test_case, launches, false);
    }
}

/* A call that must be refused: the status, one error line with the words of the rule, the launches untouched. */
static void call_refused(enum described kind, const void *description,
			 uint64_t arena_byte_count,
			 const struct linear_kernels_record *kernels,
			 bool gives_launches, const char *rule,
			 const char *what)
{
	struct kernel_launch launches[MOST_LAUNCHES];
	unsigned char untouched[sizeof(launches)];
	char logged[CAPTURE_CAPACITY];
	memset(launches, UNTOUCHED_BYTE, sizeof(launches));
	memset(untouched, UNTOUCHED_BYTE, sizeof(untouched));
	capture_begin();
	const result_status status = prepare(kind, description,
					     arena_byte_count, kernels,
					     gives_launches ? launches : NULL);
	capture_end(logged, sizeof(logged));
	expect(status == INVALID, what);
	expect_one_error_line_with(logged, rule, what);
	expect(memcmp(launches, untouched, sizeof(untouched)) == 0,
	       "a refusal leaves the launches untouched");
}

static void refused(const struct test_case *test_case, const char *rule, const char *what)
{
    call_refused(test_case->kind, &test_case->description, test_case->arena_byte_count, &made_up_kernels, true, rule,
                 what);
}

/* A launch is of this kernel, with this grid, block and shared memory (by hand) and this record. */
static void expect_launch(const struct kernel_launch *launch, uint32_t kernel,
			  uint32_t grid_columns, uint32_t grid_rows,
			  uint32_t block_threads, uint32_t shared_byte_count,
			  const void *record, size_t record_byte_count,
			  const char *what)
{
    if (launch->kernel != kernel || launch->grid[DIMENSION_X] != grid_columns ||
        launch->grid[DIMENSION_Y] != grid_rows || launch->block[DIMENSION_X] != block_threads ||
        launch->dynamic_shared_byte_count != shared_byte_count || launch->argument_byte_count != record_byte_count ||
        memcmp(launch->argument, record, record_byte_count) != 0) {
        (void)printf("FAILED: %s: %s: kernel %" PRIu32 ", grid %" PRIu32 " by %" PRIu32 ", block %" PRIu32
                     ", shared memory %" PRIu32 ", or its record\n",
                     context_text, what, launch->kernel, launch->grid[DIMENSION_X], launch->grid[DIMENSION_Y],
                     launch->block[DIMENSION_X], launch->dynamic_shared_byte_count);
        failures++;
    }
}

static void test_byte_counts(void)
{
    context_text = "the bytes of an array";
    expect(linear_values_byte_count(ROWS, HIDDEN_SIZE) == VALUES_OF_8_ROWS,
	   "values of 8 rows");
    expect(linear_codes_byte_count(ROWS, HIDDEN_SIZE) == CODES_OF_8_ROWS,
	   "codes of 8 rows");
    expect(linear_exponents_byte_count(ROWS, HIDDEN_SIZE) ==
		   EXPONENTS_OF_8_ROWS,
	   "exponents of 8 rows");
    expect(linear_sums_byte_count(ROWS, HIDDEN_SIZE) == SUMS_OF_8_ROWS,
	   "sums of 8 rows");
    expect(linear_exponents_byte_count(PAIRS_OF_8_ROWS, INNER_SIZE) ==
		   HIDDEN_EXPONENTS_OF_48_PAIRS,
	   "hidden exponents of 48 pairs");
    expect(linear_jobs_byte_count(EXPERT_COUNT) == JOBS_OF_384, "384 jobs");
    expect(linear_values_byte_count(0, HIDDEN_SIZE) == 0 &&
		   linear_codes_byte_count(ROWS, 0) == 0 &&
		   linear_sums_byte_count(0, 0) == 0 &&
		   linear_jobs_byte_count(0) == 0,
	   "no vectors, no width and no jobs are no bytes");
    expect(linear_codes_byte_count(TWO_TO_THE_32, TWO_TO_THE_32 - 1) ==
		   UINT64_MAX - TWO_TO_THE_32 + 1,
	   "2^64 - 2^32 codes are counted");
    expect(linear_codes_byte_count(TWO_TO_THE_32, TWO_TO_THE_32) == UINT64_MAX,
	   "2^64 codes are not");
    expect(linear_values_byte_count(TWO_TO_THE_63 - 1, 1) == UINT64_MAX - 1,
	   "2^64 - 2 bytes of values");
    expect(linear_values_byte_count(1, TWO_TO_THE_63) == UINT64_MAX,
	   "2^64 bytes of values");
    expect(linear_sums_byte_count(TWO_TO_THE_32, TWO_TO_THE_32) == UINT64_MAX,
	   "2^66 bytes of sums");
    expect(linear_exponents_byte_count(TWO_TO_THE_63, TWO_TO_THE_63) ==
		   UINT64_MAX,
	   "2^121 exponents");
    expect(linear_jobs_byte_count(UINT64_MAX / EXPERT_JOB_BYTES + 1) ==
		   UINT64_MAX,
	   "one job more than 64 bits count the bytes of");
}

/* The made-up handles: FIRST_MADE_UP_HANDLE and on, in the order of the members. */
static void made_up_kernels_make(void)
{
    uint32_t handles[KERNEL_COUNT];
    for (uint32_t kernel_index = 0; kernel_index < KERNEL_COUNT; kernel_index++) {
        handles[kernel_index] = FIRST_MADE_UP_HANDLE + kernel_index;
    }
    memcpy(&made_up_kernels, handles, sizeof(made_up_kernels));
}

/* The launches of the model's matrices, against constants computed by hand. */
static void test_model_launches(void)
{
	struct kernel_launch launches[MOST_LAUNCHES];
	struct test_case rows_case = case_of(QUANTIZE, NULL, ROWS, 0);
	const struct linear_quantize_description *const rows =
		&rows_case.description.quantize;
	const struct activation_quantize_arguments rows_record = {
		rows->values.byte_offset,
		rows->codes.byte_offset,
		rows->exponents.byte_offset,
		rows->operands.byte_offset,
		HIDDEN_SIZE,
		0
	};
	accepted(&rows_case, launches, "8 rows of the model quantized");
	expect_launch(&launches[0], made_up_kernels.activation_quantize,
		      ELEMENT_BLOCKS_OF_5120, ROWS, ELEMENT_THREADS, 0,
		      RECORD(rows_record),
		      "20 blocks of 256 threads for each of 8 rows");
	expect(rows->values.byte_offset > UINT32_MAX &&
		       rows->values.byte_count == VALUES_OF_8_ROWS &&
		       rows->codes.byte_count == CODES_OF_8_ROWS &&
		       rows->exponents.byte_count == EXPONENTS_OF_8_ROWS,
	       "the ranges lie beyond 32 bits and hold the bytes computed by hand");
	rows_case.description.quantize.width = A_BLOCK_AND_A_LOAD;
	lay_out_tight(&rows_case);
	accepted(&rows_case, launches, "rows of 320 values quantized");
	expect(launches[0].grid[DIMENSION_X] == ELEMENT_BLOCKS_OF_320,
	       "a width of a block and a quarter takes 2 blocks");
	for (size_t shape_index = 0; shape_index + 1 < COUNT_OF(trunk_shapes);
	     shape_index++) {
		const struct trunk_shape *const shape =
			&trunk_shapes[shape_index];
		struct test_case quantized_case = case_of(
			TRUNK_QUANTIZED, shape, ROWS, shape->slice_count);
		struct test_case bfloat16_case = case_of(
			TRUNK_BFLOAT16, shape, ROWS, shape->slice_count);
		const struct linear_trunk_quantized_description *const
			quantized = &quantized_case.description.trunk_quantized;
		const struct linear_trunk_bfloat16_description *const bfloat16 =
			&bfloat16_case.description.trunk_bfloat16;
		const struct trunk_linear_quantized_arguments quantized_record = {
			{ quantized->matrices[0].stored.byte_offset, 0 },
			{ quantized->outputs[0].byte_offset, 0 },
			quantized->codes.byte_offset,
			quantized->exponents.byte_offset,
			quantized->operands.byte_offset,
			{ shape->output_size, 0 },
			shape->input_size,
			ROWS
		};
		const struct trunk_linear_bfloat16_arguments bfloat16_record = {
			bfloat16->matrix.stored.byte_offset,
			bfloat16->values.byte_offset,
			bfloat16->output.byte_offset,
			shape->output_size,
			shape->input_size,
			ROWS,
			shape->segment_count
		};
		const uint32_t instance = shape->instance;
		accepted(&quantized_case, launches, shape->what);
		expect_launch(&launches[0],
			      made_up_kernels.trunk_linear_quantized[instance],
			      shape->tile_count, 1, shape->block_threads,
			      shape->shared_byte_count,
			      RECORD(quantized_record), "with quantized rows");
		accepted(&bfloat16_case, launches, shape->what);
		expect_launch(&launches[0],
			      made_up_kernels.trunk_linear_bfloat16[instance],
			      shape->tile_count, 1, shape->block_threads,
			      shape->shared_byte_count, RECORD(bfloat16_record),
			      "with bfloat16 rows");
	}
    for (uint32_t slice_count = FEW_SLICES; slice_count <= MANY_SLICES; slice_count *= FEW_SLICES) {
        const bool is_few = slice_count == FEW_SLICES;
        const uint32_t instance = is_few ? INSTANCE_OF_4_SLICES : INSTANCE_OF_16_SLICES;
        struct test_case routed_case = case_of(ROUTED_EXPERTS, NULL, ROWS, slice_count);
	const struct linear_routed_experts_description *const routed =
		&routed_case.description.routed_experts;
	const struct expert_gate_up_arguments gate_up_record = {
		routed->jobs.byte_offset,
		routed->codes.byte_offset,
		routed->exponents.byte_offset,
		routed->operands.byte_offset,
		routed->hidden.byte_offset,
		INNER_SIZE,
		HIDDEN_SIZE,
		PAIRS_OF_8_ROWS,
		PAIRS_PER_ROW,
		ACTIVATION_LIMIT,
		0
	};
	const struct activation_quantize_arguments pairs_record = {
		routed->hidden.byte_offset,
		routed->hidden_codes.byte_offset,
		routed->hidden_exponents.byte_offset,
		routed->hidden_operands.byte_offset,
		INNER_SIZE,
		0
	};
	const struct expert_down_arguments down_record = {
		routed->jobs.byte_offset,
		routed->hidden_codes.byte_offset,
		routed->hidden_exponents.byte_offset,
		routed->hidden_operands.byte_offset,
		routed->pair_output.byte_offset,
		HIDDEN_SIZE,
		INNER_SIZE,
		PAIRS_OF_8_ROWS,
		0
	};
	const struct expert_row_sum_arguments row_sum_record = {
		routed->pair_output.byte_offset, routed->row_output.byte_offset,
		HIDDEN_SIZE, PAIRS_PER_ROW
	};
	accepted(&routed_case, launches, "the routed experts of the model for 8 rows, with 4 and with 16 slices");
	expect_launch(&launches[LINEAR_ROUTED_EXPERTS_GATE_UP],
		      made_up_kernels.expert_gate_up[instance],
		      TILES_OF_2304_OUTPUTS, EXPERT_COUNT,
		      is_few ? GATE_UP_THREADS_OF_4_SLICES :
			       GATE_UP_THREADS_OF_16_SLICES,
		      is_few ? GATE_UP_SHARED_BYTES_OF_4_SLICES :
			       GATE_UP_SHARED_BYTES_OF_16_SLICES,
		      RECORD(gate_up_record), "gate with up");
	expect_launch(&launches[LINEAR_ROUTED_EXPERTS_QUANTIZE],
		      made_up_kernels.activation_quantize,
		      ELEMENT_BLOCKS_OF_2304, PAIRS_OF_8_ROWS, ELEMENT_THREADS,
		      0, RECORD(pairs_record),
		      "the hidden vectors of 48 pairs quantized");
	expect_launch(&launches[LINEAR_ROUTED_EXPERTS_DOWN],
		      made_up_kernels.expert_down[instance],
		      TILES_OF_5120_OUTPUTS, EXPERT_COUNT,
		      is_few ? THREADS_OF_4_SLICES : THREADS_OF_16_SLICES,
		      is_few ? SHARED_BYTES_OF_4_SLICES :
			       SHARED_BYTES_OF_16_SLICES,
		      RECORD(down_record), "down");
	expect_launch(&launches[LINEAR_ROUTED_EXPERTS_ROW_SUM],
		      made_up_kernels.expert_row_sum, ELEMENT_BLOCKS_OF_5120,
		      ROWS, ELEMENT_THREADS, 0, RECORD(row_sum_record),
		      "the sum of each row");
	expect(routed->jobs.byte_offset - routed->expert_weights.byte_offset == BYTES_OF_EVERY_EXPERT &&
                   routed->jobs.byte_count == JOBS_OF_384 && routed->hidden.byte_count == HIDDEN_VALUES_OF_48_PAIRS &&
                   routed->hidden_exponents.byte_count == HIDDEN_EXPONENTS_OF_48_PAIRS &&
                   routed->row_output.byte_count == SUMS_OF_8_ROWS,
               "the ranges of the experts hold the bytes computed by hand");
    }
    struct test_case shared_case = case_of(SHARED_EXPERT, NULL, ROWS, FEW_SLICES);
    const struct linear_shared_expert_description *const shared =
	    &shared_case.description.shared_expert;
    const struct trunk_linear_quantized_arguments gate_up_record = {
	    { shared->gate.stored.byte_offset, shared->up.stored.byte_offset },
	    { shared->gate_output.byte_offset, shared->up_output.byte_offset },
	    shared->codes.byte_offset,
	    shared->exponents.byte_offset,
	    shared->operands.byte_offset,
	    { INNER_SIZE, INNER_SIZE },
	    HIDDEN_SIZE,
	    ROWS
    };
    const struct trunk_hidden_arguments hidden_record = {
	    shared->gate_output.byte_offset, shared->up_output.byte_offset,
	    shared->hidden.byte_offset, INNER_SIZE, ACTIVATION_LIMIT
    };
    const struct activation_quantize_arguments hidden_codes_record = {
	    shared->hidden.byte_offset,
	    shared->hidden_codes.byte_offset,
	    shared->hidden_exponents.byte_offset,
	    shared->hidden_operands.byte_offset,
	    INNER_SIZE,
	    0
    };
    const struct trunk_linear_quantized_arguments shared_down_record = {
	    { shared->down.stored.byte_offset, 0 },
	    { shared->output.byte_offset, 0 },
	    shared->hidden_codes.byte_offset,
	    shared->hidden_exponents.byte_offset,
	    shared->hidden_operands.byte_offset,
	    { HIDDEN_SIZE, 0 },
	    INNER_SIZE,
	    ROWS
    };
    accepted(&shared_case, launches, "the shared expert of the model for 8 rows");
    expect_launch(&launches[LINEAR_SHARED_EXPERT_GATE_UP],
		  made_up_kernels.trunk_linear_quantized[INSTANCE_OF_4_SLICES],
		  TILES_OF_GATE_WITH_UP, 1, THREADS_OF_4_SLICES,
		  SHARED_BYTES_OF_4_SLICES, RECORD(gate_up_record),
		  "gate with up");
    expect_launch(&launches[LINEAR_SHARED_EXPERT_HIDDEN],
		  made_up_kernels.trunk_hidden, ELEMENT_BLOCKS_OF_2304, ROWS,
		  ELEMENT_THREADS, 0, RECORD(hidden_record), "the hidden step");
    expect_launch(&launches[LINEAR_SHARED_EXPERT_QUANTIZE],
		  made_up_kernels.activation_quantize, ELEMENT_BLOCKS_OF_2304,
		  ROWS, ELEMENT_THREADS, 0, RECORD(hidden_codes_record),
		  "the hidden vectors quantized");
    expect_launch(&launches[LINEAR_SHARED_EXPERT_DOWN],
		  made_up_kernels.trunk_linear_quantized[INSTANCE_OF_4_SLICES],
		  TILES_OF_5120_OUTPUTS, 1, THREADS_OF_4_SLICES,
		  SHARED_BYTES_OF_4_SLICES, RECORD(shared_down_record), "down");
}

/* One member of a description set to a value its function refuses. */
struct member_refusal {
    size_t member_offset;
    size_t member_byte_count;
    enum described kind;
    bool lays_out_again;                        /* wide, for the new sizes: nothing else is then wrong */
    uint64_t values[VALUES_PER_REFUSAL];        /* the low bytes of each are what the member is set to, in turn */
    const char *rule;                           /* words the one error line must hold */
    const char *what;
};

static const struct member_refusal member_refusals[] = {
	{ IN_QUANTIZE(width),
	  false,
	  { 0, 5152 },
	  RULE_WIDTH,
	  "no width, and one of whole blocks that is not whole loads" },
	{ IN_QUANTIZE(row_count),
	  false,
	  { 0, MODEL_WIDE_MOST_ROWS + 1u },
	  RULE_CHUNK_ROWS,
	  "no row, and one more than a chunk" },
	{ IN_TRUNK_QUANTIZED(matrices[0].weight_format),
	  false,
	  { E2M1, WEIGHT_FORMAT_FLOAT16 },
	  RULE_FORMAT,
	  "a 4-bit matrix, and a format this build does not carry" },
	{ IN_TRUNK_QUANTIZED(matrices[0].weight_format),
	  false,
	  { 0x80000005u, WEIGHT_FORMAT_COUNT },
	  RULE_FORMAT,
	  "formats outside the enumeration" },
	{ IN_TRUNK_QUANTIZED(matrices[0].shape.reserved_must_be_zero),
	  false,
	  { 1, 0x100 },
	  RULE_SHAPE_RESERVED,
	  "reserved bits" },
	{ IN_TRUNK_QUANTIZED(matrices[0].shape.bits_per_weight),
	  false,
	  { 4, 0 },
	  RULE_SHAPE,
	  "not the bits of the format" },
	{ IN_TRUNK_QUANTIZED(matrices[0].shape.output_size),
	  false,
	  { 0, 520 },
	  RULE_SHAPE,
	  "no output, and half a tile more" },
	{ IN_TRUNK_QUANTIZED(matrices[0].shape.input_size),
	  false,
	  { 0, 5136 },
	  RULE_SHAPE,
	  "no input, and half a block more" },
	{ IN_TRUNK_QUANTIZED(matrices[0].shape.input_size),
	  false,
	  { 5152, 32 },
	  RULE_LOADS,
	  "inputs the format stores" },
	{ IN_TRUNK_QUANTIZED(matrices[0].slice_count),
	  false,
	  { 0, 17 },
	  RULE_SLICES,
	  "no slice, and seventeen" },
	{ IN_TRUNK_QUANTIZED(matrices[0].wide_form),
	  false,
	  { LINEAR_WIDE_FORM_COUNT, UINT32_MAX },
	  RULE_WIDE_FORM,
	  "wide forms outside the enumeration" },
	{ IN_TRUNK_QUANTIZED(matrices[0].reserved_zero),
	  false,
	  { 1, 0x80000000u },
	  RULE_RESERVED,
	  "a reserved field of the matrix" },
	{ IN_TRUNK_QUANTIZED(row_count),
	  false,
	  { 0, LINEAR_WIDE_MOST_ROWS + 1u },
	  RULE_ROWS,
	  "no row, and one more than a wide trunk launch" },
	{ IN_TRUNK_QUANTIZED(matrix_count),
	  false,
	  { 0, 3 },
	  RULE_MATRICES,
	  "no matrix, and three" },
	{ IN_TRUNK_BFLOAT16(matrix.weight_format),
	  false,
	  { E2M1, UINT32_MAX },
	  RULE_FORMAT,
	  "not the 8-bit format" },
	{ IN_TRUNK_BFLOAT16(matrix.shape.input_size),
	  false,
	  { 4128, 4064 },
	  RULE_LOADS,
	  "half a load of inputs more, less" },
	{ IN_TRUNK_BFLOAT16(matrix.slice_count),
	  false,
	  { 0, 17 },
	  RULE_SLICES,
	  "no slice, and seventeen" },
	{ IN_TRUNK_BFLOAT16(matrix.wide_form),
	  false,
	  { LINEAR_WIDE_FORM_COUNT, 0x80000000u },
	  RULE_WIDE_FORM,
	  "wide forms outside the enumeration" },
	{ IN_TRUNK_BFLOAT16(row_count),
	  false,
	  { 0, LINEAR_WIDE_MOST_ROWS + 1u },
	  RULE_ROWS,
	  "no row, and one more than a wide trunk launch" },
	{ IN_TRUNK_BFLOAT16(segment_count),
	  false,
	  { 0, 3 },
	  RULE_SEGMENTS,
	  "no segment, and three for 512 tiles" },
	{ IN_TRUNK_BFLOAT16(segment_count),
	  false,
	  { 1024, 513 },
	  RULE_SEGMENTS,
	  "more segments than tiles" },
	{ IN_ROUTED(weight_format),
	  false,
	  { E4M3, WEIGHT_FORMAT_NONE },
	  RULE_FORMAT,
	  "not the 4-bit format" },
	{ IN_ROUTED(gate_up_shape.bits_per_weight),
	  false,
	  { 8, 5 },
	  RULE_SHAPE,
	  "gate and up of other bits" },
	{ IN_ROUTED(down_shape.reserved_must_be_zero),
	  false,
	  { 1, 0x100 },
	  RULE_SHAPE_RESERVED,
	  "a reserved field of down" },
	{ IN_ROUTED(gate_up_shape.input_size),
	  false,
	  { 5152, 0 },
	  RULE_SHAPE,
	  "inputs that are not whole loads" },
	{ IN_ROUTED(gate_up_shape.output_size),
	  true,
	  { 2240, 2368 },
	  RULE_CHAIN,
	  "gate and up of another inner size" },
	{ IN_ROUTED(down_shape.input_size),
	  true,
	  { 2240, 2368 },
	  RULE_CHAIN,
	  "down of another inner size" },
	{ IN_ROUTED(down_shape.output_size),
	  true,
	  { 5136, 5104 },
	  RULE_WIDTH,
	  "a row sum that is not whole loads wide" },
	{ IN_ROUTED(gate_up_slice_count),
	  false,
	  { 0, 17 },
	  RULE_SLICES,
	  "no slice of gate and up, and seventeen" },
	{ IN_ROUTED(down_slice_count),
	  false,
	  { 0, 17 },
	  RULE_SLICES,
	  "no slice of down, and seventeen" },
	{ IN_ROUTED(row_count),
	  false,
	  { 0, MODEL_WIDE_MOST_ROWS + 1u },
	  RULE_CHUNK_ROWS,
	  "no row, and one more than a chunk" },
	{ IN_ROUTED(pairs_per_row),
	  false,
	  { 0, 0x20000000u },
	  RULE_PAIRS,
	  "no pair, and 2^29 pairs for each of 8 rows" },
	{ IN_ROUTED(pairs_per_row),
	  false,
	  { MOST_PAIRS_PER_ROW, 7 },
	  RULE_TOUCHED,
	  "pairs that no range holds" },
	{ IN_ROUTED(job_count), false, { 0, 0 }, RULE_JOBS, "no job" },
	{ IN_ROUTED(job_count),
	  false,
	  { 385, UINT32_MAX },
	  RULE_TOUCHED,
	  "more jobs than the table holds" },
	{ IN_ROUTED(activation_limit),
	  false,
	  { 0, 0x80000000u },
	  RULE_LIMIT,
	  "a limit of zero, and of minus zero" },
	{ IN_ROUTED(activation_limit),
	  false,
	  { 0xC1200000u, 0x7FC00000u },
	  RULE_LIMIT,
	  "minus ten, and a not-a-number" },
	{ IN_ROUTED(reserved_zero),
	  false,
	  { 1, 0x80000000u },
	  RULE_RESERVED,
	  "a reserved field" },
	{ IN_SHARED(gate.weight_format),
	  false,
	  { E2M1, WEIGHT_FORMAT_EXL3 },
	  RULE_FORMAT,
	  "gate of another format" },
	{ IN_SHARED(up.weight_format),
	  false,
	  { E2M1, WEIGHT_FORMAT_INT4_SCALED },
	  RULE_FORMAT,
	  "up of another" },
	{ IN_SHARED(down.weight_format),
	  false,
	  { E2M1, WEIGHT_FORMAT_NONE },
	  RULE_FORMAT,
	  "down of another" },
	{ IN_SHARED(up.shape.output_size),
	  true,
	  { 2240, 2368 },
	  RULE_ONE_SHAPE,
	  "up of other outputs than gate" },
	{ IN_SHARED(up.shape.input_size),
	  true,
	  { 5056, 5184 },
	  RULE_PAIR,
	  "up of other inputs than gate" },
	{ IN_SHARED(down.shape.input_size),
	  true,
	  { 2240, 2368 },
	  RULE_CHAIN,
	  "down of another inner size" },
	{ IN_SHARED(down.shape.input_size),
	  false,
	  { 2336, 2272 },
	  RULE_LOADS,
	  "down of half a load of inputs more, less" },
	{ IN_SHARED(gate.slice_count),
	  false,
	  { 0, 17 },
	  RULE_SLICES,
	  "no slice of gate, and seventeen" },
	{ IN_SHARED(up.slice_count),
	  false,
	  { 0, 17 },
	  RULE_SLICES,
	  "no slice of up, and seventeen" },
	{ IN_SHARED(down.slice_count),
	  false,
	  { 0, 17 },
	  RULE_SLICES,
	  "no slice of down, and seventeen" },
	{ IN_SHARED(down.wide_form),
	  false,
	  { LINEAR_WIDE_FORM_COUNT, UINT32_MAX },
	  RULE_WIDE_FORM,
	  "a wide form of down outside the enumeration" },
	{ IN_SHARED(up.reserved_zero),
	  false,
	  { 1, 0x100 },
	  RULE_RESERVED,
	  "a reserved field of up" },
	{ IN_SHARED(row_count),
	  false,
	  { 0, LINEAR_WIDE_MOST_ROWS + 1u },
	  RULE_ROWS,
	  "no row, and one more than a wide trunk launch" },
	{ IN_SHARED(activation_limit),
	  false,
	  { 0, 0xFFC00000u },
	  RULE_LIMIT,
	  "a limit of zero, and a not-a-number" },
};

static void test_member_refusals(void)
{
	struct kernel_launch launches[MOST_LAUNCHES];
	for (size_t refusal_index = 0;
	     refusal_index < COUNT_OF(member_refusals) * VALUES_PER_REFUSAL;
	     refusal_index++) {
		const struct member_refusal *const refusal =
			&member_refusals[refusal_index / VALUES_PER_REFUSAL];
		struct test_case test_case =
			case_of(refusal->kind, NULL, ROWS, FEW_SLICES);
		context_text = refusal->what;
		memcpy((unsigned char *)&test_case.description +
			       refusal->member_offset,
		       &refusal->values[refusal_index % VALUES_PER_REFUSAL],
		       refusal->member_byte_count);
		if (refusal->lays_out_again) {
			lay_out(&test_case, true);
		}
		refused(&test_case, refusal->rule,
			"a member no launch may have");
	}
    /* what lies next to a refusal and is accepted: no limit, and as many slices as a matrix has loads */
    const struct trunk_shape *const two_loads = &trunk_shapes[COUNT_OF(trunk_shapes) - 1];
    struct test_case routed_case = case_of(ROUTED_EXPERTS, NULL, ROWS, FEW_SLICES);
    struct test_case shared_case = case_of(SHARED_EXPERT, NULL, ROWS, FEW_SLICES);
    struct test_case few_loads_case = case_of(TRUNK_QUANTIZED, two_loads, 1, two_loads->slice_count);
    routed_case.description.routed_experts.activation_limit = HUGE_VALF;
    shared_case.description.shared_expert.activation_limit = HUGE_VALF;
    accepted(&routed_case, launches, "routed experts without a limit");
    accepted(&shared_case, launches, "a shared expert without a limit");
    accepted(&few_loads_case, launches, "two slices of two loads, and four");
    few_loads_case.description.trunk_quantized.matrices[0].slice_count = FEW_SLICES;
    refused(&few_loads_case, RULE_SLICES, "four slices of two loads");
    context_text = "the experts' matrices and hidden values";
    /* the experts' range holds the larger of the two shapes: gate, then down; the smaller one lets no range grow */
    routed_case.description.routed_experts.down_shape.output_size = INNER_SIZE;
    routed_case.description.routed_experts.expert_weights.byte_count = BYTES_OF_AN_EXPERT_MATRIX - 1;
    refused(&routed_case, RULE_EXPERTS, "experts' matrices one byte short of a gate matrix");
    routed_case = case_of(ROUTED_EXPERTS, NULL, ROWS, FEW_SLICES);
    routed_case.description.routed_experts.gate_up_shape.input_size = INNER_SIZE;
    routed_case.description.routed_experts.expert_weights.byte_count = BYTES_OF_AN_EXPERT_MATRIX - 1;
    refused(&routed_case, RULE_EXPERTS, "experts' matrices one byte short of a down matrix");
    /* hidden values of more bytes than 64 bits count (one row of 2^32 - 1 pairs, the most that 32 bits count) fit no
     * range, were it of 2^64 - 1 bytes: every other range gets all the arena behind it, so only this rule refuses */
    routed_case = case_of(ROUTED_EXPERTS, NULL, 1, FEW_SLICES);
    routed_case.description.routed_experts.pairs_per_row = UINT32_MAX;
    routed_case.description.routed_experts.gate_up_shape.output_size = A_HUGE_INNER_SIZE;
    routed_case.description.routed_experts.down_shape.input_size = A_HUGE_INNER_SIZE;
    for (uint32_t range_index = 0; range_index < range_counts[ROUTED_EXPERTS]; range_index++) {
        range_of(&routed_case, range_index)->byte_count = UINT64_MAX - range_of(&routed_case, range_index)->byte_offset;
    }
    routed_case.description.routed_experts.hidden =
	    (struct linear_range){ 0, UINT64_MAX };
    routed_case.arena_byte_count = UINT64_MAX;
    refused(&routed_case, RULE_HIDDEN_VALUES, "over 2^64 bytes of hidden values in a range of 2^64 - 1 bytes");
}

/* Every kind for every number of rows, and every range of each, moved or resized alone. */
static void test_range_refusals(void)
{
	struct kernel_launch launches[MOST_LAUNCHES];
	for (uint32_t case_index = 0; case_index < DESCRIBED_COUNT * ROWS;
	     case_index++) {
		const enum described kind = (enum described)(case_index / ROWS);
		struct test_case test_case =
			case_of(kind, NULL, case_index % ROWS + 1, FEW_SLICES);
		const uint64_t arena_byte_count = test_case.arena_byte_count;
		accepted(&test_case, launches,
			 "a range of a description, for 1 to 8 rows");
		test_case.arena_byte_count = arena_byte_count - 1;
		refused(&test_case, RULE_ARENA,
			"an arena one byte short of the last range");
		for (uint32_t range_index = 0; range_index < range_counts[kind];
		     range_index++) {
			struct linear_range *const range =
				range_of(&test_case, range_index);
			const struct linear_range valid_range = *range;
			const uint64_t least_byte_count =
				is_every_expert(&test_case, range_index) ?
					BYTES_OF_AN_EXPERT_MATRIX :
					valid_range.byte_count;
			test_case.arena_byte_count = arena_byte_count;
			range->byte_offset =
				valid_range.byte_offset + HALF_AN_ALIGNMENT;
			refused(&test_case, RULE_ALIGNED,
				"a range that begins between two multiples of the alignment");
			range->byte_offset =
				UINT64_MAX - LINEAR_OFFSET_ALIGNMENT_BYTES + 1;
			refused(&test_case, RULE_ARENA,
				"a range that begins at the last multiple of the alignment in 64 bits");
			range->byte_offset = valid_range.byte_offset;
			range->byte_count = UINT64_MAX;
			refused(&test_case, RULE_ARENA,
				"a range whose end is beyond 64 bits");
			range->byte_count = least_byte_count - 1;
			refused(&test_case,
				range_index < matrix_counts[kind] ?
					RULE_STORED :
					RULE_TOUCHED,
				"a range one byte short of what the launches touch in it");
			range->byte_count = least_byte_count;
			accepted(
				&test_case, launches,
				"a range of exactly what the launches touch in it");
			range->byte_count = valid_range.byte_count + 1;
			if (range_index < matrix_counts[kind]) {
				refused(&test_case, RULE_STORED,
					"a matrix of one byte more than its format stores");
			} else if (range_index + 1 < range_counts[kind]) {
				range->byte_count =
					range_of(&test_case, range_index + 1)
						->byte_offset -
					valid_range.byte_offset;
				accepted(
					&test_case, launches,
					"a range that ends where the next begins");
				range->byte_count += 1;
				refused(&test_case, RULE_SHARED,
					"a range that ends one byte inside the next");
			}
			*range = valid_range;
			test_case.arena_byte_count = UINT64_MAX;
			for (uint32_t other_index = 0;
			     other_index < range_counts[kind]; other_index++) {
				if (other_index != range_index) {
					range->byte_offset =
						range_of(&test_case,
							 other_index)
							->byte_offset;
					refused(&test_case, RULE_SHARED,
						"a range that begins where another does");
				}
			}
			*range = valid_range;
		}
    }
}

/* What is missing: an argument, an arena, a kernel that a launch of the accepted case names. */
static void test_argument_refusals(void)
{
	struct kernel_launch launches[MOST_LAUNCHES];
	for (uint32_t case_index = 0;
	     case_index < DESCRIBED_COUNT * KERNEL_COUNT; case_index++) {
		const enum described kind =
			(enum described)(case_index / KERNEL_COUNT);
		const uint32_t kernel_index = case_index % KERNEL_COUNT;
		struct test_case test_case =
			case_of(kind, NULL, ROWS, FEW_SLICES);
		const uint64_t arena_byte_count = test_case.arena_byte_count;
		struct linear_kernels_record kernels = made_up_kernels;
		uint32_t handles[KERNEL_COUNT];
		bool is_launched = false;
		call_refused(kind, NULL, arena_byte_count, &made_up_kernels,
			     true, RULE_MISSING, "no description");
		call_refused(kind, &test_case.description, arena_byte_count,
			     NULL, true, RULE_MISSING, "no kernels");
		call_refused(kind, &test_case.description, arena_byte_count,
			     &made_up_kernels, false, RULE_MISSING,
			     "nowhere to put the launches");
		call_refused(kind, &test_case.description, 0, &made_up_kernels,
			     true, RULE_ARENA, "an arena of no bytes");
		accepted(&test_case, launches,
			 "an argument of a prepare function");
		memcpy(handles, &kernels, sizeof(handles));
		for (uint32_t launch_index = 0;
		     launch_index < launch_counts[kind]; launch_index++) {
			is_launched = is_launched ||
				      launches[launch_index].kernel ==
					      handles[kernel_index];
		}
		handles[kernel_index] = BACKEND_NO_KERNEL;
		memcpy(&kernels, handles, sizeof(kernels));
		if (is_launched) {
			call_refused(
				kind, &test_case.description, arena_byte_count,
				&kernels, true, RULE_KERNEL,
				"a kernel that is launched and was not prepared");
		} else {
			expect(prepare(kind, &test_case.description,
				       arena_byte_count, &kernels,
				       launches) == STATUS_OK,
			       "a kernel that is not launched need not be prepared");
		}
    }
}

int main(void)
{
    error_stream_descriptor = dup(STDERR_FILENO);
    capture_descriptor = mkstemp(capture_path);
    if (error_stream_descriptor < 0 || capture_descriptor < 0) {
        (void)printf("FAILED: the temporary file that stands in for the error stream could not be made\n");
        return EXIT_FAILURE;
    }
    made_up_kernels_make();
    test_byte_counts();
    test_model_launches();
    test_member_refusals();
    test_range_refusals();
    test_argument_refusals();
    context_text = "the end of the test";
    expect(close(capture_descriptor) == 0 && unlink(capture_path) == 0 && close(error_stream_descriptor) == 0 &&
               test_host_memory_backend_open_card_count() == 0,
           "the temporary file is removed and no card is left open");
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("linear_test passed (%zu trunk shapes, %zu members refused)\n", COUNT_OF(trunk_shapes) - 1,
                 COUNT_OF(member_refusals));
    return EXIT_SUCCESS;
}
