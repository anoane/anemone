/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * linear_records.h: the records of the kernels of the linear and the layout of every array a record names. C11 and
 * CUDA, nothing of the engine included: for the kernel files, the host code that prepares their launches, the tests.
 * Class-80 trunk selection above 8 rows uses relaxed_linear_records.h's ordered instance by default; the records
 * here remain the thin reference and the other classes' ABI, and the activation quantizer is shared unchanged.
 *
 * Contract:
 *   - A kernel is extern "C" void <name>(struct <name>_arguments arguments, unsigned char *arena) in the kernel
 *     file named beside its record. A record names memory by byte offsets into the arena. A kernel checks nothing:
 *     whoever prepares a launch has checked every rule of this header, that every range the launch reads or writes
 *     lies inside the arena (computed in 64 bits, without overflow), and that no range it writes overlaps another
 *     range it reads or writes. A kernel touches no byte outside the ranges stated here, uses no atomic operation,
 *     and nothing in it depends on timing. (The kernel modules of the debug and thread builds trap where the host
 *     broke a promise their checks can see: kernel_linear_common.cuh.)
 *   - Sizes. A width or input_size is a multiple of LINEAR_VALUES_PER_LOAD, an output_size a multiple of
 *     LINEAR_OUTPUTS_PER_TILE; none is zero. Every offset, in a record or in a job, is a multiple of
 *     LINEAR_OFFSET_ALIGNMENT_BYTES. A number of more than one byte is little-endian.
 *   - Arrays. A vector v is a row of the batch, or a pair: one row at one of its routed experts.
 *       values     uint16 [v][width]: bfloat16, the upper half of the float32 of the same number.
 *       codes      uint8 [v][width]: the E4M3 bytes of a quantized vector (weight_format_seam.h says what a byte
 *                  stands for) in LANE ORDER: with l = 0 .. width / 64 - 1 and t, s, j = 0 .. 3, byte
 *                  ((4 l + t) 4 + s) 4 + j of a vector is the code of its value 64 l + 16 s + 2 t + j / 2 + 8 (j % 2).
 *                  (Lane place t of a column of the tensor-core instruction m16n8k16 holds inputs 2 t and 2 t + 1
 *                  of instruction s in one register, bytes 0 and 2, and 2 t + 8 and 2 t + 9 in the other.)
 *       exponents  uint8 [v][width / 32]: entry k is the E8M0 code c of the vector's values 32 k .. 32 k + 31: a
 *                  value is what its code stands for times 2^(c - 127).
 *       operands   uint16 [v][width] of a quantized vector: the bfloat16 of what a value's code and exponent stand
 *                  for, times 2^100, the code's sign kept (exact: an E4M3 number has four significant bits), in
 *                  OPERAND ORDER: with l, t, s as above and r, h = 0 or 1, half ((4 l + t) 4 + s) 4 + 2 r + h of a
 *                  vector is the operand of its value 64 l + 16 s + 2 t + h + 8 r (half h of register r of operand B
 *                  of instruction s). Every value of a block whose exponent is above
 *                  ACTIVATION_CARRIED_EXPONENT_LARGEST is an infinity of its sign.
 *       sums       float32 [row][width].
 *       a matrix   the stored bytes of weight_format_seam.h at its offset: its codes, its scales right behind.
 *   - Quantization (activation_quantize_entry) is the reference's act_quant with scales rounded to a power of
 *     two. For each block of 32 consecutive values of a vector, all in float32: m is the block's largest
 *     magnitude, at least float32(1e-4), times float32(1 / 448); c - 127 is the exponent of m, and one more unless
 *     m is a power of two (read from the bits of m); each value is divided by 2^(c - 127), its magnitude is
 *     limited to 448 and rounded to the nearest E4M3 number, a tie to the code that is even; the code keeps the
 *     sign of the value, also where it is zero. Finite values give c from ACTIVATION_EXPONENT_SMALLEST_VALUE to
 *     ACTIVATION_EXPONENT_LARGEST. The codes, the exponents and the operands are written, on every card. A
 *     block that holds an infinity or a not-a-number gets 32 unspecified codes and operands and an unspecified
 *     exponent, and nothing else changes.
 *   - A product (the gate, up and down of an expert, a trunk linear). For one output and one vector the L =
 *     input_size / 64 loads of 64 inputs are cut into S slices: slice i holds loads floor(i L / S) to
 *     floor((i + 1) L / S) - 1. A slice is one float32 chain: from +0, for each block of 32 inputs in ascending
 *     order, chain = fma(p, 2^e, chain). p is the tensor cores' float32 sum of the block's 32 products, two
 *     instructions of 16 products, the second accumulating on the first. Their operands are a FORM, which the
 *     class of the card (arithmetic_class of backend_seam.h) fixes for each kernel:
 *       codes    the stored codes placed as 16-bit floats, which makes of an E2M1 code its value times 2^-14 and
 *                of an E4M3 byte (of the weights, or of the vector) its value times 2^-8; 2^e undoes the
 *                placements and applies the scales of both sides. The trunk's quantized rows on class 120
 *                native (ARITHMETIC_CLASS_SM_120_NATIVE: the PRO's own arithmetic, the default).
 *       carried  the weights' codes placed as bfloat16, an E2M1 code its value times 2^-126 and an E4M3 byte its
 *                value times 2^-120, by the vector's operands; 2^e undoes the placement and the operands' 2^100 and
 *                applies the weights' scale, the vector's being inside its operands. Every quantized product on
 *                class 80.
 *       block-scaled  E2M1 weights by the vector's codes, one instruction of 32 products a block with both scale
 *                codes inside it, on a zero accumulator; its result is the block's term, and the chain adds it,
 *                chain = chain + term, in place of the fused multiply-add. The routed experts on class 120 native.
 *     2^e is one exact multiplication. The S chains are added from +0 in ascending order, ((0 + chain 0) + chain 1)
 *     + .., and that sum is rounded once to bfloat16: to nearest, a tie to even, a not-a-number to 0x7FC0. S is a
 *     count of LINEAR_EVERY_SLICE_COUNT and at most L: each product kernel is stamped once for each of those
 *     counts, its function named by its stem (KERNEL_*_STEM) and the count, and its block is fixed by S.
 *     The bits of an output depend on its matrix, its vector and S: not on the other vectors of the launch, not on
 *     the column the vector takes, not on the grid. (That a column gets the same sum wherever it stands is the
 *     tensor cores' own: the cards showed it for the prototype, tools/measure/shared_rows_expert_probe.cu, and the
 *     tests show it again. So is how an instruction rounds, defined nowhere here: a host reference has the bits of p
 *     only where every addition is exact, as with 32 products that are whole multiples of one power of two and add
 *     up in magnitude to under 2^24 of it.)
 *   - Scale codes. With w the E8M0 code of the weights' scale and a the exponent of the vector's block, 2^e is
 *     the true power of two: in the codes form where w + a lies from EXPERT_SCALE_SUM_SMALLEST to
 *     EXPERT_SCALE_SUM_LARGEST (E2M1 weights) or from TRUNK_SCALE_SUM_SMALLEST to
 *     TRUNK_SCALE_SUM_LARGEST (E4M3 weights); in the carried form where w is at most
 *     EXPERT_CARRIED_SCALE_LARGEST or TRUNK_CARRIED_SCALE_LARGEST and a lies from
 *     ACTIVATION_EXPONENT_SMALLEST_VALUE to ACTIVATION_CARRIED_EXPONENT_LARGEST (below, a product of
 *     placed codes may leave the normal numbers inside the instruction); for bfloat16 rows where w is at most
 *     TRUNK_BFLOAT16_SCALE_LARGEST. There a block's term is the reference's (p times one scale, times the
 *     other, added in float32) wherever the reference's two multiplications stay normal numbers. Below the range
 *     of the codes form the block adds nothing, where the reference adds a number under 2^-120 in magnitude. Above
 *     a range p is multiplied by 2^127, a smaller power than the true one: such a block is not the reference's.
 *     In the carried form a block above the exponent bound has infinite operands, and every output it reaches is
 *     infinite or not-a-number, never a finite number. The block-scaled form has no range: its term is the exact
 *     sum of the block's products times both true scales, cut toward zero to a float32, a subnormal one included
 *     (the instruction's own rounding: batch i03a in the normal range, the exact card test at the ends of the
 *     scale codes), where the reference's is its two multiplications; beyond the largest float32 it is that number
 *     or an infinity, of the sum's sign, and an output it reaches through terms of that sign alone is infinite.
 *     In the other forms no code makes an exponent wrap or a scale infinite, and a code that weight_format_seam.h
 *     calls no number is read by its fields; the block-scaled instruction takes the scale code 0xFF as no number
 *     (a pack holds none, and no block of finite values has that exponent).
 *   - Rows that are not quantized (trunk_linear_bfloat16_value) become operands by the IEEE conversion of each
 *     bfloat16 to a 16-bit float, to nearest, a tie to even: exact for a magnitude from 2^-17 to 65280 and for
 *     zero; a smaller one is rounded to a multiple of 2^-24 (2^-25 or less: to zero, the sign kept); 65536 and
 *     above become infinity, and an output they reach is infinite or not-a-number, never a finite number.
 *   - The hidden step of an expert, in float32 in this order: g = min(gate, limit); u = up limited to -limit ..
 *     limit; h = bfloat16(((g / (1 + exp(-g))) * u) * route weight), exp being the card's expf. The fused kernel
 *     rounds gate and up to bfloat16 first, as the reference's linear returns them. A route weight of 1 changes
 *     no bit. limit is positive; +infinity where a model has none. A not-a-number stays one.
 *   - Launches (struct kernel_launch of backend_seam.h; grid[2], block[1] and block[2] are 1; <S> the instance):
 *       kernel                              grid[0]                          grid[1]   block[0]          shared bytes
 *       activation_quantize_entry         ceil(width / block[0])           vectors   a multiple of 32  0
 *       expert_gate_up_<S>          output_size / 16                 jobs      64 S              64 block[0]
 *       expert_down_<S>             output_size / 16                 jobs      32 S              64 block[0]
 *       expert_row_sum_entry              ceil(width / block[0])           rows      any               0
 *       trunk_linear_quantized_<S>  (output_size[0] + [1]) / 16      1         32 S              64 block[0]
 *       trunk_linear_bfloat16_<S>   output_size / 16                 1         32 S              64 block[0]
 *       trunk_hidden_entry                ceil(width / block[0])           vectors   any               0
 *     LINEAR_ELEMENT_THREADS is the block[0] meant where a thread takes one value; a thread beyond the
 *     width does nothing. Whatever grid[1] covers lies in the ranges below, also an empty job and a stale vector.
 *     The wide instances of the two trunk products (trunk_linear_quantized_wide_<S>,
 *     trunk_linear_bfloat16_wide_<S>) have the block and shared memory of their thin instance, and grid[0] its
 *     tiles times n, any n from 1 to the groups of 8 rows (row_count / 8 rounded up).
 *   - A wide trunk launch takes its thin instance's record with a row_count of up to LINEAR_WIDE_MOST_ROWS.
 *     Group g is rows 8 g .. 8 g + 7 (the last group fewer); of grid[0] = tiles times n, block b computes tile b / n
 *     for the groups b % n, b % n + n, .., one after the other, each exactly as the thin instance computes a launch
 *     of those rows alone. So every output has the thin instance's bits, whatever n. linear.c prepares the n of the
 *     matrix's wide form (linear.h): n = the groups, a tile's groups in adjacent blocks, which a card starts together
 *     (it hands blocks out in the order of their index), so its weights come from memory about once on every card;
 *     or n = 1, a block reading its tile again for each group, from the caches only where the tiles of the blocks a
 *     card holds at once fit its L2.
 *   - The interleaved instance (KERNEL_TRUNK_LINEAR_QUANTIZED_INTERLEAVED, quantized rows in one slice) takes
 *     the record, block and shared memory of trunk_linear_quantized_1 and the grid of its wide instance with
 *     chunks of LINEAR_INTERLEAVED_ROWS rows (4 groups) in place of groups: n from 1 to the chunks. A block
 *     works on each load of its weights for the 4 groups of a chunk before the next load, each group's chains its
 *     own: every chain is formed by the same instructions on the same operands in the same order as in the thin
 *     instance, so every output has its bits. It runs in place of a wide launch of one slice, record unchanged, where
 *     the matrix's wide form says so (linear.h).
 *   - A record has no padding and at most 80 of a launch's LAUNCH_ARGUMENT_BYTES: asserted below, C11 and
 *     CUDA.
 */
#ifndef LINEAR_RECORDS_H
#define LINEAR_RECORDS_H

#include <stdint.h>

#define LINEAR_MAXIMUM_ROWS \
	8u /* rows of one thin trunk launch (a group of a wide one); columns of
                                                     one job */
#define LINEAR_WIDE_MOST_ROWS \
	1024u /* rows of one wide trunk launch: a wide step's chunk */
#define LINEAR_INTERLEAVED_ROWS \
	32u /* a chunk of the interleaved instance: 4 groups of 8 rows */
#define LINEAR_OUTPUTS_PER_TILE \
	16u /* the outputs one block of a product owns */
#define LINEAR_VALUES_PER_BLOCK \
	32u /* values of a vector that share one scale exponent */
#define LINEAR_VALUES_PER_LOAD \
	64u /* what a width is a multiple of; the unit of a slice */
#define LINEAR_THREADS_PER_WARP 32u
#define LINEAR_MAXIMUM_SLICES \
	16u /* gate with up is then a block of 1,024 threads */
/* LINEAR_SLICE_STAMP(S) for each count S a product kernel is stamped for, ascending: the one list of them.
 * Whoever expands it defines LINEAR_SLICE_STAMP first and undefines it after. */
#define LINEAR_EVERY_SLICE_COUNT                                          \
	LINEAR_SLICE_STAMP(1)                                             \
	LINEAR_SLICE_STAMP(2) LINEAR_SLICE_STAMP(4) LINEAR_SLICE_STAMP(8) \
		LINEAR_SLICE_STAMP(16)
#define LINEAR_SLICE_INSTANCES 5u /* how many they are: linear.c asserts it */
#define LINEAR_MERGED_MATRICES 2u /* the most matrices one trunk launch reads */
#define EXPERT_GATE_UP_MATRICES \
	2u /* gate and up share a block: its block[0] is this times 32 S */
#define LINEAR_SHARED_BYTES_PER_THREAD \
	64u /* of a product: its dynamic shared memory is block[0] times it */
/* The exact profile on class 120 emulates class 80's tensor-core sums, whose registers fit blocks of 256 threads: its
 * products take at most these slices (blocks of 32 or, for gate with up, 64 threads a slice). */
#define LINEAR_CLASS_120_EXACT_MOST_SLICES 8u
#define LINEAR_CLASS_120_EXACT_MOST_GATE_UP_SLICES 4u
#define LINEAR_ELEMENT_THREADS 256u
#define LINEAR_OFFSET_ALIGNMENT_BYTES 16u
#define LINEAR_BFLOAT16_BYTES 2u /* of a value */
#define LINEAR_SUM_BYTES 4u /* of a float32 */
#define ACTIVATION_EXPONENT_SMALLEST_VALUE \
	105u /* 2^-22: what the floor of 1e-4 gives */
#define ACTIVATION_EXPONENT_LARGEST \
	247u /* 2^120: what the largest finite bfloat16 gives */
#define EXPERT_SCALE_SUM_SMALLEST 106u
#define EXPERT_SCALE_SUM_LARGEST 359u
#define TRUNK_SCALE_SUM_SMALLEST 112u
#define TRUNK_SCALE_SUM_LARGEST 365u
#define TRUNK_BFLOAT16_SCALE_LARGEST 246u
#define ACTIVATION_CARRIED_EXPONENT_LARGEST \
	146u /* largest magnitude 448 * 2^19: operands of 1.75 * 2^127 */
#define EXPERT_CARRIED_SCALE_LARGEST 228u
#define TRUNK_CARRIED_SCALE_LARGEST 234u
/* The names kernel_prepare takes: a kernel file without directory and suffix, and the functions in it. A product
 * kernel has one function for each count of LINEAR_EVERY_SLICE_COUNT: its stem, then the count. */
#define KERNEL_FILE_ACTIVATION_QUANTIZE "kernel_activation_quantize"
#define KERNEL_FILE_EXPERT_LINEAR "kernel_expert_linear"
#define KERNEL_FILE_TRUNK_LINEAR "kernel_trunk_linear"
#define KERNEL_FILE_EXPERT_LINEAR_NATIVE \
	"kernel_expert_linear_native" /* class 120 native */
#define KERNEL_FILE_TRUNK_LINEAR_NATIVE "kernel_trunk_linear_native"
#define KERNEL_ACTIVATION_QUANTIZE "activation_quantize_entry"
/* draft_bf16_operands (track L): the same record; the codes and exponents as activation_quantize_entry writes them,
 * the operands the values' own bfloat16 times 2^100 (the carried form's reads unrounded, class 80) */
#define KERNEL_ACTIVATION_CARRY "activation_carry"
#define KERNEL_EXPERT_GATE_UP_STEM "expert_gate_up_"
#define KERNEL_EXPERT_DOWN_STEM "expert_down_"
#define KERNEL_EXPERT_ROW_SUM "expert_row_sum_entry"
#define KERNEL_TRUNK_LINEAR_QUANTIZED_STEM "trunk_linear_quantized_"
#define KERNEL_TRUNK_LINEAR_BFLOAT16_STEM "trunk_linear_bfloat16_"
#define KERNEL_TRUNK_LINEAR_QUANTIZED_WIDE_STEM "trunk_linear_quantized_wide_"
#define KERNEL_TRUNK_LINEAR_BFLOAT16_WIDE_STEM "trunk_linear_bfloat16_wide_"
#define KERNEL_TRUNK_LINEAR_QUANTIZED_INTERLEAVED \
	"trunk_linear_quantized_interleaved_1"
#define KERNEL_TRUNK_HIDDEN_VALUE "trunk_hidden_entry"
#define ACTIVATION_QUANTIZE_ARGUMENT_BYTES \
	40u /* the bytes of each record, and of an entry of the job table */
#define EXPERT_JOB_BYTES 96u
#define EXPERT_GATE_UP_ARGUMENT_BYTES 64u
#define EXPERT_DOWN_ARGUMENT_BYTES 56u
#define EXPERT_ROW_SUM_ARGUMENT_BYTES 24u
#define TRUNK_LINEAR_QUANTIZED_ARGUMENT_BYTES 72u
#define TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES 40u
#define TRUNK_HIDDEN_ARGUMENT_BYTES 32u

/* kernel_activation_quantize, activation_quantize_entry. With V = grid[1] vectors:
 *   reads   values_offset, 2 V width bytes: values [V][width];
 *   writes  codes_offset, V width bytes: codes [V][width]; exponents_offset, V width / 32 bytes; operands_offset,
 *           2 V width bytes: operands [V][width]. */
struct activation_quantize_arguments {
	uint64_t values_offset;
	uint64_t codes_offset;
	uint64_t exponents_offset;
	uint64_t operands_offset;
	uint32_t width;
	uint32_t reserved_zero;
};

/* One routed expert with every row of the batch that is routed to it: an entry of the job table of the two expert
 * kernels, which a launch reads at jobs_offset, 96 grid[1] bytes. The rows are the columns of the tensor-core
 * instruction, so the expert is read once whatever their number.
 *   - row_count is 0 to LINEAR_MAXIMUM_ROWS. A job of 0 rows is empty: nothing else of it is read, and
 *     its blocks read and write nothing more. A launch may so keep one grid for every number of routed experts.
 *   - Column c below row_count carries the pair pair_index[c] = row * pairs_per_row + slot, slot counting the
 *     routed experts of that row in ascending expert order, and its route weight. Every pair_index in use is below
 *     the launch's pair_count, and no pair stands in two columns of one launch. The entries from row_count on are
 *     not read; zero keeps a table's bytes the same in every run.
 *   - The three offsets are the expert's matrices, each stored as WEIGHT_FORMAT_E2M1_SCALED: gate and up
 *     of output_size by input_size of the gate-up record, down of the down record's. The gate-up kernel reads
 *     the first two whole (output_size * input_size / 2 + output_size * input_size / 32 bytes), down the third.
 *   - expert: the routed expert's index (the routers write it; the tensor-core MoE's mixed widths read it,
 *     exl3_tc_records.h exl3_tc_width; the other kernels do not). */
struct expert_job_record {
	uint64_t gate_offset;
	uint64_t up_offset;
	uint64_t down_offset;
	uint32_t row_count;
	uint32_t expert;
	uint32_t pair_index[LINEAR_MAXIMUM_ROWS];
	float route_weight[LINEAR_MAXIMUM_ROWS];
};

/* kernel_expert_linear, expert_gate_up_<S>: gate and up of every job from the rows' quantized input, and the
 * hidden step. With R rows (pair_count = R * pairs_per_row):
 *   reads   the jobs and their gate and up matrices; codes_offset, R input_size bytes: codes [R][input_size];
 *           exponents_offset, R input_size / 32 bytes; operands_offset, 2 R input_size bytes: operands
 *           [R][input_size] (each form reads what it takes of the three);
 *   writes  hidden_offset: values [pair][output_size], only of the pairs that stand in a job: 2 output_size bytes
 *           at 2 pair output_size each.
 * pairs_per_row is at least 1. activation_limit is the limit of the hidden step. */
struct expert_gate_up_arguments {
	uint64_t jobs_offset;
	uint64_t codes_offset;
	uint64_t exponents_offset;
	uint64_t operands_offset;
	uint64_t hidden_offset;
	uint32_t output_size;
	uint32_t input_size;
	uint32_t pair_count;
	uint32_t pairs_per_row;
	float activation_limit;
	uint32_t reserved_zero;
};

/* kernel_expert_linear, expert_down_<S>: the down product of every pair that stands in a job, from the pair's
 * quantized hidden vector.
 *   reads   the jobs and their down matrices; codes_offset: codes [pair][input_size], exponents_offset:
 *           exponents [pair][input_size / 32] and operands_offset: operands [pair][input_size], only of the pairs
 *           that stand in a job;
 *   writes  output_offset: values [pair][output_size], only of those pairs. */
struct expert_down_arguments {
	uint64_t jobs_offset;
	uint64_t codes_offset;
	uint64_t exponents_offset;
	uint64_t operands_offset;
	uint64_t output_offset;
	uint32_t output_size;
	uint32_t input_size;
	uint32_t pair_count;
	uint32_t reserved_zero;
};

/* kernel_expert_linear, expert_row_sum_entry: for each of the R = grid[1] rows the float32 sum of its
 * pairs_per_row pair outputs, from +0 in ascending slot order, which is ascending expert order.
 *   reads   pair_output_offset, 2 R pairs_per_row width bytes: values [R * pairs_per_row][width], whoever wrote
 *           them (the down kernel, or an upload of a pair computed elsewhere);
 *   writes  row_output_offset, 4 R width bytes: sums [R][width]. */
struct expert_row_sum_arguments {
	uint64_t pair_output_offset;
	uint64_t row_output_offset;
	uint32_t width;
	uint32_t pairs_per_row;
};

/* kernel_trunk_linear, trunk_linear_quantized_<S>: y = W x for one or two matrices stored as
 * WEIGHT_FORMAT_E4M3_SCALED, of one input_size, and rows 0 .. row_count - 1 of quantized vectors; row_count
 * is 1 to LINEAR_MAXIMUM_ROWS (trunk_linear_quantized_wide_<S>: to LINEAR_WIDE_MOST_ROWS).
 * Matrix m takes the tiles from the first one's on; output_size[1] is 0 where
 * there is one matrix, and its offsets are then not read. A tile of either has the bits it has in a launch of its
 * matrix alone.
 *   reads   weights_offset[m], the matrix: output_size[m] * input_size + ((output_size[m] + 31) / 32) *
 *           (input_size / 32) bytes; codes_offset, row_count input_size bytes; exponents_offset, row_count
 *           input_size / 32 bytes; operands_offset, 2 row_count input_size bytes;
 *   writes  output_offset[m], 2 row_count output_size[m] bytes: values [row_count][output_size[m]]. */
struct trunk_linear_quantized_arguments {
	uint64_t weights_offset[LINEAR_MERGED_MATRICES];
	uint64_t output_offset[LINEAR_MERGED_MATRICES];
	uint64_t codes_offset;
	uint64_t exponents_offset;
	uint64_t operands_offset;
	uint32_t output_size[LINEAR_MERGED_MATRICES];
	uint32_t input_size;
	uint32_t row_count;
};

/* kernel_trunk_linear, trunk_linear_bfloat16_<S>: the same product for rows that are not quantized, row_count
 * as above (trunk_linear_bfloat16_wide_<S>: to LINEAR_WIDE_MOST_ROWS). The
 * outputs are cut into segment_count equal runs, each a whole number of tiles, and run i multiplies the values
 * i * input_size .. (i + 1) * input_size - 1 of a row: 1 for an ordinary matrix, the number of groups for one that
 * is block-diagonal over groups.
 *   reads   weights_offset, the matrix as above; values_offset, 2 row_count segment_count input_size bytes:
 *           values [row_count][segment_count * input_size];
 *   writes  output_offset, 2 row_count output_size bytes: values [row_count][output_size]. */
struct trunk_linear_bfloat16_arguments {
	uint64_t weights_offset;
	uint64_t values_offset;
	uint64_t output_offset;
	uint32_t output_size;
	uint32_t input_size;
	uint32_t row_count;
	uint32_t segment_count;
};

/* kernel_trunk_linear, trunk_hidden_entry: the hidden step alone, with a route weight of 1, for an expert whose
 * gate and up are a trunk launch of two matrices (the shared expert). With V = grid[1] vectors:
 *   reads   gate_offset and up_offset, 2 V width bytes each: values [V][width];
 *   writes  hidden_offset, 2 V width bytes: values [V][width]. */
struct trunk_hidden_arguments {
	uint64_t gate_offset;
	uint64_t up_offset;
	uint64_t hidden_offset;
	uint32_t width;
	float activation_limit;
};

#ifdef __cplusplus
#define LINEAR_RECORD_HAS(record, byte_count) \
	static_assert(sizeof(struct record) == (byte_count), #record)
#else
#define LINEAR_RECORD_HAS(record, byte_count) \
	_Static_assert(sizeof(struct record) == (byte_count), #record)
#endif
LINEAR_RECORD_HAS(activation_quantize_arguments,
		  ACTIVATION_QUANTIZE_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(expert_job_record, EXPERT_JOB_BYTES);
LINEAR_RECORD_HAS(expert_gate_up_arguments, EXPERT_GATE_UP_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(expert_down_arguments, EXPERT_DOWN_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(expert_row_sum_arguments, EXPERT_ROW_SUM_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(trunk_linear_quantized_arguments,
		  TRUNK_LINEAR_QUANTIZED_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(trunk_linear_bfloat16_arguments,
		  TRUNK_LINEAR_BFLOAT16_ARGUMENT_BYTES);
LINEAR_RECORD_HAS(trunk_hidden_arguments, TRUNK_HIDDEN_ARGUMENT_BYTES);

#endif
