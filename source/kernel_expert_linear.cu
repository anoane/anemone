/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_expert_linear.cu: the routed experts of one layer for the rows of one batch: gate and up with the hidden
 * step, down, and the sum of a row's experts. Device code only, built once per architecture into the module
 * "kernel_expert_linear". Between gate-up and down the hidden vectors are quantized by activation_quantize_entry.
 *
 * Contract: include/anemone/linear_records.h (the records, the job table, the arrays, the order of additions, the
 * hidden step, the launches); kernel_linear_common.cuh (the instruction and its operands). Here:
 *   - A block is one tile of 16 outputs of one job. The weights are operand A of the instruction and the job's
 *     rows are the columns of operand B: column c carries the vector of pair_index[c] (for gate and up the
 *     quantized input of that pair's row, for down the pair's own quantized hidden vector). A column from
 *     row_count on carries the vector of column 0 again and its results are stored nowhere. An expert is so read
 *     once whatever the number of its rows.
 *   - A warp is one slice of the input width of one matrix. Gate and up share a block: its first half of warps
 *     are the slices of gate, the second half those of up. Every warp leaves four chains per lane in the block's
 *     shared memory; after the block's one barrier, warp w finishes the results w, w + warps, .. of every lane.
 *   - A ROW KEEPS ITS BITS whatever rows share its launch and whichever column it takes, by construction: one
 *     result is one output of one column, and what is added for it, in this order, is fixed by the matrix, the
 *     vector and the slice count alone: inside an instruction the 16 products of that output and that column;
 *     the two instructions of a block of 32 inputs, the second on the sum of the first; one fused multiply-add
 *     per block, in ascending input order from +0 within a slice; the slices in ascending order from +0; one
 *     rounding to bfloat16. No sum crosses a column, no loop runs over the rows of a job, and no thread adds
 *     anything to memory another thread adds to. The row sum adds a row's pairs in slot order from +0.
 *   - No atomic operation. A block passes the barriers of its ordered slices (more than one slice: see
 *     linear_ordered_chains) and one before its results, which every thread of the block reaches or none does (a
 *     block of an empty job leaves before them, as a whole).
 *   - The form (linear_records.h): carried on class 80, block-scaled on class 120.
 *   - The producer (expert_fused_records.h) is gate with up, block for block, and then, in the block of a pair of
 *     tiles that arrives second, the quantization of the pair's 32 hidden values of each column by the quantizer's
 *     own warp function (activation_quantize_block of kernel_activation_quantize.cuh: a warp a column, lane j value
 *     j), from the values both blocks stored. The blocks of its last grid row quantize the pairs of no job alike.
 */
#include "kernel_activation_quantize.cuh"

#include "anemone/expert_fused_records.h"

#define GATE_UP_MATRICES EXPERT_GATE_UP_MATRICES
#define PRODUCER_TILES EXPERT_PRODUCER_TILES
/* The launch bounds of the instances of 4 slices: their block, and the blocks of an SM that batches i02 (class 80,
 * gate with up) and i03b (class 120) measured. Class 80's 12 blocks for down, timed only beside gate and up's 6,
 * made down of 4 slices the slowest count in batch i01_r2: there it states its block alone. The other instances
 * state no bound: a bound moves the registers the compiler takes, and none other was timed (the build checks that
 * the registers of every instance fit its block). The exact profile on class 120 (its thread sums): 2 blocks of gate
 * with up (119 registers; 1 took 130, 18% slower at 1 and 8 rows) and 4 of down (2% faster than 2), 2026-10-25.
 * Class 80's gate with up now states its block alone too, not i02's 6 blocks: its experts run one slice (25 October),
 * and the ordered slices of round 2b take about 53 registers in the checked build. */
#define EXPERT_BOUNDS_1(threads, blocks)
#define EXPERT_BOUNDS_2(threads, blocks)
#define EXPERT_BOUNDS_4(threads, blocks) __launch_bounds__(threads, blocks)
#define EXPERT_BOUNDS_8(threads, blocks)
#define EXPERT_BOUNDS_16(threads, blocks)
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) && defined(LINEAR_CLASS_FORMS)
#define GATE_UP_BLOCKS_MEASURED 1
#define DOWN_BLOCKS_MEASURED 2
#elif defined(__CUDA_ARCH_FEAT_SM120_ALL)
#define GATE_UP_BLOCKS_MEASURED 2
#define DOWN_BLOCKS_MEASURED 4
#else
#define GATE_UP_BLOCKS_MEASURED 1
#define DOWN_BLOCKS_MEASURED 1
#endif

/* The job of a block: entry blockIdx.y of the job table. */
__device__ static __forceinline__ const struct expert_job_record *
expert_job_of_block(const unsigned char *jobs)
{
	return (const struct expert_job_record *)jobs + blockIdx.y;
}

/* The pair column `column` of a job carries: a column that is not in use carries the pair of column 0. */
__device__ static __forceinline__ uint32_t
expert_column_pair(const struct expert_job_record *job, uint32_t column,
		   uint32_t pair_count)
{
    const uint32_t pair = job->pair_index[column < job->row_count ? column : 0];
    LINEAR_DEVICE_CHECK(pair < pair_count);
    return pair;
}

/* The chains of one warp of a block: one slice of one E2M1 matrix of the block's job for the block's tile, fed
 * with quantized vectors. vector_divisor makes of a pair the vector it reads: pairs_per_row for gate and up (the
 * row of the pair), 1 for down (the pair). chains is the warp's part of the block's shared memory; scratch the
 * matrix's past every warp's part (linear_thread_chains: the exact profile on class 120). */
template <uint32_t slice_count>
__device__ __forceinline__ void
expert_slice(const struct expert_job_record *job, const unsigned char *matrix,
	     uint32_t output_size, uint32_t input_size,
	     const unsigned char *codes, const unsigned char *exponents,
	     const unsigned char *operands, uint32_t vector_divisor,
	     uint32_t pair_count, uint32_t slice, float *chains, float *scratch,
	     uint32_t *fed_cache)
{
    const uint32_t thread = threadIdx.x, lane_index = thread % THREADS_PER_WARP;
#if LINEAR_THREAD_SUMS
    (void)codes;
    uint64_t low_vectors = 0, high_vectors = 0; /* 16 bits a column: its vector (linear_thread_column_vector) */
    for (uint32_t column = 0; column < job->row_count; column++) {
        const uint64_t vector = (uint64_t)(expert_column_pair(job, column, pair_count) / vector_divisor)
                                << (column % LINEAR_THREAD_COLUMNS_A_WORD * LINEAR_THREAD_COLUMN_INDEX_BITS);
        low_vectors |= column < LINEAR_THREAD_COLUMNS_A_WORD ? vector : 0u;
        high_vectors |= column < LINEAR_THREAD_COLUMNS_A_WORD ? 0u : vector;
    }
    linear_thread_chains<LINEAR_WEIGHTS_E2M1, LINEAR_EXPERT_FORM>(
	    matrix, output_size, input_size, blockIdx.x, operands,
	    (size_t)input_size * LINEAR_BFLOAT16_BYTES, exponents,
	    input_size / VALUES_PER_BLOCK, low_vectors, high_vectors,
	    job->row_count, slice * THREADS_PER_WARP + lane_index,
	    slice_count * THREADS_PER_WARP, scratch, fed_cache,
	    chains - (size_t)slice * THREADS_PER_WARP * ELEMENTS_PER_LANE);
#else
    (void)scratch;
    (void)fed_cache;
    const uint32_t lane_group = lane_index / LANES_PER_LANE_GROUP, lane_place = lane_index % LANES_PER_LANE_GROUP;
    const uint32_t load_count = input_size / VALUES_PER_LOAD;
    struct linear_lane lane;
    lane.column_count = job->row_count;
    linear_lane_weights<LINEAR_WEIGHTS_E2M1>(&lane, matrix, output_size, input_size, blockIdx.x, lane_index);
    linear_lane_quantized<LINEAR_EXPERT_FORM>(
        &lane, codes, exponents, operands, input_size, lane_place,
        expert_column_pair(job, lane_group, pair_count) / vector_divisor,
        expert_column_pair(job, lane_place * COLUMNS_PER_LANE, pair_count) / vector_divisor,
        expert_column_pair(job, lane_place * COLUMNS_PER_LANE + 1u, pair_count) / vector_divisor);
    LINEAR_CHAINS(LINEAR_WEIGHTS_E2M1, LINEAR_EXPERT_FORM, slice, load_count,
                  chains + (size_t)lane_index * ELEMENTS_PER_LANE);
#endif
}

/* The scratch of matrix `matrix` of a block of block_threads threads, matrix_threads to a matrix: past the slots of
 * every warp (none outside the exact profile on class 120). */
__device__ static __forceinline__ float *expert_scratch(float *slice_chains, uint32_t block_threads,
                                                        uint32_t matrix_threads, uint32_t matrix)
{
#if LINEAR_THREAD_SUMS
    return slice_chains + (size_t)block_threads * ELEMENTS_PER_LANE +
           (size_t)matrix * matrix_threads * LINEAR_THREAD_SCRATCH_FLOATS;
#else
    (void)block_threads;
    (void)matrix_threads;
    (void)matrix;
    return slice_chains;
#endif
}

/* The round cache of matrix `matrix`, matrix_threads to a matrix (NULL outside the exact profile on class 120). */
__device__ static __forceinline__ uint32_t *expert_cache(uint32_t *caches, uint32_t matrix_threads, uint32_t matrix)
{
#if LINEAR_THREAD_SUMS
    return caches + (size_t)matrix * matrix_threads * LINEAR_THREAD_CACHE_WORDS;
#else
    (void)matrix_threads;
    (void)matrix;
    return caches;
#endif
}

/* grid (output_size / 16, jobs), block of 64 S threads, 64 bytes of shared memory per thread. */
template <uint32_t slice_count>
__device__ __forceinline__ void
expert_gate_up(struct expert_gate_up_arguments arguments, unsigned char *arena)
{
    extern __shared__ float slice_chains[];
    const struct expert_job_record *job =
	    expert_job_of_block(arena + arguments.jobs_offset);
    const uint32_t warp = threadIdx.x / THREADS_PER_WARP, lane_index = threadIdx.x % THREADS_PER_WARP;
    const uint32_t warp_count = slice_count * GATE_UP_MATRICES;
    LINEAR_DEVICE_CHECK(blockDim.x == warp_count * THREADS_PER_WARP &&
			job->row_count <= LINEAR_MAXIMUM_ROWS);
    if (job->row_count == 0) {
        return;
    }
    LINEAR_THREAD_CACHE(fed_cache, GATE_UP_MATRICES * slice_count * THREADS_PER_WARP);
    expert_slice<slice_count>(job, arena + (warp < slice_count ? job->gate_offset : job->up_offset),
                              arguments.output_size, arguments.input_size, arena + arguments.codes_offset,
                              arena + arguments.exponents_offset, arena + arguments.operands_offset,
                              arguments.pairs_per_row, arguments.pair_count, warp % slice_count,
                              slice_chains + (size_t)warp * THREADS_PER_WARP * ELEMENTS_PER_LANE,
                              expert_scratch(slice_chains, blockDim.x, slice_count * THREADS_PER_WARP,
                                             warp < slice_count ? 0u : 1u),
                              expert_cache(fed_cache, slice_count * THREADS_PER_WARP, warp < slice_count ? 0u : 1u));
    __syncthreads();
    for (uint32_t element = warp; element < ELEMENTS_PER_LANE; element += warp_count) {
        const uint32_t column = lane_index % LANES_PER_LANE_GROUP * COLUMNS_PER_LANE + element % COLUMNS_PER_LANE;
        const uint32_t output = blockIdx.x * OUTPUTS_PER_TILE + lane_index / LANES_PER_LANE_GROUP +
                                element / COLUMNS_PER_LANE * OUTPUTS_PER_HALF_TILE;
        if (column < job->row_count) {
            /* each product rounded to bfloat16, as the reference's linear returns it */
            const float gate = linear_bfloat16_value(
                linear_bfloat16_bits(linear_sum_of_slices<slice_count>(slice_chains, 0, lane_index, element)));
            const float up = linear_bfloat16_value(linear_bfloat16_bits(
                linear_sum_of_slices<slice_count>(slice_chains, slice_count, lane_index, element)));
            uint16_t *hidden = (uint16_t *)(arena + arguments.hidden_offset) +
                               (size_t)job->pair_index[column] * arguments.output_size;
            hidden[output] =
                (uint16_t)linear_hidden_bits(gate, up, arguments.activation_limit, job->route_weight[column]);
        }
    }
}

/* grid (output_size / 16, jobs), block of 32 S threads, 64 bytes of shared memory per thread. */
template <uint32_t slice_count>
__device__ __forceinline__ void
expert_down(struct expert_down_arguments arguments, unsigned char *arena)
{
    extern __shared__ float slice_chains[];
    const struct expert_job_record *job =
	    expert_job_of_block(arena + arguments.jobs_offset);
    const uint32_t warp = threadIdx.x / THREADS_PER_WARP, lane_index = threadIdx.x % THREADS_PER_WARP;
    LINEAR_DEVICE_CHECK(blockDim.x == slice_count * THREADS_PER_WARP &&
			job->row_count <= LINEAR_MAXIMUM_ROWS);
    if (job->row_count == 0) {
        return;
    }
    LINEAR_THREAD_CACHE(fed_cache, slice_count * THREADS_PER_WARP);
    expert_slice<slice_count>(job, arena + job->down_offset, arguments.output_size, arguments.input_size,
                              arena + arguments.codes_offset, arena + arguments.exponents_offset,
                              arena + arguments.operands_offset, 1u, arguments.pair_count, warp,
                              slice_chains + (size_t)warp * THREADS_PER_WARP * ELEMENTS_PER_LANE,
                              expert_scratch(slice_chains, blockDim.x, blockDim.x, 0u), fed_cache);
    __syncthreads();
    for (uint32_t element = warp; element < ELEMENTS_PER_LANE; element += slice_count) {
        const uint32_t column = lane_index % LANES_PER_LANE_GROUP * COLUMNS_PER_LANE + element % COLUMNS_PER_LANE;
        const uint32_t output = blockIdx.x * OUTPUTS_PER_TILE + lane_index / LANES_PER_LANE_GROUP +
                                element / COLUMNS_PER_LANE * OUTPUTS_PER_HALF_TILE;
        if (column < job->row_count) {
            uint16_t *pair_output = (uint16_t *)(arena + arguments.output_offset) +
                                    (size_t)job->pair_index[column] * arguments.output_size;
            pair_output[output] = (uint16_t)linear_bfloat16_bits(
                linear_sum_of_slices<slice_count>(slice_chains, 0, lane_index, element));
        }
    }
}

/* One arrival at a counter: its value before. The addition of 1 is a release of everything the block stored before the
 * barrier that precedes it (at the scope of the card); the second arrival, which finds 1, then acquires what the first
 * released by a fence (its strong read and the fence: an acquire pattern). The first takes no acquire: an acquire
 * empties the L1 of its multiprocessor (CCTL.IVALL), which the other blocks there would pay for. */
__device__ static __forceinline__ uint32_t producer_arrive(uint32_t *counter)
{
    uint32_t before = 0;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;\n" : "=r"(before) : "l"(counter) : "memory");
    if (before != 0) {
        asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
    }
    return before;
}

/* The pairs of the job table's columns in use, marked in is_routed (shared memory, a byte a pair) by a block of the
 * producer's last grid row. Every thread of the block passes its two barriers. */
__device__ static __forceinline__ void
producer_mark_routed(const struct expert_gate_up_quantize_arguments *arguments,
		     const unsigned char *arena, unsigned char *is_routed)
{
	const struct expert_job_record *const jobs =
		(const struct expert_job_record *)(arena +
						   arguments->jobs_offset);
	const uint32_t pair_count = arguments->pair_count;
	const uint32_t entry_count =
		blockIdx.y * LINEAR_MAXIMUM_ROWS; /* the columns of every job */
	LINEAR_DEVICE_CHECK(pair_count <= EXPERT_PRODUCER_MOST_PAIRS);
	for (uint32_t pair = threadIdx.x; pair < pair_count;
	     pair += blockDim.x) {
		is_routed[pair] = 0;
	}
    __syncthreads();
    for (uint32_t entry = threadIdx.x; entry < entry_count; entry += blockDim.x) {
	    const struct expert_job_record *const job =
		    &jobs[entry / LINEAR_MAXIMUM_ROWS];
	    const uint32_t column = entry % LINEAR_MAXIMUM_ROWS;
	    if (column < job->row_count &&
		job->pair_index[column] < pair_count) {
		    is_routed[job->pair_index[column]] = 1;
	    }
    }
    __syncthreads();
}

/* The producer (grid (output_size / 16, jobs + 1), block of 64 S threads, 64 bytes of shared memory per thread). Grid
 * rows 0 .. jobs - 1: gate with up as expert_gate_up_<S> computes and stores it, block for block; then the
 * arrival of the block at its pair's counter (block x is of pair x / 2: one block of 32 values of the quantization):
 * the block meets at a barrier and its thread 0 arrives (the second block puts the counter back to 0); behind a second
 * barrier the block that arrived second quantizes the pair's 32 values of each column. Block x of the last grid row (x
 * below output_size / 32) quantizes block x of each pair that stands in no job (nothing else of the launch writes or
 * reads one). Both read the values past their L1 (a pair of a job's where both blocks stored them) and quantize them in
 * the one loop below, a warp a pair, lane j value j. */
template <uint32_t slice_count, bool packed_wire = false>
__device__ __forceinline__ void
expert_gate_up_quantize(struct expert_gate_up_quantize_arguments arguments,
			unsigned char *arena)
{
	__shared__ unsigned char is_routed[EXPERT_PRODUCER_MOST_PAIRS];
	__shared__ uint32_t arrived_before;
	const struct expert_job_record *job = expert_job_of_block(
		arena + arguments.jobs_offset); /* rows 0 .. jobs - 1 */
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP,
		       lane_index = threadIdx.x % THREADS_PER_WARP;
	const uint32_t block_count = arguments.output_size / VALUES_PER_BLOCK;
	const bool is_orphan_row =
		blockIdx.y == gridDim.y - 1u; /* the same for the whole block */
	const uint32_t block_index =
		is_orphan_row ? blockIdx.x : blockIdx.x / PRODUCER_TILES;
	uint32_t item_count = 0;
	if (is_orphan_row) {
		if (block_index >= block_count) {
			return;
		}
		producer_mark_routed(&arguments, arena, is_routed);
		item_count = arguments.pair_count;
	} else {
		const struct expert_gate_up_arguments gate_up = {
			arguments.jobs_offset,
			arguments.codes_offset,
			arguments.exponents_offset,
			arguments.operands_offset,
			arguments.hidden_offset,
			arguments.output_size,
			arguments.input_size,
			arguments.pair_count,
			arguments.pairs_per_row,
			arguments.activation_limit,
			0
		};
		expert_gate_up<slice_count>(gate_up, arena);
		if (job->row_count == 0) {
			return;
		}
		__syncthreads();
		if (threadIdx.x == 0) {
			uint32_t *const arrivals =
				(uint32_t *)(arena +
					     arguments.arrivals_offset) +
				(size_t)blockIdx.y * block_count + block_index;
			arrived_before = producer_arrive(arrivals);
			if (arrived_before != 0) {
				*arrivals = 0;
			}
		}
		__syncthreads();
		if (arrived_before == 0) {
			return;
		}
		item_count = job->row_count;
	}
    for (uint32_t item = warp; item < item_count; item += blockDim.x / THREADS_PER_WARP) {
        const size_t pair = is_orphan_row ? item : job->pair_index[item], pair_values = pair * arguments.output_size;
        if (!is_orphan_row || is_routed[pair] == 0) { /* the same for the whole warp */
            const unsigned short *const pair_hidden =
                (const unsigned short *)(arena + arguments.hidden_offset) + pair_values;
            unsigned char *const codes = arena + arguments.hidden_codes_offset +
                pair * (packed_wire ? arguments.reserved_zero : arguments.output_size);
            unsigned char *const exponents = packed_wire ? codes + arguments.output_size :
                arena + arguments.hidden_exponents_offset + pair * block_count;
            activation_quantize_block(__ldcg(pair_hidden + (size_t)block_index * VALUES_PER_BLOCK + lane_index),
                                      block_index, lane_index, codes, exponents,
                                      (uint16_t *)(arena + arguments.hidden_operands_offset) + pair_values);
            if (packed_wire && block_index + 1u == block_count)
                for (uint32_t padding = block_count + lane_index;
                     padding < arguments.reserved_zero - arguments.output_size; padding += THREADS_PER_WARP)
                    codes[arguments.output_size + padding] = 0;
        }
    }
}

#define LINEAR_SLICE_STAMP(slices)                                             \
	extern "C" __global__ void EXPERT_BOUNDS_##slices(                     \
		slices * GATE_UP_MATRICES * THREADS_PER_WARP,                  \
		GATE_UP_BLOCKS_MEASURED)                                       \
		expert_gate_up_##slices(                                       \
			struct expert_gate_up_arguments arguments,             \
			unsigned char *arena)                                  \
	{                                                                      \
		LINEAR_SLICED_BODY(slices,                                     \
				   LINEAR_CLASS_120_EXACT_MOST_GATE_UP_SLICES, \
				   expert_gate_up<slices>(arguments, arena))   \
	}                                                                      \
	extern "C" __global__ void EXPERT_BOUNDS_##slices(                     \
		slices *THREADS_PER_WARP, DOWN_BLOCKS_MEASURED)                \
		expert_down_##slices(struct expert_down_arguments arguments,   \
				     unsigned char *arena)                     \
	{                                                                      \
		LINEAR_SLICED_BODY(slices, LINEAR_CLASS_120_EXACT_MOST_SLICES, \
				   expert_down<slices>(arguments, arena))      \
	}
LINEAR_EVERY_SLICE_COUNT
#undef LINEAR_SLICE_STAMP

/* The producer's instances state the bounds of gate with up's: its blocks are theirs (but those of its last row). */
#define LINEAR_SLICE_STAMP(slices)                                          \
	extern "C" __global__ void EXPERT_BOUNDS_##slices(                  \
		slices * GATE_UP_MATRICES * THREADS_PER_WARP,               \
		GATE_UP_BLOCKS_MEASURED)                                    \
		expert_gate_up_quantize_##slices(                           \
			struct expert_gate_up_quantize_arguments arguments, \
			unsigned char *arena)                               \
	{                                                                   \
		LINEAR_SLICED_BODY(                                         \
			slices, LINEAR_CLASS_120_EXACT_MOST_GATE_UP_SLICES, \
			expert_gate_up_quantize<slices>(arguments, arena))  \
	}
LINEAR_EVERY_SLICE_COUNT
#undef LINEAR_SLICE_STAMP

/* grid (ceil(width / blockDim.x), rows): a thread takes one output of one row. */
extern "C" __global__ void
expert_row_sum_entry(struct expert_row_sum_arguments arguments,
		     unsigned char *arena)
{
    const size_t column = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y;
    if (column >= arguments.width) {
        return;
    }
    const uint16_t *pair_outputs =
        (const uint16_t *)(arena + arguments.pair_output_offset) + row * arguments.pairs_per_row * arguments.width;
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < arguments.pairs_per_row; slot++) {
        sum += linear_bfloat16_value(pair_outputs[(size_t)slot * arguments.width + column]);
    }
    ((float *)(arena + arguments.row_output_offset))[row * arguments.width + column] = sum;
}

/* Exact transport producer: reserved_zero is the prepared byte pitch of one code/scale wire row.
 * Codes retain their original lane order; exponents follow the codes, with immutable zero padding. */
extern "C" __global__ void
expert_gate_up_wire_1(struct expert_gate_up_quantize_arguments arguments,
		      unsigned char *arena)
{
	LINEAR_DEVICE_CHECK(
		arguments.pair_count <= EXPERT_PRODUCER_MOST_PAIRS &&
		arguments.output_size % LINEAR_VALUES_PER_LOAD == 0 &&
		arguments.reserved_zero >=
			arguments.output_size +
				arguments.output_size / VALUES_PER_BLOCK &&
		arguments.reserved_zero % LINEAR_OFFSET_ALIGNMENT_BYTES == 0);
	expert_gate_up_quantize<1, true>(arguments, arena);
}
