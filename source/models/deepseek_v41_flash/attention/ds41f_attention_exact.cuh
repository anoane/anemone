/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_attention_exact.cuh: the exact profile on class 120, the attention core's reference order with class 80's
 * sums and without its tensor cores (exact_tile_scores, exact_tile_values). Device code only: ds41f_attention_core.cu
 * includes it after its own helpers (the row, its keys' indices and rows, the weights, the epilogue) and nothing else
 * does.
 * Contract: ds41f_attention_core_records.h (the reference order, the launches on class 120).
 */
#ifndef DS41F_ATTENTION_EXACT_CUH
#define DS41F_ATTENTION_EXACT_CUH

#if !defined(KERNEL_ATTENTION_CORE_HELPERS)
#error "ds41f_attention_core.cu includes ds41f_attention_exact.cuh after its helpers"
#endif

#if LINEAR_THREAD_SUMS
/* ---- the reference order on class 120 in the exact profile ----
 * The sums are class 80's, a group of 8 products formed by one thread (class_80_thread_group of
 * kernel_mma_common.cuh): the instructions emulated lane by lane cost every lane shuffles and a decode of all the
 * inputs it gathers. Each score is formed once, by many small blocks: a card of many multiprocessors at a few rows.
 *   - Scores: a block is a row, DS41F_EXACT_SCORE_HEADS heads and a tile of 32 keys. The queries are decoded into
 *     shared memory (class_80_thread_decode): a magnitude times 16, a sign mask, an exponent; keys the same in
 *     registers, without the 16. Lane 4 j + s of warp w takes slice s of the tile's key 8 w + j: its heads' chains of
 *     16 groups of 8 dims from +0. Lane 4 j adds the 4 slices from +0 in ascending order; a key of index -1 is
 *     -infinity.
 *   - Values: a block is a row, heads_per_block heads and a tile of 128 dims. The softmax of the reference order: m
 *     the largest score from the floor up; p = expf(s - m); a group's 16 p added as core_softmax adds them; the
 *     groups' sums added in ascending order from +0 (the factor is 1: no group passes m); the weights of P V decoded
 *     into shared memory. Thread d takes dim d of the tile for every head of the block, a group's 16 keys as an
 *     instruction's two groups of 8, the next group's values read while one is summed. The epilogue is
 *     core_output_bits; the partner of a rotated dim is the next lane's.
 * Groups of no key are not run (a group of zero products would leave its sum as it is). */
#define DS41F_EXACT_THREADS DS41F_ATTENTION_CORE_THREADS
#define DS41F_EXACT_SCORE_HEADS DS41F_ATTENTION_CORE_EXACT_SCORE_HEADS
#define DS41F_EXACT_VALUE_HEADS DS41F_ATTENTION_CORE_EXACT_VALUE_HEADS
#define DS41F_EXACT_TILE_KEYS DS41F_ATTENTION_CORE_EXACT_TILE_KEYS
#define DS41F_EXACT_TILE_DIMS DS41F_ATTENTION_CORE_TILE_DIMS
#define DS41F_EXACT_WARPS (DS41F_EXACT_THREADS / THREADS_PER_WARP)
#define DS41F_EXACT_GROUPS (CORE_KEYS / CORE_GROUP_KEYS)
#define DS41F_EXACT_PRODUCTS CLASS_80_GROUP_PRODUCTS /* of a group */
#define DS41F_EXACT_SLICE_GROUPS (CORE_SLICE_DIMS / DS41F_EXACT_PRODUCTS)
#define DS41F_EXACT_KEYS_PER_WARP (THREADS_PER_WARP / CORE_SLICES)
#define DS41F_EXACT_QUERY_SLICE_WORDS \
	(CORE_SLICE_DIMS + 4u) /* slices apart in the banks */
#define DS41F_EXACT_QUERY_HEAD_WORDS \
	(CORE_SLICES * DS41F_EXACT_QUERY_SLICE_WORDS)
#define DS41F_EXACT_QUERY_WORDS    \
	(DS41F_EXACT_SCORE_HEADS * \
	 DS41F_EXACT_QUERY_HEAD_WORDS) /* of one field */
#define DS41F_EXACT_WEIGHT_WORDS \
	(DS41F_EXACT_VALUE_HEADS * CORE_KEYS) /* of one field */
#define DS41F_EXACT_WORD_BYTES 4u
#define DS41F_EXACT_VECTOR_MEMBERS 4u /* of a uint4 or an int4 */
/* shared memory, in bytes. Scores: the queries' magnitudes, signs and exponents, the tile's indices. Values: the
 * weights' magnitudes and exponents [head][key], the scores then p [head][key], the keys' rows, the groups' sums
 * [head][group], the warps' maxima, the heads' maxima and denominators. */
#define DS41F_EXACT_QUERY_SIGNS_OFFSET \
	(DS41F_EXACT_QUERY_WORDS * DS41F_EXACT_WORD_BYTES)
#define DS41F_EXACT_QUERY_EXPONENTS_OFFSET \
	(2u * DS41F_EXACT_QUERY_WORDS * DS41F_EXACT_WORD_BYTES)
#define DS41F_EXACT_TILE_INDICES_OFFSET \
	(3u * DS41F_EXACT_QUERY_WORDS * DS41F_EXACT_WORD_BYTES)
#define DS41F_EXACT_SCORES_SHARED_END      \
	(DS41F_EXACT_TILE_INDICES_OFFSET + \
	 DS41F_EXACT_TILE_KEYS * DS41F_EXACT_WORD_BYTES)
#define DS41F_EXACT_WEIGHT_EXPONENTS_OFFSET \
	(DS41F_EXACT_WEIGHT_WORDS * DS41F_EXACT_WORD_BYTES)
#define DS41F_EXACT_PROBABILITIES_OFFSET \
	(2u * DS41F_EXACT_WEIGHT_WORDS * DS41F_EXACT_WORD_BYTES)
#define DS41F_EXACT_KEY_ROWS_OFFSET \
	(3u * DS41F_EXACT_WEIGHT_WORDS * DS41F_EXACT_WORD_BYTES)
#define DS41F_EXACT_GROUP_SUMS_OFFSET  \
	(DS41F_EXACT_KEY_ROWS_OFFSET + \
	 CORE_KEYS * sizeof(const unsigned char *))
#define DS41F_EXACT_WARP_MAXIMA_OFFSET                                \
	(DS41F_EXACT_GROUP_SUMS_OFFSET + DS41F_EXACT_VALUE_HEADS *    \
						 DS41F_EXACT_GROUPS * \
						 DS41F_EXACT_WORD_BYTES)
#define DS41F_EXACT_MAXIMA_OFFSET         \
	(DS41F_EXACT_WARP_MAXIMA_OFFSET + \
	 DS41F_EXACT_WARPS * DS41F_EXACT_WORD_BYTES)
#define DS41F_EXACT_DENOMINATORS_OFFSET \
	(DS41F_EXACT_MAXIMA_OFFSET +    \
	 DS41F_EXACT_VALUE_HEADS * DS41F_EXACT_WORD_BYTES)
#define DS41F_EXACT_VALUES_SHARED_END      \
	(DS41F_EXACT_DENOMINATORS_OFFSET + \
	 DS41F_EXACT_VALUE_HEADS * DS41F_EXACT_WORD_BYTES)

static_assert(DS41F_EXACT_SCORES_SHARED_END ==
			      DS41F_ATTENTION_CORE_EXACT_SCORES_SHARED_BYTES &&
		      DS41F_EXACT_VALUES_SHARED_END ==
			      DS41F_ATTENTION_CORE_EXACT_VALUES_SHARED_BYTES,
	      "the layouts are the record header's sizes");
static_assert(DS41F_EXACT_TILE_KEYS ==
			      DS41F_EXACT_WARPS * DS41F_EXACT_KEYS_PER_WARP &&
		      CORE_SLICES * DS41F_EXACT_KEYS_PER_WARP ==
			      THREADS_PER_WARP &&
		      CORE_KEYS % DS41F_EXACT_TILE_KEYS == 0,
	      "a tile of keys: four lanes a key");
static_assert(
	DS41F_EXACT_THREADS == DS41F_EXACT_TILE_DIMS &&
		DS41F_HEAD_DIMS % DS41F_EXACT_TILE_DIMS == 0 &&
		DS41F_FIRST_ROTATED_DIM % THREADS_PER_WARP == 0 &&
		DS41F_HEADS_VALUE % DS41F_EXACT_SCORE_HEADS == 0 &&
		DS41F_EXACT_THREADS %
				(DS41F_EXACT_VALUE_HEADS * THREADS_PER_WARP) ==
			0 &&
		DS41F_EXACT_KEY_ROWS_OFFSET % sizeof(const unsigned char *) ==
			0,
	"a thread a dim of the values; whole warps a head of the softmax and of rotated dims");

/* The four words of a vector of shared memory into place .. place + 3 of an array, in order. */
__device__ static __forceinline__ void
exact_unsigned_quarter(const uint32_t *source, uint32_t *array, uint32_t place)
{
	const uint4 vector = *(const uint4 *)source;
	const uint32_t members[DS41F_EXACT_VECTOR_MEMBERS] = {
		vector.x, vector.y, vector.z, vector.w
	};
#pragma unroll
	for (uint32_t member = 0; member < DS41F_EXACT_VECTOR_MEMBERS;
	     member++) {
		array[place + member] = members[member];
	}
}

__device__ static __forceinline__ void
exact_signed_quarter(const int32_t *source, int32_t *array, uint32_t place)
{
	const int4 vector = *(const int4 *)source;
	const int32_t members[DS41F_EXACT_VECTOR_MEMBERS] = {
		vector.x, vector.y, vector.z, vector.w
	};
#pragma unroll
	for (uint32_t member = 0; member < DS41F_EXACT_VECTOR_MEMBERS;
	     member++) {
		array[place + member] = members[member];
	}
}

/* The block's heads' chains of one key over one slice: 16 groups of 8 dims, from +0. slice_keys: the slice's 128
 * dims of the key; the queries' fields at the slice of head 0. */
__device__ static __forceinline__ void
exact_slice_chains(const unsigned char *slice_keys,
		   const uint32_t *query_magnitudes, const int32_t *query_signs,
		   const int32_t *query_exponents,
		   float chains[DS41F_EXACT_SCORE_HEADS])
{
	const uint4 *chunks = (const uint4 *)slice_keys;
	uint4 chunk = chunks[0];
#pragma unroll 1
	for (uint32_t group = 0; group < DS41F_EXACT_SLICE_GROUPS; group++) {
		const uint32_t words[DS41F_EXACT_PRODUCTS / COLUMNS_PER_LANE] = {
			chunk.x, chunk.y, chunk.z, chunk.w
		};
		if (group + 1u < DS41F_EXACT_SLICE_GROUPS) {
			chunk = chunks[group + 1u];
		}
		uint32_t key_magnitudes[DS41F_EXACT_PRODUCTS];
		int32_t key_exponents[DS41F_EXACT_PRODUCTS],
			key_signs[DS41F_EXACT_PRODUCTS];
#pragma unroll
		for (uint32_t k = 0; k < DS41F_EXACT_PRODUCTS; k++) {
			class_80_thread_decode<false>(
				(words[k / COLUMNS_PER_LANE] >>
				 (k % COLUMNS_PER_LANE * BITS_PER_HALF)) &
					CLASS_80_HALF_BITS,
				&key_magnitudes[k], &key_exponents[k],
				&key_signs[k]);
		}
#pragma unroll
		for (uint32_t head = 0; head < DS41F_EXACT_SCORE_HEADS;
		     head++) {
			const uint32_t place =
				head * DS41F_EXACT_QUERY_HEAD_WORDS +
				group * DS41F_EXACT_PRODUCTS;
			uint32_t column_magnitudes[DS41F_EXACT_PRODUCTS];
			int32_t column_exponents[DS41F_EXACT_PRODUCTS],
				product_signs[DS41F_EXACT_PRODUCTS];
#pragma unroll
			for (uint32_t quarter = 0;
			     quarter < DS41F_EXACT_PRODUCTS;
			     quarter += DS41F_EXACT_VECTOR_MEMBERS) {
				exact_unsigned_quarter(
					query_magnitudes + place + quarter,
					column_magnitudes, quarter);
				exact_signed_quarter(query_exponents + place +
							     quarter,
						     column_exponents, quarter);
				exact_signed_quarter(query_signs + place +
							     quarter,
						     product_signs, quarter);
			}
#pragma unroll
			for (uint32_t k = 0; k < DS41F_EXACT_PRODUCTS; k++) {
				product_signs[k] =
					(key_signs[k] ^ product_signs[k]) | 1;
			}
			chains[head] = class_80_thread_group(
				chains[head], key_magnitudes, key_exponents,
				column_magnitudes, column_exponents,
				product_signs, CLASS_80_HALF_PAIR_BIAS);
		}
	}
}

/* The 16-bit values of dim `dim` of group `group`'s keys (0 for a key of none). */
__device__ static __forceinline__ void
exact_group_load(const unsigned char *const *key_rows, uint32_t group,
		 uint32_t dim, uint32_t half_bits[CORE_GROUP_KEYS])
{
#pragma unroll
	for (uint32_t k = 0; k < CORE_GROUP_KEYS; k++) {
		const unsigned char *const key_row =
			key_rows[group * CORE_GROUP_KEYS + k];
		half_bits[k] =
			key_row == NULL ? 0u : ((const uint16_t *)key_row)[dim];
	}
}

/* O += P V of one group of 16 keys for one dim of the block's first `heads` heads: keys 0 .. 7, then 8 .. 15. */
__device__ static __forceinline__ void
exact_group_values(const uint32_t half_bits[CORE_GROUP_KEYS],
		   const uint32_t *weight_magnitudes,
		   const int32_t *weight_exponents, uint32_t group,
		   uint32_t heads, float values[DS41F_EXACT_VALUE_HEADS])
{
#pragma unroll
	for (uint32_t half = 0; half < CORE_KEY_HALVES; half++) {
		uint32_t value_magnitudes[DS41F_EXACT_PRODUCTS];
		int32_t value_exponents[DS41F_EXACT_PRODUCTS],
			value_signs[DS41F_EXACT_PRODUCTS];
#pragma unroll
		for (uint32_t k = 0; k < DS41F_EXACT_PRODUCTS; k++) {
			class_80_thread_decode<false>(
				half_bits[half * CORE_HALF_GROUP + k],
				&value_magnitudes[k], &value_exponents[k],
				&value_signs[k]);
			value_signs[k] |= 1; /* a weight is never negative */
		}
#pragma unroll
		for (uint32_t head = 0; head < DS41F_EXACT_VALUE_HEADS;
		     head++) {
			if (head < heads) {
				const uint32_t place = head * CORE_KEYS +
						       group * CORE_GROUP_KEYS +
						       half * CORE_HALF_GROUP;
				uint32_t column_magnitudes[DS41F_EXACT_PRODUCTS];
				int32_t column_exponents[DS41F_EXACT_PRODUCTS];
#pragma unroll
				for (uint32_t quarter = 0;
				     quarter < DS41F_EXACT_PRODUCTS;
				     quarter += DS41F_EXACT_VECTOR_MEMBERS) {
					exact_unsigned_quarter(
						weight_magnitudes + place +
							quarter,
						column_magnitudes, quarter);
					exact_signed_quarter(
						weight_exponents + place +
							quarter,
						column_exponents, quarter);
				}
				values[head] = class_80_thread_group(
					values[head], value_magnitudes,
					value_exponents, column_magnitudes,
					column_exponents, value_signs,
					CLASS_80_HALF_PAIR_BIAS);
			}
		}
	}
}

/* A 16-bit float decoded as a column of class_80_thread_group into place `place` of the three fields. */
__device__ static __forceinline__ void
exact_column_store(uint32_t half_bits, uint32_t place, uint32_t *magnitudes,
		   int32_t *signs, int32_t *exponents)
{
	uint32_t magnitude;
	int32_t sign;
	class_80_thread_decode<false>(half_bits, &magnitude, &exponents[place],
				      &sign);
	magnitudes[place] = magnitude << CLASS_80_HALF_COLUMN_SHIFT;
	if (signs != NULL) {
		signs[place] = sign;
	}
}

/* The scores of a block (above): its row, DS41F_EXACT_SCORE_HEADS heads and the tile of keys z, into scores. */
__device__ static __forceinline__ void
exact_tile_scores(const struct ds41f_attention_core_arguments_record arguments,
		  unsigned char *arena, unsigned char *shared)
{
	struct core_row row;
	uint32_t first_group, end_group;
	const uint32_t tile_key = blockIdx.z * DS41F_EXACT_TILE_KEYS;
	if (!core_row_open(arguments, arena, &row, &first_group, &end_group) ||
	    tile_key >= end_group * CORE_GROUP_KEYS ||
	    tile_key + DS41F_EXACT_TILE_KEYS <= first_group * CORE_GROUP_KEYS) {
		return; /* every thread: no key of the row in the tile */
	}
	LINEAR_DEVICE_CHECK(
		blockDim.x == DS41F_EXACT_THREADS &&
		arguments.first_head + gridDim.y * DS41F_EXACT_SCORE_HEADS <=
			DS41F_HEADS_VALUE &&
		gridDim.z * DS41F_EXACT_TILE_KEYS == CORE_KEYS);
	const uint32_t first_key = first_group * CORE_GROUP_KEYS,
		       end_key = end_group * CORE_GROUP_KEYS;
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP,
		       lane = threadIdx.x % THREADS_PER_WARP;
	const uint32_t first_head =
		arguments.first_head + blockIdx.y * DS41F_EXACT_SCORE_HEADS;
	uint32_t *query_magnitudes = (uint32_t *)shared;
	int32_t *query_signs =
		(int32_t *)(shared + DS41F_EXACT_QUERY_SIGNS_OFFSET);
	int32_t *query_exponents =
		(int32_t *)(shared + DS41F_EXACT_QUERY_EXPONENTS_OFFSET);
	int32_t *tile_indices =
		(int32_t *)(shared + DS41F_EXACT_TILE_INDICES_OFFSET);
	const uint16_t *block_queries =
		(const uint16_t *)(arena + arguments.queries_offset) +
		((size_t)blockIdx.x * DS41F_HEADS_VALUE + first_head) *
			DS41F_HEAD_DIMS;
	for (uint32_t element = threadIdx.x;
	     element < DS41F_EXACT_SCORE_HEADS * DS41F_HEAD_DIMS;
	     element += DS41F_EXACT_THREADS) {
		const uint32_t head = element / DS41F_HEAD_DIMS,
			       dim = element % DS41F_HEAD_DIMS;
		exact_column_store(
			block_queries[element],
			head * DS41F_EXACT_QUERY_HEAD_WORDS +
				dim / CORE_SLICE_DIMS *
					DS41F_EXACT_QUERY_SLICE_WORDS +
				dim % CORE_SLICE_DIMS,
			query_magnitudes, query_signs, query_exponents);
	}
	if (threadIdx.x < DS41F_EXACT_TILE_KEYS) {
		const uint32_t key = tile_key + threadIdx.x;
		tile_indices[threadIdx.x] = key >= first_key && key < end_key ?
						    core_key_index(&row, key) :
						    -1;
	}
	__syncthreads();
	const uint32_t slice = lane % CORE_SLICES,
		       tile_place = warp * DS41F_EXACT_KEYS_PER_WARP +
				    lane / CORE_SLICES;
	const uint32_t key = tile_key + tile_place;
	const int32_t index = tile_indices[tile_place];
	float chains[DS41F_EXACT_SCORE_HEADS];
#pragma unroll
	for (uint32_t head = 0; head < DS41F_EXACT_SCORE_HEADS; head++) {
		chains[head] = 0.0f;
	}
	if (index >= 0) {
		const uint32_t slice_place =
			slice * DS41F_EXACT_QUERY_SLICE_WORDS;
		exact_slice_chains(core_key_row(&row, key, index) +
					   slice * CORE_SLICE_DIMS *
						   sizeof(uint16_t),
				   query_magnitudes + slice_place,
				   query_signs + slice_place,
				   query_exponents + slice_place, chains);
	}
	float *const scores =
		(float *)(arena + arguments.scores_offset) +
		((size_t)blockIdx.x * DS41F_HEADS_VALUE + first_head) *
			CORE_KEYS +
		key;
#pragma unroll
	for (uint32_t head = 0; head < DS41F_EXACT_SCORE_HEADS;
	     head++) { /* every lane takes part in the shuffles */
		float score = 0.0f + chains[head];
#pragma unroll
		for (uint32_t other = 1; other < CORE_SLICES; other++) {
			score = score + __shfl_down_sync(CORE_EVERY_LANE,
							 chains[head], other);
		}
		if (slice == 0 && key >= first_key && key < end_key) {
			scores[head * CORE_KEYS] =
				index < 0 ? -INFINITY :
					    score * DS41F_ATTENTION_CORE_SCALE;
		}
	}
}

/* The softmax and values of a block (above): its row, heads_per_block heads and the tile of dims z. */
__device__ static __forceinline__ void
exact_tile_values(const struct ds41f_attention_core_arguments_record arguments,
		  unsigned char *arena, unsigned char *shared)
{
	struct core_row row;
	uint32_t first_group, end_group;
	if (!core_row_open(arguments, arena, &row, &first_group, &end_group)) {
		return;
	}
	const uint32_t heads = arguments.heads_per_block;
	LINEAR_DEVICE_CHECK(
		blockDim.x == DS41F_EXACT_THREADS && heads != 0 &&
		heads <= DS41F_EXACT_VALUE_HEADS &&
		DS41F_EXACT_THREADS % (heads * THREADS_PER_WARP) == 0 &&
		arguments.first_head + gridDim.y * heads <= DS41F_HEADS_VALUE &&
		gridDim.z * DS41F_EXACT_TILE_DIMS == DS41F_HEAD_DIMS);
	const uint32_t row_index = blockIdx.x;
	const uint32_t first_key = first_group * CORE_GROUP_KEYS,
		       end_key = end_group * CORE_GROUP_KEYS;
	const uint32_t warp = threadIdx.x / THREADS_PER_WARP,
		       lane = threadIdx.x % THREADS_PER_WARP;
	const uint32_t first_head = arguments.first_head + blockIdx.y * heads;
	uint32_t *weight_magnitudes = (uint32_t *)shared;
	int32_t *weight_exponents =
		(int32_t *)(shared + DS41F_EXACT_WEIGHT_EXPONENTS_OFFSET);
	float *probabilities =
		(float *)(shared +
			  DS41F_EXACT_PROBABILITIES_OFFSET); /* [head][key]: the scores, then p */
	const unsigned char **key_rows =
		(const unsigned char **)(shared + DS41F_EXACT_KEY_ROWS_OFFSET);
	float *group_sums =
		(float *)(shared +
			  DS41F_EXACT_GROUP_SUMS_OFFSET); /* [head][group] */
	float *warp_maxima = (float *)(shared + DS41F_EXACT_WARP_MAXIMA_OFFSET);
	float *maxima = (float *)(shared + DS41F_EXACT_MAXIMA_OFFSET);
	float *denominators =
		(float *)(shared + DS41F_EXACT_DENOMINATORS_OFFSET);
	for (uint32_t key = first_key + threadIdx.x; key < end_key;
	     key += DS41F_EXACT_THREADS) {
		const int32_t index = core_key_index(&row, key);
		key_rows[key] = index < 0 ? NULL :
					    core_key_row(&row, key, index);
	}
	/* the softmax: m, then p and the weights, then the groups' sums, then the denominators */
	const uint32_t head_threads = DS41F_EXACT_THREADS / heads;
	const uint32_t softmax_head = threadIdx.x / head_threads,
		       head_thread = threadIdx.x % head_threads;
	const float *const head_scores =
		(const float *)(arena + arguments.scores_offset) +
		((size_t)row_index * DS41F_HEADS_VALUE + first_head +
		 softmax_head) *
			CORE_KEYS;
	float largest = DS41F_ATTENTION_CORE_SCORE_FLOOR;
	for (uint32_t key = first_key + head_thread; key < end_key;
	     key += head_threads) {
		const float score = head_scores[key];
		probabilities[softmax_head * CORE_KEYS + key] = score;
		largest = fmaxf(largest, score);
	}
	for (uint32_t distance = 1u; distance < THREADS_PER_WARP;
	     distance <<= 1u) {
		largest =
			fmaxf(largest, __shfl_xor_sync(CORE_EVERY_LANE, largest,
						       (int)distance));
	}
	if (lane == 0) {
		warp_maxima[warp] = largest;
	}
	__syncthreads();
	float maximum = DS41F_ATTENTION_CORE_SCORE_FLOOR;
	for (uint32_t head_warp = 0;
	     head_warp < head_threads / THREADS_PER_WARP; head_warp++) {
		maximum = fmaxf(maximum,
				warp_maxima[softmax_head * (head_threads /
							    THREADS_PER_WARP) +
					    head_warp]);
	}
	if (head_thread == 0) {
		maxima[softmax_head] = maximum;
	}
	for (uint32_t key = first_key + head_thread; key < end_key;
	     key += head_threads) {
		const uint32_t place = softmax_head * CORE_KEYS + key;
		const float probability = expf(probabilities[place] - maximum);
		probabilities[place] = probability;
		exact_column_store(core_weight<true>(probability), place,
				   weight_magnitudes, NULL, weight_exponents);
	}
	__syncthreads();
	const uint32_t group_count = end_group - first_group;
	for (uint32_t item = threadIdx.x; item < heads * group_count;
	     item += DS41F_EXACT_THREADS) {
		const uint32_t head = item / group_count,
			       group = first_group + item % group_count;
		const float *group_probabilities = probabilities +
						   head * CORE_KEYS +
						   group * CORE_GROUP_KEYS;
		float lane_sums[CORE_SOFTMAX_LANES];
#pragma unroll
		for (uint32_t head_lane = 0; head_lane < CORE_SOFTMAX_LANES;
		     head_lane++) {
			lane_sums[head_lane] =
				(0.0f + group_probabilities[head_lane]) +
				group_probabilities[head_lane + CORE_HALF_GROUP];
		}
#pragma unroll
		for (uint32_t stride = 1; stride < CORE_SOFTMAX_LANES;
		     stride += stride) { /* ((0 + 1) + (2 + 3)) + .. */
#pragma unroll
			for (uint32_t first = 0; first < CORE_SOFTMAX_LANES;
			     first += stride + stride) {
				lane_sums[first] = lane_sums[first] +
						   lane_sums[first + stride];
			}
		}
		group_sums[head * DS41F_EXACT_GROUPS + group] = lane_sums[0];
	}
	__syncthreads();
	if (threadIdx.x < heads) {
		float total = 0.0f;
		for (uint32_t group = first_group; group < end_group; group++) {
			total += group_sums[threadIdx.x * DS41F_EXACT_GROUPS +
					    group];
		}
		denominators[threadIdx.x] =
			total +
			expf(((const float *)(arena + arguments.sinks_offset))
				     [first_head + threadIdx.x] -
			     maxima[threadIdx.x]);
	}
	/* the values */
	const uint32_t dim = blockIdx.z * DS41F_EXACT_TILE_DIMS + threadIdx.x;
	float values[DS41F_EXACT_VALUE_HEADS];
#pragma unroll
	for (uint32_t head = 0; head < DS41F_EXACT_VALUE_HEADS; head++) {
		values[head] = 0.0f;
	}
	uint32_t next_bits[CORE_GROUP_KEYS];
	exact_group_load(key_rows, first_group, dim, next_bits);
#pragma unroll 1
	for (uint32_t group = first_group; group < end_group; group++) {
		uint32_t half_bits[CORE_GROUP_KEYS];
#pragma unroll
		for (uint32_t k = 0; k < CORE_GROUP_KEYS; k++) {
			half_bits[k] = next_bits[k];
		}
		if (group + 1u < end_group) {
			exact_group_load(key_rows, group + 1u, dim, next_bits);
		}
		exact_group_values(half_bits, weight_magnitudes,
				   weight_exponents, group, heads, values);
	}
	__syncthreads(); /* the denominators */
	const float *rotation =
		(const float *)(arena + arguments.rotation_offset) +
		(size_t)row_index * CORE_ROTATION_ROW_FLOATS;
	uint16_t *output = (uint16_t *)(arena + arguments.output_offset) +
			   core_output_head(arguments, row_index, first_head) *
				   DS41F_HEAD_DIMS +
			   dim;
#pragma unroll
	for (uint32_t head = 0; head < DS41F_EXACT_VALUE_HEADS; head++) {
		if (head < heads) { /* the same in the whole block */
			output[(size_t)head * DS41F_HEAD_DIMS] =
				(uint16_t)core_output_bits<true>(
					values[head], denominators[head],
					rotation, dim, 1u);
		}
	}
}
#endif

#endif
