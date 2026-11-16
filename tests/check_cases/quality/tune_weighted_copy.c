/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Isolated private-helper regressions of the storage tuner. Load points steal (each copy's readers take the next chunk
 * from one cursor), so every load point reads the same volume and is ranked by its time; only the full copy set
 * measures load, since only a load that reads every copy can win; the profile's load shares are the winner's chunks
 * per copy. A saved state's cache mode is validated like its other fields. */
#include "../../../source/storage_tune.c"

#define TEST_COPIES 2u
#define TEST_SET_COPIES 3u
#define TEST_PAIR_SET 3u /* copies 0 and 1 of three */
#define TEST_FULL_SET 7u
#define TEST_CHUNK_NS 1000000u /* one reader takes this many ns a load chunk */
#define TEST_FAST_CHUNKS \
	340u /* of TUNE_LOAD_CHUNKS: the fast copy's readers took about twice the slow copy's */
#define TEST_TWO_THREADS 2u
#define TEST_SOLO_SPEEDUP \
	4u /* a solo point, faster still, that must never win the load */

/* The busiest copy's share is TUNE_WEIGHT_SCALE, the others in proportion; a member that took no chunk keeps 1, a copy
 * outside the point 0. */
static bool steal_shares(void)
{
	struct tune_point point = { .threads = 1u };
	point.weights[0] = point.weights[1] = point.weights[2] = 1u;
	point.copy_chunks[0] = TEST_FAST_CHUNKS;
	point.copy_chunks[1] = TUNE_LOAD_CHUNKS - TEST_FAST_CHUNKS;
	struct tune_point outside = point;
	outside.weights[2] = 0;
	return load_share(&point, 0) == TUNE_WEIGHT_SCALE &&
	       load_share(&point, 1) == (TUNE_LOAD_CHUNKS - TEST_FAST_CHUNKS) *
						TUNE_WEIGHT_SCALE /
						TEST_FAST_CHUNKS &&
	       load_share(&point, 2) == 1u && load_share(&outside, 2) == 0;
}

/* Two full-set points of the same volume: the faster wins, whatever its threads; a solo point never wins the load. */
static bool load_ranked_by_time(void)
{
	struct tune_state *const state = calloc(1, sizeof(*state));
	if (state == NULL)
		return false;
	state->count = TEST_COPIES;
	const uint64_t bytes = (uint64_t)TUNE_LOAD_CHUNKS * LOADER_CHUNK_BYTES;
	const uint64_t one_ns = (uint64_t)TUNE_LOAD_CHUNKS * TEST_CHUNK_NS;
	struct tune_point *const one = &state->values[state->points++];
	*one = (struct tune_point){ .threads = 1u,
				    .bytes = bytes,
				    .ns = one_ns };
	one->weights[0] = one->weights[1] = 1u;
	struct tune_point *const two = &state->values[state->points++];
	*two = (struct tune_point){ .threads = TEST_TWO_THREADS,
				    .bytes = bytes,
				    .ns = one_ns / TEST_TWO_THREADS };
	two->weights[0] = two->weights[1] = 1u;
	struct tune_point *const solo = &state->values[state->points++];
	*solo = (struct tune_point){ .threads = 1u,
				     .bytes = bytes,
				     .ns = one_ns / TEST_SOLO_SPEEDUP };
	solo->weights[0] = 1u;
	const bool passed = point_cost(two) < point_cost(one) &&
			    best_point(state, 0, 0, true, 0) == two;
	free(state);
	return passed;
}

/* With every lookup point of copies 0 and 1 measured and the tune stopped (any point still to measure times out),
 * that pair measures nothing more, while the full set still has its load points to measure. */
static bool subsets_measure_lookup_only(void)
{
	struct tune_state *const state = calloc(1, sizeof(*state));
	struct prepare_tune_options options = { .deadline_ns = 0 };
	struct tune_context context = { .options = &options,
					.copies = TEST_SET_COPIES };
	if (state == NULL)
		return false;
	state->count = TEST_SET_COPIES;
	for (uint32_t mode = 0; mode < TUNE_CACHE_MODES; mode++)
		for (uint32_t io = 0; io < TUNE_IO_MECHANISMS; io++) {
			struct tune_point *const point =
				&state->values[state->points++];
			point->action = 1u;
			point->io = io;
			point->threads = 1u;
			point->cached = mode == 0 ? 1u : 0u;
			point->weights[0] = point->weights[1] = 1u;
		}
	const bool passed = set_measure(&context, state, "", TEST_PAIR_SET) ==
				    STATUS_OK &&
			    set_measure(&context, state, "", TEST_FULL_SET) ==
				    STATUS_TIMED_OUT;
	free(state);
	return passed;
}

static bool cache_mode_validated(void)
{
	struct tune_state *const state = calloc(1, sizeof(*state));
	if (state == NULL)
		return false;
	state->count = TEST_COPIES;
	state->points = 1u;
	struct tune_point *const point = &state->values[0];
	point->action = 1u;
	point->threads = 1u;
	point->weights[0] = 1u;
	point->cached = 1u;
	bool passed = state_valid(state);
	point->cached = TUNE_CACHE_MODES;
	passed = passed && !state_valid(state);
	point->action = 0;
	point->cached = 1u;
	passed = passed && !state_valid(state);
	point->cached = 0;
	passed = passed && state_valid(state);
	free(state);
	return passed;
}

int main(void)
{
	const bool passed = steal_shares() && load_ranked_by_time() &&
			    subsets_measure_lookup_only() &&
			    cache_mode_validated();
	return passed ? 0 : 1;
}
