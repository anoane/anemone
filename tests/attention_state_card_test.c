/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * attention_state_card_test: the replay of the attention state on the real cards: each kernel fed the reference
 * model's own input of it, from a capture folder of tools/oracle/stepwise_capture.py, and held to the reference's
 * own output (design/first_token_plan.md 3.2, the rows of U4).
 *
 *     attention_state_card_test [--break weight|frequency|rounding] [<capture folder> <pack copy> [<pack copy> ...]]
 *
 * Without a capture, as `make card_tests` runs every card test, there is nothing to replay: it runs the rotation step
 * of every card on words of zero (cos 1, sin 0), says that it replayed nothing and passes
 * (attention_state_synthetic_card_test holds the kernels to the host reference). With a capture of
 * the u16 run with published keys and --frequencies, and the pack whose norm weights the model read, on every card:
 *   - the rotation step at every position below 16,384 and at the sampled ones, from the words the reference formed
 *     (frequencies.theta_<T>.words), against its freqs_cis: every cos and sin within one float32 place, the share
 *     bit-equal printed;
 *   - at every capture point (layer_<L>.position_<P>), one row at position P, each kernel fed the captured input:
 *     qr from attn.wq_a against attn.qr; the window row from attn.wkv against attn.window_row and, where the indexer
 *     ran (P + 1 >= the layer's ratio), the index queries from indexer.wq_b against indexer.query; the main queries
 *     from attn.wq_b against attn.q bit for bit; on the layers that pool, from compressor.wkv and .wgate, with
 *     position P - 1 from compressor.kv_state and .score_state in its slot: the latent against
 *     indexer.latent_bfloat16, the pool entry against compressor.pool_entry and the key from indexer.wk against
 *     indexer.key; at a position that does not emit, its slot against the captured one bit for bit. A rounded value
 *     within one place of its format, and of each check at least 99.9% bit-equal on every card (the order of the
 *     norms' sums is the engine's). Every check must have counted values on every card, or the test FAILS: a
 *     capture that lacks a kind proves nothing of it.
 *   - Risk 8: of each kind the reference rounds to FP16 (window rows, index queries, pool entries, keys: their
 *     *_unrounded records) and of the main queries, the values outside FP16's normal range (not 0, and below 2^-14
 *     or above 65,504) are counted and printed. A main query below 2^-14 is held within 2^-25 of the reference's
 *     (FP16 keeps fewer bits there), one above 65,504 FAILS (FP16 cannot hold it).
 *   - ROTATION, once, on the host: how often each form of each part of the complex product turns attn.kv_normed
 *     into attn.window_row_unrounded bit for bit by the model's own cos and sin, at every point below 16,384 (none
 *     FAILS): the records' (fmaf(a, c, -(b s)), fmaf(b, c, a s)); the products rounded, then added; the other
 *     contraction (fmaf(-b, s, a c), fmaf(a, s, b c)).
 *   - --break weight: one kv_norm weight changed on the card; --break frequency: one captured word one place off;
 *     --break rounding: every rounded value expected one place toward zero, inside the tolerance, so that only the
 *     shares can make the test fail. Each must make the test FAIL: shown failing.
 * Every count the capture and the pack state is bounded before it is used: a record that is missing, or of another
 * type or size, fails the test with its name.
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/attention_state.h"
#include "anemone/attention_state_records.h"
#include "anemone/backend_seam.h"
#include "anemone/pack.h"
#include "anemone/status.h"
#include "anemone/storage_seam.h"
#include "attention_state_card_fixture.h"
#include "attention_state_reference.h"
#include "capture_reader.h"
#include "linear_card_fixture.h"
#include "linear_reference.h"

#define TEST_TIME_LIMIT_SECONDS 3600
#define WORK_CAPACITY (64ull << 20u)
#define WEIGHTS_CAPACITY (1ull << 20u)
#define ROWS LINEAR_MAXIMUM_ROWS
#define QUERY_RANK MODEL_QUERY_RANK
#define DIMS MODEL_HEAD_DIMS
#define INDEX_DIMS MODEL_INDEX_HEAD_DIMS
#define INDEX_VALUES ((uint64_t)MODEL_INDEX_HEADS * MODEL_INDEX_HEAD_DIMS)
#define HEAD_VALUES ((uint64_t)MODEL_HEADS_VALUE * MODEL_HEAD_DIMS)
#define FREQUENCIES MODEL_FREQUENCIES
#define WORDS (MODEL_THETAS * MODEL_FREQUENCIES)
#define PAIR MODEL_PAIR_VALUES
#define PAIR_FLOATS                    \
	((uint64_t)MODEL_FREQUENCIES * \
	 MODEL_PAIR_VALUES) /* a position's table row */
#define RATIO_TWO MODEL_POOLED_RATIO
#define SLOT_FLOATS ((uint64_t)ATTENTION_SLOT_VECTORS * MODEL_HEAD_DIMS)
#define TABLE_POSITIONS 16384u                  /* stepwise_capture.py's FREQUENCY_ROWS */
#define SAMPLED_POSITIONS 64u                   /* its FREQUENCY_SAMPLES */
#define TABLE_BATCHES (TABLE_POSITIONS / ROWS)
#define SAMPLED_BATCHES (SAMPLED_POSITIONS / ROWS)
#define LAST_SAMPLED_POSITION 1048575           /* its FREQUENCY_LAST_POSITION */
#define MOST_POINTS 4096u
#define NAME_CAPACITY 160
#define DECIMAL 10
#define BFLOAT16_LOW_HALF 0xFFFFu
#define BFLOAT16_SHIFT 16u
#define HALF_LARGEST 65504.0
#define HALF_SMALLEST_NORMAL 6.103515625e-05
#define HALF_SUBNORMAL_ERROR 2.98023223876953125e-08      /* 2^-25: half of FP16's place below 2^-14 */
#define BROKEN_WEIGHT 7u
#define BROKEN_WORD FREQUENCIES                 /* the first word of theta 1 */
#define BREAK_FLAG "--break"
#define ARGUMENTS_OF_A_BREAK 2
#define FORMS 3u
#define ROTATED_PAIR_FIRST (MODEL_FIRST_ROTATED_DIM / MODEL_PAIR_VALUES)

/* The model: CFG:52 compress_ratios, CFG:33 index_source_layers, CFG:32 kv_source_layers, the thetas' names. */
static const uint32_t compress_ratios[MODEL_LAYERS] = {
	0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};
static const uint32_t index_layers[] = {2, 8, 14, 20, 24, 28, 32, 36};
static const uint32_t source_layers[] = {2, 8, 14, 20};
static const char *const theta_records[MODEL_THETAS] = {
	"frequencies.theta_10000", "frequencies.theta_160000"
};
static const char *const norm_names[TEST_ATTENTION_NORMS] = {"attn.q_norm.weight", "attn.kv_norm.weight",
                                                             "attn.compressor.norm.weight",
                                                             "attn.indexer.k_norm.weight"};
static const uint32_t norm_widths[TEST_ATTENTION_NORMS] = {QUERY_RANK, DIMS, DIMS, INDEX_DIMS};
static const char *const form_names[FORMS] = {"the records' fmaf(a, c, -(b s)), fmaf(b, c, a s)",
                                              "products rounded, then added",
                                              "the other contraction fmaf(-b, s, a c), fmaf(a, s, b c)"};

struct capture_point {
    uint32_t layer;
    uint32_t position;
};

/* What the replay reads from: the capture, the pack, and what was found in them. */
struct replay_sources {
    struct test_capture *capture;
    struct storage_reader *reader;
    struct pack_record *pack;
    uint32_t block_alignment_bytes;
    uint32_t point_count;
    uint32_t context;                           /* above every captured position, whole granules */
    uint32_t reserved_zero;
    float words[WORDS];
    struct capture_point points[MOST_POINTS];
};

static struct replay_sources sources;
static struct test_attention_tally tallies[TEST_ATTENTION_CHECK_COUNT];
static const char *broken = NULL;
static float tables[MODEL_THETAS][TABLE_POSITIONS * PAIR_FLOATS];
static float sampled[MODEL_THETAS][SAMPLED_POSITIONS * PAIR_FLOATS];
static int64_t sampled_positions[MODEL_THETAS][SAMPLED_POSITIONS];
static uint32_t index_source_of[INDEX_VALUES];
static uint32_t key_source_of[INDEX_DIMS];
static uint16_t halves[HEAD_VALUES];
static float floats[HEAD_VALUES];
static float more_floats[DIMS];
static uint64_t form_equal[FORMS][PAIR];        /* real, imaginary */
static uint64_t form_pairs;

static bool breaks(const char *what)
{
    return broken != NULL && strcmp(broken, what) == 0;
}

/* The ordinal a card's value is held to: the reference's, or under --break rounding one place toward zero. */
static int32_t expected_ordinal(int32_t ordinal)
{
    return !breaks("rounding") ? ordinal : ordinal > 0 ? ordinal - 1 : ordinal < 0 ? ordinal + 1 : 0;
}

static bool is_listed(uint32_t layer, const uint32_t *list, size_t count)
{
    for (size_t index = 0; index < count; index++) {
        if (list[index] == layer) {
            return true;
        }
    }
    return false;
}

static struct test_attention_kind kind_of(uint32_t layer)
{
    const bool is_source = is_listed(layer, source_layers, sizeof(source_layers) / sizeof(source_layers[0]));
    return (struct test_attention_kind){compress_ratios[layer] != 0 ? 1u : 0u,
                                        is_listed(layer, index_layers, sizeof(index_layers) / sizeof(index_layers[0])),
                                        is_source ? compress_ratios[layer] : 0u, 0};
}

static bool name_make(char name[NAME_CAPACITY], const char *first, uint32_t layer, uint32_t position, const char *part)
{
    const int length = first == NULL ? snprintf(name, NAME_CAPACITY, "layer_%" PRIu32 ".position_%" PRIu32 ".%s",
                                                layer, position, part)
                                     : snprintf(name, NAME_CAPACITY, "%s.%s", first, part);
    return test_linear_expect(length > 0 && length < NAME_CAPACITY, "a record's name fits");
}

/* A record of count values of a type, read whole; false behind a failure line where it is not such a record. */
static bool record_read(const char *name, enum test_capture_type type, uint64_t count, void *destination)
{
    const struct test_capture_tensor *const tensor = test_capture_find(sources.capture, name);
    if (tensor == NULL || tensor->type != (uint32_t)type || tensor->element_count != count) {
        test_linear_fail("the capture holds no record %s of %" PRIu64 " values of %s", name, count,
                         test_capture_type_name(type));
        return false;
    }
    return test_linear_expect_status(
	    test_capture_read(sources.capture, name, type, tensor->dimensions,
			      tensor->dimension_count, 0, count, destination,
			      count * test_capture_element_bytes(type)),
	    STATUS_OK, name);
}

static bool point_record(uint32_t layer, uint32_t position, const char *part, enum test_capture_type type,
                         uint64_t count, void *destination)
{
    char name[NAME_CAPACITY];
    return name_make(name, NULL, layer, position, part) && record_read(name, type, count, destination);
}

/* bfloat16 values of a record of bfloat16, or of float32 that are all bfloat16 (the u16 run widens some). */
static bool point_bfloat16(uint32_t layer, uint32_t position, const char *part, uint64_t count, uint16_t *out)
{
    char name[NAME_CAPACITY];
    if (!name_make(name, NULL, layer, position, part) || count > HEAD_VALUES) {
        return false;
    }
    const struct test_capture_tensor *const tensor = test_capture_find(sources.capture, name);
    if (tensor == NULL || tensor->type != TEST_CAPTURE_FLOAT32) {
        return record_read(name, TEST_CAPTURE_BFLOAT16, count, out);
    }
    if (!record_read(name, TEST_CAPTURE_FLOAT32, count, floats)) {
        return false;
    }
    for (uint64_t index = 0; index < count; index++) {
        uint32_t bits = 0;
        memcpy(&bits, &floats[index], sizeof(bits));
        if ((bits & BFLOAT16_LOW_HALF) != 0) {
            test_linear_fail("%s: value %" PRIu64 " is no bfloat16", name, index);
            return false;
        }
        out[index] = (uint16_t)(bits >> BFLOAT16_SHIFT);
    }
    return true;
}

/* FP16 bits of a record of float32 that are all FP16 values (u16's state). */
static bool point_half(uint32_t layer, uint32_t position, const char *part, uint64_t count, uint16_t *out)
{
    if (count > HEAD_VALUES || !point_record(layer, position, part, TEST_CAPTURE_FLOAT32, count, floats)) {
        return false;
    }
    for (uint64_t index = 0; index < count; index++) {
        out[index] = test_attention_half_nearest(floats[index]);
        if (test_attention_half_value(out[index]) != (double)floats[index]) {
            test_linear_fail("layer %" PRIu32 " position %" PRIu32 " %s: value %" PRIu64 " is no FP16 value", layer,
                             position, part, index);
            return false;
        }
    }
    return true;
}

/* The capture points (every layer_<L>.position_<P>.attn.wq_a), ascending by layer and position. */
static bool points_find(void)
{
    static const char layer_prefix[] = "layer_", position_prefix[] = ".position_", marker[] = ".attn.wq_a";
    uint32_t largest_position = 0;
    for (uint32_t index = 0; index < test_capture_tensor_count(sources.capture); index++) {
        const char *const name = test_capture_tensor_at(sources.capture, index)->name;
        char *end = NULL;
        if (strncmp(name, layer_prefix, sizeof(layer_prefix) - 1u) != 0) {
            continue;
        }
        const unsigned long layer = strtoul(name + sizeof(layer_prefix) - 1u, &end, DECIMAL);
        if (strncmp(end, position_prefix, sizeof(position_prefix) - 1u) != 0) {
            continue;
        }
        const unsigned long position = strtoul(end + sizeof(position_prefix) - 1u, &end, DECIMAL);
        if (strcmp(end, marker) != 0) {
            continue;
        }
	if (layer >= MODEL_LAYERS ||
	    position >= MODEL_CONTEXT_WITHOUT_CANDIDATES ||
	    sources.point_count == MOST_POINTS) {
		test_linear_fail(
			"%s: a layer, a position or a number of points beyond the test's",
			name);
		return false;
	}
	uint32_t place = sources.point_count++;
        while (place > 0 &&
               (sources.points[place - 1u].layer > layer ||
                (sources.points[place - 1u].layer == layer && sources.points[place - 1u].position > position))) {
            sources.points[place] = sources.points[place - 1u];
            place--;
        }
        sources.points[place] = (struct capture_point){(uint32_t)layer, (uint32_t)position};
        largest_position = (uint32_t)position > largest_position ? (uint32_t)position : largest_position;
    }
    sources.context = (largest_position / MODEL_CONTEXT_GRANULE + 1u) *
		      MODEL_CONTEXT_GRANULE;
    return test_linear_expect(sources.point_count != 0, "the capture holds capture points");
}

/* The words, the tables of cos and sin, with the --break; the placements. */
static bool frequencies_read(void)
{
    char name[NAME_CAPACITY];
    static uint16_t identity[INDEX_VALUES], placed[INDEX_VALUES];
    for (uint32_t theta = 0; theta < MODEL_THETAS; theta++) {
	    if (!name_make(name, theta_records[theta], 0, 0, "words") ||
		!record_read(name, TEST_CAPTURE_FLOAT32, FREQUENCIES,
			     &sources.words[(size_t)theta * FREQUENCIES]) ||
		!name_make(name, theta_records[theta], 0, 0,
			   "cos_sin_below_16384") ||
		!record_read(name, TEST_CAPTURE_FLOAT32,
			     (uint64_t)TABLE_POSITIONS * PAIR_FLOATS,
			     tables[theta]) ||
		!name_make(name, theta_records[theta], 0, 0, "positions") ||
		!record_read(name, TEST_CAPTURE_INT64, SAMPLED_POSITIONS,
			     sampled_positions[theta]) ||
		!name_make(name, theta_records[theta], 0, 0,
			   "cos_sin_sampled") ||
		!record_read(name, TEST_CAPTURE_FLOAT32,
			     SAMPLED_POSITIONS * PAIR_FLOATS, sampled[theta])) {
		    return false;
	    }
	    for (uint32_t index = 0; index < SAMPLED_POSITIONS; index++) {
		    if (sampled_positions[theta][index] < 0 ||
			sampled_positions[theta][index] >
				LAST_SAMPLED_POSITION) {
			    test_linear_fail(
				    "%s: a sampled position beyond 2^20 - 1",
				    theta_records[theta]);
			    return false;
		    }
	    }
    }
    if (breaks("frequency")) {
        uint32_t bits = 0;
        memcpy(&bits, &sources.words[BROKEN_WORD], sizeof(bits));
        bits++;
        memcpy(&sources.words[BROKEN_WORD], &bits, sizeof(bits));
    }
    for (uint32_t index = 0; index < INDEX_VALUES; index++) {
        identity[index] = (uint16_t)index;
    }
    test_attention_layout_index_queries(identity, placed);
    for (uint32_t index = 0; index < INDEX_VALUES; index++) {
        index_source_of[index] = placed[index];
    }
    test_attention_layout_key(identity, placed);
    for (uint32_t index = 0; index < INDEX_DIMS; index++) {
        key_source_of[index] = placed[index];
    }
    return true;
}

static bool same_bits(float first, float second)
{
    uint32_t first_bits = 0, second_bits = 0;
    memcpy(&first_bits, &first, sizeof(first_bits));
    memcpy(&second_bits, &second, sizeof(second_bits));
    return first_bits == second_bits;
}

/* ROTATION: the three forms of the complex product against the model's own rotated window rows. */
static void forms_count(void)
{
    for (uint32_t point = 0; point < sources.point_count; point++) {
        const uint32_t layer = sources.points[point].layer, position = sources.points[point].position;
        if (position >= TABLE_POSITIONS ||
            !point_record(layer, position, "attn.kv_normed", TEST_CAPTURE_FLOAT32, DIMS, floats) ||
            !point_record(layer, position, "attn.window_row_unrounded", TEST_CAPTURE_FLOAT32, DIMS, more_floats)) {
            continue;
        }
        const float *const row = &tables[kind_of(layer).theta][(uint64_t)position * PAIR_FLOATS];
        for (uint32_t pair = ROTATED_PAIR_FIRST; pair < DIMS / PAIR; pair++) {
            const size_t dim = (size_t)pair * PAIR, angle = (size_t)(pair - ROTATED_PAIR_FIRST) * PAIR;
            const float a = floats[dim], b = floats[dim + 1u], c = row[angle], s = row[angle + 1u];
            const float expected[PAIR] = {more_floats[dim], more_floats[dim + 1u]};
            const float forms[FORMS][PAIR] = {{fmaf(a, c, -(b * s)), fmaf(b, c, a * s)},
                                              {a * c - b * s, a * s + b * c},
                                              {fmaf(-b, s, a * c), fmaf(a, s, b * c)}};
            for (uint32_t form = 0; form < FORMS; form++) {
                form_equal[form][0] += same_bits(forms[form][0], expected[0]) ? 1u : 0u;
                form_equal[form][1] += same_bits(forms[form][1], expected[1]) ? 1u : 0u;
            }
            form_pairs++;
        }
    }
    for (uint32_t form = 0; form < FORMS; form++) {
        (void)printf("ROTATION form=\"%s\" pairs=%" PRIu64 " real_bit_equal=%" PRIu64 " imaginary_bit_equal=%" PRIu64
                     "\n", form_names[form], form_pairs, form_equal[form][0], form_equal[form][1]);
    }
    (void)test_linear_expect(form_pairs != 0, "ROTATION counted the pairs of a capture point below 16,384");
}

/* The rotation step at every position of the tables and at the sampled ones, eight rows a launch. */
static void rotation_replay(struct test_linear_card *card,
			    const struct attention_state_kernels *kernels)
{
	struct attention_rotation_description rotation = {
		{ 0, 0 }, { 0, 0 }, { 0, 0 }, ROWS, 0
	};
	struct kernel_launch launch;
	test_linear_card_clear(card);
	rotation.step = test_linear_card_work(
		card,
		STEP_BLOCK_BYTES_VALUE + (uint64_t)STEP_ROW_BYTES_VALUE * ROWS);
	rotation.frequencies =
		test_linear_card_work(card, sizeof(sources.words));
	rotation.rotation = test_linear_card_work(
		card, (uint64_t)MODEL_ROTATION_ROW_BYTES * ROWS);
	if (!test_linear_card_has_room(card) ||
	    !test_linear_card_put(card, rotation.frequencies, 0, sources.words,
				  sizeof(sources.words)) ||
	    !test_linear_expect_status(
		    attention_rotation_prepare(&rotation,
					       card->arena_byte_count, kernels,
					       &launch),
		    STATUS_OK, "the rotation step is prepared") ||
	    !test_linear_card_load(card)) {
		return;
	}
	for (uint32_t batch = 0;
	     batch < TABLE_BATCHES + MODEL_THETAS * SAMPLED_BATCHES; batch++) {
		const bool is_sampled = batch >= TABLE_BATCHES;
		const uint32_t theta =
			is_sampled ? (batch - TABLE_BATCHES) / SAMPLED_BATCHES :
				     0;
		const uint32_t first_sample =
			is_sampled ? (batch - TABLE_BATCHES) % SAMPLED_BATCHES *
					     ROWS :
				     0;
		struct test_attention_row rows[ROWS];
		for (uint32_t row = 0; row < ROWS; row++) {
			rows[row].position =
				is_sampled ?
					(int32_t)sampled_positions
						[theta][first_sample + row] :
					(int32_t)(batch * ROWS + row);
			rows[row].session = 0;
		}
		if (!test_attention_step_put(card, rotation.step, rows, ROWS,
					     ROWS, batch) ||
		    !test_linear_card_run(card, &launch, 1) ||
		    !test_linear_card_fetch(card)) {
			return;
		}
		for (uint32_t row = 0; row < ROWS; row++) {
			const uint32_t position = (uint32_t)rows[row].position;
			for (uint32_t use = 0; use < MODEL_ROTATION_USES;
			     use++) {
				const uint32_t table_theta =
					use == ATTENTION_ROTATION_WINDOW_ONLY ?
						0 :
						1;
				const bool is_earlier =
					use == ATTENTION_ROTATION_GROUP_START;
				const float *expected = NULL;
				if (!is_sampled &&
				    !(is_earlier && position == 0)) {
					expected =
						&tables[table_theta]
						       [(uint64_t)(position -
								   (is_earlier ?
									    1u :
									    0u)) *
							PAIR_FLOATS];
				} else if (is_sampled && !is_earlier &&
					   table_theta == theta) {
					expected =
						&sampled[theta]
							[(uint64_t)(first_sample +
								    row) *
							 PAIR_FLOATS];
				}
				const uint64_t first_value =
					((uint64_t)row * MODEL_ROTATION_USES +
					 use) *
					PAIR_FLOATS;
				for (uint32_t value = 0;
				     expected != NULL && value < PAIR_FLOATS;
				     value++) {
					const int32_t distance =
						test_attention_float_ordinal(
							test_attention_float_at(
								card,
								rotation.rotation,
								first_value +
									value)) -
						test_attention_float_ordinal(
							expected[value]);
					test_attention_tally(
						tallies,
						TEST_ATTENTION_ROTATION,
						distance == 0,
						distance >= -1 && distance <= 1,
						"cos and sin against freqs_cis",
						position);
				}
			}
		}
	}
}

/* A norm's weight of a layer, read from the pack into the weights region; its range, or {0, 0} behind a failure. */
static struct linear_range norm_read(struct test_linear_card *card,
				     uint32_t layer, uint32_t norm)
{
    char name[NAME_CAPACITY];
    const int length = snprintf(name, sizeof(name), "layers.%" PRIu32 ".%s", layer, norm_names[norm]);
    const uint32_t tensor_index =
	    length > 0 && length < NAME_CAPACITY ?
		    pack_find_tensor(sources.pack, name, (uint32_t)length) :
		    PACK_NOT_FOUND;
    const uint64_t alignment = sources.block_alignment_bytes, byte_count = sizeof(uint16_t) * norm_widths[norm];
    if (tensor_index == PACK_NOT_FOUND) {
	    test_linear_fail("the pack holds no tensor %s", name);
	    return (struct linear_range){ 0, 0 };
    }
    const struct pack_tensor *const tensor =
	    &pack_tensors_of(sources.pack)[tensor_index];
    const struct pack_extent *const extent =
	    &pack_extents_of(sources.pack)[tensor->extent_index];
    if (tensor->element_type != PACK_ELEMENT_BFLOAT16 ||
	tensor->dimension_count != 1 ||
	tensor->dimensions[0] != norm_widths[norm] ||
	tensor->byte_count != byte_count) {
	    test_linear_fail("%s is not bfloat16 [%" PRIu32 "] in the pack",
			     name, norm_widths[norm]);
	    return (struct linear_range){ 0, 0 };
    }
    /* a read is whole blocks of the reader, from the block the tensor begins in to the end of the one it ends in */
    const uint64_t file_byte_offset = extent->byte_offset + tensor->byte_offset_in_extent;
    const uint64_t lead_byte_count = file_byte_offset % alignment;
    const uint64_t read_byte_count = (lead_byte_count + byte_count + alignment - 1u) / alignment * alignment;
    const struct linear_range read_range =
	    test_linear_card_weights(card, read_byte_count);
    unsigned char *const destination = test_linear_card_writable(card, read_range);
    if (destination == NULL ||
	!test_linear_expect_status(
		storage_direct.reader_read(sources.reader, 0,
					   file_byte_offset - lead_byte_count,
					   destination, read_byte_count),
		STATUS_OK, "a norm is read from the pack")) {
	    return (struct linear_range){ 0, 0 };
    }
    if (breaks("weight") && norm == TEST_ATTENTION_KV_NORM) {
        uint16_t weight = 0;
        memcpy(&weight, destination + lead_byte_count + BROKEN_WEIGHT * sizeof(weight), sizeof(weight));
        weight = test_linear_bfloat16_at(test_linear_bfloat16_ordinal(weight) + 1);
        memcpy(destination + lead_byte_count + BROKEN_WEIGHT * sizeof(weight), &weight, sizeof(weight));
    }
    return (struct linear_range){ read_range.byte_offset + lead_byte_count,
				  byte_count };
}

/* Captured values into an input array of the layer, uploaded. */
static bool input_put(struct test_linear_card *card, struct linear_range range,
		      uint64_t byte_offset, const void *values,
		      uint64_t byte_count)
{
    return test_linear_card_put(card, range, byte_offset, values, byte_count) &&
           test_linear_card_upload(card, range, byte_offset, byte_count);
}

/* Halves of the card against FP16 bits expected (placed: through source_of), within one place. */
static void halves_check(const struct test_linear_card *card,
			 struct linear_range range, uint64_t first,
			 const uint16_t *expected, const uint32_t *source_of,
			 uint32_t count, enum test_attention_check check,
			 const char *what)
{
    for (uint32_t index = 0; index < count; index++) {
        const uint16_t bits = test_attention_half_at(card, range, first + index);
        const uint16_t reference_bits = expected[source_of == NULL ? index : source_of[index]];
        const int32_t distance = test_attention_half_ordinal(bits) -
                                 expected_ordinal(test_attention_half_ordinal(reference_bits));
        test_attention_tally(tallies, check, distance == 0, distance >= -1 && distance <= 1, what, index);
    }
}

static void bfloat16_check(const struct test_linear_card *card,
			   struct linear_range range, const uint16_t *expected,
			   uint32_t count, enum test_attention_check check,
			   const char *what)
{
    for (uint32_t index = 0; index < count; index++) {
        const int32_t distance = test_linear_bfloat16_ordinal(test_attention_half_at(card, range, index)) -
                                 expected_ordinal(test_linear_bfloat16_ordinal(expected[index]));
        test_attention_tally(tallies, check, distance == 0, distance >= -1 && distance <= 1, what, index);
    }
}

/* Risk 8: a value FP16 keeps with fewer bits (below 2^-14) or not at all (above 65,504). */
static bool is_outside_half_range(double value)
{
    const double magnitude = value < 0.0 ? -value : value;
    return value != 0.0 && (magnitude < HALF_SMALLEST_NORMAL || magnitude > HALF_LARGEST);
}

/* The values of a kind before the reference's FP16 rounding that lie outside FP16's normal range, counted apart. */
static void range_count(struct capture_point point, const char *part, uint64_t count, enum test_attention_check check)
{
    if (count <= HEAD_VALUES && point_record(point.layer, point.position, part, TEST_CAPTURE_FLOAT32, count, floats)) {
        for (uint64_t index = 0; index < count; index++) {
            tallies[check].out_of_range += is_outside_half_range(floats[index]) ? 1u : 0u;
        }
    }
}

/* One capture point: its inputs, the launches, the checks. */
static void point_replay(struct test_linear_card *card, const struct test_attention_layer *on_card,
                         struct capture_point point, uint64_t sequence)
{
	const struct attention_state_description *const layer = &on_card->layer;
	const uint32_t ratio = layer->pool_ratio, position = point.position;
	const bool emits = ratio != 0 && (position + 1u) % ratio == 0;
	const bool has_index_queries = layer->has_index != 0 &&
				       position + 1u >=
					       compress_ratios[point.layer];
	const struct test_attention_row row = { (int32_t)position, 0 };
	bool is_ready = test_attention_step_put(card, layer->step, &row, 1, 1,
						sequence) &&
			point_bfloat16(point.layer, position, "attn.wq_a",
				       QUERY_RANK, halves) &&
			input_put(card, layer->query_rank, 0, halves,
				  sizeof(uint16_t) * QUERY_RANK) &&
			point_bfloat16(point.layer, position, "attn.wkv", DIMS,
				       halves) &&
			input_put(card, layer->window_input, 0, halves,
				  sizeof(uint16_t) * DIMS) &&
			point_bfloat16(point.layer, position, "attn.wq_b",
				       HEAD_VALUES, halves) &&
			input_put(card, layer->heads, 0, halves,
				  sizeof(uint16_t) * HEAD_VALUES);
	if (is_ready && has_index_queries) {
		is_ready = point_bfloat16(point.layer, position, "indexer.wq_b",
					  INDEX_VALUES, halves) &&
			   input_put(card, layer->index_heads, 0, halves,
				     sizeof(uint16_t) * INDEX_VALUES);
	}
    if (is_ready && ratio == 1) {
        is_ready = point_bfloat16(point.layer, position, "compressor.wkv", DIMS, halves) &&
                   input_put(card, layer->pool_input, 0, halves, sizeof(uint16_t) * DIMS);
    }
    if (is_ready && ratio == RATIO_TWO) {
        is_ready = point_record(point.layer, position, "compressor.wkv", TEST_CAPTURE_FLOAT32, DIMS, floats) &&
                   input_put(card, layer->pool_input, 0, floats, sizeof(float) * DIMS) &&
                   point_record(point.layer, position, "compressor.wgate", TEST_CAPTURE_FLOAT32, DIMS, floats) &&
                   input_put(card, layer->pool_score, 0, floats, sizeof(float) * DIMS);
    }
    if (is_ready && ratio == RATIO_TWO && emits) {             /* position P - 1 in its slot, as an earlier step */
	    const uint64_t slot =
		    (uint64_t)((position - 1u) % ATTENTION_COMPRESSOR_SLOTS) *
		    SLOT_FLOATS;
	    is_ready =
		    point_record(point.layer, position, "compressor.kv_state",
				 TEST_CAPTURE_FLOAT32, SLOT_FLOATS, floats) &&
		    input_put(card, layer->slots, sizeof(float) * slot, floats,
			      sizeof(float) * DIMS) &&
		    point_record(point.layer, position,
				 "compressor.score_state", TEST_CAPTURE_FLOAT32,
				 SLOT_FLOATS, floats) &&
		    input_put(card, layer->slots, sizeof(float) * (slot + DIMS),
			      floats, sizeof(float) * DIMS);
    }
    if (is_ready && emits) {
        is_ready = point_bfloat16(point.layer, position, "indexer.wk", INDEX_DIMS, halves) &&
                   input_put(card, layer->key_input, 0, halves, sizeof(uint16_t) * INDEX_DIMS);
    }
    if (!is_ready || !test_linear_card_run(card, on_card->launches, on_card->launch_count) ||
        !test_linear_card_fetch(card)) {
        return;
    }
    if (point_bfloat16(point.layer, position, "attn.qr", QUERY_RANK, halves)) {
        bfloat16_check(card, layer->query_normed, halves, QUERY_RANK, TEST_ATTENTION_QR, "qr");
    }
    if (point_half(point.layer, position, "attn.window_row", DIMS, halves)) {
	    halves_check(card, layer->ring,
			 (uint64_t)(position % MODEL_RING_SLOTS) * DIMS, halves,
			 NULL, DIMS, TEST_ATTENTION_WINDOW, "a window row");
    }
    range_count(point, "attn.window_row_unrounded", DIMS, TEST_ATTENTION_WINDOW);
    if (point_bfloat16(point.layer, position, "attn.q", HEAD_VALUES, halves)) {
        for (uint32_t index = 0; index < HEAD_VALUES; index++) {
            const double expected = test_linear_bfloat16_value(halves[index]);
            const double card_value = test_attention_half_value(test_attention_half_at(card, layer->queries, index));
            const double magnitude = expected < 0.0 ? -expected : expected;
            const double error = card_value > expected ? card_value - expected : expected - card_value;
            tallies[TEST_ATTENTION_QUERIES].out_of_range += is_outside_half_range(expected) ? 1u : 0u;
            test_attention_tally(tallies, TEST_ATTENTION_QUERIES, card_value == expected,
                                 magnitude <= HALF_LARGEST &&
                                     (card_value == expected ||
                                      (magnitude < HALF_SMALLEST_NORMAL && error <= HALF_SUBNORMAL_ERROR)),
                                 "a main query, against the reference's bfloat16 q", index);
        }
    }
    if (has_index_queries && point_half(point.layer, position, "indexer.query", INDEX_VALUES, halves)) {
        halves_check(card, layer->index_queries, 0, halves, index_source_of, INDEX_VALUES,
                     TEST_ATTENTION_INDEX_QUERIES, "an index query");
        range_count(point, "indexer.query_unrounded", INDEX_VALUES, TEST_ATTENTION_INDEX_QUERIES);
    }
    if (ratio == RATIO_TWO && !emits &&
        point_record(point.layer, position, "compressor.kv_state", TEST_CAPTURE_FLOAT32, SLOT_FLOATS, floats) &&
        point_record(point.layer, position, "compressor.score_state", TEST_CAPTURE_FLOAT32, SLOT_FLOATS,
                     floats + SLOT_FLOATS)) {
	    const uint64_t slot =
		    (uint64_t)(position % ATTENTION_COMPRESSOR_SLOTS) *
		    SLOT_FLOATS;
	    for (uint32_t dim = 0; dim < DIMS;
		 dim++) { /* slot 0 of the reference's two: this position */
		    const bool is_kept =
			    same_bits(test_attention_float_at(
					      card, layer->slots, slot + dim),
				      floats[dim]) &&
			    same_bits(
				    test_attention_float_at(card, layer->slots,
							    slot + DIMS + dim),
				    floats[SLOT_FLOATS + dim]);
		    test_attention_tally(tallies, TEST_ATTENTION_SLOTS, is_kept,
					 is_kept, "a slot", dim);
	    }
    }
    if (!emits) {
        return;
    }
    const uint64_t entry = position / ratio;
    if (point_bfloat16(point.layer, position, "indexer.latent_bfloat16", DIMS, halves)) {
        bfloat16_check(card, layer->latent, halves, DIMS, TEST_ATTENTION_LATENT, "the latent wk reads");
    }
    if (point_half(point.layer, position, "compressor.pool_entry", DIMS, halves)) {
        halves_check(card, layer->pool, entry * DIMS, halves, NULL, DIMS, TEST_ATTENTION_POOL, "a pool entry");
    }
    if (point_half(point.layer, position, "indexer.key", INDEX_DIMS, halves)) {
        halves_check(card, layer->keys, entry * INDEX_DIMS, halves, key_source_of, INDEX_DIMS, TEST_ATTENTION_KEYS,
                     "a key");
    }
    range_count(point, "compressor.pool_entry_unrounded", DIMS, TEST_ATTENTION_POOL);
    range_count(point, "indexer.key_unrounded", INDEX_DIMS, TEST_ATTENTION_KEYS);
}

/* Every capture point of every layer, a layer laid out afresh with its norms from the pack. */
static void layers_replay(struct test_linear_card *card,
			  const struct attention_state_kernels *kernels)
{
    uint64_t sequence = 1;
    for (uint32_t first = 0; first < sources.point_count;) {
        const uint32_t layer = sources.points[first].layer;
        const struct test_attention_kind kind = kind_of(layer);
	struct linear_range norms[TEST_ATTENTION_NORMS] = {
		{ 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 }
	};
	struct test_attention_layer on_card;
        uint32_t end = first;
        while (end < sources.point_count && sources.points[end].layer == layer) {
            end++;
        }
        test_linear_context("card %" PRIu32 ", the replay of layer %" PRIu32, card->card_index, layer);
        test_linear_card_clear(card);
        bool is_ready = true;
        for (uint32_t norm = 0; norm < TEST_ATTENTION_NORMS && is_ready; norm++) {
            if (norm < TEST_ATTENTION_POOL_NORM || kind.pool_ratio != 0) {
                norms[norm] = norm_read(card, layer, norm);
                is_ready = norms[norm].byte_count != 0;
            }
        }
	const struct linear_range frequencies =
		test_linear_card_work(card, sizeof(sources.words));
	if (is_ready && test_linear_card_put(card, frequencies, 0, sources.words, sizeof(sources.words)) &&
            test_attention_layer_set_aside(card, kernels, &on_card, kind, 1, 1, sources.context, norms,
                                           frequencies) &&
            test_linear_card_load(card)) {
            for (uint32_t point = first; point < end; point++) {
                point_replay(card, &on_card, sources.points[point], sequence++);
            }
        }
        first = end;
    }
}

/* Without a capture: the module loads on the card and its rotation step runs, words of zero giving cos 1 and sin 0
 * (so that compute-sanitizer, which `make card_tests` runs every card test under, sees the kernels at work). */
static void smoke_run(uint32_t card_index)
{
    struct test_linear_card card;
    struct attention_state_kernels kernels;
    struct attention_rotation_description rotation = {
	    { 0, 0 }, { 0, 0 }, { 0, 0 }, ROWS, 0
    };
    struct kernel_launch launch;
    struct test_attention_row rows[ROWS];
    for (uint32_t row = 0; row < ROWS; row++) {
        rows[row] = (struct test_attention_row){(int32_t)row, 0};
    }
    if (test_linear_card_open(&card, card_index, WORK_CAPACITY,
			      WEIGHTS_CAPACITY) &&
	test_linear_expect_status(
		attention_state_kernels_prepare(&backend_cuda_value, card.card,
						&kernels),
		STATUS_OK, "the kernels of the attention state are prepared")) {
	    rotation.step = test_linear_card_work(
		    &card, STEP_BLOCK_BYTES_VALUE +
				   (uint64_t)STEP_ROW_BYTES_VALUE * ROWS);
	    rotation.frequencies =
		    test_linear_card_work(&card, sizeof(sources.words));
	    rotation.rotation = test_linear_card_work(
		    &card, (uint64_t)MODEL_ROTATION_ROW_BYTES * ROWS);
	    if (test_linear_card_has_room(&card) &&
		test_linear_expect_status(
			attention_rotation_prepare(&rotation,
						   card.arena_byte_count,
						   &kernels, &launch),
			STATUS_OK, "the rotation step is prepared") &&
		test_linear_card_load(&card) &&
		test_attention_step_put(&card, rotation.step, rows, ROWS, ROWS,
					1) &&
		test_linear_card_run(&card, &launch, 1) &&
		test_linear_card_fetch(&card)) {
		    for (uint64_t value = 0;
			 value <
			 (uint64_t)ROWS * MODEL_ROTATION_USES * PAIR_FLOATS;
			 value += PAIR) {
			    (void)test_linear_expect(
				    test_attention_float_at(&card,
							    rotation.rotation,
							    value) == 1.0f &&
					    test_attention_float_at(
						    &card, rotation.rotation,
						    value + 1u) == 0.0f,
				    "an angle of zero has cos 1 and sin 0");
		    }
	    }
    }
    test_linear_card_close(&card);
}

static void card_run(uint32_t card_index)
{
    struct test_linear_card card;
    struct attention_state_kernels kernels;
    if (test_linear_card_open(&card, card_index, WORK_CAPACITY,
			      WEIGHTS_CAPACITY) &&
	test_linear_expect_status(
		attention_state_kernels_prepare(&backend_cuda_value, card.card,
						&kernels),
		STATUS_OK, "the kernels of the attention state are prepared")) {
	    test_linear_context("card %" PRIu32
				", the rotation step against freqs_cis",
				card_index);
	    rotation_replay(&card, &kernels);
	    layers_replay(&card, &kernels);
    }
    test_linear_card_close(&card);
    char what[NAME_CAPACITY];
    (void)snprintf(what, sizeof(what), "card %" PRIu32, card_index);
    test_attention_tallies_print(tallies, what);
    test_attention_tallies_expect_counted(tallies, TEST_ATTENTION_ALONE);     /* every check but the synthetic one */
    test_attention_tallies_expect_shares(tallies);
    memset(tallies, 0, sizeof(tallies));
}

int main(int argument_count, char **arguments)
{
	struct storage_description description = { 0, 0, 0 };
	uint32_t card_count = 0;
	const bool has_break = argument_count > ARGUMENTS_OF_A_BREAK &&
			       strcmp(arguments[1], BREAK_FLAG) == 0;
	const int first_argument = has_break ? 1 + ARGUMENTS_OF_A_BREAK : 1;
	broken = has_break ? arguments[ARGUMENTS_OF_A_BREAK] : NULL;
	if (argument_count == first_argument && broken == NULL) {
		if (test_linear_expect_status(
			    backend_cuda_value.card_count(&card_count),
			    STATUS_OK, "the cards are counted") &&
		    test_linear_expect(card_count > 0 &&
					       card_count <=
						       BACKEND_MAXIMUM_CARDS,
				       "the machine has cards for the test")) {
			for (uint32_t card_index = 0; card_index < card_count;
			     card_index++) {
				smoke_run(card_index);
			}
		}
		if (test_linear_failures() != 0) {
			(void)fprintf(
				stderr,
				"attention_state_card_test FAILED: %d failures\n",
				test_linear_failures());
			return EXIT_FAILURE;
		}
		(void)printf(
			"attention_state_card_test passed on %" PRIu32
			" cards (nothing replayed: the replay needs a "
			"capture folder and a pack; the rotation step ran)\n",
			card_count);
		return EXIT_SUCCESS;
	}
	if (argument_count - first_argument < ARGUMENTS_OF_A_BREAK ||
	    argument_count - first_argument - 1 > STORAGE_MAXIMUM_COPIES ||
	    (broken != NULL && !breaks("weight") && !breaks("frequency") &&
	     !breaks("rounding"))) {
		(void)fprintf(
			stderr,
			"usage: attention_state_card_test [%s weight|frequency|rounding] [<capture folder> "
			"<pack copy> [<pack copy> ...]]\n",
			BREAK_FLAG);
		return EXIT_FAILURE;
	}
    (void)alarm(TEST_TIME_LIMIT_SECONDS);
    if (broken != NULL) {
        (void)printf("BREAK %s: the test must fail\n", broken);
    }
    const char *const *const copy_paths = (const char *const *)&arguments[first_argument + 1];
    const uint32_t copy_count = (uint32_t)(argument_count - first_argument - 1);
    result_status status =
	    storage_direct.reader_open(copy_paths, copy_count, &sources.reader);
    status = status == STATUS_OK ?
		     storage_direct.reader_describe(sources.reader,
						    &description) :
		     status;
    status = status == STATUS_OK ?
		     pack_open_entry(&storage_direct, sources.reader,
				     &sources.pack) :
		     status;
    sources.block_alignment_bytes = description.block_alignment_bytes;
    if (test_linear_expect_status(status, STATUS_OK, "the pack opens") &&
	test_linear_expect(
		description.block_alignment_bytes != 0 &&
			BACKEND_STAGING_ALIGNMENT_BYTES %
					description.block_alignment_bytes ==
				0,
		"the reader's blocks divide a staging page") &&
	test_linear_expect_status(test_capture_open(arguments[first_argument],
						    &sources.capture),
				  STATUS_OK, "the capture opens") &&
	points_find() && frequencies_read() &&
	test_linear_expect_status(backend_cuda_value.card_count(&card_count),
				  STATUS_OK, "the cards are counted") &&
	test_linear_expect(card_count > 0 &&
				   card_count <= BACKEND_MAXIMUM_CARDS,
			   "the machine has cards for the test")) {
	    forms_count();
	    for (uint32_t card_index = 0; card_index < card_count;
		 card_index++) {
		    card_run(card_index);
	    }
    }
    test_capture_close(sources.capture);
    pack_close(sources.pack);
    storage_direct.reader_close(sources.reader);
    if (test_linear_failures() != 0) {
        (void)fprintf(stderr, "attention_state_card_test FAILED: %d failures\n", test_linear_failures());
        return EXIT_FAILURE;
    }
    (void)printf("attention_state_card_test passed: %" PRIu32 " capture points on %" PRIu32 " cards\n",
                 sources.point_count, card_count);
    return EXIT_SUCCESS;
}
