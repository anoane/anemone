/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * teacher_forced_test: the teacher-forced harness (teacher_forced.h) and the program view (program_view.h) on the
 * backend in host memory with the synthetic model of teacher_forced_synthetic.h, and the capture writer
 * (capture_writer.h) through the capture reader.
 *
 *   - A synthetic capture of five positions (written by the capture writer, read by the harness through the reader)
 *     with capture points at layer 1 position 2 and layer 3 position 4.
 *   - Each run on cards opened again (their mirrors zero, the backend's counts from zero). Injected: every record of
 *     the folder, read back through the capture reader, equals what the host computes from the capture's inputs
 *     (layer 0 from the embedding): the streams after the n-gram gate, the streams and pre leaving each layer, the
 *     experts and weights in the gate's order, the biased scores, every record of a capture point, the logits; the
 *     other segment ran once a step; the pool entries reached card 1 and nothing else did; card 0 had the most copies
 *     a card may have in flight, and no more. Free: each layer from the one before, across the cards. The break: the
 *     layer broken alone differs, as computed with a zero pre. A subset: a layer left out is not written, the next is
 *     given the capture's streams.
 *   - Fused: the same model whose layers are the fused launches that stand for their sites (site front, output
 *     projection, FFN front with the select, join with its row sums): every record of the folder the host's, as above.
 *     The model's own programs are read with each selected arithmetic class and reversed physical card order.
 *   - Front split: fused and unfused gate layer 1 in two physical fragments, injected and free. Every capture
 *     value is the host's, each physical launch runs once, and the pool hand-off at the front boundary is preserved.
 *   - Refused: an output folder that exists, a layer segment without its router, an n-gram gate in a segment of no
 *     layer or behind its layer's attention site, a program that is not thin, a card with views, a hand-off of a
 *     view, a fused program whose join stores no row sums, positions beyond the capture's, a free run with a layer
 *     left out, a break of layer 0; a writer's record given twice, a slab put before the one ahead of it, again, or
 *     beyond a record.
 *   - The model's own program (model_v41_program.c) for the five cards of first_token_plan.md 1.3, from the tables of
 *     its pack (model_v41_pack.h) on cards in host memory: the view reads it whole (each layer's sites around its
 *     router on its card's streams and carried pre, the n-gram gates of layers 1 and 14 before them; the embedding on
 *     its card, the head on the result card, their records on the streams their cards hand on) and its hand-offs fit
 *     the harness.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/model_layout.h"
#include "capture_reader.h"
#include "capture_writer.h"
#include "host_memory_backend.h"
#include "memory_storage.h"
#include "model_v41.h"
#include "model_v41_pack.h"
#include "program_view.h"
#include "teacher_forced.h"
#include "teacher_forced_synthetic.h"

#define CARDS TEST_SYNTHETIC_CARDS
#define LAYERS TEST_SYNTHETIC_LAYERS
#define POSITIONS TEST_SYNTHETIC_POSITIONS
#define BREAK_LAYER 2u
#define POINT_LAYER 1u
#define POINT_POSITION 2u
#define LATE_POINT_LAYER 3u
#define LATE_POINT_POSITION 4u
#define WAIT_LIMIT_NANOSECONDS 1000000000ull
#define PATH_CAPACITY 512u
#define NAME_CAPACITY 96u
#define D MODEL_HIDDEN
#define K MODEL_STREAMS
#define SCRATCH_BYTES 1048576u                  /* the widest record read back: a row of logits */
#define TEMPORARY_TEMPLATE "teacher_forced_test_XXXXXX"
#define MODEL_CARDS TEST_MODEL_V41_CARDS
#define MODEL_RESULT_CARD 4u                    /* the PRO of first_token_plan.md 1.3 */
#define MODEL_CARD_BYTES 4096u                  /* no launch runs: a program only names offsets */
#define MODEL_CARD_MEMORY_BYTES 4194304ull
#define HARNESS_HAND_OFF_BYTES 131072u          /* teacher_forced.c's staging of a hand-off */
#define TOKEN_DIMENSIONS 2u                     /* [1, positions] */
#define MATRIX_DIMENSIONS 2u
#define FIRST_LAYER_RECORD 1u                   /* of the synthetic capture: the token ids, then each layer's two, */
#define RECORDS_PER_LAYER 2u                    /* its streams and its pre, then the two points */
#define CAPTURE_POINTS 2u
#define CAPTURE_RECORDS (FIRST_LAYER_RECORD + RECORDS_PER_LAYER * LAYERS + CAPTURE_POINTS)
#define FIRST_LAYER_SEGMENT 1u                  /* the embedding's segment, then layer 0's and layer 1's */
#define GATE_LAYER_SEGMENT 2u
#define GATE_IN_ITS_SEGMENT 1u                  /* the gate layer's segment: the rows-to-values launch, the gate */
#define SPLIT_SEGMENTS (TEST_SYNTHETIC_SEGMENTS + 1u)
#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
enum { ATTENTION, FFN };

static const struct backend_operations *const backend =
	&test_host_memory_backend;
static int failures = 0;

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)fprintf(stderr, "FAILED: %s\n", what);
        failures++;
    }
}

/* The synthetic capture of the module text, through the capture writer, with its refusals. */
static bool capture_write(const char *folder, struct test_synthetic_row *row)
{
    const uint64_t tokens[] = {1, POSITIONS}, output[] = {1, POSITIONS, K, D}, carried[] = {1, POSITIONS, K};
    const uint64_t hidden[] = {D};
    const uint32_t point_layers[] = {POINT_LAYER, LATE_POINT_LAYER}, point_positions[] = {POINT_POSITION,
                                                                                         LATE_POINT_POSITION};
    struct test_capture_writer *writer = NULL;
    char name[NAME_CAPACITY];
    uint32_t index = 0;
    if (row == NULL) {
        return false;
    }
    bool held = test_capture_writer_create(CAPTURE_RECORDS + 1u, &writer) ==
			STATUS_OK &&
		test_capture_writer_add(writer, "input.token_ids",
					TEST_CAPTURE_INT64, tokens,
					TOKEN_DIMENSIONS, POSITIONS,
					TEST_CAPTURE_NO_LAYER,
					TEST_CAPTURE_WRITER_NO_POSITION,
					"synthetic", &index) == STATUS_OK;
    for (uint32_t layer = 0; held && layer < LAYERS; layer++) {
        (void)snprintf(name, sizeof(name), "layer_%" PRIu32 ".block.output", layer);
	held = test_capture_writer_add(writer, name, TEST_CAPTURE_BFLOAT16,
				       output, COUNT_OF(output), POSITIONS,
				       layer, TEST_CAPTURE_WRITER_NO_POSITION,
				       "synthetic", &index) == STATUS_OK;
	(void)snprintf(name, sizeof(name), "layer_%" PRIu32 ".block.output_collapse_weights", layer);
	held = held &&
	       test_capture_writer_add(writer, name, TEST_CAPTURE_FLOAT32,
				       carried, COUNT_OF(carried), POSITIONS,
				       layer, TEST_CAPTURE_WRITER_NO_POSITION,
				       "synthetic", &index) == STATUS_OK;
    }
    for (uint32_t point = 0; held && point < COUNT_OF(point_layers); point++) {
        (void)snprintf(name, sizeof(name), "layer_%" PRIu32 ".position_%" PRIu32 ".attn_site.normed",
                       point_layers[point], point_positions[point]);
	held = test_capture_writer_add(writer, name, TEST_CAPTURE_BFLOAT16,
				       hidden, 1, 1, point_layers[point],
				       point_positions[point], "synthetic",
				       &index) == STATUS_OK;
    }
    held = held &&
	   test_capture_writer_add(writer, "input.token_ids",
				   TEST_CAPTURE_INT64, tokens, TOKEN_DIMENSIONS,
				   POSITIONS, TEST_CAPTURE_NO_LAYER,
				   TEST_CAPTURE_WRITER_NO_POSITION, "synthetic",
				   &index) == STATUS_INVALID_ARGUMENT;
    held = held && test_capture_writer_open(writer, folder) == STATUS_OK &&
	   test_capture_writer_put(writer, FIRST_LAYER_RECORD, 1,
				   row->streams) == STATUS_INVALID_ARGUMENT;
    for (uint32_t position = 0; held && position < POSITIONS; position++) {
        const int64_t token = test_synthetic_token_at(position);
	held = test_capture_writer_put(writer, 0, position, &token) ==
	       STATUS_OK;
	for (uint32_t layer = 0; held && layer < LAYERS; layer++) {
            test_synthetic_capture_row(layer, position, row);
            const uint32_t record = FIRST_LAYER_RECORD + RECORDS_PER_LAYER * layer;
	    held = test_capture_writer_put(writer, record, position,
					   row->streams) == STATUS_OK &&
		   test_capture_writer_put(writer, record + 1u, position,
					   row->pre) == STATUS_OK;
	}
    }
    for (uint32_t point = 0; held && point < COUNT_OF(point_layers); point++) {
	    held = test_capture_writer_put(writer,
					   FIRST_LAYER_RECORD +
						   RECORDS_PER_LAYER * LAYERS +
						   point,
					   0, row->streams[0]) == STATUS_OK;
    }
    held = held &&
	   test_capture_writer_put(writer, 0, POSITIONS, row->pre) ==
		   STATUS_INVALID_ARGUMENT &&
	   test_capture_writer_put(writer, 0, POSITIONS - 1u, row->pre) ==
		   STATUS_INVALID_ARGUMENT;
    return test_capture_writer_close(writer, held, "{}", "{}") == STATUS_OK &&
	   held;
}

static bool record_is(const struct test_capture *capture, const char *name, enum test_capture_type type,
                      const uint64_t *dimensions, uint32_t dimension_count, uint64_t position, const void *expected,
                      uint64_t byte_count, unsigned char *scratch)
{
    const uint64_t count = byte_count / test_capture_element_bytes(type);
    const uint64_t first = dimension_count > 1u ? position * count : 0;
    return test_capture_find(capture, name) !=
		   NULL && /* a record not written: no error line */
	   test_capture_read(capture, name, type, dimensions, dimension_count,
			     first, count, scratch, byte_count) == STATUS_OK &&
	   memcmp(scratch, expected, byte_count) == 0;
}

/* A layer's records at a position against the host's values: those of every position, and its capture point's
 * records where it has one; returns the records that differ. */
static uint32_t layer_check(const struct test_capture *capture, uint32_t layer, uint32_t position, bool runs,
                            const struct test_synthetic_layer *value, unsigned char *scratch)
{
    const uint64_t output[] = {1, POSITIONS, K, D}, carried[] = {1, POSITIONS, K};
    const uint64_t picks[] = { POSITIONS, MODEL_EXPERTS_PER_ROW },
		   scores[] = { POSITIONS, MODEL_EXPERTS };
    const uint64_t hidden[] = { D }, streams[] = { K, D },
		   experts[] = { MODEL_EXPERTS }, four[] = { K };
    const uint64_t mixes[] = { MODEL_MIXES }, comb[] = { K, K };
    const bool is_point = runs && ((layer == POINT_LAYER && position == POINT_POSITION) ||
                                   (layer == LATE_POINT_LAYER && position == LATE_POINT_POSITION));
    const bool has_gate = runs && layer == TEST_SYNTHETIC_GATE_LAYER;
    const struct { const char *suffix; const uint64_t *dimensions; const void *expected; uint64_t bytes;
                   enum test_capture_type type; uint32_t count; bool written; } records[] = {
        {"block.output", output, value->output.streams, sizeof(value->output.streams), TEST_CAPTURE_BFLOAT16,
         COUNT_OF(output), runs},
        {"block.output_collapse_weights", carried, value->output.pre, sizeof(value->output.pre),
         TEST_CAPTURE_FLOAT32, COUNT_OF(carried), runs},
        {"mlp.routed_expert_ids", picks, value->gate_ids, sizeof(value->gate_ids), TEST_CAPTURE_INT64,
         MATRIX_DIMENSIONS, runs},
        {"mlp.route_weights", picks, value->gate_weights, sizeof(value->gate_weights), TEST_CAPTURE_FLOAT32,
         MATRIX_DIMENSIONS, runs},
        {"mlp.biased_scores", scores, value->biased, sizeof(value->biased), TEST_CAPTURE_FLOAT32, MATRIX_DIMENSIONS,
         runs},
        {"ngram.output", output, value->ngram, sizeof(value->ngram), TEST_CAPTURE_BFLOAT16, COUNT_OF(output),
         has_gate}};
    const struct { const char *suffix; const uint64_t *dimensions; const void *expected; uint64_t bytes;
                   enum test_capture_type type; uint32_t count; } points[] = {
        {"attn_site.mixes", mixes, value->mixes[ATTENTION], sizeof(value->mixes[0]), TEST_CAPTURE_FLOAT32, 1},
        {"attn_site.pre", four, value->pre[ATTENTION], sizeof(value->pre[0]), TEST_CAPTURE_FLOAT32, 1},
        {"attn_site.post", four, value->post[ATTENTION], sizeof(value->post[0]), TEST_CAPTURE_FLOAT32, 1},
        {"attn_site.comb", comb, value->comb[ATTENTION], sizeof(value->comb[0]), TEST_CAPTURE_FLOAT32,
         MATRIX_DIMENSIONS},
        {"attn_site.normed", hidden, value->hidden[ATTENTION], sizeof(value->hidden[0]), TEST_CAPTURE_BFLOAT16, 1},
        {"attn.wo_b", hidden, value->wo_b, sizeof(value->wo_b), TEST_CAPTURE_BFLOAT16, 1},
        {"ffn_site.streams", streams, value->ffn_streams, sizeof(value->ffn_streams), TEST_CAPTURE_BFLOAT16,
         MATRIX_DIMENSIONS},
        {"ffn_site.mixes", mixes, value->mixes[FFN], sizeof(value->mixes[0]), TEST_CAPTURE_FLOAT32, 1},
        {"ffn_site.pre", four, value->pre[FFN], sizeof(value->pre[0]), TEST_CAPTURE_FLOAT32, 1},
        {"ffn_site.post", four, value->post[FFN], sizeof(value->post[0]), TEST_CAPTURE_FLOAT32, 1},
        {"ffn_site.comb", comb, value->comb[FFN], sizeof(value->comb[0]), TEST_CAPTURE_FLOAT32, MATRIX_DIMENSIONS},
        {"ffn_site.normed", hidden, value->hidden[FFN], sizeof(value->hidden[0]), TEST_CAPTURE_BFLOAT16, 1},
        {"router.logits", experts, value->hidden[FFN], sizeof(value->affinities), TEST_CAPTURE_FLOAT32, 1},
        {"router.scores", experts, value->affinities, sizeof(value->affinities), TEST_CAPTURE_FLOAT32, 1},
        {"router.biased_scores", experts, value->biased, sizeof(value->biased), TEST_CAPTURE_FLOAT32, 1},
        {"experts.row_sum", hidden, value->sums, sizeof(value->sums), TEST_CAPTURE_FLOAT32, 1},
        {"shared_expert.output", hidden, value->shared, sizeof(value->shared), TEST_CAPTURE_BFLOAT16, 1}};
    char name[NAME_CAPACITY];
    uint32_t differing = 0;
    for (uint32_t record = 0; record < COUNT_OF(records); record++) {
        (void)snprintf(name, sizeof(name), "layer_%" PRIu32 ".%s", layer, records[record].suffix);
        differing += records[record].written != record_is(capture, name, records[record].type,
                                                          records[record].dimensions, records[record].count, position,
                                                          records[record].expected, records[record].bytes, scratch)
                   ? 1u : 0;
    }
    for (uint32_t point = 0; point < COUNT_OF(points); point++) {
        (void)snprintf(name, sizeof(name), "layer_%" PRIu32 ".position_%" PRIu32 ".%s", layer, position,
                       points[point].suffix);
        differing += is_point != record_is(capture, name, points[point].type, points[point].dimensions,
                                           points[point].count, 0, points[point].expected, points[point].bytes,
                                           scratch) ? 1u : 0;
    }
    return differing;
}

/* Every record of a run against the host's values; returns the records that differ. */
static uint32_t folder_check(const char *folder, enum test_teacher_forced_mode mode, uint64_t layers,
                             uint32_t broken_layer, struct test_synthetic_layer *values, unsigned char *scratch)
{
    const uint64_t tokens[] = {1, POSITIONS}, embeddings[] = {1, POSITIONS, D};
    const uint64_t logits[] = { 1, POSITIONS, MODEL_VOCABULARY_VALUE };
    struct test_capture *capture = NULL;
    struct test_synthetic_row *const input = calloc(LAYERS, sizeof(struct test_synthetic_row));
    float *const logit_row = calloc(MODEL_VOCABULARY_VALUE, sizeof(float));
    uint32_t differing = test_capture_open(folder, &capture) == STATUS_OK &&
					 input != NULL && logit_row != NULL ?
				 0 :
				 1u;
    for (uint32_t position = 0; differing == 0 && position < POSITIONS; position++) {
        const int64_t token = test_synthetic_token_at(position);
        test_synthetic_embedding_row(position, &input[0]);
        differing += record_is(capture, "input.token_ids", TEST_CAPTURE_INT64, tokens, TOKEN_DIMENSIONS, position,
                               &token, sizeof(token), scratch) ? 0 : 1u;
        differing += record_is(capture, "input.embeddings", TEST_CAPTURE_BFLOAT16, embeddings, COUNT_OF(embeddings),
                               position, input[0].streams[0], sizeof(input[0].streams[0]), scratch) ? 0 : 1u;
        for (uint32_t layer = 0; layer < LAYERS; layer++) {
            if (layer > 0 && mode == TEST_TEACHER_FORCED_INJECTED) {
                test_synthetic_capture_row(layer - 1u, position, &input[layer]);
            } else if (layer > 0) {
                input[layer] = values[layer - 1u].output;
            }
            if (layer == broken_layer) {
                memset(input[layer].pre, 0, sizeof(input[layer].pre));
            }
            test_synthetic_layer_compute(layer, &input[layer], &values[layer]);
            differing += layer_check(capture, layer, position, (layers >> layer & 1u) != 0, &values[layer], scratch);
        }
        struct test_synthetic_row *const head = &input[0];             /* the embedding's row is not read again */
        if (mode == TEST_TEACHER_FORCED_INJECTED) {
            test_synthetic_capture_row(LAYERS - 1u, position, head);
        } else {
            *head = values[LAYERS - 1u].output;
        }
	for (uint32_t index = 0; index < MODEL_VOCABULARY_VALUE; index++) {
		logit_row[index] = test_synthetic_logit_of(
			test_synthetic_token_at(position), index,
			head->streams[TEST_SYNTHETIC_HEAD_INPUT_STREAM]);
	}
	differing += record_is(capture, "logits.all_rows", TEST_CAPTURE_FLOAT32,
			       logits, COUNT_OF(logits), position, logit_row,
			       MODEL_LOGITS_ROW_BYTES, scratch) ?
			     0 :
			     1u;
    }
    test_capture_close(capture);
    free(input);
    free(logit_row);
    return differing;
}

static void folder_remove(const char *base, const char *folder)
{
    static const char *const files[] = {"manifest.json", "tensors.bin"};
    char path[PATH_CAPACITY];
    for (uint32_t file = 0; file < COUNT_OF(files); file++) {
        const int length = snprintf(path, sizeof(path), "%s/%s/%s", base, folder, files[file]);
        if (length > 0 && (size_t)length < sizeof(path)) {
            (void)unlink(path);
        }
    }
    const int length = snprintf(path, sizeof(path), "%s/%s", base, folder);
    if (length > 0 && (size_t)length < sizeof(path)) {
        (void)rmdir(path);
    }
}

/* The view of the model's own program for the five cards of first_token_plan.md 1.3, from its pack's tables. */
static void test_model_program(uint32_t result_class)
{
    static uint32_t card_of_unit[TEST_MODEL_V41_UNIT_COUNT];
    const struct model_limits limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES,
					 1,
					 1,
					 STATE_WIDTH_16,
					 LINEAR_PROFILE_PRECISION,
					 LINEAR_ORDER_FIXED,
					 CMP_ORDER_REFERENCE,
					 0,
					 0,
					 0 };
    const struct model_operations *const model =
	    model_operations_of(V41_DATA_MODEL_NAME);
    struct device_card *cards[MODEL_CARDS] = { NULL };
    struct card_description_record descriptions[MODEL_CARDS];
    struct pack_record *pack = NULL;
    struct load_layout *layout = NULL;
    struct model_plan_record *plan = NULL;
    struct execution_program *program = NULL;
    struct test_program_view *view = NULL;
    bool held = model != NULL &&
		test_model_v41_pack_open(
			(struct test_model_v41_change){
				TEST_MODEL_V41_CHANGE_NONE, NULL, 0 },
			&pack) == STATUS_OK;
    test_host_memory_backend_reset(MODEL_CARDS, MODEL_CARD_MEMORY_BYTES);
    for (uint32_t card = 0; held && card < MODEL_CARDS; card++) {
        const uint32_t physical_card = MODEL_CARDS - 1u - card;
	const uint32_t selected_class = card == MODEL_RESULT_CARD ?
						result_class :
						ARITHMETIC_CLASS_SM_80;
	const uint32_t physical_class =
		selected_class == ARITHMETIC_CLASS_SM_120_NATIVE ?
			ARITHMETIC_CLASS_SM_120 :
			selected_class;
	test_host_memory_backend_set_arithmetic_class(physical_card, physical_class);
	held = backend->card_describe(physical_card, &descriptions[card]) ==
		       STATUS_OK &&
	       backend->card_open(physical_card, MODEL_CARD_BYTES,
				  &cards[card]) == STATUS_OK;
	descriptions[card].arithmetic_class = selected_class;
    }
    if (held) {
        test_model_v41_five_cards(card_of_unit, 0, 0);
    }
    held = held &&
	   load_layout_create(pack, card_of_unit, TEST_MODEL_V41_UNIT_COUNT,
			      MODEL_CARDS, &layout) == STATUS_OK &&
	   model->plan_create(pack, layout, &limits, &plan) == STATUS_OK &&
	   model->program_prepare(plan, backend, cards, descriptions,
				  PROGRAM_THIN, 1, &program) == STATUS_OK &&
	   test_program_view_create(backend, cards, descriptions, program,
				    MODEL_LAYERS, (1ull << MODEL_LAYERS) - 1u,
				    &view) == STATUS_OK;
    expect(held, "the model's program for the five cards is made and the view reads it");
    if (held) {
	    const struct program_card_record *const first =
		    &program->cards[program->segments[view->embed_segment].card];
	    const struct program_card_record *const result =
		    &program->cards[MODEL_RESULT_CARD];
	    bool fits = view->head_segment != TEST_PROGRAM_VIEW_NONE &&
			program->result_card == MODEL_RESULT_CARD &&
			program->segments[view->head_segment].card ==
				MODEL_RESULT_CARD &&
			view->embed.streams_offset == first->streams_offset &&
			view->embed.pre_offset == first->carried_pre_offset &&
			view->final.streams_offset == result->streams_offset &&
			view->final.pre_offset == result->carried_pre_offset;
	    for (uint32_t index = 0; index < program->hand_off_count; index++) {
		    fits = fits && program->hand_offs[index].byte_count <=
					   HARNESS_HAND_OFF_BYTES;
	    }
	    for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
		    fits = fits && (view->layers[layer].gate_launch !=
				    TEST_PROGRAM_VIEW_NONE) ==
					   ((V41_DATA_ENGRAM_LAYERS >> layer &
					     1u) != 0);
	    }
	expect(fits, "the embedding on its card and the head on the result card read and write the streams and pre "
                     "the hand-offs carry, every hand-off fits the harness's staging, the n-gram gates are 1 and 14's");
    }
    test_program_view_destroy(view);
    if (program != NULL) {
        model->program_destroy(program);
    }
    if (plan != NULL) {
        model->plan_destroy(plan);
    }
    load_layout_destroy(layout);
    for (uint32_t card = 0; card < MODEL_CARDS; card++) {
        backend->card_close(cards[card], WAIT_LIMIT_NANOSECONDS);
    }
    pack_close(pack);
    test_memory_storage_reset();
}

/* A copy of the program whose own arrays the test may change. */
static void program_copy(const struct test_synthetic_program *made, struct test_synthetic_program *copy)
{
    *copy = *made;
    for (uint32_t card = 0; card < CARDS; card++) {
        copy->cards[card].launches = copy->launches[card];
    }
    copy->program = (struct execution_program){ copy->cards,
						copy->segments,
						copy->hand_offs,
						made->program.limits,
						CARDS,
						TEST_SYNTHETIC_SEGMENTS,
						TEST_SYNTHETIC_HAND_OFFS,
						1,
						1,
						NULL,
						NULL,
						0,
						0,
						PROGRAM_THIN,
						0,
						NULL,
						0,
						0,
						0,
						0,
						NULL,
						NULL,
						TEST_SYNTHETIC_SEGMENTS };
}

/* The cards opened again for a run: their arenas zero, the backend's counts from zero, the program's handles. */
static bool cards_reopen(struct device_card **cards,
			 const struct test_synthetic_kernels *kernels,
			 struct test_synthetic_calls *calls)
{
    struct test_synthetic_kernels again[CARDS];
    for (uint32_t card = 0; card < CARDS; card++) {
        backend->card_close(cards[card], WAIT_LIMIT_NANOSECONDS);
        cards[card] = NULL;
    }
    *calls = (struct test_synthetic_calls){{0}};
    return test_synthetic_cards_set(cards, again, calls) && memcmp(again, kernels, sizeof(again)) == 0;
}

/* Every launch of the thin physical program, exactly once at each position, including both front fragments. */
static bool physical_launches_once(const struct execution_program *program)
{
    uint64_t launches[CARDS] = {0};
    const uint64_t work_count = test_host_memory_backend_work_count();
    for (uint64_t index = 0; index < work_count; index++) {
        struct test_host_memory_backend_work work;
        if (!test_host_memory_backend_work_at(index, &work)) {
            return false;
        }
        if (work.kind == TEST_HOST_MEMORY_BACKEND_LAUNCH) {
            if (work.card_index >= CARDS) {
                return false;
            }
            launches[work.card_index]++;
        }
    }
    for (uint32_t card = 0; card < CARDS; card++) {
        if (launches[card] != (uint64_t)program->cards[card].launch_count * POSITIONS) {
            return false;
        }
    }
    return true;
}

/* Split after attention/pool work, exactly before the FFN finish. This puts capture moments on both sides and
 * directly on the boundary, makes reinjecting at the continuation wrong, and needs the first boundary's pool copy. */
static void split_capture(const struct test_teacher_forced_run *base_run,
			  const struct test_synthetic_program *made,
			  const struct test_synthetic_kernels *kernels,
			  struct device_card **cards,
			  struct test_synthetic_calls *calls,
			  const char *folder, bool is_free,
			  struct test_synthetic_layer *values,
			  unsigned char *scratch)
{
    struct test_program_view *view = NULL;
    if (!cards_reopen(cards, kernels, calls) ||
	test_program_view_create(backend, cards, NULL, &made->program, LAYERS,
				 base_run->run_layers, &view) != STATUS_OK) {
	    expect(false,
		   "front-split fixture opens and the original view is readable");
	    test_program_view_destroy(view);
	    return;
    }
    const uint32_t cut = view->layers[TEST_SYNTHETIC_GATE_LAYER].finish_launch[FFN];
    test_program_view_destroy(view);
    struct segment_record segments[SPLIT_SEGMENTS];
    for (uint32_t segment = 0; segment < TEST_SYNTHETIC_SEGMENTS; segment++) {
        segments[segment + (segment > GATE_LAYER_SEGMENT ? 1u : 0u)] = made->segments[segment];
    }
    segments[GATE_LAYER_SEGMENT].launch_count = cut - made->segments[GATE_LAYER_SEGMENT].first_launch;
    segments[GATE_LAYER_SEGMENT + 1u] = made->segments[GATE_LAYER_SEGMENT];
    segments[GATE_LAYER_SEGMENT + 1u].first_launch = cut;
    segments[GATE_LAYER_SEGMENT + 1u].launch_count -= segments[GATE_LAYER_SEGMENT].launch_count;
    struct hand_off_record hand_offs[TEST_SYNTHETIC_HAND_OFFS];
    hand_offs[0] = made->hand_offs[TEST_SYNTHETIC_HAND_OFFS - 1u];
    hand_offs[0].after_segment = GATE_LAYER_SEGMENT;
    for (uint32_t index = 1; index < TEST_SYNTHETIC_HAND_OFFS; index++) {
        hand_offs[index] = made->hand_offs[index - 1u];
        hand_offs[index].after_segment = GATE_LAYER_SEGMENT + 1u;
    }
    struct execution_program program = made->program;
    program.segments = segments;
    program.segment_count = SPLIT_SEGMENTS;
    program.front_segment_count = GATE_LAYER_SEGMENT + 1u;
    program.draft_segment_first = SPLIT_SEGMENTS;
    program.hand_offs = hand_offs;
    struct test_teacher_forced_run run = *base_run;
    run.program = &program;
    run.folder = folder;
    run.mode = is_free ? TEST_TEACHER_FORCED_FREE : TEST_TEACHER_FORCED_INJECTED;
    const result_status status = test_teacher_forced_write(&run);
    expect(status == STATUS_OK && physical_launches_once(&program) &&
		   folder_check(folder, run.mode, run.run_layers,
				run.break_layer, values, scratch) == 0 &&
		   calls->others[1] == POSITIONS &&
		   test_synthetic_mirror_holds(cards[1]),
	   "split layer: all records correct, each launch once, input injected once, both hand-off boundaries kept");
}

int main(void)
{
    static const char *const folders[] = {"capture", "injected", "free", "broken", "subset", "refused", "fused",
                                         "split_injected", "split_free", "split_fused_injected", "split_fused_free"};
    enum { CAPTURE, INJECTED, FREE, BROKEN, SUBSET, REFUSED, FUSED, SPLIT_INJECTED, SPLIT_FREE, SPLIT_FUSED_INJECTED,
           SPLIT_FUSED_FREE };
    const char *const temporary = getenv("TMPDIR") != NULL ? getenv("TMPDIR") : "/tmp";
    struct device_card *cards[CARDS] = { NULL };
    struct test_synthetic_kernels kernels[CARDS];
    struct test_synthetic_calls calls = {{0}};
    struct test_capture *capture = NULL;
    struct test_synthetic_program *const made = calloc(1, sizeof(struct test_synthetic_program));
    struct test_synthetic_program *const broken = calloc(1, sizeof(struct test_synthetic_program));
    struct test_synthetic_program *const fused = calloc(1, sizeof(struct test_synthetic_program));
    struct test_synthetic_layer *const values = calloc(LAYERS, sizeof(struct test_synthetic_layer));
    struct test_synthetic_row *const row = calloc(1, sizeof(struct test_synthetic_row));
    unsigned char *const scratch = malloc(SCRATCH_BYTES);
    char base[PATH_CAPACITY], paths[COUNT_OF(folders)][PATH_CAPACITY];
    const int length = snprintf(base, sizeof(base), "%s/" TEMPORARY_TEMPLATE, temporary);
    static const uint32_t model_classes[] = { ARITHMETIC_CLASS_SM_80,
					      ARITHMETIC_CLASS_SM_120,
					      ARITHMETIC_CLASS_SM_120_NATIVE };
    for (uint32_t index = 0; index < COUNT_OF(model_classes); index++) {
        test_model_program(model_classes[index]);
    }
    bool ready = made != NULL && broken != NULL && fused != NULL && values != NULL && row != NULL && scratch != NULL &&
                 length > 0 && (size_t)length < sizeof(base) && mkdtemp(base) != NULL &&
                 test_synthetic_cards_set(cards, kernels, &calls);
    for (uint32_t folder = 0; folder < COUNT_OF(folders); folder++) {
        const int written = snprintf(paths[folder], sizeof(paths[folder]), "%s/%s", base, folders[folder]);
        ready = ready && written > 0 && (size_t)written < sizeof(paths[folder]);
    }
    if (ready) {
        test_synthetic_program_build(made, &kernels[0], false);
        test_synthetic_program_build(fused, &kernels[0], true);
    }
    ready = ready && capture_write(paths[CAPTURE], row) &&
	    test_capture_open(paths[CAPTURE], &capture) == STATUS_OK;
    expect(ready, "the cards, kernels, stand-ins, program and capture were set up");
    const struct test_teacher_forced_run base_run = {
        .backend = backend, .cards = cards, .program = made != NULL ? &made->program : NULL, .capture = capture,
        .folder = paths[INJECTED], .conditions = "{\"synthetic\": true}", .run_layers = (1u << LAYERS) - 1u,
        .wait_limit_nanoseconds = WAIT_LIMIT_NANOSECONDS, .layer_count = LAYERS, .positions = POSITIONS,
        .break_layer = TEST_TEACHER_FORCED_NO_BREAK, .mode = TEST_TEACHER_FORCED_INJECTED, .logits = true};
    const struct {
        uint32_t folder;
        enum test_teacher_forced_mode mode;
        uint64_t layers;
        uint32_t break_layer;
        const char *what;
    } cases[] = {{INJECTED, TEST_TEACHER_FORCED_INJECTED, base_run.run_layers, TEST_TEACHER_FORCED_NO_BREAK,
                  "injected: every record is the host's, the other segment ran each step, the pool reached card 1"},
                 {FREE, TEST_TEACHER_FORCED_FREE, base_run.run_layers, TEST_TEACHER_FORCED_NO_BREAK,
                  "free: each layer from the one before, across the cards"},
                 {BROKEN, TEST_TEACHER_FORCED_INJECTED, base_run.run_layers, BREAK_LAYER,
                  "broken: the layer broken alone, as with a zero pre"},
                 {SUBSET, TEST_TEACHER_FORCED_INJECTED, base_run.run_layers & ~(1u << BREAK_LAYER),
                  TEST_TEACHER_FORCED_NO_BREAK, "a subset: the layer left out is not written, the next is given"}};
    for (uint32_t index = 0; ready && index < COUNT_OF(cases); index++) {
        struct test_teacher_forced_run run = base_run;
        run.folder = paths[cases[index].folder];
        run.mode = cases[index].mode;
        run.run_layers = cases[index].layers;
        run.break_layer = cases[index].break_layer;
        const bool reopened = cards_reopen(cards, kernels, &calls);
	const result_status status = reopened ?
					     test_teacher_forced_write(&run) :
					     STATUS_DEVICE_ERROR;
	expect(status == STATUS_OK &&
		       folder_check(run.folder, run.mode, run.run_layers,
				    run.break_layer, values, scratch) == 0 &&
		       calls.others[1] == POSITIONS &&
		       test_synthetic_mirror_holds(cards[1]) &&
		       test_host_memory_backend_most_uploads_in_flight(0) ==
			       BACKEND_MAXIMUM_UPLOADS_IN_FLIGHT,
	       cases[index].what);
    }
    if (ready) {                                /* the same model in the fused launches: the same records */
        struct test_teacher_forced_run run = base_run;
        run.folder = paths[FUSED];
        run.program = &fused->program;
        const bool reopened = cards_reopen(cards, kernels, &calls);
	const result_status status = reopened ?
					     test_teacher_forced_write(&run) :
					     STATUS_DEVICE_ERROR;
	expect(status == STATUS_OK &&
		       folder_check(run.folder, run.mode, run.run_layers,
				    run.break_layer, values, scratch) == 0,
	       "fused: the site fronts, projections and joins read as the launches they stand for, every record the "
	       "host's");
    }
    if (ready) {
        split_capture(&base_run, made, kernels, cards, &calls, paths[SPLIT_INJECTED], false, values, scratch);
        split_capture(&base_run, made, kernels, cards, &calls, paths[SPLIT_FREE], true, values, scratch);
        split_capture(&base_run, fused, kernels, cards, &calls, paths[SPLIT_FUSED_INJECTED], false, values, scratch);
        split_capture(&base_run, fused, kernels, cards, &calls, paths[SPLIT_FUSED_FREE], true, values, scratch);
    }
    if (ready) {
        struct test_teacher_forced_run run = base_run;
        run.folder = paths[CAPTURE];
	expect(test_teacher_forced_write(&run) != STATUS_OK,
	       "an output folder that exists is refused");
	run.folder = paths[REFUSED];
        run.positions = POSITIONS + 1u;
	expect(test_teacher_forced_write(&run) == STATUS_INVALID_ARGUMENT,
	       "positions beyond the capture's are refused");
	run.positions = POSITIONS;
        run.mode = TEST_TEACHER_FORCED_FREE;
        run.run_layers = base_run.run_layers & ~1u;
	expect(test_teacher_forced_write(&run) == STATUS_INVALID_ARGUMENT,
	       "a free run with a layer left out is refused");
	run = base_run;
        run.folder = paths[REFUSED];
        run.break_layer = 0;
	expect(test_teacher_forced_write(&run) == STATUS_INVALID_ARGUMENT,
	       "a break of layer 0 is refused");
	run.break_layer = TEST_TEACHER_FORCED_NO_BREAK;
        run.program = &broken->program;
	const struct segment_record *const first_layer =
		&made->segments[FIRST_LAYER_SEGMENT];
	const struct segment_record *const gate_layer =
		&made->segments[GATE_LAYER_SEGMENT];
	program_copy(made, broken);
        broken->launches[0][first_layer->first_launch + first_layer->launch_count -
                            TEST_SYNTHETIC_ROUTER_FROM_THE_END] = broken->launches[1][0];   /* card 1's own fill */
	expect(test_teacher_forced_write(&run) == STATUS_INVALID_ARGUMENT,
	       "a layer's segment without its router is refused");
	program_copy(made, broken);
        broken->launches[1][0] = made->launches[0][gate_layer->first_launch + GATE_IN_ITS_SEGMENT];
	expect(test_teacher_forced_write(&run) == STATUS_INVALID_ARGUMENT,
	       "an n-gram gate in a segment of no layer is refused");
	program_copy(made, broken);
	struct kernel_launch *const gate =
		&broken->launches[0][gate_layer->first_launch +
				     GATE_IN_ITS_SEGMENT];
	gate[0] = gate[1];                                          /* the attention site's mix finish, then the gate */
        gate[1] = made->launches[0][gate_layer->first_launch + GATE_IN_ITS_SEGMENT];
	expect(test_teacher_forced_write(&run) == STATUS_INVALID_ARGUMENT,
	       "an n-gram gate behind its layer's attention site is refused");
	program_copy(made, broken);
	broken->program.kind = PROGRAM_WIDE;
	expect(test_teacher_forced_write(&run) == STATUS_INVALID_ARGUMENT,
	       "a program that is not thin is refused");
	static const struct step_view view = {
		0, 0, 1
	}; /* the step's first row; refused before read */
	program_copy(made, broken);
        broken->cards[0].views = &view;
        broken->cards[0].view_count = 1;
	expect(test_teacher_forced_write(&run) == STATUS_INVALID_ARGUMENT,
	       "a card with views is refused");
	program_copy(made, broken);
        broken->hand_offs[0].view = 1;
	expect(test_teacher_forced_write(&run) == STATUS_INVALID_ARGUMENT,
	       "a hand-off of a view is refused");
	program_copy(fused, broken);
	const struct segment_record *const fused_first =
		&fused->segments[FIRST_LAYER_SEGMENT];
	struct kernel_launch *const join =
		&broken->launches[0][fused_first->first_launch +
				     fused_first->launch_count - 1u];
	struct expert_join_arguments unsummed;
	memcpy(&unsummed, join->argument, sizeof(unsummed));
	unsummed.row_sums_offset = EXPERT_JOIN_NO_ROW_SUMS;
	memcpy(join->argument, &unsummed, sizeof(unsummed));
	expect(test_teacher_forced_write(&run) == STATUS_INVALID_ARGUMENT,
	       "a fused program whose join stores no row sums is refused");
	program_copy(fused, broken);
        join->argument_byte_count--;
	expect(test_teacher_forced_write(&run) == STATUS_INVALID_ARGUMENT,
	       "a fused join with a truncated record is refused");
	program_copy(fused, broken);
        broken->launches[0][fused_first->first_launch + 1u].argument_byte_count--;
	expect(test_teacher_forced_write(&run) == STATUS_INVALID_ARGUMENT,
	       "a fused projection with a truncated record is refused");
	program_copy(fused, broken);
	struct expert_join_arguments plain_join;
	memcpy(&plain_join, join->argument, sizeof(plain_join));
	const struct expert_shared_join_arguments shared_join = {
		.output_offset = plain_join.shared_output_offset,
		.pair_output_offset = plain_join.pair_output_offset,
		.streams_offset = plain_join.streams_offset,
		.post_offset = plain_join.post_offset,
		.comb_offset = plain_join.comb_offset,
		.row_sums_offset = plain_join.row_sums_offset,
		.row_count = 1,
		.pairs_per_row = MODEL_EXPERTS_PER_ROW
	};
	join->kernel = kernels[0].joins.shared_join[0];
        join->argument_byte_count = sizeof(shared_join);
        memcpy(join->argument, &shared_join, sizeof(shared_join));
        struct test_program_view *shared_view = NULL;
	expect(test_program_view_create(backend, cards, NULL, &broken->program,
					LAYERS, base_run.run_layers,
					&shared_view) == STATUS_OK &&
		       shared_view != NULL &&
		       shared_view->layers[0]
				       .post[TEST_PROGRAM_FFN_SITE]
				       .sums_offset ==
			       plain_join.row_sums_offset &&
		       shared_view->layers[0]
				       .post[TEST_PROGRAM_FFN_SITE]
				       .output_offset ==
			       plain_join.shared_output_offset,
	       "a shared join exposes the same captured post and row-sum arrays");
	test_program_view_destroy(shared_view);
        join->argument_byte_count--;
	expect(test_teacher_forced_write(&run) == STATUS_INVALID_ARGUMENT,
	       "a fused shared join with a truncated record is refused");
	expect(access(paths[REFUSED], F_OK) != 0, "a refused run makes no folder");
    }
    test_capture_close(capture);
    for (uint32_t card = 0; card < CARDS; card++) {
        backend->card_close(cards[card], WAIT_LIMIT_NANOSECONDS);
    }
    for (uint32_t folder = 0; ready && folder < COUNT_OF(folders); folder++) {
        folder_remove(base, folders[folder]);
    }
    if (ready && rmdir(base) != 0) {
        expect(false, "the temporary folder was emptied and removed");
    }
    free(made);
    free(broken);
    free(fused);
    free(values);
    free(row);
    free(scratch);
    if (failures != 0) {
        (void)fprintf(stderr, "teacher_forced_test FAILED: %d failures\n", failures);
        return EXIT_FAILURE;
    }
    (void)printf("teacher_forced_test passed\n");
    return EXIT_SUCCESS;
}
