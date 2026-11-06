/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * teacher_forced.c: the harness of teacher_forced.h. Test support: no part of the engine.
 *
 * Contract: teacher_forced.h. How this file keeps it:
 *   - The program is read once (program_view.h). The folder's records are laid out before the first step
 *     (capture_writer.h), from the capture's capture points, so that a step writes each value in its place.
 *   - Staging is one block in regions: the step area and the session settings, the n-gram rows of each table, the
 *     hand-offs, and the region of a segment, taken in order from its start for every segment. A copy waits for the
 *     card's last ticket when BACKEND_MAXIMUM_UPLOADS_IN_FLIGHT are unfinished; a segment ends with a wait for
 *     its card's copies and launches, after which what came back is written; a step ends with a wait on every card,
 *     so no region is written again while a copy may read it.
 *   - A layer's launches run in parts: up to its n-gram gate, if it has one, after which the streams come back; at a
 *     capture point also up to the end of the attention site's mix finish, the beginning of the FFN site's, its end;
 *     then the rest of the segment. Each part's records come back after it. wo_b's output comes back before the FFN
 *     site, which may write that array again.
 */
#include "teacher_forced.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "anemone/hooks.h"
#include "anemone/linear_records.h"
#include "anemone/model_layout.h"
#include "anemone/step_records.h"
#include "capture_writer.h"
#include "program_view.h"

#define NONE TEST_PROGRAM_VIEW_NONE
#define NO_POSITION TEST_CAPTURE_WRITER_NO_POSITION
#define NAME_CAPACITY TEST_CAPTURE_WRITER_NAME_CAPACITY
#define STAGING_BYTES 2097152ull
#define SESSION_STAGING 256u                    /* behind the step area, in the first region */
#define ENGRAM_STAGING 4096u
#define HAND_OFF_STAGING 131072u
#define HAND_OFF_STAGING_BYTES 131072u
#define SEGMENT_STAGING 262144u
#define STAGING_ALIGNMENT 64u
#define MOST_PENDING 32u                        /* downloads a segment writes into the folder */
#define PROGRESS_POSITIONS 256u
#define NANOSECONDS_PER_SECOND 1e9
#define TOP_P_OF_GREEDY 1.0f
#define RESULTS_CAPACITY 1024u                  /* the results object: four members and up to 64 layers */
#define INPUT_RECORDS 3u                        /* the token ids, the embeddings, the logits */
#define STEP_AREA_BYTES (STEP_BLOCK_BYTES_VALUE + STEP_ROW_BYTES_VALUE)
#define RESULT_AREA_BYTES (STEP_RESULT_BYTES + STEP_ROW_RESULT_BYTES)
#define JOBS_BYTES ((uint64_t)MODEL_EXPERTS_PER_ROW * EXPERT_JOB_BYTES)
#define STREAM_VALUES ((uint64_t)MODEL_STREAMS * MODEL_HIDDEN)
#define TOKEN_DIMENSIONS 2u                     /* [1, positions] */
#define MATRIX_DIMENSIONS 2u                    /* [positions, 6], [positions, 384], [4, 4], [4, 5120] */
#define OUTPUT_DIMENSIONS 4u                    /* [1, positions, 4, 5120] */
#define CARRIED_DIMENSIONS 3u                   /* [1, positions, 4] */
#define TOKEN_IDS_NAME "input.token_ids"
#define OUTPUT_FORMAT "layer_%" PRIu32 ".block.output"
#define CARRIED_FORMAT "layer_%" PRIu32 ".block.output_collapse_weights"
#define POINT_FORMAT "layer_%" PRIu32 ".position_%" PRIu32 ".%s"
#define ATTENTION TEST_PROGRAM_ATTENTION_SITE
#define FFN TEST_PROGRAM_FFN_SITE
#define BFLOAT16 TEST_CAPTURE_BFLOAT16
#define FLOAT32 TEST_CAPTURE_FLOAT32
#define INT64 TEST_CAPTURE_INT64

_Static_assert(ENGRAM_STAGING + ENGRAM_MAXIMUM_TABLES *
					       ENGRAM_MAXIMUM_ROWS_PER_TOKEN *
					       ENGRAM_ROW_BYTES <=
			       HAND_OFF_STAGING &&
		       HAND_OFF_STAGING + HAND_OFF_STAGING_BYTES <=
			       SEGMENT_STAGING &&
		       SESSION_STAGING >= STEP_AREA_BYTES &&
		       SEGMENT_STAGING < STAGING_BYTES,
	       "the regions of staging lie apart");
_Static_assert(STREAM_VALUES <= TEST_CAPTURE_WRITER_MOST_SLAB_VALUES &&
                   TEST_TEACHER_FORCED_MOST_LAYERS == TEST_PROGRAM_VIEW_MOST_LAYERS,
               "a row's streams are one slab of the writer, and a view takes every model the harness takes");

enum layer_record { LAYER_OUTPUT, LAYER_CARRIED, LAYER_IDS, LAYER_WEIGHTS, LAYER_BIASED, LAYER_ROW_IDS,
                    LAYER_NGRAM_OUTPUT, LAYER_RECORD_COUNT };
enum moment { AFTER_ENGRAM_GATE, AFTER_ATTENTION_FINISH, BEFORE_FFN_FINISH, AFTER_FFN_FINISH, AFTER_LAYER,
              MOMENT_COUNT };
enum field { FIELD_MIXES, FIELD_PRE, FIELD_POST, FIELD_COMB, FIELD_HIDDEN, FIELD_STREAMS, FIELD_OUTPUT, FIELD_SUMS,
             FIELD_SCORES, FIELD_AFFINITIES, FIELD_BIASED, FIELD_COUNT };

/* A record of a capture point (stepwise_capture.py's POINT_RECORDS of the sites, wo_b and the expert layer): when and
 * from which array of which site's records it comes back. */
struct point_form {
    const char *suffix;
    uint32_t moment;
    uint32_t site;
    uint32_t field;
    uint32_t type;
    uint32_t dimension_count;
    uint64_t dimensions[TEST_CAPTURE_WRITER_MOST_DIMENSIONS];
};
static const struct point_form point_forms[] = {
	{ "attn_site.mixes",
	  AFTER_ATTENTION_FINISH,
	  ATTENTION,
	  FIELD_MIXES,
	  FLOAT32,
	  1,
	  { MODEL_MIXES } },
	{ "attn_site.pre",
	  AFTER_ATTENTION_FINISH,
	  ATTENTION,
	  FIELD_PRE,
	  FLOAT32,
	  1,
	  { MODEL_STREAMS } },
	{ "attn_site.post",
	  AFTER_ATTENTION_FINISH,
	  ATTENTION,
	  FIELD_POST,
	  FLOAT32,
	  1,
	  { MODEL_STREAMS } },
	{ "attn_site.comb",
	  AFTER_ATTENTION_FINISH,
	  ATTENTION,
	  FIELD_COMB,
	  FLOAT32,
	  MATRIX_DIMENSIONS,
	  { MODEL_STREAMS, MODEL_STREAMS } },
	{ "attn_site.normed",
	  AFTER_ATTENTION_FINISH,
	  ATTENTION,
	  FIELD_HIDDEN,
	  BFLOAT16,
	  1,
	  { MODEL_HIDDEN } },
	{ "attn.wo_b",
	  BEFORE_FFN_FINISH,
	  ATTENTION,
	  FIELD_OUTPUT,
	  BFLOAT16,
	  1,
	  { MODEL_HIDDEN } },
	{ "ffn_site.streams",
	  BEFORE_FFN_FINISH,
	  FFN,
	  FIELD_STREAMS,
	  BFLOAT16,
	  MATRIX_DIMENSIONS,
	  { MODEL_STREAMS, MODEL_HIDDEN } },
	{ "ffn_site.mixes",
	  AFTER_FFN_FINISH,
	  FFN,
	  FIELD_MIXES,
	  FLOAT32,
	  1,
	  { MODEL_MIXES } },
	{ "ffn_site.pre",
	  AFTER_FFN_FINISH,
	  FFN,
	  FIELD_PRE,
	  FLOAT32,
	  1,
	  { MODEL_STREAMS } },
	{ "ffn_site.post",
	  AFTER_FFN_FINISH,
	  FFN,
	  FIELD_POST,
	  FLOAT32,
	  1,
	  { MODEL_STREAMS } },
	{ "ffn_site.comb",
	  AFTER_FFN_FINISH,
	  FFN,
	  FIELD_COMB,
	  FLOAT32,
	  MATRIX_DIMENSIONS,
	  { MODEL_STREAMS, MODEL_STREAMS } },
	{ "ffn_site.normed",
	  AFTER_FFN_FINISH,
	  FFN,
	  FIELD_HIDDEN,
	  BFLOAT16,
	  1,
	  { MODEL_HIDDEN } },
	{ "router.logits",
	  AFTER_LAYER,
	  FFN,
	  FIELD_SCORES,
	  FLOAT32,
	  1,
	  { MODEL_EXPERTS } },
	{ "router.scores",
	  AFTER_LAYER,
	  FFN,
	  FIELD_AFFINITIES,
	  FLOAT32,
	  1,
	  { MODEL_EXPERTS } },
	{ "router.biased_scores",
	  AFTER_LAYER,
	  FFN,
	  FIELD_BIASED,
	  FLOAT32,
	  1,
	  { MODEL_EXPERTS } },
	{ "experts.row_sum",
	  AFTER_LAYER,
	  FFN,
	  FIELD_SUMS,
	  FLOAT32,
	  1,
	  { MODEL_HIDDEN } },
	{ "shared_expert.output",
	  AFTER_LAYER,
	  FFN,
	  FIELD_OUTPUT,
	  BFLOAT16,
	  1,
	  { MODEL_HIDDEN } },
};
#define POINT_FORMS ((uint32_t)(sizeof(point_forms) / sizeof(point_forms[0])))
_Static_assert(sizeof(point_forms) / sizeof(point_forms[0]) + LAYER_RECORD_COUNT <= MOST_PENDING,
               "the downloads of a segment fit its list");

struct layer_state {
	uint64_t expert_table[MODEL_EXPERTS][MODEL_EXPERT_MATRICES];
	float bias[MODEL_EXPERTS];
	uint32_t records[LAYER_RECORD_COUNT]; /* or NONE */
};

struct record_facts {
    uint64_t slab_bytes;                        /* as a card holds them */
    bool every_position;                        /* a slab a position; else one */
};

struct pending {
    const unsigned char *source;                /* in staging */
    uint32_t record;
};

struct card_copies {
    uint64_t last_ticket;
    uint32_t unfinished;                        /* copies not yet reported finished */
};

struct harness {
    const struct test_teacher_forced_run *run;
    const struct backend_operations *backend;
    const struct execution_program *program;
    struct test_program_view *view;
    struct test_capture_writer *writer;
    struct layer_state *layers;
    struct record_facts *facts;
    uint32_t *point_first;                      /* [layer][position]: its first point record plus 1, or 0 */
    uint32_t *token_ids;
    unsigned char *staging;
    struct card_copies copies[BACKEND_MAXIMUM_CARDS];
    struct pending pending[MOST_PENDING];
    uint64_t segment_used;                      /* of the segment region */
    uint64_t row_ids[ENGRAM_MAXIMUM_TABLES * ENGRAM_MAXIMUM_ROWS_PER_TOKEN];
    uint32_t pending_count;
    uint32_t capture_positions;                 /* of the capture's tokens */
    uint32_t token_record;
    uint32_t embedding_record;
    uint32_t logits_record;                     /* or NONE */
    bool head_runs;
    bool engram_is_read;                        /* a card reads n-gram rows */
};

static bool runs(const struct harness *harness, uint32_t layer)
{
    return layer < harness->run->layer_count && (harness->run->run_layers >> layer & 1u) != 0;
}

static double seconds_now(void)
{
    struct timespec now = {0};
    return clock_gettime(CLOCK_MONOTONIC, &now) == 0 ? (double)now.tv_sec + (double)now.tv_nsec / NANOSECONDS_PER_SECOND
                                                     : 0.0;
}

/* ---- copies, waits and records --------------------------------------------------------------------------------- */
MUST_CHECK static result_status card_settle(struct harness *harness,
					    uint32_t card)
{
    struct card_copies *const copies = &harness->copies[card];
    const uint64_t limit = harness->run->wait_limit_nanoseconds;
    result_status status =
	    copies->unfinished == 0 ?
		    STATUS_OK :
		    harness->backend->upload_wait(harness->run->cards[card],
						  copies->last_ticket, limit);
    copies->unfinished = status == STATUS_OK ? 0 : copies->unfinished;
    status = status == STATUS_OK ?
		     harness->backend->launch_wait(harness->run->cards[card],
						   limit) :
		     status;
    if (status != STATUS_OK) {
	    LOG_ERROR("teacher-forced: card %" PRIu32
		      " did not finish its copies and launches: %s",
		      card, status_text(status));
    }
    return status;
}

MUST_CHECK static result_status
copy_begin(struct harness *harness, uint32_t card, bool is_download,
	   uint64_t arena_offset, unsigned char *staging, uint64_t byte_count)
{
    struct card_copies *const copies = &harness->copies[card];
    struct device_card *const handle = harness->run->cards[card];
    uint64_t ticket = 0;
    result_status status = STATUS_OK;
    if (copies->unfinished >= BACKEND_MAXIMUM_UPLOADS_IN_FLIGHT) {
	    status = harness->backend->upload_wait(
		    handle, copies->last_ticket,
		    harness->run->wait_limit_nanoseconds);
	    copies->unfinished = status == STATUS_OK ? 0 : copies->unfinished;
    }
    if (status == STATUS_OK) {
	    status = is_download ? harness->backend->download_begin(
					   handle, arena_offset, staging,
					   byte_count, &ticket) :
				   harness->backend->upload_begin(
					   handle, arena_offset, staging,
					   byte_count, &ticket);
    }
    if (status != STATUS_OK) {
	    LOG_ERROR("teacher-forced: card %" PRIu32 ": the %s of %" PRIu64
		      " bytes at %" PRIu64 " failed: %s",
		      card, is_download ? "download" : "upload", byte_count,
		      arena_offset, status_text(status));
	    return status;
    }
    copies->last_ticket = ticket;
    copies->unfinished++;
    return STATUS_OK;
}

/* byte_count bytes of the segment region, or NULL (one error line) when they do not fit. */
static unsigned char *segment_take(struct harness *harness, uint64_t byte_count)
{
    const uint64_t start = harness->segment_used;
    const uint64_t aligned = (byte_count + STAGING_ALIGNMENT - 1u) / STAGING_ALIGNMENT * STAGING_ALIGNMENT;
    if (byte_count > STAGING_BYTES || aligned > STAGING_BYTES - SEGMENT_STAGING - start) {
	    LOG_ERROR(
		    "teacher-forced: a segment's copies do not fit the harness's staging");
	    return NULL;
    }
    harness->segment_used = start + aligned;
    return harness->staging + SEGMENT_STAGING + start;
}

MUST_CHECK static result_status record_write(struct harness *harness,
					     uint32_t record, uint32_t position,
					     const void *values)
{
    return test_capture_writer_put(harness->writer, record, harness->facts[record].every_position ? position : 0,
                                   values);
}

/* A download of a record's slab into the segment region, written into the folder once the segment is done. */
MUST_CHECK static result_status download_for(struct harness *harness,
					     uint32_t card,
					     uint64_t arena_offset,
					     uint32_t record)
{
    unsigned char *const staging = segment_take(harness, harness->facts[record].slab_bytes);
    if (staging == NULL || harness->pending_count >= MOST_PENDING) {
	    return STATUS_INVALID_ARGUMENT;
    }
    harness->pending[harness->pending_count++] = (struct pending){staging, record};
    return copy_begin(harness, card, true, arena_offset, staging, harness->facts[record].slab_bytes);
}

/* A record of the folder: of every position (a slab a position) or, at a point, of one. */
MUST_CHECK static result_status
record_add(struct harness *harness, const char *name, uint32_t type,
	   const uint64_t *dimensions, uint32_t dimension_count, uint32_t layer,
	   uint32_t position, const char *description, uint32_t *index_out)
{
    const uint64_t slabs = position == NO_POSITION ? harness->run->positions : 1u;
    uint64_t values = 1;
    for (uint32_t dimension = 0; dimension < dimension_count; dimension++) {
        values *= dimensions[dimension];        /* the writer refuses a product beyond 63 bits */
    }
    const result_status status = test_capture_writer_add(
	    harness->writer, name, (enum test_capture_type)type, dimensions,
	    dimension_count, slabs, layer, position, description, index_out);
    if (status == STATUS_OK) {
	    harness->facts[*index_out] = (struct record_facts){
		    values / slabs *
			    test_capture_element_bytes(
				    (enum test_capture_type)type),
		    position == NO_POSITION
	    };
    }
    return status;
}

static bool capture_holds(const struct harness *harness, const char *name, uint32_t type, const uint64_t *dimensions,
                          uint32_t dimension_count)
{
    const struct test_capture_tensor *const tensor = test_capture_find(harness->run->capture, name);
    return tensor != NULL && tensor->type == type && tensor->dimension_count == dimension_count &&
           memcmp(tensor->dimensions, dimensions, dimension_count * sizeof(dimensions[0])) == 0;
}

/* Each run layer's records of every position, after the inputs and the logits. */
MUST_CHECK static result_status layer_records_plan(struct harness *harness,
						   uint32_t layer)
{
    const struct test_teacher_forced_run *const run = harness->run;
    const uint64_t positions = run->positions;
    const uint32_t tables =
	    run->engram == NULL ?
		    0 :
		    engram_description_of(run->engram)->table_count;
    const uint64_t output[] = { 1, positions, MODEL_STREAMS, MODEL_HIDDEN };
    const uint64_t picks[] = { positions, MODEL_EXPERTS_PER_ROW },
		   scores[] = { positions, MODEL_EXPERTS };
    const uint64_t row_ids[] = {
	    1, positions,
	    run->engram == NULL ?
		    0 :
		    engram_description_of(run->engram)->rows_per_token
    };
    const struct { const char *suffix; const uint64_t *dimensions; uint32_t type; uint32_t count; } forms[] = {
        [LAYER_OUTPUT] = {"block.output", output, BFLOAT16, OUTPUT_DIMENSIONS},
        [LAYER_CARRIED] = {"block.output_collapse_weights", output, FLOAT32, CARRIED_DIMENSIONS},
        [LAYER_IDS] = {"mlp.routed_expert_ids", picks, INT64, MATRIX_DIMENSIONS},
        [LAYER_WEIGHTS] = {"mlp.route_weights", picks, FLOAT32, MATRIX_DIMENSIONS},
        [LAYER_BIASED] = {"mlp.biased_scores", scores, FLOAT32, MATRIX_DIMENSIONS},
        [LAYER_ROW_IDS] = {"ngram.row_ids", row_ids, INT64, CARRIED_DIMENSIONS},
        [LAYER_NGRAM_OUTPUT] = {"ngram.output", output, BFLOAT16, OUTPUT_DIMENSIONS}};
    const bool has_gate = runs(harness, layer) && harness->view->layers[layer].gate_launch != NONE;
    bool is_engram_layer = false;
    for (uint32_t table = 0; table < tables && harness->engram_is_read; table++) {
        is_engram_layer = is_engram_layer || run->engram_layers[table] == layer;
    }
    if (runs(harness, layer) && harness->engram_is_read && is_engram_layer != has_gate) {
	    LOG_ERROR("teacher-forced: layer %" PRIu32
		      " has %s n-gram gate but %s the layer of a table read",
		      layer, has_gate ? "an" : "no",
		      is_engram_layer ? "is" : "is not");
	    return STATUS_INVALID_ARGUMENT;
    }
    char name[NAME_CAPACITY];
    result_status status = STATUS_OK;
    for (uint32_t record = 0; record < LAYER_RECORD_COUNT; record++) {
        const bool is_left_out = (record == LAYER_ROW_IDS && !is_engram_layer) ||
                                 (record == LAYER_NGRAM_OUTPUT && !has_gate);
        harness->layers[layer].records[record] = NONE;
        (void)snprintf(name, sizeof(name), "layer_%" PRIu32 ".%s", layer, forms[record].suffix);
	status = status != STATUS_OK || !runs(harness, layer) || is_left_out ?
			 status :
			 record_add(harness, name, forms[record].type,
				    forms[record].dimensions,
				    forms[record].count, layer, NO_POSITION,
				    "of the engine",
				    &harness->layers[layer].records[record]);
    }
    return status;
}

/* The records of the folder, in the order of the file: the inputs, the logits, each run layer's, the points. */
MUST_CHECK static result_status records_plan(struct harness *harness)
{
    const struct test_teacher_forced_run *const run = harness->run;
    const uint64_t positions = run->positions;
    const uint64_t hidden[] = { MODEL_HIDDEN };
    char name[NAME_CAPACITY];
    uint32_t points = 0;
    harness->point_first = calloc((size_t)run->layer_count * positions, sizeof(uint32_t));
    for (uint64_t index = 0; harness->point_first != NULL && index < run->layer_count * positions; index++) {
        const uint32_t layer = (uint32_t)(index / positions), position = (uint32_t)(index % positions);
        (void)snprintf(name, sizeof(name), POINT_FORMAT, layer, position, "attn_site.normed");
        harness->point_first[index] = runs(harness, layer) && capture_holds(harness, name, BFLOAT16, hidden, 1);
        points += harness->point_first[index];
    }
    const uint32_t capacity = LAYER_RECORD_COUNT * run->layer_count + points * POINT_FORMS + INPUT_RECORDS;
    harness->facts = calloc(capacity, sizeof(struct record_facts));
    if (harness->point_first == NULL || harness->facts == NULL) {
	    LOG_ERROR("teacher-forced: no memory for the plan of %" PRIu32
		      " records",
		      capacity);
    }
    result_status status =
	    harness->point_first == NULL || harness->facts == NULL ?
		    STATUS_OUT_OF_MEMORY :
		    test_capture_writer_create(capacity, &harness->writer);
    const uint64_t tokens[] = { 1, positions },
		   embeddings[] = { 1, positions, MODEL_HIDDEN };
    const uint64_t logits[] = { 1, positions, MODEL_VOCABULARY_VALUE };
    status = status != STATUS_OK ?
		     status :
		     record_add(harness, TOKEN_IDS_NAME, INT64, tokens,
				TOKEN_DIMENSIONS, TEST_CAPTURE_NO_LAYER,
				NO_POSITION, "the tokens of the capture",
				&harness->token_record);
    status = status != STATUS_OK ?
		     status :
		     record_add(harness, "input.embeddings", BFLOAT16,
				embeddings, CARRIED_DIMENSIONS,
				TEST_CAPTURE_NO_LAYER, NO_POSITION,
				"stream 0 of the embedding segment",
				&harness->embedding_record);
    status = status != STATUS_OK || !run->logits ?
		     status :
		     record_add(harness, "logits.all_rows", FLOAT32, logits,
				CARRIED_DIMENSIONS, TEST_CAPTURE_NO_LAYER,
				NO_POSITION, "the logits of the head",
				&harness->logits_record);
    for (uint32_t layer = 0; status == STATUS_OK && layer < run->layer_count;
	 layer++) {
	    status = layer_records_plan(harness, layer);
    }
    for (uint64_t index = 0;
	 status == STATUS_OK && index < run->layer_count * positions; index++) {
	    const uint32_t layer = (uint32_t)(index / positions),
			   position = (uint32_t)(index % positions);
	    for (uint32_t form = 0, first = 0;
		 status == STATUS_OK && harness->point_first[index] != 0 &&
		 form < POINT_FORMS;
		 form++) {
		    (void)snprintf(name, sizeof(name), POINT_FORMAT, layer,
				   position, point_forms[form].suffix);
		    status = record_add(harness, name, point_forms[form].type,
					point_forms[form].dimensions,
					point_forms[form].dimension_count,
					layer, position, "a point", &first);
		    harness->point_first[index] =
			    form == 0 ? first + 1u :
					harness->point_first[index];
	    }
    }
    return status;
}

/* ---- a step ---------------------------------------------------------------------------------------------------- */
/* The capture's streams and pre leaving source_layer at position into the arrays a segment reads them from (a layer:
 * its card's streams and carried pre; the head: its final collapse's), the pre as zeros where it is broken. */
MUST_CHECK static result_status inject(struct harness *harness, uint32_t card,
				       uint64_t streams_offset,
				       uint64_t pre_offset,
				       uint32_t source_layer, uint32_t position,
				       bool is_broken)
{
    const struct test_teacher_forced_run *const run = harness->run;
    const uint64_t output[] = { 1, harness->capture_positions, MODEL_STREAMS,
				MODEL_HIDDEN };
    unsigned char *const streams =
	    segment_take(harness, MODEL_STREAMS_ROW_BYTES);
    unsigned char *const pre =
	    streams != NULL ? segment_take(harness, MODEL_SITE_ROW_BYTES) :
			      NULL;
    char name[NAME_CAPACITY];
    if (pre == NULL) {
	    return STATUS_INVALID_ARGUMENT;
    }
    (void)snprintf(name, sizeof(name), OUTPUT_FORMAT, source_layer);
    result_status status = test_capture_read(run->capture, name, BFLOAT16,
					     output, OUTPUT_DIMENSIONS,
					     (uint64_t)position * STREAM_VALUES,
					     STREAM_VALUES, streams,
					     MODEL_STREAMS_ROW_BYTES);
    (void)snprintf(name, sizeof(name), CARRIED_FORMAT, source_layer);
    memset(pre, 0, MODEL_SITE_ROW_BYTES); /* a broken step keeps these zeros */
    status = status != STATUS_OK || is_broken ?
		     status :
		     test_capture_read(run->capture, name, FLOAT32, output,
				       CARRIED_DIMENSIONS,
				       (uint64_t)position * MODEL_STREAMS,
				       MODEL_STREAMS, pre,
				       MODEL_SITE_ROW_BYTES);
    status = status != STATUS_OK ?
		     status :
		     copy_begin(harness, card, false, streams_offset, streams,
				MODEL_STREAMS_ROW_BYTES);
    return status != STATUS_OK ? status :
				 copy_begin(harness, card, false, pre_offset,
					    pre, MODEL_SITE_ROW_BYTES);
}

MUST_CHECK static result_status launches_run(struct harness *harness,
					     uint32_t card, uint32_t first,
					     uint32_t end)
{
	const result_status status =
		end <= first ?
			STATUS_OK :
			harness->backend->launch_run(
				harness->run->cards[card],
				&harness->program->cards[card].launches[first],
				end - first);
	if (status != STATUS_OK) {
		LOG_ERROR("teacher-forced: card %" PRIu32
			  " refused launches %" PRIu32 " to %" PRIu32 ": %s",
			  card, first, end, status_text(status));
	}
    return status;
}

static uint64_t field_offset(const struct test_program_layer *layer, uint32_t site, uint32_t field)
{
	const struct hyper_finish_arguments *const finish =
		&layer->finish[site];
	const uint64_t offsets[FIELD_COUNT] = {
		[FIELD_MIXES] = finish->mixes_offset,
		[FIELD_PRE] = finish->pre_offset,
		[FIELD_POST] = finish->post_offset,
		[FIELD_COMB] = finish->comb_offset,
		[FIELD_HIDDEN] = finish->hidden_offset,
		[FIELD_STREAMS] = finish->streams_offset,
		[FIELD_OUTPUT] = layer->post[site].output_offset,
		[FIELD_SUMS] = layer->post[site].sums_offset,
		[FIELD_SCORES] = layer->router.scores_offset,
		[FIELD_AFFINITIES] = layer->router.affinities_offset
	};
	return offsets[field];
}

/* The jobs of the layer's router, decoded through its expert table: ids and weights in the gate's order. */
MUST_CHECK static result_status
routing_write(struct harness *harness, uint32_t layer_index, uint32_t position,
	      const unsigned char *jobs, const unsigned char *affinities,
	      uint32_t point)
{
    const struct layer_state *const layer = &harness->layers[layer_index];
    float biased[MODEL_EXPERTS], weights[MODEL_EXPERTS_PER_ROW];
    uint32_t picks[MODEL_EXPERTS_PER_ROW] = { 0 }, previous = 0;
    int64_t ids[MODEL_EXPERTS_PER_ROW];
    bool decoded = true;
    memcpy(biased, affinities, sizeof(biased));
    for (uint32_t expert = 0; expert < MODEL_EXPERTS; expert++) {
	    biased[expert] += layer->bias[expert];
    }
    for (uint32_t slot = 0; decoded && slot < MODEL_EXPERTS_PER_ROW; slot++) {
	    struct expert_job_record job;
	    uint32_t expert = 0;
	    memcpy(&job, jobs + slot * sizeof(job), sizeof(job));
	    while (expert < MODEL_EXPERTS &&
		   (layer->expert_table[expert][ROUTER_GATE] !=
			    job.gate_offset ||
		    layer->expert_table[expert][ROUTER_UP] != job.up_offset ||
		    layer->expert_table[expert][ROUTER_DOWN] !=
			    job.down_offset)) {
		    expert++;
	    }
	    decoded = expert < MODEL_EXPERTS && job.row_count == 1 &&
		      job.pair_index[0] == slot &&
		      (slot == 0 || expert > previous);
	    previous = expert;
	    picks[slot] = expert;
	    weights[slot] = job.route_weight[0];
	    /* the gate's order: biased descending; the jobs come in ascending experts, so among equals the lower first */
	    for (uint32_t place = slot;
		 decoded && place > 0 &&
		 biased[picks[place]] > biased[picks[place - 1u]];
		 place--) {
		    const uint32_t pick = picks[place - 1u];
		    const float weight = weights[place - 1u];
		    picks[place - 1u] = picks[place];
		    weights[place - 1u] = weights[place];
		    picks[place] = pick;
		    weights[place] = weight;
	    }
    }
    if (!decoded) {
	    LOG_ERROR("teacher-forced: layer %" PRIu32 " at position %" PRIu32
		      ": the router's jobs are not six of "
		      "one row, each an expert of the layer's table, ascending",
		      layer_index, position);
	    return STATUS_DEVICE_ERROR;
    }
    for (uint32_t slot = 0; slot < MODEL_EXPERTS_PER_ROW; slot++) {
	    ids[slot] = picks[slot];
    }
    result_status status =
	    record_write(harness, layer->records[LAYER_IDS], position, ids);
    status = status != STATUS_OK ?
		     status :
		     record_write(harness, layer->records[LAYER_WEIGHTS],
				  position, weights);
    status = status != STATUS_OK ?
		     status :
		     record_write(harness, layer->records[LAYER_BIASED],
				  position, biased);
    for (uint32_t form = 0;
	 status == STATUS_OK && point != 0 && form < POINT_FORMS; form++) {
	    status = point_forms[form].field == FIELD_BIASED ?
			     record_write(harness, point - 1u + form, position,
					  biased) :
			     status;
    }
    return status;
}

/* One physical fragment of a run layer; capture each moment only in the fragment that finishes it. */
MUST_CHECK static result_status
layer_run(struct harness *harness, uint32_t segment_index, uint32_t position,
	  unsigned char **jobs_out, unsigned char **affinities_out)
{
	const struct segment_record *const segment =
		&harness->program->segments[segment_index];
	const uint32_t layer_index = segment->layer;
	const struct test_program_layer *const layer =
		&harness->view->layers[layer_index];
	const struct segment_record *const last =
		&harness->program->segments[layer->segment];
	const struct program_card_record *const card =
		&harness->program->cards[segment->card];
	const uint32_t *const records = harness->layers[layer_index].records;
	const uint32_t point =
		harness->point_first[(uint64_t)layer_index *
					     harness->run->positions +
				     position];
	const bool has_gate = layer->gate_launch != NONE;
	const uint32_t ends[MOMENT_COUNT] = {
		has_gate ? layer->gate_launch + 1u : segment->first_launch,
		layer->finish_launch[ATTENTION] + 1u, layer->finish_launch[FFN],
		layer->finish_launch[FFN] + 1u,
		last->first_launch + last->launch_count
	};
	const uint32_t segment_end =
		segment->first_launch + segment->launch_count;
	uint32_t first = segment->first_launch;
	result_status status = STATUS_OK;
	for (uint32_t moment = 0; status == STATUS_OK && moment < MOMENT_COUNT;
	     moment++) {
		if (ends[moment] <= segment->first_launch ||
		    ends[moment] > segment_end ||
		    (moment != AFTER_LAYER &&
		     (moment == AFTER_ENGRAM_GATE ? !has_gate : point == 0))) {
			continue; /* no cut there */
		}
		status = launches_run(harness, segment->card, first,
				      ends[moment]);
		first = ends[moment];
		status = status != STATUS_OK || moment != AFTER_ENGRAM_GATE ?
				 status :
				 download_for(harness, segment->card,
					      card->streams_offset,
					      records[LAYER_NGRAM_OUTPUT]);
		for (uint32_t form = 0;
		     status == STATUS_OK && point != 0 && form < POINT_FORMS;
		     form++) {
			const struct point_form *const shape =
				&point_forms[form];
			status =
				shape->moment != moment ||
						shape->field == FIELD_BIASED ?
					status :
					download_for(harness, segment->card,
						     field_offset(layer,
								  shape->site,
								  shape->field),
						     point - 1u + form);
		}
	}
	status = status != STATUS_OK ? status :
				       launches_run(harness, segment->card,
						    first, segment_end);
	if (status != STATUS_OK || segment_index != layer->segment) {
		return status; /* only the last fragment leaves a layer */
	}
	status = status != STATUS_OK ? status :
				       download_for(harness, segment->card,
						    card->streams_offset,
						    records[LAYER_OUTPUT]);
	status = status != STATUS_OK ? status :
				       download_for(harness, segment->card,
						    card->carried_pre_offset,
						    records[LAYER_CARRIED]);
	*jobs_out = status == STATUS_OK ? segment_take(harness, JOBS_BYTES) :
					  NULL;
	*affinities_out =
		*jobs_out != NULL ?
			segment_take(harness, MODEL_ROUTER_ROW_BYTES) :
			NULL;
	status = status == STATUS_OK && *affinities_out == NULL ?
			 STATUS_INVALID_ARGUMENT :
			 status;
	status = status != STATUS_OK ? status :
				       copy_begin(harness, segment->card, true,
						  layer->router.jobs_offset,
						  *jobs_out, JOBS_BYTES);
	return status != STATUS_OK ?
		       status :
		       copy_begin(harness, segment->card, true,
				  layer->router.affinities_offset,
				  *affinities_out, MODEL_ROUTER_ROW_BYTES);
}

/* The embedding segment's output: four equal streams and the pre (1, 0, 0, 0). */
static bool embedding_is_whole(const unsigned char *streams, const unsigned char *pre)
{
	float carried[MODEL_STREAMS];
	memcpy(carried, pre, sizeof(carried));
	bool whole = carried[0] == 1.0f;
	for (uint32_t stream = 1; stream < MODEL_STREAMS; stream++) {
		whole = whole && carried[stream] == 0.0f &&
			memcmp(streams,
			       streams +
				       (size_t)stream * MODEL_HIDDEN_ROW_BYTES,
			       MODEL_HIDDEN_ROW_BYTES) == 0;
	}
    return whole;
}

/* Whether the result area answers the step at position with a token. */
MUST_CHECK static result_status head_answers(const unsigned char *result,
					     uint32_t position)
{
	struct step_result answer;
	struct step_row_result row;
	memcpy(&answer, result, sizeof(answer));
	memcpy(&row, result + sizeof(answer), sizeof(row));
	if (answer.sequence == (uint64_t)position + 1u &&
	    answer.row_count == 1 && row.token_id >= 0 &&
	    (uint32_t)row.token_id < MODEL_VOCABULARY_VALUE) {
		return STATUS_OK;
	}
	LOG_ERROR("teacher-forced: at position %" PRIu32
		  " the result card's result does not answer the step "
		  "with a token",
		  position);
	return STATUS_DEVICE_ERROR;
}

/* What came back of a segment: its records, then a layer's routing, the embedding's streams or the head's result. */
MUST_CHECK static result_status segment_finish(struct harness *harness,
					       uint32_t segment_index,
					       uint32_t position,
					       const unsigned char *first,
					       const unsigned char *second)
{
	const struct segment_record *const segment =
		&harness->program->segments[segment_index];
	const uint32_t kind = harness->view->segment_kinds[segment_index];
	result_status status = STATUS_OK;
	for (uint32_t index = 0;
	     status == STATUS_OK && index < harness->pending_count; index++) {
		status = record_write(harness, harness->pending[index].record,
				      position, harness->pending[index].source);
	}
	if (status != STATUS_OK || kind == TEST_PROGRAM_SEGMENT_OTHER) {
		return status;
	}
    if (kind == TEST_PROGRAM_SEGMENT_LAYER &&
        segment_index != harness->view->layers[segment->layer].segment) {
        return status;                                                  /* the front has no final routing/output */
    }
    if (first == NULL || (second == NULL && kind != TEST_PROGRAM_SEGMENT_HEAD)) {
	    return STATUS_INVALID_ARGUMENT; /* segment_run took no staging: it said so */
    }
    if (kind == TEST_PROGRAM_SEGMENT_HEAD) {
        return head_answers(first, position);
    }
    if (kind == TEST_PROGRAM_SEGMENT_LAYER) {
        return routing_write(harness, segment->layer, position, first, second,
                             harness->point_first[(uint64_t)segment->layer * harness->run->positions + position]);
    }
    if (embedding_is_whole(first, second)) {
        return record_write(harness, harness->embedding_record, position, first);
    }
    LOG_ERROR("teacher-forced: at position %" PRIu32
	      " the embedding gave unequal streams or another pre "
	      "than (1, 0, 0, 0)",
	      position);
    return STATUS_DEVICE_ERROR;
}

MUST_CHECK static result_status
segment_run(struct harness *harness, uint32_t segment_index, uint32_t position)
{
    const struct test_teacher_forced_run *const run = harness->run;
    const struct segment_record *const segment =
	    &harness->program->segments[segment_index];
    const struct program_card_record *const card =
	    &harness->program->cards[segment->card];
    const uint32_t kind = harness->view->segment_kinds[segment_index];
    const bool is_injected = run->mode == TEST_TEACHER_FORCED_INJECTED;
    unsigned char *first = NULL, *second = NULL;
    result_status status = STATUS_OK;
    harness->segment_used = 0;
    harness->pending_count = 0;
    if (kind == TEST_PROGRAM_SEGMENT_LAYER) {
        status = is_injected && segment->layer > 0 &&
                 segment_index == harness->view->layers[segment->layer].first_segment
               ? inject(harness, segment->card, card->streams_offset, card->carried_pre_offset, segment->layer - 1u,
                        position, segment->layer == run->break_layer) : status;
	status = status != STATUS_OK ? status :
				       layer_run(harness, segment_index,
						 position, &first, &second);
    } else {
        status = kind == TEST_PROGRAM_SEGMENT_HEAD && is_injected
               ? inject(harness, segment->card, harness->view->final.streams_offset, harness->view->final.pre_offset,
                        run->layer_count - 1u, position, false) : status;
	status = status != STATUS_OK ?
			 status :
			 launches_run(
				 harness, segment->card, segment->first_launch,
				 segment->first_launch + segment->launch_count);
    }
    if (status == STATUS_OK && kind == TEST_PROGRAM_SEGMENT_EMBED) {
	    first = segment_take(harness, MODEL_STREAMS_ROW_BYTES);
	    second = first != NULL ?
			     segment_take(harness, MODEL_SITE_ROW_BYTES) :
			     NULL;
	    status = second == NULL ?
			     STATUS_INVALID_ARGUMENT :
			     copy_begin(harness, segment->card, true,
					harness->view->embed.streams_offset,
					first, MODEL_STREAMS_ROW_BYTES);
	    status = status != STATUS_OK ?
			     status :
			     copy_begin(harness, segment->card, true,
					harness->view->embed.pre_offset, second,
					MODEL_SITE_ROW_BYTES);
    }
    if (status == STATUS_OK && kind == TEST_PROGRAM_SEGMENT_HEAD) {
	    status = harness->logits_record == NONE ?
			     status :
			     download_for(harness, segment->card,
					  card->logits_offset,
					  harness->logits_record);
	    first = status == STATUS_OK ?
			    segment_take(harness, RESULT_AREA_BYTES) :
			    NULL;
	    status = first == NULL ? STATUS_INVALID_ARGUMENT :
				     copy_begin(harness, segment->card, true,
						card->result_offset, first,
						RESULT_AREA_BYTES);
    }
    status = status != STATUS_OK ? status : card_settle(harness, segment->card);
    return status != STATUS_OK ? status :
				 segment_finish(harness, segment_index,
						position, first, second);
}

/* The hand-offs after a segment, as step.h moves them, for the one row of a step at position. */
MUST_CHECK static result_status
hand_offs_run(struct harness *harness, uint32_t segment, uint32_t position)
{
	const struct execution_program *const program = harness->program;
	unsigned char *const staging = harness->staging + HAND_OFF_STAGING;
	result_status status = STATUS_OK;
	for (uint32_t index = 0;
	     status == STATUS_OK && index < program->hand_off_count; index++) {
		const struct hand_off_record *const hand_off =
			&program->hand_offs[index];
		const uint32_t ratio = hand_off->entry_ratio;
		if (hand_off->after_segment != segment ||
		    (ratio != 0 && ((uint64_t)position + 1u) % ratio != 0)) {
			continue;
		}
		const uint64_t step =
			(uint64_t)(ratio == 0 ? 0 : position / ratio) *
			hand_off->byte_count;
		if (step > UINT64_MAX - hand_off->source_offset ||
		    step > UINT64_MAX - hand_off->destination_offset) {
			LOG_ERROR("teacher-forced: hand-off %" PRIu32
				  " reaches beyond 64 bits",
				  index);
			return STATUS_INVALID_ARGUMENT;
		}
		status = copy_begin(harness, hand_off->from_card, true,
				    hand_off->source_offset + step, staging,
				    hand_off->byte_count);
		status = status != STATUS_OK ?
				 status :
				 card_settle(harness, hand_off->from_card);
		status = status != STATUS_OK ?
				 status :
				 copy_begin(harness, hand_off->to_card, false,
					    hand_off->destination_offset + step,
					    staging, hand_off->byte_count);
		status = status != STATUS_OK ?
				 status :
				 card_settle(harness, hand_off->to_card);
	}
    return status;
}

/* The step area and the n-gram rows of a step, to every card that reads them; the token into the folder. */
MUST_CHECK static result_status step_put(struct harness *harness,
					 uint32_t position)
{
    const struct test_teacher_forced_run *const run = harness->run;
    const struct execution_program *const program = harness->program;
    const struct step_block block = { (uint64_t)position + 1u, 1, 0 };
    const struct step_row_record row = { (int32_t)position,
					 (int32_t)harness->token_ids[position],
					 0, STEP_ROW_LOGITS };
    const int64_t token = harness->token_ids[position];
    memcpy(harness->staging, &block, sizeof(block));
    memcpy(harness->staging + sizeof(block), &row, sizeof(row));
    result_status status =
	    record_write(harness, harness->token_record, position, &token);
    for (uint32_t card = 0; status == STATUS_OK && card < program->card_count;
	 card++) {
	    status = program->cards[card].step_offset == PROGRAM_NOWHERE ?
			     status :
			     copy_begin(harness, card, false,
					program->cards[card].step_offset,
					harness->staging, STEP_AREA_BYTES);
    }
    if (status != STATUS_OK || !harness->engram_is_read ||
	run->engram == NULL) {
	    return status; /* a program that reads rows has an engram */
    }
    const struct engram_description *const engram =
	    engram_description_of(run->engram);
    const uint64_t table_bytes =
	    (uint64_t)engram->rows_per_token * ENGRAM_ROW_BYTES;
    status = engram_row_ids_entry(run->engram, harness->token_ids, NULL,
				  position, harness->row_ids);
    for (uint32_t table = 0; status == STATUS_OK && table < engram->table_count;
	 table++) {
	    const uint32_t layer = run->engram_layers[table];
	    status = engram_gather_entry(
		    run->engram, table,
		    harness->row_ids + (size_t)table * engram->rows_per_token,
		    engram->rows_per_token,
		    harness->staging + ENGRAM_STAGING + table * table_bytes);
	    status = status != STATUS_OK || !runs(harness, layer) ?
			     status :
			     record_write(
				     harness,
				     harness->layers[layer]
					     .records[LAYER_ROW_IDS],
				     position,
				     harness->row_ids +
					     (size_t)table *
						     engram->rows_per_token);
    }
    for (uint32_t card = 0; status == STATUS_OK && card < program->card_count;
	 card++) {
	    const struct program_card_record *const holder =
		    &program->cards[card];
	    for (uint32_t table = 0, below = 0;
		 status == STATUS_OK && table < engram->table_count; table++) {
		    if ((holder->engram_tables >> table & 1u) != 0) {
			    status = copy_begin(
				    harness, card, false,
				    holder->engram_rows_offset +
					    (uint64_t)below *
						    program->limits.row_capacity *
						    table_bytes,
				    harness->staging + ENGRAM_STAGING +
					    table * table_bytes,
				    table_bytes);
			    below++;
		    }
	    }
    }
    return status;
}

MUST_CHECK static result_status step_run(struct harness *harness,
					 uint32_t position)
{
	const struct execution_program *const program = harness->program;
	result_status status = step_put(harness, position);
	for (uint32_t segment = 0;
	     status == STATUS_OK && segment < program->segment_count;
	     segment++) {
		const uint32_t kind = harness->view->segment_kinds[segment];
		if ((kind == TEST_PROGRAM_SEGMENT_LAYER &&
		     !runs(harness, program->segments[segment].layer)) ||
		    (kind == TEST_PROGRAM_SEGMENT_HEAD &&
		     !harness->head_runs)) {
			continue;
		}
		status = segment_run(harness, segment, position);
		status = status != STATUS_OK ?
				 status :
				 hand_offs_run(harness, segment, position);
	}
	for (uint32_t card = 0;
	     status == STATUS_OK && card < program->card_count; card++) {
		status = harness->run->cards[card] == NULL ?
				 status :
				 card_settle(harness, card);
	}
    return status;
}

/* ---- the run --------------------------------------------------------------------------------------------------- */
static bool run_is_valid(const struct test_teacher_forced_run *run)
{
    if (run == NULL || run->backend == NULL || run->cards == NULL || run->program == NULL || run->capture == NULL ||
        run->folder == NULL || run->conditions == NULL || run->layer_count - 1u >= TEST_TEACHER_FORCED_MOST_LAYERS ||
        run->positions - 1u >= INT32_MAX || (run->mode != TEST_TEACHER_FORCED_INJECTED &&
                                             run->mode != TEST_TEACHER_FORCED_FREE)) {
        return false;
    }
    const uint64_t every = run->layer_count == TEST_TEACHER_FORCED_MOST_LAYERS ? UINT64_MAX
                         : (1ull << run->layer_count) - 1u;
    const bool breaks = run->break_layer == TEST_TEACHER_FORCED_NO_BREAK ||
                        (run->mode == TEST_TEACHER_FORCED_INJECTED && run->break_layer - 1u < run->layer_count - 1u &&
                         (run->run_layers >> run->break_layer & 1u) != 0);
    return (run->run_layers & ~every) == 0 && breaks &&
           (run->mode == TEST_TEACHER_FORCED_INJECTED || run->run_layers == every);
}

/* What the program reads beyond the view: steps of one row, hand-offs that fit staging, n-gram tables it was given. */
static bool program_fits(struct harness *harness)
{
	const struct execution_program *const program = harness->program;
	const uint32_t tables =
		harness->run->engram == NULL ?
			0 :
			engram_description_of(harness->run->engram)->table_count;
	bool fits = program->kind == PROGRAM_THIN && program->row_count == 1 &&
		    program->limits.row_capacity != 0;
	for (uint32_t index = 0; fits && index < program->hand_off_count;
	     index++) {
		fits = program->hand_offs[index].byte_count - 1u <
			       HAND_OFF_STAGING_BYTES &&
		       program->hand_offs[index].view == 0;
	}
    for (uint32_t card = 0; fits && card < program->card_count; card++) {
        const uint32_t names = program->cards[card].engram_tables;
	fits = (names >> tables) == 0 && program->cards[card].view_count == 0 &&
	       (names == 0 ||
		program->cards[card].engram_rows_offset != PROGRAM_NOWHERE);
	harness->engram_is_read = harness->engram_is_read || names != 0;
    }
    if (!fits) {
	    LOG_ERROR(
		    "teacher-forced: the program is not thin, runs steps of more than one row, has a hand-off "
		    "of more than %u bytes or of a view, or names n-gram tables the harness was not given",
		    HAND_OFF_STAGING_BYTES);
    }
    return fits;
}

/* The capture's tokens; in injected mode the streams and pre each run layer is given, and whether the head runs. */
MUST_CHECK static result_status capture_read(struct harness *harness)
{
    const struct test_teacher_forced_run *const run = harness->run;
    const struct test_capture_tensor *const tokens = test_capture_find(run->capture, TOKEN_IDS_NAME);
    if (tokens == NULL || tokens->type != INT64 || tokens->dimension_count != TOKEN_DIMENSIONS ||
        tokens->dimensions[0] != 1 || tokens->dimensions[1] < run->positions || tokens->dimensions[1] > UINT32_MAX) {
	    LOG_ERROR("teacher-forced: the capture holds no " TOKEN_IDS_NAME
		      " int64 [1, %" PRIu32 " or more]",
		      run->positions);
	    return STATUS_INVALID_ARGUMENT;
    }
    harness->capture_positions = (uint32_t)tokens->dimensions[1];
    int64_t *const ids = calloc(run->positions, sizeof(int64_t));
    harness->token_ids = calloc(run->positions, sizeof(uint32_t));
    result_status status =
	    ids == NULL || harness->token_ids == NULL ?
		    STATUS_OUT_OF_MEMORY :
		    test_capture_read(run->capture, TOKEN_IDS_NAME, INT64,
				      tokens->dimensions, TOKEN_DIMENSIONS, 0,
				      run->positions, ids,
				      (uint64_t)run->positions *
					      sizeof(int64_t));
    for (uint32_t position = 0;
	 status == STATUS_OK && position < run->positions; position++) {
	    status = ids[position] >= 0 &&
				     ids[position] < MODEL_VOCABULARY_VALUE ?
			     status :
			     STATUS_INVALID_ARGUMENT;
	    harness->token_ids[position] =
		    (uint32_t)(ids[position] >= 0 ? ids[position] : 0);
    }
    free(ids);
    if (status == STATUS_OUT_OF_MEMORY) {
	    LOG_ERROR("teacher-forced: no memory for %" PRIu32 " token ids",
		      run->positions);
    }
    const uint64_t output[] = { 1, harness->capture_positions, MODEL_STREAMS,
				MODEL_HIDDEN };
    const bool is_injected = run->mode == TEST_TEACHER_FORCED_INJECTED;
    const bool has_head = harness->view->head_segment != NONE;
    char name[NAME_CAPACITY];
    harness->head_runs = has_head && !is_injected;
    for (uint32_t layer = 0; status == STATUS_OK && layer < run->layer_count;
	 layer++) {
	    (void)snprintf(name, sizeof(name), OUTPUT_FORMAT, layer);
	    bool holds = capture_holds(harness, name, BFLOAT16, output,
				       OUTPUT_DIMENSIONS);
	    (void)snprintf(name, sizeof(name), CARRIED_FORMAT, layer);
	    holds = holds && capture_holds(harness, name, FLOAT32, output,
					   CARRIED_DIMENSIONS);
	    harness->head_runs = harness->head_runs ||
				 (is_injected && holds && has_head &&
				  layer + 1u == run->layer_count);
	    status = is_injected && runs(harness, layer + 1u) && !holds ?
			     STATUS_INVALID_ARGUMENT :
			     status;
    }
    status = status == STATUS_OK && run->logits && !harness->head_runs ?
		     STATUS_INVALID_ARGUMENT :
		     status;
    if (status == STATUS_INVALID_ARGUMENT) {
	    LOG_ERROR(
		    "teacher-forced: the capture holds a token outside the vocabulary, lacks the streams leaving "
		    "a layer before a run layer, or no head runs for the logits asked");
    }
    return status;
}

/* The greedy session settings to the result card; each run layer's expert table and bias from its card. */
MUST_CHECK static result_status layers_prepare(struct harness *harness)
{
	const struct execution_program *const program = harness->program;
	const struct program_card_record *const result =
		&program->cards[program->result_card];
	const struct step_session greedy = { 0, 0.0f, TOP_P_OF_GREEDY };
	result_status status = STATUS_OK;
	harness->layers =
		calloc(harness->run->layer_count, sizeof(struct layer_state));
	if (harness->layers == NULL) {
		LOG_ERROR("teacher-forced: no memory for the state of %" PRIu32
			  " layers",
			  harness->run->layer_count);
		return STATUS_OUT_OF_MEMORY;
	}
	if (harness->view->head_segment != NONE &&
	    result->session_offset != PROGRAM_NOWHERE) {
		memcpy(harness->staging + SESSION_STAGING, &greedy,
		       sizeof(greedy));
		status = copy_begin(harness, program->result_card, false,
				    result->session_offset,
				    harness->staging + SESSION_STAGING,
				    sizeof(greedy));
		status = status != STATUS_OK ?
				 status :
				 card_settle(harness, program->result_card);
	}
	for (uint32_t index = 0;
	     status == STATUS_OK && index < harness->run->layer_count;
	     index++) {
		const struct test_program_layer *const read =
			&harness->view->layers[index];
		struct layer_state *const layer = &harness->layers[index];
		const uint32_t card =
			runs(harness, index) ?
				program->segments[read->segment].card :
				NONE;
		harness->segment_used = 0;
		unsigned char *const table =
			card == NONE ?
				NULL :
				segment_take(harness,
					     sizeof(layer->expert_table));
		unsigned char *const bias =
			table != NULL ?
				segment_take(harness, sizeof(layer->bias)) :
				NULL;
		status = card == NONE ?
				 status :
			 bias == NULL ?
				 STATUS_INVALID_ARGUMENT :
				 copy_begin(harness, card, true,
					    read->router.expert_offsets_offset,
					    table, sizeof(layer->expert_table));
		status = status != STATUS_OK || card == NONE ?
				 status :
				 copy_begin(harness, card, true,
					    read->router.bias_offset, bias,
					    sizeof(layer->bias));
		status = status != STATUS_OK || card == NONE ?
				 status :
				 card_settle(harness, card);
		if (status == STATUS_OK && card != NONE) {
			memcpy(layer->expert_table, table,
			       sizeof(layer->expert_table));
			memcpy(layer->bias, bias, sizeof(layer->bias));
		}
	}
    return status;
}

/* The manifest with the run's results: complete when the run and every write succeeded. */
MUST_CHECK static result_status folder_close(struct harness *harness,
					     bool complete, double seconds)
{
    const struct test_teacher_forced_run *const run = harness->run;
    char results[RESULTS_CAPACITY];
    size_t used = 0;
    int length = snprintf(results, sizeof(results), "{\"positions\": %" PRIu32 ", \"mode\": \"%s\", \"head\": %s, "
                          "\"seconds\": %.3f, \"layers\": [", run->positions,
                          run->mode == TEST_TEACHER_FORCED_FREE ? "free" : "injected",
                          harness->head_runs ? "true" : "false", seconds);
    for (uint32_t layer = 0; length >= 0 && (size_t)length < sizeof(results) - used && layer <= run->layer_count;
         layer++) {
        used += (size_t)length;
        length = layer == run->layer_count ? snprintf(results + used, sizeof(results) - used, "]}")
               : !runs(harness, layer) ? 0
               : snprintf(results + used, sizeof(results) - used, "%s%" PRIu32,
                          results[used - 1u] == '[' ? "" : ", ", layer);
    }
    const bool fits = length >= 0 && (size_t)length < sizeof(results) - used;
    if (!fits) {
	    LOG_ERROR(
		    "teacher-forced: the results of the run do not fit %u bytes",
		    RESULTS_CAPACITY);
    }
    return test_capture_writer_close(harness->writer, complete && fits, run->conditions, fits ? results : "{}");
}

result_status
test_teacher_forced_write(const struct test_teacher_forced_run *run)
{
    if (!run_is_valid(run)) {
	    LOG_ERROR(
		    "teacher-forced: a missing member, or a setting out of range (layers, positions, mode, "
		    "the break, free mode with a layer not run)");
	    return STATUS_INVALID_ARGUMENT;
    }
    struct harness *const harness = calloc(1, sizeof(struct harness));
    if (harness == NULL) {
	    LOG_ERROR("teacher-forced: no memory for the harness");
	    return STATUS_OUT_OF_MEMORY;
    }
    *harness = (struct harness){.run = run, .backend = run->backend, .program = run->program, .logits_record = NONE};
    void *staging = NULL;
    result_status status = test_program_view_create(
	    run->backend, run->cards, run->card_descriptions, run->program,
	    run->layer_count, run->run_layers, &harness->view);
    status = status != STATUS_OK || program_fits(harness) ?
		     status :
		     STATUS_INVALID_ARGUMENT;
    status = status != STATUS_OK ?
		     status :
		     run->backend->staging_allocate(STAGING_BYTES, &staging);
    harness->staging = staging;
    status = status != STATUS_OK ? status : layers_prepare(harness);
    status = status != STATUS_OK ? status : capture_read(harness);
    status = status != STATUS_OK ? status : records_plan(harness);
    status = status != STATUS_OK ?
		     status :
		     test_capture_writer_open(harness->writer, run->folder);
    const bool is_open = status == STATUS_OK;
    const double started = seconds_now();
    for (uint32_t position = 0;
	 status == STATUS_OK && position < run->positions; position++) {
	    status = step_run(harness, position);
	    if (status == STATUS_OK &&
		((position + 1u) % PROGRESS_POSITIONS == 0 ||
		 position + 1u == run->positions)) {
		    (void)printf("TEACHER_FORCED position=%" PRIu32
				 " seconds=%.1f\n",
				 position, seconds_now() - started);
	    }
    }
    bool settled = true;
    for (uint32_t card = 0; harness->view != NULL && card < run->program->card_count; card++) {
	    settled = settled && (harness->copies[card].unfinished == 0 ||
				  card_settle(harness, card) == STATUS_OK);
    }
    const result_status closed =
	    is_open ? folder_close(harness, status == STATUS_OK,
				   seconds_now() - started) :
		      test_capture_writer_close(harness->writer, false, "{}",
						"{}");
    status = status != STATUS_OK || !is_open ? status : closed;
    if (settled) {
        run->backend->staging_free(staging);
    } else {
	    LOG_ERROR(
		    "teacher-forced: staging stays allocated: a copy may still read it");
    }
    test_program_view_destroy(harness->view);
    free(harness->layers);
    free(harness->facts);
    free(harness->point_first);
    free(harness->token_ids);
    free(harness);
    return status;
}
