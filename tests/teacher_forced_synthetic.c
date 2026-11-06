/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * teacher_forced_synthetic.c: the synthetic model of teacher_forced_synthetic.h. Test support: no part of the engine.
 *
 * Contract: teacher_forced_synthetic.h. How this file keeps it: each stand-in reads its record's arrays from the
 * card's arena and calls the host function the test calls, so the two compute the same by construction; the arrays
 * of a card lie at the same offsets on both cards.
 */
#include "teacher_forced_synthetic.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/backend_records.h"
#include "anemone/engram_rows_records.h"
#include "anemone/sampler_records.h"
#include "anemone/step_records.h"
#include "host_memory_backend.h"

#define CARDS TEST_SYNTHETIC_CARDS
#define CARD_MEMORY_BYTES 4194304ull
#define ARENA_BYTES 1048576ull
#define STEP 0u                                 /* the arrays of a card, the same offsets on both */
#define SESSIONS 256u
#define RESULT 512u
#define STREAMS 4096u
#define PRE 49152u
#define MIXES 49408u
#define POST 49664u
#define COMB 49920u
#define HIDDEN 50176u
#define OUTPUT 61440u
#define SUMS 86016u
#define AFFINITIES 110592u
#define JOBS 114688u
#define TABLE 118784u
#define BIAS 131072u
#define POOL 135168u
#define MIRROR 139264u
#define ENGRAM_VALUES 143360u                   /* named by the rows-to-values launch: never read */
#define SCRATCH 196608u /* the fronts' (struct site_front_scratch): the mixes */
/* in the fronts' order */
#define SITE_FRONT_OF_ONE_ROW \
	((uint32_t)(SITE_FRONT_KIND_SITE * SITE_FRONT_ROW_FORMS))
#define FFN_FRONT_OF_ONE_ROW \
	((uint32_t)(SITE_FRONT_KIND_FFN * SITE_FRONT_ROW_FORMS))
#define LOGITS 262144u
#define POOL_ENTRY_BYTES 64u
#define POOL_RATIO 2u
#define EXPERT_BASE 655360u                     /* the expert matrices the table names: never read */
#define EXPERT_STRIDE 64u
#define FINITE_MASK 0x7F7Fu                     /* a bfloat16 with no all-ones exponent: finite */
#define AFFINITY_STEPS 1024u
#define AFFINITY_SCALE 64.0f
#define WEIGHT_SCALE 8.0f
#define BIAS_STEPS 13u
#define BIAS_FACTOR 7u
#define BIAS_SCALE 4.0f
#define ATTENTION_OUTPUT_ADD 7u
#define SHARED_FACTOR 3u
#define GATE_ADD 5u
#define SUMS_SCALE 0.5f
#define LOGIT_FACTOR 31u
#define LOGIT_STEPS 997u
#define HEAD_INPUT_STEPS 8u                     /* what the head's input adds to a logit */
#define VALUE_PRIME 7919u
#define POSITION_PRIME 131u
#define STREAM_PRIME 17u
#define PRE_STEP 0.25f
#define PRE_STREAM_STEP 0.5f
#define POST_FACTOR 2.0f
#define FIRST_TOKEN 11u
#define TOKEN_STEP 11u
#define WAIT_LIMIT_NANOSECONDS 1000000000ull
#define MIRROR_ENTRIES 3u                       /* positions 1 and 3 write entries 0 and 1; 2 stays zero */
#define MIRROR_BYTES ((uint64_t)MIRROR_ENTRIES * POOL_ENTRY_BYTES)
#define POOL_RUBBISH 0xEEu                      /* what a pool entry holds before its position writes it */
#define POLLS_BEFORE_A_COPY_FINISHES 1u         /* so that copies stay in flight until a wait */
#define LAYERS_A_CARD 2u
#define HEAD_LAUNCHES 2u
#define PICKS MODEL_EXPERTS_PER_ROW
#define JOBS_BYTES_OF_ONE_ROW ((uint64_t)PICKS * EXPERT_JOB_BYTES)
#define D MODEL_HIDDEN
#define K MODEL_STREAMS
#define HALF_SHIFT 16u
#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))
enum { ATTENTION, FFN };

static const struct backend_operations *const backend =
	&test_host_memory_backend;

/* ---- what the stand-ins compute, on host arrays ------------------------------------------------------------------ */
static uint16_t masked(uint32_t value)
{
    return (uint16_t)(value & FINITE_MASK);
}

static uint16_t high_half(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> HALF_SHIFT);
}

static float bias_of(uint32_t expert)
{
    return (float)(expert * BIAS_FACTOR % BIAS_STEPS) / BIAS_SCALE;
}

/* The n-gram gate: something added to every value of the streams. */
static void gate_add(uint16_t streams[K][D])
{
    for (uint32_t stream = 0; stream < K; stream++) {
        for (uint32_t value = 0; value < D; value++) {
            streams[stream][value] = masked((uint32_t)streams[stream][value] + GATE_ADD + stream);
        }
    }
}

/* A site's finish: the normed input, the mixes, post and comb, and the pre carried on. */
static void site_finish(struct test_synthetic_row *row, uint16_t hidden[D],
			float mixes[MODEL_MIXES], float post[K],
			float comb[K * K])
{
    for (uint32_t value = 0; value < D; value++) {
        hidden[value] = masked((uint32_t)row->streams[0][value] + row->streams[K - 1u][value] + high_half(row->pre[1]));
    }
    for (uint32_t mix = 0; mix < MODEL_MIXES; mix++) {
	    mixes[mix] = row->pre[0] + (float)mix;
    }
    for (uint32_t stream = 0; stream < K; stream++) {
        post[stream] = row->pre[stream] * POST_FACTOR;
        for (uint32_t into = 0; into < K; into++) {
            comb[stream * K + into] = row->pre[stream] + (float)into;
        }
    }
    for (uint32_t stream = 0; stream < K; stream++) {
        row->pre[stream] += 1.0f;
    }
}

/* A post: its output (and with the combine the routed sums) written first, then the streams. */
static void site_post(struct test_synthetic_row *row, bool is_combined, uint16_t output[D], float sums[D])
{
    for (uint32_t value = 0; value < D; value++) {
        output[value] = is_combined ? masked(value * SHARED_FACTOR + 1u) : masked(value + ATTENTION_OUTPUT_ADD);
        if (is_combined) {
            sums[value] = (float)value * SUMS_SCALE;
        }
        for (uint32_t stream = 0; stream < K; stream++) {
            row->streams[stream][value] = masked((uint32_t)row->streams[stream][value] + output[value] + stream +
                                                 (is_combined ? value : 0u));
        }
    }
}

/* The router over the FFN site's normed input: affinities, the six picks ascending and their weights. */
static void route(const uint16_t hidden[D], const float *bias, struct test_synthetic_layer *values)
{
	bool taken[MODEL_EXPERTS] = { false };
	for (uint32_t expert = 0; expert < MODEL_EXPERTS; expert++) {
		values->affinities[expert] =
			(float)(hidden[expert] % AFFINITY_STEPS) /
			AFFINITY_SCALE;
		values->biased[expert] =
			values->affinities[expert] + bias[expert];
	}
    for (uint32_t pick = 0; pick < PICKS; pick++) {
	    uint32_t best = MODEL_EXPERTS;
	    for (uint32_t expert = 0; expert < MODEL_EXPERTS; expert++) {
		    best = !taken[expert] && (best == MODEL_EXPERTS ||
					      values->biased[expert] >
						      values->biased[best]) ?
				   expert :
				   best;
	    }
	taken[best] = true;
        values->gate_ids[pick] = best;
        values->gate_weights[pick] = values->affinities[best] / WEIGHT_SCALE;
    }
    for (uint32_t expert = 0, pick = 0; expert < MODEL_EXPERTS; expert++) {
	    if (taken[expert]) {
		    values->picks[pick] = expert;
		    values->weights[pick++] =
			    values->affinities[expert] / WEIGHT_SCALE;
	    }
    }
}

void test_synthetic_layer_compute(uint32_t layer, const struct test_synthetic_row *input,
                                  struct test_synthetic_layer *values)
{
	float bias[MODEL_EXPERTS];
	for (uint32_t expert = 0; expert < MODEL_EXPERTS; expert++) {
		bias[expert] = bias_of(expert);
	}
    values->output = *input;
    if (layer == TEST_SYNTHETIC_GATE_LAYER) {
        gate_add(values->output.streams);
    }
    memcpy(values->ngram, values->output.streams, sizeof(values->ngram));
    site_finish(&values->output, values->hidden[ATTENTION], values->mixes[ATTENTION], values->post[ATTENTION],
                values->comb[ATTENTION]);
    memcpy(values->pre[ATTENTION], values->output.pre, sizeof(values->pre[ATTENTION]));
    site_post(&values->output, false, values->wo_b, values->sums);
    memcpy(values->ffn_streams, values->output.streams, sizeof(values->ffn_streams));
    site_finish(&values->output, values->hidden[FFN], values->mixes[FFN], values->post[FFN], values->comb[FFN]);
    memcpy(values->pre[FFN], values->output.pre, sizeof(values->pre[FFN]));
    route(values->hidden[FFN], bias, values);
    site_post(&values->output, true, values->shared, values->sums);
}

/* A logit of the stand-in head: of the token, and of the final collapse's output (its record's weights). */
float test_synthetic_logit_of(int32_t token, uint32_t vocabulary_index, const uint16_t *head_input)
{
    return (float)(((uint32_t)token * LOGIT_FACTOR + vocabulary_index) % LOGIT_STEPS +
                   head_input[vocabulary_index % D] % HEAD_INPUT_STEPS);
}

int32_t test_synthetic_token_at(uint32_t position)
{
    return (int32_t)(FIRST_TOKEN + position * TOKEN_STEP);
}

void test_synthetic_capture_row(uint32_t layer, uint32_t position, struct test_synthetic_row *row)
{
    for (uint32_t stream = 0; stream < K; stream++) {
        for (uint32_t value = 0; value < D; value++) {
            row->streams[stream][value] = masked(layer * VALUE_PRIME + position * POSITION_PRIME +
                                                 stream * STREAM_PRIME + value);
        }
        row->pre[stream] = (float)(layer + 1u) * PRE_STEP + (float)position + (float)stream * PRE_STREAM_STEP;
    }
}

void test_synthetic_embedding_row(uint32_t position, struct test_synthetic_row *row)
{
    static const float one_hot[K] = {1.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t stream = 0; stream < K; stream++) {
        for (uint32_t value = 0; value < D; value++) {
            row->streams[stream][value] = masked((uint32_t)test_synthetic_token_at(position) + value);
        }
    }
    memcpy(row->pre, one_hot, sizeof(one_hot));
}

/* ---- the stand-ins ----------------------------------------------------------------------------------------------- */
static bool inside(uint64_t offset, uint64_t byte_count, uint64_t arena_byte_count)
{
    return offset <= arena_byte_count && byte_count <= arena_byte_count - offset;
}

static bool embed_stand_in(const struct kernel_launch *launch,
			   uint32_t card_index, unsigned char *arena,
			   uint64_t arena_byte_count, void *context)
{
	struct hyper_embed_arguments record;
	struct step_row_record step_row;
	static const float one_hot[K] = { 1.0f, 0.0f, 0.0f, 0.0f };
	(void)card_index;
	(void)context;
	memcpy(&record, launch->argument, sizeof(record));
	if (!inside(record.streams_offset, MODEL_STREAMS_ROW_BYTES,
		    arena_byte_count)) {
		return false;
	}
	memcpy(&step_row, arena + record.step_offset + STEP_BLOCK_BYTES_VALUE,
	       sizeof(step_row));
	for (uint32_t stream = 0; stream < K; stream++) {
		for (uint32_t value = 0; value < D; value++) {
			const uint16_t half =
				masked((uint32_t)step_row.token_id + value);
			memcpy(arena + record.streams_offset +
				       (stream * D + value) * sizeof(half),
			       &half, sizeof(half));
		}
    }
    memcpy(arena + record.pre_offset, one_hot, sizeof(one_hot));
    return true;
}

static bool gate_stand_in(const struct kernel_launch *launch,
			  uint32_t card_index, unsigned char *arena,
			  uint64_t arena_byte_count, void *context)
{
	struct engram_gate_arguments record;
	struct test_synthetic_row *const row =
		calloc(1, sizeof(struct test_synthetic_row));
	(void)card_index;
	(void)context;
	memcpy(&record, launch->argument, sizeof(record));
	const bool held = row != NULL &&
			  inside(record.streams_offset, sizeof(row->streams),
				 arena_byte_count);
	if (held) {
		memcpy(row->streams, arena + record.streams_offset,
		       sizeof(row->streams));
		gate_add(row->streams);
		memcpy(arena + record.streams_offset, row->streams,
		       sizeof(row->streams));
	}
    free(row);
    return held;
}

/* A site's finish on the arrays of a finish record (of a front: its mixes in its scratch). */
static bool finish_apply(const struct hyper_finish_arguments *record,
			 unsigned char *arena, uint64_t arena_byte_count)
{
    struct test_synthetic_row *const row = calloc(1, sizeof(struct test_synthetic_row));
    uint16_t hidden[D];
    float mixes[MODEL_MIXES], post[K], comb[K * K];
    const bool held = row != NULL && inside(record->hidden_offset, sizeof(hidden), arena_byte_count) &&
                      inside(record->mixes_offset, sizeof(mixes), arena_byte_count);
    if (held) {
        memcpy(row->streams, arena + record->streams_offset, sizeof(row->streams));
        memcpy(row->pre, arena + record->pre_offset, sizeof(row->pre));
        site_finish(row, hidden, mixes, post, comb);
        memcpy(arena + record->hidden_offset, hidden, sizeof(hidden));
        memcpy(arena + record->mixes_offset, mixes, sizeof(mixes));
        memcpy(arena + record->post_offset, post, sizeof(post));
        memcpy(arena + record->comb_offset, comb, sizeof(comb));
        memcpy(arena + record->pre_offset, row->pre, sizeof(row->pre));
    }
    free(row);
    return held;
}

static bool finish_stand_in(const struct kernel_launch *launch,
			    uint32_t card_index, unsigned char *arena,
			    uint64_t arena_byte_count, void *context)
{
	struct hyper_finish_arguments record;
	(void)card_index;
	(void)context;
	memcpy(&record, launch->argument, sizeof(record));
	return launch->argument_byte_count == sizeof(record) &&
	       finish_apply(&record, arena, arena_byte_count);
}

/* A post on the arrays of a post record (of a projection or a join, the record it stands for). */
static bool post_apply(const struct hyper_post_arguments *record,
		       unsigned char *arena, uint64_t arena_byte_count)
{
    struct test_synthetic_row *const row = calloc(1, sizeof(struct test_synthetic_row));
    uint16_t output[D];
    float sums[D];
    const bool held = row != NULL && (record->is_combined == 0 || inside(record->sums_offset, sizeof(sums),
                                                                         arena_byte_count));
    if (held) {
        memcpy(row->streams, arena + record->streams_offset, sizeof(row->streams));
        site_post(row, record->is_combined != 0, output, sums);
        memcpy(arena + record->output_offset, output, sizeof(output));
        if (record->is_combined != 0) {
            memcpy(arena + record->sums_offset, sums, sizeof(sums));
        }
        memcpy(arena + record->streams_offset, row->streams, sizeof(row->streams));
    }
    free(row);
    return held;
}

static bool post_stand_in(const struct kernel_launch *launch,
			  uint32_t card_index, unsigned char *arena,
			  uint64_t arena_byte_count, void *context)
{
	struct hyper_post_arguments record;
	(void)card_index;
	(void)context;
	memcpy(&record, launch->argument, sizeof(record));
	return post_apply(&record, arena, arena_byte_count);
}

/* The router on the arrays of a select record (of the FFN front: its select). */
static bool route_apply(const struct router_select_arguments *selected,
			unsigned char *arena, uint64_t arena_byte_count)
{
	const struct router_select_arguments record = *selected;
	struct test_synthetic_layer *const values =
		calloc(1, sizeof(struct test_synthetic_layer));
	uint16_t hidden[D];
	float bias[MODEL_EXPERTS];
	uint64_t table[MODEL_EXPERTS][MODEL_EXPERT_MATRICES];
	const bool held = values != NULL &&
			  inside(record.jobs_offset, JOBS_BYTES_OF_ONE_ROW,
				 arena_byte_count);
	if (held) {
		memcpy(hidden, arena + record.scores_offset, sizeof(hidden));
		memcpy(bias, arena + record.bias_offset, sizeof(bias));
		memcpy(table, arena + record.expert_offsets_offset,
		       sizeof(table));
		route(hidden, bias, values);
		memcpy(arena + record.affinities_offset, values->affinities,
		       sizeof(values->affinities));
		for (uint32_t slot = 0; slot < PICKS; slot++) {
			const uint32_t expert = values->picks[slot];
			struct expert_job_record job = {
				table[expert][ROUTER_GATE],
				table[expert][ROUTER_UP],
				table[expert][ROUTER_DOWN],
				1,
				0,
				{ slot },
				{ values->weights[slot] }
			};
			memcpy(arena + record.jobs_offset + slot * sizeof(job),
			       &job, sizeof(job));
		}
    }
    free(values);
    return held;
}

static bool router_stand_in(const struct kernel_launch *launch,
			    uint32_t card_index, unsigned char *arena,
			    uint64_t arena_byte_count, void *context)
{
	struct router_select_arguments record;
	(void)card_index;
	(void)context;
	memcpy(&record, launch->argument, sizeof(record));
	return route_apply(&record, arena, arena_byte_count);
}

/* The fronts: the finish of their record's arrays, the mixes in their scratch; the FFN front's also the select. */
static bool front_apply(const struct kernel_launch *launch,
			unsigned char *arena, uint64_t arena_byte_count,
			bool is_ffn)
{
	struct site_front_arguments record;
	memcpy(&record, launch->argument, sizeof(record));
	const struct hyper_finish_arguments finish = {
		record.streams_offset,
		record.pre_offset,
		record.scratch_offset +
			offsetof(struct site_front_scratch, mixes),
		record.scale_offset,
		record.base_offset,
		record.post_offset,
		record.comb_offset,
		record.norm_weight_offset,
		record.hidden_offset
	};
	return launch->argument_byte_count == sizeof(record) &&
	       finish_apply(&finish, arena, arena_byte_count) &&
	       (!is_ffn ||
		route_apply(&record.select, arena, arena_byte_count));
}
static bool site_front_stand_in(const struct kernel_launch *launch,
				uint32_t card_index, unsigned char *arena,
				uint64_t arena_byte_count, void *context)
{
    (void)card_index;
    (void)context;
    return front_apply(launch, arena, arena_byte_count, false);
}
static bool ffn_front_stand_in(const struct kernel_launch *launch,
			       uint32_t card_index, unsigned char *arena,
			       uint64_t arena_byte_count, void *context)
{
    (void)card_index;
    (void)context;
    return front_apply(launch, arena, arena_byte_count, true);
}

/* The output projection: the attention site's post, its output the projection's. */
static bool projection_stand_in(const struct kernel_launch *launch,
				uint32_t card_index, unsigned char *arena,
				uint64_t arena_byte_count, void *context)
{
	struct attention_projection_arguments record;
	(void)card_index;
	(void)context;
	memcpy(&record, launch->argument, sizeof(record));
	const struct hyper_post_arguments post = { record.streams_offset,
						   record.post_offset,
						   record.comb_offset,
						   record.output_offset,
						   0,
						   0,
						   0 };
	return launch->argument_byte_count == sizeof(record) &&
	       post_apply(&post, arena, arena_byte_count);
}

/* The join: the FFN site's post with the combine, its output the shared expert's, its sums its row sums. */
static bool join_stand_in(const struct kernel_launch *launch,
			  uint32_t card_index, unsigned char *arena,
			  uint64_t arena_byte_count, void *context)
{
	struct expert_join_arguments record;
	(void)card_index;
	(void)context;
	memcpy(&record, launch->argument, sizeof(record));
	const struct hyper_post_arguments post = { record.streams_offset,
						   record.post_offset,
						   record.comb_offset,
						   record.shared_output_offset,
						   record.row_sums_offset,
						   1,
						   0 };
	return launch->argument_byte_count == sizeof(record) &&
	       post_apply(&post, arena, arena_byte_count);
}

static bool fill_stand_in(const struct kernel_launch *launch,
			  uint32_t card_index, unsigned char *arena,
			  uint64_t arena_byte_count, void *context)
{
	struct fill_record record;
	struct step_row_record step_row;
	struct test_synthetic_calls *const calls = context;
	memcpy(&record, launch->argument, sizeof(record));
	memcpy(&step_row, arena + STEP + STEP_BLOCK_BYTES_VALUE,
	       sizeof(step_row));
	const uint32_t position = (uint32_t)step_row.position;
	if (record.word_count ==
	    0) { /* the segment of its own of a card: counted */
		calls->others[card_index < CARDS ? card_index : 0]++;
		return card_index < CARDS;
	}
    const uint64_t entry = record.arena_byte_offset + (uint64_t)(position / POOL_RATIO) * POOL_ENTRY_BYTES;
    if ((position + 1u) % POOL_RATIO == 0 && inside(entry, POOL_ENTRY_BYTES, arena_byte_count)) {
        for (uint32_t byte = 0; byte < POOL_ENTRY_BYTES; byte++) {
            arena[entry + byte] = (unsigned char)(position * STREAM_PRIME + byte);
        }
    }
    return true;
}

static bool final_stand_in(const struct kernel_launch *launch,
			   uint32_t card_index, unsigned char *arena,
			   uint64_t arena_byte_count, void *context)
{
	struct hyper_finish_arguments record;
	(void)card_index;
	(void)context;
	memcpy(&record, launch->argument, sizeof(record));
	const uint64_t stream = record.streams_offset +
				(uint64_t)TEST_SYNTHETIC_HEAD_INPUT_STREAM *
					MODEL_HIDDEN_ROW_BYTES;
	const bool held =
		inside(record.hidden_offset, MODEL_HIDDEN_ROW_BYTES,
		       arena_byte_count) &&
		record.streams_offset < arena_byte_count &&
		inside(stream, MODEL_HIDDEN_ROW_BYTES, arena_byte_count);
	if (held) {
		memmove(arena + record.hidden_offset, arena + stream,
			MODEL_HIDDEN_ROW_BYTES);
	}
    return held;
}

static bool sampler_stand_in(const struct kernel_launch *launch,
			     uint32_t card_index, unsigned char *arena,
			     uint64_t arena_byte_count, void *context)
{
	struct sampler_arguments record;
	struct step_block block;
	struct step_row_record step_row;
	(void)card_index;
	(void)context;
	memcpy(&record, launch->argument, sizeof(record));
	if (!inside(record.logits_offset, MODEL_LOGITS_ROW_BYTES,
		    arena_byte_count)) {
		return false;
	}
    uint16_t head_input[D];
    memcpy(&block, arena + record.step_offset, sizeof(block));
    memcpy(&step_row, arena + record.step_offset + sizeof(block), sizeof(step_row));
    memcpy(head_input, arena + record.weights_offset, sizeof(head_input));
    struct step_row_result best = {
	    0, test_synthetic_logit_of(step_row.token_id, 0, head_input)
    };
    for (uint32_t index = 0; index < MODEL_VOCABULARY_VALUE; index++) {
	    const float logit = test_synthetic_logit_of(step_row.token_id,
							index, head_input);
	    memcpy(arena + record.logits_offset + index * sizeof(logit), &logit,
		   sizeof(logit));
	    best = logit > best.logit ?
			   (struct step_row_result){ (int32_t)index, logit } :
			   best;
    }
    const struct step_result result = { block.sequence, block.row_count, 0 };
    memcpy(arena + record.result_offset, &result, sizeof(result));
    memcpy(arena + record.result_offset + sizeof(result), &best, sizeof(best));
    return true;
}

/* ---- the program ------------------------------------------------------------------------------------------------- */
static struct kernel_launch launch_of(uint32_t kernel, const void *record,
				      uint32_t byte_count)
{
	struct kernel_launch launch = { .kernel = kernel,
					.grid = { 1, 1, 1 },
					.block = { 1, 1, 1 },
					.argument_byte_count = byte_count };
	memcpy(launch.argument, record, byte_count);
	return launch;
}

/* A layer's launches from launches[0]: the sites around the router (with is_fused the fused launches that stand for
 * them: the site front, the projection, the FFN front with the select, the join); the gate layer's n-gram launches
 * first and a pool writer between its sites. */
static uint32_t layer_launches(struct kernel_launch *launches,
			       const struct test_synthetic_kernels *kernels,
			       uint32_t layer, bool is_fused)
{
	const struct hyper_finish_arguments finish = { STREAMS, PRE, MIXES,
						       0,	0,   POST,
						       COMB,	0,   HIDDEN };
	const struct hyper_post_arguments attention = { STREAMS, POST, COMB,
							OUTPUT,	 0,    0,
							0 };
	const struct hyper_post_arguments ffn = { STREAMS, POST, COMB, OUTPUT,
						  SUMS,	   1,	 0 };
	const struct router_select_arguments router = {
		STEP,	     HIDDEN,	  BIAS,	       TABLE, AFFINITIES, JOBS,
		EXPERT_BASE, EXPERT_BASE, EXPERT_BASE, 1,     0
	};
	const struct site_front_arguments site_front = {
		.select = { .row_count = 1 },
		.streams_offset = STREAMS,
		.pre_offset = PRE,
		.post_offset = POST,
		.comb_offset = COMB,
		.hidden_offset = HIDDEN,
		.scratch_offset = SCRATCH
	};
	struct site_front_arguments ffn_front = site_front;
	const struct attention_projection_arguments projection = {
		.output_offset = OUTPUT,
		.streams_offset = STREAMS,
		.post_offset = POST,
		.comb_offset = COMB,
		.output_size = D
	};
	const struct expert_join_arguments join = { .shared_output_offset =
							    OUTPUT,
						    .streams_offset = STREAMS,
						    .post_offset = POST,
						    .comb_offset = COMB,
						    .row_sums_offset = SUMS,
						    .pairs_per_row = PICKS };
	const struct fill_record pool = { POOL, 1, 0, 0 };
	const struct engram_values_arguments values = { 0, ENGRAM_VALUES };
	const struct engram_gate_arguments gate = { 0, 0, 0, STREAMS };
	const bool is_gate_layer = layer == TEST_SYNTHETIC_GATE_LAYER;
	uint32_t count = 0;
	ffn_front.select = router;
	if (is_gate_layer) {
		launches[count++] = launch_of(kernels->engram.values, &values,
					      sizeof(values));
		launches[count++] =
			launch_of(kernels->engram.gate, &gate, sizeof(gate));
	}
    launches[count++] = is_fused ? launch_of(kernels->fronts.fronts[SITE_FRONT_OF_ONE_ROW], &site_front,
                                             sizeof(site_front))
                                 : launch_of(kernels->hyper.mix_finish, &finish, sizeof(finish));
    launches[count++] = is_fused ? launch_of(kernels->outputs.projection[0], &projection, sizeof(projection))
                                 : launch_of(kernels->hyper.post, &attention, sizeof(attention));
    if (is_gate_layer) {
        launches[count++] = launch_of(kernels->fill, &pool, sizeof(pool));
    }
    if (is_fused) {
        launches[count++] = launch_of(kernels->fronts.fronts[FFN_FRONT_OF_ONE_ROW], &ffn_front, sizeof(ffn_front));
        launches[count++] = launch_of(kernels->joins.join, &join, sizeof(join));
        return count;
    }
    launches[count++] = launch_of(kernels->hyper.mix_finish, &finish, sizeof(finish));
    launches[count++] = launch_of(kernels->router.select, &router, sizeof(router));
    launches[count++] = launch_of(kernels->hyper.post, &ffn, sizeof(ffn));
    return count;
}

void test_synthetic_program_build(struct test_synthetic_program *made, const struct test_synthetic_kernels *kernels,
                                  bool is_fused)
{
	const struct hyper_embed_arguments embed = {
		STEP, 0, STREAMS, PRE, MODEL_VOCABULARY_VALUE, 0
	};
	const struct hyper_finish_arguments final = { STREAMS, PRE, 0, 0,     0,
						      0,       0,   0, HIDDEN };
	const struct sampler_arguments sampler = {
		STEP, SESSIONS, LOGITS, HIDDEN, RESULT, MODEL_VOCABULARY_VALUE,
		1
	};
	const struct fill_record own = { 0, 0, 0, 0 };
	struct kernel_launch *const first = made->launches[0],
				    *const second = made->launches[1];
	const uint32_t none = SEGMENT_NO_LAYER;
	uint32_t count = 0, segment = 0, handed_after = 0,
		 card_launches[CARDS] = { 0, 0 };
	first[count++] = launch_of(kernels->hyper.embed, &embed, sizeof(embed));
	made->segments[segment++] =
		(struct segment_record){ 0, 0, none, 0, 1, 0 };
	for (uint32_t layer = 0; layer < TEST_SYNTHETIC_LAYERS; layer++) {
		const uint32_t card = layer / LAYERS_A_CARD;
		struct kernel_launch *const launches = made->launches[card];
		if (layer ==
		    LAYERS_A_CARD) { /* card 1 begins with a segment of its own */
			handed_after = segment - 1u;
			card_launches[0] = count;
			count = 0;
			second[count++] =
				launch_of(kernels->fill, &own, sizeof(own));
			made->segments[segment++] =
				(struct segment_record){ card, card, none,
							 0,    1,    0 };
		}
		const uint32_t launch_count = layer_launches(
			launches + count, kernels, layer, is_fused);
		made->segments[segment++] =
			(struct segment_record){ card,	card,	      layer,
						 count, launch_count, 0 };
		count += launch_count;
	}
	made->segments[segment] =
		(struct segment_record){ 1, 1, none, count, HEAD_LAUNCHES, 0 };
	second[count++] =
		launch_of(kernels->hyper.final_collapse, &final, sizeof(final));
	second[count++] =
		launch_of(kernels->sampler, &sampler, sizeof(sampler));
	card_launches[1] = count;
	const struct hand_off_record hand_offs[TEST_SYNTHETIC_HAND_OFFS] = {
		{ STREAMS, STREAMS, 0, MODEL_STREAMS_ROW_BYTES, 0, 0, 1,
		  handed_after, 0 },
		{ PRE, PRE, 0, MODEL_SITE_ROW_BYTES, 0, 0, 1, handed_after, 0 },
		{ POOL, MIRROR, 0, POOL_ENTRY_BYTES, POOL_RATIO, 0, 1,
		  handed_after, 0 }
	};
	memcpy(made->hand_offs, hand_offs, sizeof(hand_offs));
	for (uint32_t card = 0; card < CARDS; card++) {
		const bool is_result = card == 1u;
		made->cards[card] = (struct program_card_record){
			made->launches[card],
			ARENA_BYTES,
			STEP,
			is_result ? SESSIONS : PROGRAM_NOWHERE,
			is_result ? RESULT : PROGRAM_NOWHERE,
			is_result ? LOGITS : PROGRAM_NOWHERE,
			STREAMS,
			PRE,
			PROGRAM_NOWHERE,
			0,
			card_launches[card],
			NULL,
			0,
			PROGRAM_NOWHERE
		};
	}
	made->program = (struct execution_program){
		made->cards,
		made->segments,
		made->hand_offs,
		{ MODEL_CONTEXT_GRANULE, 1, 1, STATE_WIDTH_16,
		  LINEAR_PROFILE_PRECISION, LINEAR_ORDER_FIXED,
		  CMP_ORDER_REFERENCE, 0, 0, 0 },
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
		TEST_SYNTHETIC_SEGMENTS
	};
}

bool test_synthetic_cards_set(struct device_card **cards,
			      struct test_synthetic_kernels *kernels,
			      struct test_synthetic_calls *calls)
{
    void *block = NULL;
    uint64_t ticket = 0;
    test_host_memory_backend_reset(CARDS, CARD_MEMORY_BYTES);
    bool held = backend->staging_allocate(MODEL_EXPERT_OFFSETS_BYTES +
						  MODEL_ROUTER_ROW_BYTES +
						  MIRROR_BYTES,
					  &block) == STATUS_OK;
    unsigned char *const staging = block;
    unsigned char *const rubbish = held ? staging + MODEL_EXPERT_OFFSETS_BYTES +
						   MODEL_ROUTER_ROW_BYTES :
					  NULL;
    if (held) {
        memset(rubbish, POOL_RUBBISH, MIRROR_BYTES);
    }
    for (uint32_t expert = 0; held && expert < MODEL_EXPERTS; expert++) {
	    const float bias = bias_of(expert);
	    for (uint32_t matrix = 0; matrix < MODEL_EXPERT_MATRICES;
		 matrix++) {
		    const uint64_t offset =
			    EXPERT_BASE +
			    (expert * MODEL_EXPERT_MATRICES + matrix) *
				    EXPERT_STRIDE;
		    memcpy(staging + (expert * MODEL_EXPERT_MATRICES + matrix) *
					     sizeof(offset),
			   &offset, sizeof(offset));
	    }
	    memcpy(staging + MODEL_EXPERT_OFFSETS_BYTES + expert * sizeof(bias),
		   &bias, sizeof(bias));
    }
    for (uint32_t card = 0; held && card < CARDS; card++) {
	    test_host_memory_backend_set_arithmetic_class(
		    card, ARITHMETIC_CLASS_SM_80);
	    test_host_memory_backend_set_polls_before_an_upload_finishes(
		    card, POLLS_BEFORE_A_COPY_FINISHES);
	    held = backend->card_open(card, ARENA_BYTES, &cards[card]) ==
			   STATUS_OK &&
		   hyper_kernels_prepare(backend, cards[card],
					 &kernels[card].hyper) == STATUS_OK &&
		   router_kernels_prepare(backend, cards[card],
					  &kernels[card].router) == STATUS_OK &&
		   engram_rows_kernels_prepare(backend, cards[card],
					       &kernels[card].engram) ==
			   STATUS_OK &&
		   site_front_kernels_prepare(backend, cards[card],
					      &kernels[card].fronts) ==
			   STATUS_OK &&
		   attention_output_kernels_prepare(backend, cards[card],
						    ARITHMETIC_CLASS_SM_80,
						    &kernels[card].outputs) ==
			   STATUS_OK &&
		   expert_fused_kernels_prepare(backend, cards[card],
						ARITHMETIC_CLASS_SM_80,
						&kernels[card].joins) ==
			   STATUS_OK &&
		   backend->kernel_prepare(cards[card], KERNEL_FILE_FILL,
					   KERNEL_FILL_WORDS, FILL_RECORD_BYTES,
					   &kernels[card].fill) == STATUS_OK &&
		   backend->kernel_prepare(cards[card], KERNEL_FILE_SAMPLER,
					   KERNEL_SAMPLER,
					   SAMPLER_ARGUMENT_BYTES,
					   &kernels[card].sampler) ==
			   STATUS_OK &&
		   backend->upload_begin(cards[card], TABLE, staging,
					 MODEL_EXPERT_OFFSETS_BYTES,
					 &ticket) == STATUS_OK &&
		   backend->upload_begin(cards[card], BIAS,
					 staging + MODEL_EXPERT_OFFSETS_BYTES,
					 MODEL_ROUTER_ROW_BYTES,
					 &ticket) == STATUS_OK &&
		   backend->upload_begin(cards[card], POOL, rubbish,
					 MIRROR_BYTES, &ticket) == STATUS_OK &&
		   backend->upload_wait(cards[card], ticket,
					WAIT_LIMIT_NANOSECONDS) == STATUS_OK;
    }
    const struct { const char *file, *function; test_host_memory_backend_stand_in stand_in;
    } stand_ins[] = {
	    { KERNEL_FILE_HYPER_CONNECTION, KERNEL_HYPER_EMBED,
	      embed_stand_in },
	    { KERNEL_FILE_HYPER_CONNECTION, KERNEL_HYPER_MIX_FINISH,
	      finish_stand_in },
	    { KERNEL_FILE_HYPER_CONNECTION, KERNEL_HYPER_POST, post_stand_in },
	    { KERNEL_FILE_HYPER_CONNECTION, KERNEL_HYPER_FINAL,
	      final_stand_in },
	    { KERNEL_FILE_ROUTER, KERNEL_ROUTER_SELECT, router_stand_in },
	    { KERNEL_FILE_ENGRAM_ROWS, KERNEL_ENGRAM_GATE, gate_stand_in },
	    { KERNEL_FILE_FILL, KERNEL_FILL_WORDS, fill_stand_in },
	    { KERNEL_FILE_SAMPLER, KERNEL_SAMPLER, sampler_stand_in },
	    { KERNEL_FILE_SITE_FRONT, KERNEL_SITE_FRONT_STEM "1",
	      site_front_stand_in },
	    { KERNEL_FILE_SITE_FRONT, KERNEL_FFN_FRONT_STEM "1",
	      ffn_front_stand_in },
	    { KERNEL_FILE_ATTENTION_OUTPUT,
	      KERNEL_ATTENTION_PROJECTION_STEM "1", projection_stand_in },
	    { KERNEL_FILE_HYPER_CONNECTION, KERNEL_EXPERT_JOIN, join_stand_in }
    };
    for (uint32_t index = 0; held && index < COUNT_OF(stand_ins); index++) {
	    held = test_host_memory_backend_set_stand_in(
			   stand_ins[index].file, stand_ins[index].function,
			   stand_ins[index].stand_in, calls) == STATUS_OK;
    }
    backend->staging_free(block);
    return held && memcmp(&kernels[0], &kernels[1], sizeof(kernels[0])) == 0;
}

bool test_synthetic_mirror_holds(struct device_card *card)
{
    void *block = NULL;
    uint64_t ticket = 0;
    bool held = backend->staging_allocate(MIRROR_BYTES, &block) == STATUS_OK &&
		backend->download_begin(card, MIRROR, block, MIRROR_BYTES,
					&ticket) == STATUS_OK &&
		backend->upload_wait(card, ticket, WAIT_LIMIT_NANOSECONDS) ==
			STATUS_OK;
    const unsigned char *const bytes = block;
    for (uint32_t entry = 0; held && entry < MIRROR_ENTRIES; entry++) {
        const uint32_t position = entry * POOL_RATIO + 1u;
        for (uint32_t byte = 0; held && byte < POOL_ENTRY_BYTES; byte++) {
            held = bytes[entry * POOL_ENTRY_BYTES + byte] ==
                   (position < TEST_SYNTHETIC_POSITIONS ? (unsigned char)(position * STREAM_PRIME + byte) : 0);
        }
    }
    backend->staging_free(block);
    return held;
}
