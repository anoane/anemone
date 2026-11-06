/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41_program_test: the programs of DeepSeek-V4.1-Flash (source/model_v41_program.c), from plans of the model's
 * pack held as tables only (model_v41_pack.h), on the backend in host memory with cards of both classes.
 *
 *   - On the five cards of first_token_plan.md 1.3 (cards 0 to 3 of class 80, card 4 of class 120), at 1 and 8 rows
 *     and in both orders of additions: every layer's launches are, by their functions' names, the sequence of 2.1
 *     for its kind of layer (window only, n-gram, 2-to-1 and 1-to-1 source, reuse, index), written below from the
 *     reference; each product leads its record with its own layer's tensor, cut as round 2's table says for the
 *     card's class and the order, and each dense product rounds and scales as 2.3 says; each launch reads what the
 *     launch before it in the reference's data flow wrote, or the card's streams, carried pre, rotation and step
 *     area; the n-gram rows, the attention state, the scan, the core and the router read and write the arrays of
 *     their layer, of its source and group, on its card.
 *   - The program of 8 rows is that of 1 row with every row count, grid row and pair count 8 times its.
 *   - The segments of a stage per card (the step segment where the visit holds a layer, the embedding, the layers,
 *     the head last) and the hand-offs, exactly; each card's fields; each segment run through launch_run in the
 *     program's order, and logged so.
 *   - Placements: 1.3; the embedding on the last card and the head on the first; layer 14 on the card of layer 1 (a
 *     card used twice, two n-gram tables on one card); the embedding and the head on a card that runs no layer.
 *   - Wide programs of 64, 65 and 1,024 rows, in a plan of 1,024-row chunks on 1.3: a front step's program (layer
 *     20's front last, no head); a view a slab on every card that runs it; the hand-offs of the thin program's
 *     front. Full programs of 64, 65, 1,024 and 1,100 rows (two pieces) in a plan of 2,048-row chunks: the thin
 *     program's segments but the head, layer 20 in its front and its rest. Of both, the embedding one launch over the
 *     chunk, and each layer (chunk_layout): every launch by name, tensor and rows, those over a slab the thin
 *     program's of its rows; every launch that reads no state (the trunk and dense products in their class's wide
 *     form, the n-gram rows, the hyper sites) over each piece of up to 1,024 rows, the queries over each span of 8
 *     slabs ahead of its first slab, the routed experts once; every read of every launch, by the reference's data
 *     flow, of bytes a launch of the step the flow names wrote before it in the layer at the same field and place of
 *     row 0, rows of the same bytes, and nothing wrote again since; areas at the launch's rows. Shown failing once on
 *     copies of two layers: wo_a ahead of the last slab's core, span 1's queries at span 0's heads, slab 8's core
 *     ahead of span 1's queries; the indexer's weights behind slab 0's scan, slab 1's compressor at slab 0's rows of
 *     wkv's product.
 *   - Interleaved layouts (cards visited twice a step: B1's W11, I12 and R11, B7's recommended layout, and one whose
 *     card 0 takes the candidates in at a visit that reads none): in thin, wide and full programs every visit of a
 *     card that holds a layer begins with its step segment and no other segment is one; the pool and key copies of
 *     layer 20 behind its front go to every card that reads them; W11, I12 and R11 name no lease, B7 the two of card
 *     2 (the selection of layer 28 from it to 31, the candidates from 28's visit to 36), the last the candidates' of
 *     card 0 from its visit of 21 to 28 and of card 2 from 32's visit to 36, a wide program none; a step driver takes
 *     each layout's thin, wide and full programs (I12's full ones leave out the streams that only its head on another
 *     card reads). The production layout D, which visits each card once, passes the same checks with no lease.
 *     Both 16k and 1M context plans exercise those drivers, with legacy and split index selection respectively.
 *   - At 1M context, isolate layer 14 or 20 with the embedding and head on one card: thin 1/8-row and full 4,096-row
 *     programs fit their launch allocations; every split chain retains its records, scratch bounds and outputs.
 *   model_v41_program_refusal_test holds the refusals and the model's table of the seam.
 * A failure is one line on the standard output: the case, and what was expected of it.
 */
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/attention_core_records.h"
#include "anemone/attention_output_records.h"
#include "anemone/attention_state_records.h"
#include "anemone/backend_records.h"
#include "anemone/dense_linear.h"
#include "anemone/dense_linear_records.h"
#include "anemone/engram_rows_records.h"
#include "anemone/hooks.h"
#include "anemone/hyper_connection_records.h"
#include "anemone/index_scan.h"
#include "anemone/index_scan_records.h"
#include "anemone/loader.h"
#include "anemone/relaxed_linear_records.h"
#include "anemone/router.h"
#include "anemone/expert_relaxed.h"
#include "anemone/expert_fused_records.h"
#include "anemone/router_records.h"
#include "anemone/sampler_records.h"
#include "anemone/site_front_records.h"
#include "anemone/official_hc.h"
#include "anemone/official_head.h"
#include "anemone/sampler.h"
#include "anemone/official_reference.h"
#include "anemone/step.h"
#include "engram_pack.h"
#include "host_memory_backend.h"
#include "memory_storage.h"
#include "model_v41.h"
#include "model_v41_pack.h"
#include "program_view.h"

#define CARDS TEST_MODEL_V41_CARDS
#define LAYERS MODEL_LAYERS
#define LAYERS_PER_CARD TEST_MODEL_V41_LAYERS_PER_CARD
#define UNIT_COUNT TEST_MODEL_V41_UNIT_COUNT
#define CONTEXT 16384u
#define ROWS 8u
#define SESSIONS 3u
#define PRO_CARD 4u
#define HOST_CARD_BYTES 4096u                      /* an arena no launch reads: the programs only name offsets */
#define NAME_CAPACITY TEST_HOST_MEMORY_BACKEND_FUNCTION_NAME_CAPACITY
#define TOKEN_ROWS_BYTES \
	((uint64_t)V41_DATA_ENGRAM_ROWS_PER_TOKEN * ENGRAM_ROW_BYTES)
#define HEAD_LAUNCHES 3u                           /* final collapse, head, sampler */
#define HEAD_DENSE 1u                              /* the head among them */
#define REVISITING_LAYER 14u                       /* moved to card 0: a card used twice, two tables on one card */
#define LAST_CARD_LAYER 32u                        /* the first layer of card 4 in 1.3 */
#define ORDERS (LINEAR_ORDER_REFERENCE + 1u)
#define WIDE_CAPACITY 1024u                        /* the chunk of the wide programs' plan */
#define NO_SEGMENT UINT32_MAX
#define PIECE_ROWS \
	LINEAR_WIDE_MOST_ROWS /* of a piece of a chunk: a wide trunk launch's */
#define SPAN_ROWS \
	V41_DATA_SPAN_ROWS /* of a span of a chunk's slabs: its queries launch's */
#define SPAN_SLABS (SPAN_ROWS / ROWS)
#define MOST_LAYER_LAUNCHES 8192u                  /* of a chunk's layer of PIECES_ROWS: 10 a slab and more */
#define FULL_CAPACITY 2048u                        /* the chunk of the full programs' plan */
#define PIECES_ROWS 1100u                          /* a full program of two pieces (1,024 and 76 rows) */
#define PRO_CLASSES 2u                             /* of the PRO's full programs: class 120, then 120 native */
#define FULL_INDEX_SOURCES 4u                       /* layers 2, 8, 14 and 20 */
#define MOST_WRITES 3u                             /* fields a launch writes that a flow reads: codes, exponents,
                                                      operands; post, comb and hidden */
#define MOST_GAPS 8u                               /* of a read, not yet found written */
#define CONTROL_LAYER 3u                           /* of card 0: the layers the flows are shown failing on */
#define POOL_CONTROL_LAYER 2u                      /* a 2-to-1 source and index layer */
#define EMBEDDING LAYERS                           /* in place of a layer: the segment of the embedding */
#define NO_TENSOR V41_DATA_LAYER_TENSOR_COUNT
#define LATE_CARD 3u                               /* a 170HX of layers 24 to 31 */
#define ROTATION_USE_BYTES          \
	(MODEL_ROTATION_ROW_BYTES / \
	 MODEL_ROTATION_USES) /* use 1: with a pool */
#define LAUNCH_WORDS \
	((uint32_t)(sizeof(struct kernel_launch) / sizeof(uint32_t)))
#define KERNEL_WORDS \
	((uint32_t)(offsetof(struct kernel_launch, grid) / sizeof(uint32_t)))
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define TEST_INTERLEAVED_LEASES 2u                 /* the most an interleaved layout below names */
#define TEST_LEASE_ENDS 2u                         /* a lease's layers: of its first visit, of its last */
#define WAIT_LIMIT 1000000000ull                   /* of the step drivers of the interleaved layouts' programs */
#define TEST_CONTEXTS 2u                           /* shallow legacy and deep split selection */
_Static_assert(sizeof(struct kernel_launch) % sizeof(uint32_t) == 0,
	       "a launch is whole words");

static int failures = 0;
static const char *context_text = "";

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)printf("FAILED: %s: %s\n", context_text, what);
        failures++;
    }
}

static void expect_status(result_status status, result_status expected,
			  const char *what)
{
    if (status != expected) {
	    (void)printf("FAILED: %s: %s: %s, expected %s\n", context_text,
			 what, status_text(status), status_text(expected));
	    failures++;
    }
}

/* ---- the reference's sequence (first_token_plan.md 2.1) ---- */

/* The kinds of layer a launch runs on, by model_v41_pack.h's kinds; DIRECT: a source of ratio 1. */
enum { EVERY = 1u << TEST_MODEL_V41_EVERY, SOURCE = 1u << TEST_MODEL_V41_SOURCE, POOLED = 1u << TEST_MODEL_V41_POOLED,
       INDEX = 1u << TEST_MODEL_V41_INDEX, ENGRAM = 1u << TEST_MODEL_V41_ENGRAM,
       DIRECT = 1u << TEST_MODEL_V41_LAYER_KINDS, CANDIDATE_SOURCE = 1u << (TEST_MODEL_V41_LAYER_KINDS + 1u),
       REFERENCE_ORDER = 1u << (TEST_MODEL_V41_LAYER_KINDS + 2u), /* every layer of a plan of that order */
       CHUNK = 1u << (TEST_MODEL_V41_LAYER_KINDS + 3u) /* a chunk's layer alone */ };
/* How a launch's function is named: as written, the stem and 1 or 8 for its rows (dense; _narrow behind for the narrow
 * form), the stem and the slice count of round 2's table for the family + 1 (FAMILY + family). */
enum { NAMED = 0, ROWS_NAMED = 1, FAMILY = 2 };
#define TRUNK_STEM KERNEL_TRUNK_LINEAR_QUANTIZED_STEM
#define OFFICIAL_HC_SITES 2u
#define DENSE_BF16 KERNEL_DENSE_LINEAR_STEM "bfloat16_"
#define DENSE_F32 KERNEL_DENSE_LINEAR_STEM "float32_"
#define DENSE_WIDE "wide_"                         /* behind a type's stem: a wide launch's form */
#define BF16_OUT DENSE_BFLOAT16
#define F32_OUT DENSE_FLOAT32
#define LAUNCH(function, kinds) {function, kinds, NAMED, NO_TENSOR, NO_TENSOR, 0, 0}
#define QUANTIZE(kinds) LAUNCH(KERNEL_ACTIVATION_QUANTIZE, kinds)
#define PRODUCT(stem, family, kinds, tensor, second) {stem, kinds, FAMILY + (family), tensor, second, 0, 0}
#define TRUNK(kinds, tensor, second)                                      \
	PRODUCT(TRUNK_STEM, LINEAR_FAMILY_TRUNK_QUANTIZED, kinds, tensor, \
		second)
#define DENSE(stem, kinds, tensor, output_type, flags) {stem, kinds, ROWS_NAMED, tensor, NO_TENSOR, output_type, flags}

struct expected_launch {
    const char *function;                       /* its name, or the stem the count completes */
    uint32_t kinds;
    uint32_t naming;
    uint32_t tensor, second_tensor;             /* of a product: the layer's tensors its record leads with */
    uint32_t output_type, flags;                /* of a dense product */
};
/* The launches of 2.1 in the program's order: the n-gram rows (E), the attention site (A; a source's pool entry and
 * key, 10 to 13, right behind the window, 5: layer 20's front ends there), the FFN site (F); then, in place of a
 * launch, the areas of the card a launch may read. */
enum sequence_step {
    E_VALUES, E_QUANTIZE, E_PRODUCT, E_GATE, A_MIX, A_FINISH, A_QUANTIZE, A_DOWN, A_QUERY_NORM, A_PREPARE, A_POOLED_KV,
    A_DIRECT_KV, A_POOLED_SCORE, A_COMPRESSOR, A_KEY_PRODUCT, A_KEY, A_QUANTIZE_NORMED, A_UP, A_INDEX_UP,
    A_INDEX_WEIGHTS, A_QUERIES, A_SCAN, A_SELECT, A_CANDIDATES, A_CORE_SCORES, A_CORE, A_GROUPS, A_QUANTIZE_GROUPS,
    A_OUTPUT, A_POST, F_MIX, F_FINISH, F_ROUTER, F_SELECT, W_SELECT, W_COUNT, W_PLACE, W_JOBS, F_QUANTIZE, F_GATE_UP,
    F_QUANTIZE_PAIRS, F_DOWN, F_ROW_SUM, F_SHARED_GATE_UP, F_SHARED_HIDDEN, F_QUANTIZE_SHARED, F_SHARED_DOWN, F_POST,
    SEQUENCE_LENGTH, STREAMS_AREA, PRE_AREA, ROTATION_AREA, STEP_AREA
};
/* REF is the reference's inference/model.py, as first_token_plan.md cites it. */
static const struct expected_launch reference_sequence[SEQUENCE_LENGTH] = {
	[E_VALUES] = LAUNCH(KERNEL_ENGRAM_VALUES, ENGRAM), /* E1, REF:317-320 */
	[E_QUANTIZE] = QUANTIZE(ENGRAM),
	[E_PRODUCT] = TRUNK(ENGRAM, V41_DATA_ENGRAM, NO_TENSOR), /* REF:353 */
	[E_GATE] = LAUNCH(KERNEL_ENGRAM_GATE, ENGRAM), /* REF:355-365 */
	[A_MIX] = DENSE(DENSE_F32, EVERY, V41_DATA_ATTENTION_MIX, F32_OUT,
			DENSE_ROW_SCALE), /* 1, 948-955 */
	[A_FINISH] =
		LAUNCH(KERNEL_HYPER_MIX_FINISH, EVERY), /* 2, REF:982-984 */
	[A_QUANTIZE] = QUANTIZE(EVERY),
	[A_DOWN] = TRUNK(EVERY, V41_DATA_QUERY_DOWN,
			 V41_DATA_WINDOW), /* 4, REF:770, 705 */
	[A_QUERY_NORM] = LAUNCH(KERNEL_ATTENTION_PREPARE,
				CHUNK), /* its qr alone, a chunk's */
	[A_PREPARE] =
		LAUNCH(KERNEL_ATTENTION_PREPARE, EVERY), /* 5, REF:705-718 */
	[A_QUANTIZE_NORMED] = QUANTIZE(EVERY),
	[A_UP] = TRUNK(EVERY, V41_DATA_QUERY_UP, NO_TENSOR), /* 7, REF:771 */
	[A_INDEX_UP] =
		TRUNK(INDEX, V41_DATA_INDEX_QUERY_UP, NO_TENSOR), /* REF:550 */
	[A_INDEX_WEIGHTS] = DENSE(DENSE_BF16, INDEX, V41_DATA_INDEX_WEIGHTS,
				  BF16_OUT, 0), /* 8, REF:555 */
	[A_QUERIES] =
		LAUNCH(KERNEL_ATTENTION_QUERIES, EVERY), /* 9, REF:772, 551 */
	[A_POOLED_KV] = DENSE(DENSE_BF16, POOLED, V41_DATA_COMPRESSOR, F32_OUT,
			      0), /* 10, REF:464 */
	[A_DIRECT_KV] = DENSE(DENSE_BF16, DIRECT, V41_DATA_COMPRESSOR, BF16_OUT,
			      0), /* REF:462 */
	[A_POOLED_SCORE] = DENSE(DENSE_BF16, POOLED, V41_DATA_COMPRESSOR_GATE,
				 F32_OUT, 0), /* REF:465 */
	[A_COMPRESSOR] = LAUNCH(KERNEL_ATTENTION_COMPRESSOR,
				SOURCE), /* 11, REF:458-485 */
	[A_KEY_PRODUCT] = DENSE(DENSE_BF16, SOURCE, V41_DATA_KEY, BF16_OUT,
				0), /* 12, REF:544 */
	[A_KEY] = LAUNCH(KERNEL_ATTENTION_KEY, SOURCE), /* 13, REF:537-548 */
	[A_SCAN] = LAUNCH(KERNEL_INDEX_SCAN, INDEX), /* 14, REF:554-567 */
	[A_SELECT] = LAUNCH(KERNEL_INDEX_SELECT, INDEX), /* 15, REF:577-580 */
	[A_CANDIDATES] = LAUNCH(KERNEL_INDEX_CANDIDATES,
				CANDIDATE_SOURCE), /* the candidate stage */
	[A_CORE_SCORES] = LAUNCH(KERNEL_ATTENTION_CORE_SCORES,
				 REFERENCE_ORDER), /* the scores */
	[A_CORE] = LAUNCH(KERNEL_ATTENTION_CORE, EVERY), /* 16, REF:774-781 */
	[A_GROUPS] = PRODUCT(
		KERNEL_TRUNK_LINEAR_BFLOAT16_STEM, LINEAR_FAMILY_TRUNK_BFLOAT16,
		EVERY, V41_DATA_OUTPUT_GROUPS, NO_TENSOR), /* 17, REF:785-787 */
	[A_QUANTIZE_GROUPS] = QUANTIZE(EVERY),
	[A_OUTPUT] = TRUNK(EVERY, V41_DATA_OUTPUT, NO_TENSOR), /* 19, REF:788 */
	[A_POST] = LAUNCH(KERNEL_HYPER_POST, EVERY), /* 20, REF:986 */
	[F_MIX] = DENSE(DENSE_F32, EVERY, V41_DATA_FFN_MIX, F32_OUT,
			DENSE_ROW_SCALE), /* 21 */
	[F_FINISH] =
		LAUNCH(KERNEL_HYPER_MIX_FINISH, EVERY), /* 22, REF:989-991 */
	[F_ROUTER] = DENSE(DENSE_BF16, EVERY, V41_DATA_ROUTER, F32_OUT,
			   0), /* 23, REF:811 */
	[F_SELECT] = LAUNCH(KERNEL_ROUTER_SELECT, EVERY), /* 24, REF:812-827 */
	[W_SELECT] = LAUNCH(KERNEL_ROUTER_WIDE_SELECT,
			    CHUNK), /* a chunk's: router.h's four */
	[W_COUNT] = LAUNCH(KERNEL_ROUTER_WIDE_COUNT, CHUNK),
	[W_PLACE] = LAUNCH(KERNEL_ROUTER_WIDE_PLACE, CHUNK),
	[W_JOBS] = LAUNCH(KERNEL_ROUTER_WIDE_JOBS, CHUNK),
	[F_QUANTIZE] = QUANTIZE(EVERY),
	[F_GATE_UP] = PRODUCT(KERNEL_EXPERT_GATE_UP_STEM,
			      LINEAR_FAMILY_EXPERT_GATE_UP, EVERY, NO_TENSOR,
			      NO_TENSOR), /* 26, REF:841-847 */
	[F_QUANTIZE_PAIRS] = QUANTIZE(EVERY),
	[F_DOWN] = PRODUCT(KERNEL_EXPERT_DOWN_STEM, LINEAR_FAMILY_EXPERT_DOWN,
			   EVERY, NO_TENSOR, NO_TENSOR),
	[F_ROW_SUM] =
		LAUNCH(KERNEL_EXPERT_ROW_SUM, EVERY), /* 29, REF:893-900 */
	[F_SHARED_GATE_UP] = TRUNK(EVERY, V41_DATA_SHARED_GATE,
				   V41_DATA_SHARED_UP), /* 30, REF:887 */
	[F_SHARED_HIDDEN] = LAUNCH(KERNEL_TRUNK_HIDDEN_VALUE, EVERY),
	[F_QUANTIZE_SHARED] = QUANTIZE(EVERY),
	[F_SHARED_DOWN] = TRUNK(EVERY, V41_DATA_SHARED_DOWN, NO_TENSOR),
	[F_POST] = LAUNCH(KERNEL_HYPER_POST, EVERY), /* 34, REF:903-904 */
};

/* The data flow of a layer (REF:968-994 through the functions 2.1 cites): the consumer reads at that byte of its
 * record the offsets (words of them, in a row) that the producer wrote at its byte, or the card's area. A flow whose
 * consumer or producer the layer does not run is not one of its flows. */
struct data_flow {
    uint32_t consumer, consumer_at, producer, producer_at, words;
};
#define AT(record, field) ((uint32_t)offsetof(struct record, field))
#define PAIRED 2u                                 /* post and comb; gate and up */
#define QUANTIZED 3u                              /* codes, exponents, operands */
#define FLOW(consumer, record, field, producer, producer_record, producer_field) \
    {consumer, AT(record, field), producer, AT(producer_record, producer_field), 1}
#define AREA(consumer, record, field, area) {consumer, AT(record, field), area, 0, 1}
#define STEP_AND_ROTATION(consumer, record) \
    AREA(consumer, record, step_offset, STEP_AREA), AREA(consumer, record, rotation_offset, ROTATION_AREA)
#define SITE_AREAS(finish)                                                  \
	AREA(finish, hyper_finish_arguments, streams_offset, STREAMS_AREA), \
		AREA(finish, hyper_finish_arguments, pre_offset, PRE_AREA)
#define QUANTIZED_BY(consumer, record, producer)        \
	{ consumer, AT(record, codes_offset), producer, \
	  AT(activation_quantize_arguments, codes_offset), QUANTIZED }
#define HIDDEN_OF(consumer, record, field, site)                    \
	FLOW(consumer, record, field, site, hyper_finish_arguments, \
	     hidden_offset)
#define DENSE_OUT(consumer, record, field, producer)                    \
	FLOW(consumer, record, field, producer, dense_linear_arguments, \
	     output_offset)
#define TRUNK_OUT(consumer, record, field, producer) \
	FLOW(consumer, record, field, producer,      \
	     trunk_linear_quantized_arguments, output_offset)
static const struct data_flow data_flows[] = {
	FLOW(E_QUANTIZE, activation_quantize_arguments, values_offset, E_VALUES,
	     engram_values_arguments, values_offset),
	QUANTIZED_BY(E_PRODUCT, trunk_linear_quantized_arguments, E_QUANTIZE),
	TRUNK_OUT(E_GATE, engram_gate_arguments, projection_offset, E_PRODUCT),
	AREA(E_GATE, engram_gate_arguments, streams_offset, STREAMS_AREA),
	AREA(A_MIX, dense_linear_arguments, rows_offset, STREAMS_AREA),
	SITE_AREAS(A_FINISH),
	DENSE_OUT(A_FINISH, hyper_finish_arguments, mixes_offset, A_MIX),
	HIDDEN_OF(A_QUANTIZE, activation_quantize_arguments, values_offset,
		  A_FINISH),
	QUANTIZED_BY(A_DOWN, trunk_linear_quantized_arguments, A_QUANTIZE),
	STEP_AND_ROTATION(A_PREPARE, attention_prepare_arguments),
	TRUNK_OUT(A_PREPARE, attention_prepare_arguments, query_rank_offset,
		  A_DOWN),
	{ A_PREPARE, AT(attention_prepare_arguments, window_input_offset),
	  A_DOWN,
	  AT(trunk_linear_quantized_arguments, output_offset) +
		  (uint32_t)sizeof(uint64_t),
	  1 }, /* wkv's: the second */
	FLOW(A_QUANTIZE_NORMED, activation_quantize_arguments, values_offset,
	     A_PREPARE, attention_prepare_arguments, query_normed_offset),
	QUANTIZED_BY(A_UP, trunk_linear_quantized_arguments, A_QUANTIZE_NORMED),
	QUANTIZED_BY(A_INDEX_UP, trunk_linear_quantized_arguments,
		     A_QUANTIZE_NORMED),
	HIDDEN_OF(A_INDEX_WEIGHTS, dense_linear_arguments, rows_offset,
		  A_FINISH),
	AREA(A_QUERIES, attention_queries_arguments, rotation_offset,
	     ROTATION_AREA),
	TRUNK_OUT(A_QUERIES, attention_queries_arguments, heads_offset, A_UP),
	TRUNK_OUT(A_QUERIES, attention_queries_arguments, index_heads_offset,
		  A_INDEX_UP),
	HIDDEN_OF(A_POOLED_KV, dense_linear_arguments, rows_offset, A_FINISH),
	HIDDEN_OF(A_DIRECT_KV, dense_linear_arguments, rows_offset, A_FINISH),
	HIDDEN_OF(A_POOLED_SCORE, dense_linear_arguments, rows_offset,
		  A_FINISH),
	STEP_AND_ROTATION(A_COMPRESSOR, attention_compressor_arguments),
	DENSE_OUT(A_COMPRESSOR, attention_compressor_arguments, kv_offset,
		  A_POOLED_KV),
	DENSE_OUT(A_COMPRESSOR, attention_compressor_arguments, kv_offset,
		  A_DIRECT_KV),
	DENSE_OUT(A_COMPRESSOR, attention_compressor_arguments, score_offset,
		  A_POOLED_SCORE),
	FLOW(A_KEY_PRODUCT, dense_linear_arguments, rows_offset, A_COMPRESSOR,
	     attention_compressor_arguments, latent_offset),
	STEP_AND_ROTATION(A_KEY, attention_key_arguments),
	DENSE_OUT(A_KEY, attention_key_arguments, key_input_offset,
		  A_KEY_PRODUCT),
	AREA(A_SCAN, index_scan_arguments_record, step_offset, STEP_AREA),
	AREA(A_SELECT, index_select_arguments_record, step_offset, STEP_AREA),
	FLOW(A_SCAN, index_scan_arguments_record, queries_offset, A_QUERIES,
	     attention_queries_arguments, index_queries_offset),
	DENSE_OUT(A_SCAN, index_scan_arguments_record, weights_offset,
		  A_INDEX_WEIGHTS),
	FLOW(A_SELECT, index_select_arguments_record, scores_offset, A_SCAN,
	     index_scan_arguments_record, scores_offset),
	AREA(A_CORE_SCORES, attention_core_arguments_record, step_offset,
	     STEP_AREA),
	FLOW(A_CORE_SCORES, attention_core_arguments_record, queries_offset,
	     A_QUERIES, attention_queries_arguments, queries_offset),
	FLOW(A_CORE, attention_core_arguments_record, scores_offset,
	     A_CORE_SCORES, attention_core_arguments_record, scores_offset),
	AREA(A_CORE, attention_core_arguments_record, step_offset, STEP_AREA),
	FLOW(A_CORE, attention_core_arguments_record, queries_offset, A_QUERIES,
	     attention_queries_arguments, queries_offset),
	FLOW(A_GROUPS, trunk_linear_bfloat16_arguments, values_offset, A_CORE,
	     attention_core_arguments_record, output_offset),
	FLOW(A_QUANTIZE_GROUPS, activation_quantize_arguments, values_offset,
	     A_GROUPS, trunk_linear_bfloat16_arguments, output_offset),
	QUANTIZED_BY(A_OUTPUT, trunk_linear_quantized_arguments,
		     A_QUANTIZE_GROUPS),
	AREA(A_POST, hyper_post_arguments, streams_offset, STREAMS_AREA),
	TRUNK_OUT(A_POST, hyper_post_arguments, output_offset, A_OUTPUT),
	{ A_POST, AT(hyper_post_arguments, post_offset), A_FINISH,
	  AT(hyper_finish_arguments, post_offset), PAIRED },
	AREA(F_MIX, dense_linear_arguments, rows_offset, STREAMS_AREA),
	SITE_AREAS(F_FINISH),
	DENSE_OUT(F_FINISH, hyper_finish_arguments, mixes_offset, F_MIX),
	HIDDEN_OF(F_ROUTER, dense_linear_arguments, rows_offset, F_FINISH),
	AREA(F_SELECT, router_select_arguments, step_offset, STEP_AREA),
	DENSE_OUT(F_SELECT, router_select_arguments, scores_offset, F_ROUTER),
	HIDDEN_OF(F_QUANTIZE, activation_quantize_arguments, values_offset,
		  F_FINISH),
	QUANTIZED_BY(F_GATE_UP, expert_gate_up_arguments, F_QUANTIZE),
	FLOW(F_GATE_UP, expert_gate_up_arguments, jobs_offset, F_SELECT,
	     router_select_arguments, jobs_offset),
	FLOW(F_QUANTIZE_PAIRS, activation_quantize_arguments, values_offset,
	     F_GATE_UP, expert_gate_up_arguments, hidden_offset),
	QUANTIZED_BY(F_DOWN, expert_down_arguments, F_QUANTIZE_PAIRS),
	FLOW(F_DOWN, expert_down_arguments, jobs_offset, F_SELECT,
	     router_select_arguments, jobs_offset),
	FLOW(F_ROW_SUM, expert_row_sum_arguments, pair_output_offset, F_DOWN,
	     expert_down_arguments, output_offset),
	QUANTIZED_BY(F_SHARED_GATE_UP, trunk_linear_quantized_arguments,
		     F_QUANTIZE),
	{ F_SHARED_HIDDEN, AT(trunk_hidden_arguments, gate_offset),
	  F_SHARED_GATE_UP, AT(trunk_linear_quantized_arguments, output_offset),
	  PAIRED },
	FLOW(F_QUANTIZE_SHARED, activation_quantize_arguments, values_offset,
	     F_SHARED_HIDDEN, trunk_hidden_arguments, hidden_offset),
	QUANTIZED_BY(F_SHARED_DOWN, trunk_linear_quantized_arguments,
		     F_QUANTIZE_SHARED),
	AREA(F_POST, hyper_post_arguments, streams_offset, STREAMS_AREA),
	TRUNK_OUT(F_POST, hyper_post_arguments, output_offset, F_SHARED_DOWN),
	{ F_POST, AT(hyper_post_arguments, post_offset), F_FINISH,
	  AT(hyper_finish_arguments, post_offset), PAIRED },
	FLOW(F_POST, hyper_post_arguments, sums_offset, F_ROW_SUM,
	     expert_row_sum_arguments, row_output_offset),
	/* a chunk's own: qr alone ahead of the slabs' prepares; the wide select's scores and jobs */
	TRUNK_OUT(A_QUERY_NORM, attention_prepare_arguments, query_rank_offset,
		  A_DOWN),
	FLOW(A_QUANTIZE_NORMED, activation_quantize_arguments, values_offset,
	     A_QUERY_NORM, attention_prepare_arguments, query_normed_offset),
	AREA(W_SELECT, router_wide_arguments, step_offset, STEP_AREA),
	DENSE_OUT(W_SELECT, router_wide_arguments, scores_offset, F_ROUTER),
	FLOW(F_GATE_UP, expert_gate_up_arguments, jobs_offset, W_JOBS,
	     router_wide_arguments, jobs_offset),
	FLOW(F_DOWN, expert_down_arguments, jobs_offset, W_JOBS,
	     router_wide_arguments, jobs_offset),
	FLOW(F_GATE_UP, expert_gate_up_arguments, jobs_offset, W_PLACE,
	     router_wide_arguments, tiles_offset),
	FLOW(F_DOWN, expert_down_arguments, jobs_offset, W_PLACE,
	     router_wide_arguments, tiles_offset),
};

/* The segments: a card's step segment, the embedding's, a layer's, the head's; those outside the layers by name. */
enum segment_kind { STEP_SEGMENT, EMBEDDING_SEGMENT, LAYER_SEGMENT, HEAD_SEGMENT };
/* What of its layer a layer's segment runs: all of it, or layer 20's front (model_seam.h) and then the rest. */
enum layer_part { WHOLE_LAYER, FRONT_PART, LATER_PART };
static const char *const outside_sequences[][HEAD_LAUNCHES] = {
	[STEP_SEGMENT] = { KERNEL_ATTENTION_ROTATION },
	[EMBEDDING_SEGMENT] = { KERNEL_HYPER_EMBED },
	[HEAD_SEGMENT] = { KERNEL_HYPER_FINAL, DENSE_BF16, KERNEL_SAMPLER }
}; /* the head: and its rows */
static const uint32_t outside_launches[] = {[STEP_SEGMENT] = 1, [EMBEDDING_SEGMENT] = 1, [LAYER_SEGMENT] = 0,
                                            [HEAD_SEGMENT] = HEAD_LAUNCHES};
static const struct matrix_shape_record expert_shapes[] = {
	{ V41_DATA_EXPERT_INNER, MODEL_HIDDEN, 0, 0 },
	{ MODEL_HIDDEN, V41_DATA_EXPERT_INNER, 0, 0 }
};
/* The outputs a block of a dense product's wide form, by its weights' type, on class 80 and on class 120 in either form
 * (dense_linear.h). */
static const uint32_t dense_wide_outputs[][DENSE_TYPE_COUNT] = {
	{ [DENSE_BFLOAT16] = 1u, [DENSE_FLOAT32] = 8u },
	{ [DENSE_BFLOAT16] = 8u, [DENSE_FLOAT32] = 4u }
};
/* Class 80's skinny forms (dense_linear.c, 00509d7): a float32 matrix of at most DENSE_SKINNY_MOST_OUTPUTS_VALUE
 * outputs (the hyper mixes) takes 1 output a block to DENSE_SKINNY_MANY_ROWS rows, 4 above. */
static const uint32_t dense_skinny_outputs[DENSE_SKINNY_REGIMES] = { 1u, 4u };

/* The kinds of a layer; its source (the latest source at or before it) and group (the latest index layer). */
static uint32_t kinds_of(uint32_t layer)
{
    uint32_t kinds = DIRECT * ((test_model_v41_layers_of_kind[TEST_MODEL_V41_SOURCE] &
                                ~test_model_v41_layers_of_kind[TEST_MODEL_V41_POOLED]) >> layer & 1u);
    kinds |= layer == MODEL_CANDIDATE_SOURCE ? CANDIDATE_SOURCE : 0u;
    for (uint32_t kind = 0; kind < TEST_MODEL_V41_LAYER_KINDS; kind++) {
        kinds |= (uint32_t)(test_model_v41_layers_of_kind[kind] >> layer & 1u) << kind;
    }
    return kinds;
}

static uint32_t latest_of(uint32_t kind, uint32_t layer, uint32_t none)
{
    uint32_t latest = none;
    for (uint32_t earlier = 0; earlier <= layer; earlier++) {
        latest = (test_model_v41_layers_of_kind[kind] >> earlier & 1u) != 0 ? earlier : latest;
    }
    return latest;
}

/* ---- a program against its plan ---- */

struct program_case {
	const struct model_plan_record *plan;
	const struct execution_program *program;
	const struct card_description_record *descriptions;
	uint32_t rows;
};

/* The function a launch runs, as the card prepared it. */
static void name_of(uint32_t card, const struct kernel_launch *launch,
		    char *name_out)
{
    struct test_host_memory_backend_kernel kernel;
    const bool is_known = test_host_memory_backend_kernel_of(card, launch->kernel, &kernel);
    (void)snprintf(name_out, NAME_CAPACITY, "%s", is_known ? kernel.function_name : "(no kernel)");
}

/* The name the reference's sequence gives a launch on that card and layer; a trunk product of more than 8 rows is of
 * its wide form's instance (linear.h): the wide stem and its count, or the interleaved instance; a dense product of
 * more than 8 rows its class's wide form (dense_linear.h). */
static void expected_name_of(const struct program_case *made, uint32_t card, uint32_t layer,
                             const struct expected_launch *expected, char *name_out)
{
	const struct model_limits *const limits = &made->plan->limits;
	const uint32_t family =
		expected->naming >= FAMILY ? expected->naming - FAMILY : 0u;
	const bool is_product = expected->naming >= FAMILY &&
				family <= LINEAR_FAMILY_TRUNK_BFLOAT16;
	const bool is_wide = is_product && made->rows > ROWS &&
			     family >= LINEAR_FAMILY_TRUNK_QUANTIZED;
	if (is_product && family < LINEAR_FAMILY_TRUNK_QUANTIZED &&
	    made->rows >= EXPERT_ORDERED_MINIMUM_ROWS &&
	    made->descriptions[card].arithmetic_class ==
		    ARITHMETIC_CLASS_SM_80) {
		(void)snprintf(name_out, NAME_CAPACITY, "%s",
			       family == LINEAR_FAMILY_EXPERT_GATE_UP ?
				       KERNEL_EXPERT_ORDERED_WIDE_GATE_UP :
				       KERNEL_EXPERT_ORDERED_WIDE_DOWN);
		return;
	}
	if (is_wide && made->descriptions[card].arithmetic_class ==
			       ARITHMETIC_CLASS_SM_80) {
		(void)snprintf(name_out, NAME_CAPACITY, "%s",
			       family == LINEAR_FAMILY_TRUNK_QUANTIZED ?
				       KERNEL_ORDERED_WIDE_QUANTIZED_SMALL :
				       KERNEL_ORDERED_WIDE_BFLOAT16_SMALL);
		return;
	}
	enum linear_wide_form form = LINEAR_WIDE_GROUP_BLOCKS;
	uint32_t slice_count = 0;
	if (is_product) {
		const struct matrix_shape_record *const shape =
			expected->tensor != NO_TENSOR ?
				&made->plan
					 ->layer_tensors[layer][expected->tensor]
					 .shape :
				&expert_shapes[family < COUNT_OF(expert_shapes) ?
						       family :
						       0u];
		expect_status(
			is_wide ?
				linear_default_wide_form(
					limits->linear_profile,
					limits->linear_order,
					(enum linear_family_code)family, shape,
					made->descriptions[card]
						.arithmetic_class,
					&slice_count, &form) :
				linear_default_slice_count(
					limits->linear_profile,
					limits->linear_order,
					(enum linear_family_code)family, shape,
					made->descriptions[card]
						.arithmetic_class,
					&slice_count),
			STATUS_OK,
			"round 2's table has the slice count of a product");
	}
	if (is_wide && form >= LINEAR_INTERLEAVED_CHUNK_BLOCKS) {
		(void)snprintf(name_out, NAME_CAPACITY, "%s",
			       KERNEL_TRUNK_LINEAR_QUANTIZED_INTERLEAVED);
		return;
	}
	const char *const wide_stem =
		family == LINEAR_FAMILY_TRUNK_QUANTIZED ?
			KERNEL_TRUNK_LINEAR_QUANTIZED_WIDE_STEM :
			KERNEL_TRUNK_LINEAR_BFLOAT16_WIDE_STEM;
	const char *const stem = is_wide ? wide_stem : expected->function;
	const bool is_wide_dense = expected->naming == ROWS_NAMED &&
				   made->rows > ROWS;
	const uint32_t class_index =
		made->descriptions[card].arithmetic_class ==
				ARITHMETIC_CLASS_SM_80 ?
			0u :
			1u;
	const uint32_t dense_type = strcmp(expected->function, DENSE_F32) == 0 ?
					    DENSE_FLOAT32 :
					    DENSE_BFLOAT16;
	const uint32_t output_size =
		expected->naming == ROWS_NAMED &&
				expected->tensor != NO_TENSOR ?
			made->plan->layer_tensors[layer][expected->tensor]
				.shape.output_size :
			0u;
	const uint32_t skinny =
		dense_skinny_outputs[made->rows > DENSE_SKINNY_MANY_ROWS ? 1u :
									   0u];
	const bool is_skinny = is_wide_dense && class_index == 0 &&
			       dense_type == DENSE_FLOAT32 &&
			       output_size <= DENSE_SKINNY_MOST_OUTPUTS_VALUE &&
			       output_size % skinny == 0;
	const uint32_t count =
		is_skinny     ? skinny :
		is_wide_dense ? dense_wide_outputs[class_index][dense_type] :
		expected->naming == ROWS_NAMED ?
				(made->rows == 1 ? 1u : LINEAR_MAXIMUM_ROWS) :
				slice_count;
	/* a dense product of fewer than DENSE_NARROW_BELOW_BLOCKS blocks of 4 outputs is the narrow form */
	const bool is_narrow = expected->naming == ROWS_NAMED &&
			       !is_wide_dense &&
			       output_size / DENSE_OUTPUTS_PER_BLOCK <
				       DENSE_NARROW_BELOW_BLOCKS;
	if (expected->naming ==
	    NAMED) { /* the core's kernels are the plan's order's (attention_core.h) */
		const int reference_core =
			strcmp(expected->function, KERNEL_ATTENTION_CORE) ==
				0 &&
			limits->linear_order == LINEAR_ORDER_REFERENCE;
		(void)snprintf(name_out, NAME_CAPACITY, "%s",
			       reference_core ? KERNEL_ATTENTION_CORE_VALUES :
						expected->function);
	} else {
		(void)snprintf(name_out, NAME_CAPACITY, "%s%s%" PRIu32 "%s",
			       stem, is_wide_dense ? DENSE_WIDE : "", count,
			       is_narrow ? "_narrow" : "");
	}
}

/* What the records of a layer's launches read and write: its card's arrays of it, its source and its group. Copies of
 * records are static here and below: gcc 15.2 at -march=znver4 lays out 64-byte locals of a realigned frame for a base
 * that the address sanitizer's fake stack does not give (a vmovdqa64 to an address 32 bytes off), so no record lies
 * on the stack of this single-threaded test. */
static void expect_record(const struct program_case *made, uint32_t card_index,
			  uint32_t layer, const char *name,
			  const struct kernel_launch *launch)
{
	const struct v41_card *const card = &made->plan->cards[card_index];
	const struct program_card_record *const program_card =
		&made->program->cards[card_index];
	const uint32_t source = latest_of(TEST_MODEL_V41_SOURCE, layer, LAYERS);
	const uint32_t group = latest_of(TEST_MODEL_V41_INDEX, layer, LAYERS);
	const uint32_t ratio = source == LAYERS ? 0 :
			       (kinds_of(source) & POOLED) != 0 ?
						  MODEL_POOLED_RATIO :
						  MODEL_DIRECT_RATIO;
	const uint64_t pool = source == LAYERS ? 0 : card->pool_offset[source];
	const uint64_t selection =
		group == LAYERS ? 0 : card->selection_offset[group];
	if (strcmp(name, KERNEL_ATTENTION_PREPARE) == 0) {
		static struct attention_prepare_arguments prepare;
		memcpy(&prepare, launch->argument, sizeof(prepare));
		expect(prepare.ring_offset == card->ring_offset[layer] &&
			       prepare.rotation_use == (ratio == 0 ? 0 : 1u),
		       "the window row goes to the layer's ring, rotated at its theta");
	} else if (strcmp(name, KERNEL_ATTENTION_COMPRESSOR) == 0) {
		static struct attention_compressor_arguments compressor;
		memcpy(&compressor, launch->argument, sizeof(compressor));
		expect(compressor.pool_offset == card->pool_offset[layer] &&
			       compressor.ratio == ratio &&
			       compressor.slots_offset ==
				       (ratio == MODEL_POOLED_RATIO ?
						card->compressor_offset[layer] :
						0),
		       "a source writes its own pool with its slots at its ratio");
	} else if (strcmp(name, KERNEL_ATTENTION_KEY) == 0) {
		static struct attention_key_arguments key;
		memcpy(&key, launch->argument, sizeof(key));
		expect(key.keys_offset == card->key_tiles_offset[layer],
		       "a source writes its own key tiles");
	} else if (strcmp(name, KERNEL_INDEX_SCAN) == 0) {
		static struct index_scan_arguments_record scan;
		memcpy(&scan, launch->argument, sizeof(scan));
		expect(source < LAYERS &&
			       scan.keys_offset ==
				       card->key_tiles_offset[source] &&
			       scan.ratio == ratio,
		       "an index layer scores its source's keys (no shared slot)");
	} else if (strcmp(name, KERNEL_INDEX_SELECT) == 0) {
		static struct index_select_arguments_record select;
		memcpy(&select, launch->argument, sizeof(select));
		expect(select.selection_offset == card->selection_offset[layer],
		       "an index layer writes its own selection");
	} else if (strcmp(name, KERNEL_ATTENTION_CORE) == 0 ||
		   strcmp(name, KERNEL_ATTENTION_CORE_SCORES) == 0 ||
		   strcmp(name, KERNEL_ATTENTION_CORE_VALUES) == 0) {
		static struct attention_core_arguments_record core;
		memcpy(&core, launch->argument, sizeof(core));
		expect(core.ring_offset == card->ring_offset[layer] &&
			       core.pool_offset == (ratio == 0 ? 0 : pool) &&
			       core.rotation_offset ==
				       card->array_offset[V41_DATA_ARRAY_ROTATION] +
					       (ratio == 0 ?
							0 :
							ROTATION_USE_BYTES) &&
			       core.selection_offset ==
				       (ratio == 0 ? 0 : selection) &&
			       core.pool_ratio == ratio &&
			       core.sinks_offset ==
				       made->plan
					       ->layer_tensors[layer]
							      [V41_DATA_SINK]
					       .stored.byte_offset,
		       "the core reads its ring, its source's pool, its group's selection, its sinks and its theta's use");
	} else if (strcmp(name, KERNEL_ROUTER_SELECT) == 0) {
		static struct router_select_arguments router;
		memcpy(&router, launch->argument, sizeof(router));
		expect(router.expert_offsets_offset ==
				       card->expert_table_offset[layer] &&
			       router.bias_offset ==
				       made->plan
					       ->layer_tensors
						       [layer]
						       [V41_DATA_ROUTER_BIAS]
					       .stored.byte_offset &&
			       router.reference_order ==
				       (made->plan->limits.linear_order ==
					LINEAR_ORDER_REFERENCE),
		       "the router reads its layer's table and bias, in the plan's order");
	} else if (strcmp(name, KERNEL_ENGRAM_VALUES) == 0) {
		static struct engram_values_arguments values;
		memcpy(&values, launch->argument, sizeof(values));
		uint64_t below =
			0; /* the card's tables before this one (model_seam.h) */
		for (uint32_t earlier = 0; earlier < layer; earlier++) {
			below +=
				(kinds_of(earlier) & ENGRAM) != 0 &&
						made->plan->layer_card[earlier] ==
							card_index ?
					1u :
					0u;
		}
		expect(values.rows_offset ==
			       program_card->engram_rows_offset +
				       below * made->plan->limits.row_capacity *
					       TOKEN_ROWS_BYTES,
		       "an n-gram layer reads its table's rows where model_seam.h lays them");
	}
}

/* A layer's launches, by their steps of the sequence (NULL: not run), against the data flow of a layer. */
static void expect_data_flow(const struct program_case *made,
			     uint32_t card_index, uint32_t layer,
			     const struct kernel_launch *const *by_step)
{
	const struct program_card_record *const card =
		&made->program->cards[card_index];
	const uint64_t areas[] = {
		[STREAMS_AREA - STREAMS_AREA] = card->streams_offset,
		[PRE_AREA - STREAMS_AREA] = card->carried_pre_offset,
		[ROTATION_AREA - STREAMS_AREA] =
			made->plan->cards[card_index]
				.array_offset[V41_DATA_ARRAY_ROTATION],
		[STEP_AREA - STREAMS_AREA] = card->step_offset
	};
	for (uint32_t index = 0; index < COUNT_OF(data_flows); index++) {
		const struct data_flow *const flow = &data_flows[index];
		const struct kernel_launch *const consumer =
			by_step[flow->consumer];
		const struct kernel_launch *const producer =
			flow->producer < SEQUENCE_LENGTH ?
				by_step[flow->producer] :
				NULL;
		uint64_t read[QUANTIZED] = { 0 }, written[QUANTIZED] = { 0 };
		if (consumer == NULL ||
		    (producer == NULL && flow->producer < SEQUENCE_LENGTH)) {
			continue;
		}
		memcpy(read, consumer->argument + flow->consumer_at,
		       flow->words * sizeof(uint64_t));
		if (producer != NULL) {
			memcpy(written, producer->argument + flow->producer_at,
			       flow->words * sizeof(uint64_t));
		} else {
			written[0] = areas[flow->producer - STREAMS_AREA];
		}
		if (memcmp(read, written, sizeof(read)) != 0 ||
		    written[0] == PROGRAM_NOWHERE) {
			(void)printf(
				"FAILED: %s: layer %" PRIu32
				": %s reads at byte %" PRIu32 " %" PRIu64
				", not %" PRIu64 " of %s\n",
				context_text, layer,
				reference_sequence[flow->consumer].function,
				flow->consumer_at, read[0], written[0],
				producer != NULL ?
					reference_sequence[flow->producer]
						.function :
					"the card's area");
			failures++;
		}
    }
}

/* Fused launches expose the same semantic reads and writes as the original steps. These records are test views;
 * the actual launch sequence, names, slice counts and fused-only offsets are checked before expanding it. */
static void semantic_put(struct kernel_launch *records,
			 const struct kernel_launch **by_step, uint32_t step,
			 const void *record, uint32_t bytes)
{
    memcpy(records[step].argument, record, bytes);
    records[step].argument_byte_count = bytes;
    by_step[step] = &records[step];
}
#define SEMANTIC(step, record) semantic_put(records, by_step, step, &(record), (uint32_t)sizeof(record))

static void expect_fused_name(const struct program_case *made, uint32_t card,
			      uint32_t layer, uint32_t step,
			      const struct kernel_launch *launch,
			      const char *stem)
{
    char original[NAME_CAPACITY], expected[NAME_CAPACITY], actual[NAME_CAPACITY];
    expected_name_of(made, card, layer, &reference_sequence[step], original);
    const char *const suffix = strrchr(original, '_');
    (void)snprintf(expected, sizeof(expected), "%s%s", stem, suffix != NULL ? suffix + 1 : "missing");
    name_of(card, launch, actual);
    expect(strcmp(expected, actual) == 0, "a fused product keeps its class's selected slice count");
}

static void expand_front(const struct program_case *made, uint32_t card,
			 uint32_t layer, uint32_t step,
			 const struct kernel_launch *launch,
			 struct kernel_launch *records,
			 const struct kernel_launch **by_step)
{
	struct site_front_arguments front;
	memcpy(&front, launch->argument, sizeof(front));
	const bool ffn = step == F_MIX;
	const struct v41_tensor *const tensors =
		made->plan->layer_tensors[layer];
	const uint64_t mixes = front.scratch_offset +
			       offsetof(struct site_front_scratch, mixes);
	const struct dense_linear_arguments mix = {
		.weights_offset = front.mix_weights_offset,
		.rows_offset = front.streams_offset,
		.output_offset = mixes
	};
	const struct hyper_finish_arguments finish = { front.streams_offset,
						       front.pre_offset,
						       mixes,
						       front.scale_offset,
						       front.base_offset,
						       front.post_offset,
						       front.comb_offset,
						       front.norm_weight_offset,
						       front.hidden_offset };
	const struct activation_quantize_arguments quantize = {
		front.hidden_offset,	front.codes_offset,
		front.exponents_offset, front.operands_offset,
		MODEL_HIDDEN,		0
	};
	char name[NAME_CAPACITY], expected[NAME_CAPACITY];
	name_of(card, launch, name);
	(void)snprintf(expected, sizeof(expected), "%s_front_%" PRIu32,
		       ffn ? "ffn" : "site", made->rows == 1u ? 1u : ROWS);
	expect(strcmp(name, expected) == 0 &&
		       front.select.row_count == made->rows &&
		       front.scratch_offset ==
			       made->plan->cards[card].front_scratch_offset &&
		       front.mix_weights_offset ==
			       tensors[ffn ? V41_DATA_FFN_MIX :
					     V41_DATA_ATTENTION_MIX]
				       .stored.byte_offset &&
		       front.scale_offset ==
			       tensors[ffn ? V41_DATA_FFN_MIX_SCALE :
					     V41_DATA_ATTENTION_MIX_SCALE]
				       .stored.byte_offset &&
		       front.base_offset ==
			       tensors[ffn ? V41_DATA_FFN_MIX_BASE :
					     V41_DATA_ATTENTION_MIX_BASE]
				       .stored.byte_offset &&
		       front.norm_weight_offset ==
			       tensors[ffn ? V41_DATA_FFN_NORM :
					     V41_DATA_ATTENTION_NORM]
				       .stored.byte_offset,
	       "the fused front uses its site's tensors, row form and reserved scratch");
	SEMANTIC(step, mix);
	SEMANTIC(ffn ? F_FINISH : A_FINISH, finish);
	SEMANTIC(ffn ? F_QUANTIZE : A_QUANTIZE, quantize);
	if (ffn) {
		const struct dense_linear_arguments router = {
			.weights_offset = front.router_weights_offset,
			.rows_offset = front.hidden_offset,
			.output_offset = front.select.scores_offset
		};
		expect(front.router_weights_offset ==
			       tensors[V41_DATA_ROUTER].stored.byte_offset,
		       "the FFN front reads its own router tensor");
		SEMANTIC(F_ROUTER, router);
		SEMANTIC(F_SELECT, front.select);
		expect_record(made, card, layer, KERNEL_ROUTER_SELECT,
			      by_step[F_SELECT]);
	}
}

static void expand_groups(const struct program_case *made, uint32_t card,
			  uint32_t layer, const struct kernel_launch *launch,
			  struct kernel_launch *records,
			  const struct kernel_launch **by_step)
{
	struct attention_groups_arguments groups;
	memcpy(&groups, launch->argument, sizeof(groups));
	const struct trunk_linear_bfloat16_arguments product = {
		groups.weights_offset, groups.values_offset,
		groups.groups_offset,  groups.output_size,
		groups.input_size,     groups.row_count,
		groups.segment_count
	};
	const struct activation_quantize_arguments quantize = {
		groups.groups_offset,	 groups.codes_offset,
		groups.exponents_offset, groups.operands_offset,
		groups.output_size,	 0
	};
	expect_fused_name(made, card, layer, A_GROUPS, launch,
			  KERNEL_ATTENTION_GROUPS_STEM);
	expect(groups.weights_offset ==
			       made->plan
				       ->layer_tensors[layer]
						      [V41_DATA_OUTPUT_GROUPS]
				       .stored.byte_offset &&
		       groups.row_count == made->rows &&
		       groups.output_size == V41_DATA_WIDEST_ROW &&
		       groups.input_size == MODEL_HEADS_VALUE *
						    MODEL_HEAD_DIMS /
						    MODEL_OUTPUT_GROUPS &&
		       groups.segment_count == MODEL_OUTPUT_GROUPS,
	       "the fused groups use the grouped output matrix and complete row shape");
	SEMANTIC(A_GROUPS, product);
	SEMANTIC(A_QUANTIZE_GROUPS, quantize);
}

static void expand_projection(const struct program_case *made, uint32_t card,
			      uint32_t layer,
			      const struct kernel_launch *launch,
			      struct kernel_launch *records,
			      const struct kernel_launch **by_step)
{
	struct attention_projection_arguments projection;
	memcpy(&projection, launch->argument, sizeof(projection));
	const struct trunk_linear_quantized_arguments product = {
		{ projection.weights_offset, 0 },
		{ projection.output_offset, 0 },
		projection.codes_offset,
		projection.exponents_offset,
		projection.operands_offset,
		{ projection.output_size, 0 },
		projection.input_size,
		projection.row_count
	};
	const struct hyper_post_arguments post = { projection.streams_offset,
						   projection.post_offset,
						   projection.comb_offset,
						   projection.output_offset,
						   0,
						   0,
						   0 };
	expect_fused_name(made, card, layer, A_OUTPUT, launch,
			  KERNEL_ATTENTION_PROJECTION_STEM);
	expect(projection.weights_offset ==
			       made->plan->layer_tensors[layer][V41_DATA_OUTPUT]
				       .stored.byte_offset &&
		       projection.row_count == made->rows &&
		       projection.output_size == MODEL_HIDDEN &&
		       projection.input_size == V41_DATA_WIDEST_ROW,
	       "the fused projection reads its output matrix at the complete row shape");
	SEMANTIC(A_OUTPUT, product);
	SEMANTIC(A_POST, post);
}

static void expand_producer(const struct program_case *made, uint32_t card,
			    uint32_t layer, const struct kernel_launch *launch,
			    struct kernel_launch *records,
			    const struct kernel_launch **by_step)
{
	struct expert_gate_up_quantize_arguments producer;
	memcpy(&producer, launch->argument, sizeof(producer));
	const struct expert_gate_up_arguments product = {
		producer.jobs_offset,	   producer.codes_offset,
		producer.exponents_offset, producer.operands_offset,
		producer.hidden_offset,	   producer.output_size,
		producer.input_size,	   producer.pair_count,
		producer.pairs_per_row,	   producer.activation_limit,
		producer.reserved_zero
	};
	const struct activation_quantize_arguments quantize = {
		producer.hidden_offset,
		producer.hidden_codes_offset,
		producer.hidden_exponents_offset,
		producer.hidden_operands_offset,
		producer.output_size,
		0
	};
	expect_fused_name(made, card, layer, F_GATE_UP, launch,
			  KERNEL_EXPERT_GATE_UP_QUANTIZE_STEM);
	expect(producer.arrivals_offset == made->plan->cards[card]
						   .producer_arrivals_offset &&
		       producer.pair_count ==
			       made->rows * MODEL_EXPERTS_PER_ROW &&
		       producer.pairs_per_row == MODEL_EXPERTS_PER_ROW &&
		       producer.output_size == V41_DATA_EXPERT_INNER &&
		       producer.input_size == MODEL_HIDDEN,
	       "the fused producer uses the reserved arrival counters and every target pair");
	SEMANTIC(F_GATE_UP, product);
	SEMANTIC(F_QUANTIZE_PAIRS, quantize);
}

static void expand_join(const struct program_case *made, uint32_t card,
			uint32_t layer, const struct kernel_launch *launch,
			struct kernel_launch *records,
			const struct kernel_launch **by_step)
{
	struct expert_shared_join_arguments join;
	memcpy(&join, launch->argument, sizeof(join));
	const struct trunk_linear_quantized_arguments product = {
		{ join.weights_offset, 0 },
		{ join.output_offset, 0 },
		join.codes_offset,
		join.exponents_offset,
		join.operands_offset,
		{ MODEL_HIDDEN, 0 },
		join.input_size,
		join.row_count
	};
	const struct expert_row_sum_arguments row_sum = {
		join.pair_output_offset, join.row_sums_offset, MODEL_HIDDEN,
		join.pairs_per_row
	};
	const struct hyper_post_arguments post = { join.streams_offset,
						   join.post_offset,
						   join.comb_offset,
						   join.output_offset,
						   join.row_sums_offset,
						   1,
						   0 };
	expect_fused_name(made, card, layer, F_SHARED_DOWN, launch,
			  KERNEL_EXPERT_SHARED_JOIN_STEM);
	expect(join.weights_offset ==
			       made->plan
				       ->layer_tensors[layer]
						      [V41_DATA_SHARED_DOWN]
				       .stored.byte_offset &&
		       join.row_sums_offset ==
			       made->plan->cards[card]
				       .array_offset[V41_DATA_ARRAY_ROW_SUMS] &&
		       join.row_count == made->rows &&
		       join.pairs_per_row == MODEL_EXPERTS_PER_ROW,
	       "the shared join reads its down matrix and retains the exact layer gate's row sums");
	SEMANTIC(F_SHARED_DOWN, product);
	SEMANTIC(F_ROW_SUM, row_sum);
	SEMANTIC(F_POST, post);
}
#undef SEMANTIC

/* A layer's segment against the reference's sequence for its kind (of its part's steps), and its data flow once its
 * last part is seen: the front's launches stay in by_step for the later part's. */
static void expect_layer(const struct program_case *made,
			 const struct segment_record *segment, uint32_t part)
{
    const uint32_t layer = segment->layer;
    const uint32_t kinds =
	    kinds_of(layer) | EVERY |
	    (made->plan->limits.linear_order == LINEAR_ORDER_REFERENCE ?
		     REFERENCE_ORDER :
		     0u);
    const uint32_t first_step = part == LATER_PART ? A_QUANTIZE_NORMED : 0u;
    const uint32_t end_step = part == FRONT_PART ? A_QUANTIZE_NORMED : SEQUENCE_LENGTH;
    const struct kernel_launch *const launches =
	    made->program->cards[segment->card].launches +
	    segment->first_launch;
    static const struct kernel_launch
	    *by_step[SEQUENCE_LENGTH]; /* static: see expect_record */
    static struct kernel_launch records[SEQUENCE_LENGTH];
    char name[NAME_CAPACITY], expected_name[NAME_CAPACITY];
    uint32_t index = 0;
    for (uint32_t step = first_step; step < end_step; step++) {
        by_step[step] = NULL;
    }
    for (uint32_t step = first_step; step < end_step; step++) {
        const struct expected_launch *const expected = &reference_sequence[step];
        if ((expected->kinds & kinds) == 0 || by_step[step] != NULL || step == F_ROW_SUM) {
            continue;
        }
        if (index == segment->launch_count) {
            (void)printf("FAILED: %s: layer %" PRIu32 " ends before %s\n", context_text, layer, expected->function);
            failures++;
            return;
        }
	const struct kernel_launch *const launch = &launches[index++];
	const uint32_t arithmetic_class = made->descriptions[segment->card].arithmetic_class;
        if (step == A_MIX || step == F_MIX) {
            expand_front(made, segment->card, layer, step, launch, records, by_step);
            continue;
        }
	if (step == A_GROUPS &&
	    arithmetic_class == ARITHMETIC_CLASS_SM_120_NATIVE) {
		expand_groups(made, segment->card, layer, launch, records,
			      by_step);
		continue;
	}
	if (step == A_OUTPUT) {
            expand_projection(made, segment->card, layer, launch, records, by_step);
            continue;
        }
	if (step == F_GATE_UP && arithmetic_class == ARITHMETIC_CLASS_SM_120) {
		expand_producer(made, segment->card, layer, launch, records,
				by_step);
		continue;
	}
	if (step == F_SHARED_DOWN) {
            expand_join(made, segment->card, layer, launch, records, by_step);
            continue;
        }
        by_step[step] = launch;
        name_of(segment->card, launch, name);
        expected_name_of(made, segment->card, layer, expected, expected_name);
        if (strcmp(name, expected_name) != 0) {
            (void)printf("FAILED: %s: launch %" PRIu32 " of layer %" PRIu32 " is %s, the reference's sequence has %s\n",
                         context_text, index - 1u, layer, name, expected_name);
            failures++;
            return;
        }
	uint64_t leading[LINEAR_MERGED_MATRICES];
	memcpy(leading, launch->argument, sizeof(leading));
	const struct v41_tensor *const tensors =
		made->plan->layer_tensors[layer];
	expect(expected->tensor == NO_TENSOR || (leading[0] == tensors[expected->tensor].stored.byte_offset &&
                                                 (expected->second_tensor == NO_TENSOR ||
                                                  leading[1] == tensors[expected->second_tensor].stored.byte_offset)),
               "a product reads its own layer's tensor");
        if (expected->naming == ROWS_NAMED) {
		static struct dense_linear_arguments dense;
		memcpy(&dense, launch->argument, sizeof(dense));
		const float scale = expected->tensor == V41_DATA_INDEX_WEIGHTS ?
					    DENSE_INDEX_WEIGHTS_SCALE :
					    1.0f;
		expect(dense.output_type == expected->output_type &&
			       dense.flags == expected->flags &&
			       dense.output_scale == scale &&
			       dense.row_count == made->rows,
		       "a dense product rounds and scales its output as first_token_plan.md 2.3 says");
        }
        expect_record(made, segment->card, layer, name, launch);
    }
    expect(index == segment->launch_count, "a layer runs the reference's sequence and nothing more");
    if (part != FRONT_PART) {
        expect_data_flow(made, segment->card, layer, by_step);
    }
}

/* A segment outside the layers by its functions' names, and what it reads and writes: the step segment's rotation
 * from the card's words; the embedding from embed.weight into its card's streams and carried pre; the head's final
 * collapse with norm.weight, head.weight over the flagged rows, the sampler to the result. */
static void expect_outside_layers(const struct program_case *made,
				  const struct segment_record *segment,
				  uint32_t kind)
{
	const struct v41_card *const card = &made->plan->cards[segment->card];
	const struct kernel_launch *launches =
		made->program->cards[segment->card].launches +
		segment->first_launch;
	const uint64_t *const arrays = card->array_offset;
	const struct v41_tensor *const globals = made->plan->globals;
	char name[NAME_CAPACITY], expected_name[NAME_CAPACITY];
	if (kind == STEP_SEGMENT &&
	    made->plan->limits.row_capacity >= MODEL_WIDE_LEAST_ROWS) {
		struct fill_record clear;
		memcpy(&clear, launches->argument, sizeof(clear));
		name_of(segment->card, launches, name);
		expect(strcmp(name, KERNEL_FILL_WORDS) == 0 &&
			       clear.arena_byte_offset ==
				       card->front_scratch_offset &&
			       clear.arena_byte_offset ==
				       arrays[V41_DATA_ARRAY_ROUTER_PICKS] &&
			       clear.word_count ==
				       V41_DATA_FUSED_SCRATCH_BYTES /
					       FILL_WORD_BYTES &&
			       clear.value == 0 && clear.reserved_zero == 0 &&
			       card->producer_arrivals_offset ==
				       card->front_scratch_offset +
					       SITE_FRONT_SCRATCH_BYTES &&
			       clear.arena_byte_offset +
					       clear.word_count *
						       FILL_WORD_BYTES <=
				       arrays[V41_DATA_ARRAY_JOBS],
		       "each thin visit clears only its borrowed wide-routing scratch before the rotation");
		launches++;
	}
    for (uint32_t launch = 0; launch < outside_launches[kind]; launch++) {
        name_of(segment->card, &launches[launch], name);
        if (kind == HEAD_SEGMENT && launch == HEAD_DENSE) {
		(void)snprintf(expected_name, sizeof(expected_name),
			       "%s%" PRIu32, outside_sequences[kind][launch],
			       made->rows == 1 ? 1u : LINEAR_MAXIMUM_ROWS);
	} else {
            (void)snprintf(expected_name, sizeof(expected_name), "%s", outside_sequences[kind][launch]);
        }
        expect(strcmp(name, expected_name) == 0, "the rotation; the embedding; the collapse, head, sampler");
    }
    static struct attention_rotation_arguments rotation;
    static struct hyper_embed_arguments embed;
    static struct hyper_finish_arguments final;
    static struct dense_linear_arguments head;
    static struct sampler_arguments sampler;
    if (kind == STEP_SEGMENT) {
        memcpy(&rotation, launches[0].argument, sizeof(rotation));
	expect(card->layers != 0 &&
		       rotation.step_offset == arrays[V41_DATA_ARRAY_STEP] &&
		       rotation.frequencies_offset ==
			       arrays[V41_DATA_ARRAY_FREQUENCIES] &&
		       rotation.rotation_offset ==
			       arrays[V41_DATA_ARRAY_ROTATION],
	       "a card that runs layers forms its rotation step from its words");
    } else if (kind == EMBEDDING_SEGMENT) {
        memcpy(&embed, launches[0].argument, sizeof(embed));
	expect(segment->card == made->plan->embedding_card &&
		       embed.step_offset == arrays[V41_DATA_ARRAY_STEP] &&
		       embed.embedding_offset ==
			       globals[V41_DATA_EMBEDDING].stored.byte_offset &&
		       embed.streams_offset == arrays[V41_DATA_ARRAY_STREAMS] &&
		       embed.pre_offset == arrays[V41_DATA_ARRAY_CARRIED_PRE] &&
		       embed.vocabulary_size == MODEL_VOCABULARY_VALUE,
	       "the embedding goes into its card's streams and carried pre");
    } else {
        memcpy(&final, launches[0].argument, sizeof(final));
        memcpy(&head, launches[HEAD_DENSE].argument, sizeof(head));
        memcpy(&sampler, launches[HEAD_LAUNCHES - 1u].argument, sizeof(sampler));
	expect(segment->card == made->plan->result_card &&
		       final.streams_offset == arrays[V41_DATA_ARRAY_STREAMS] &&
		       final.pre_offset == arrays[V41_DATA_ARRAY_CARRIED_PRE] &&
		       final.weight_offset ==
			       globals[V41_DATA_NORM].stored.byte_offset &&
		       final.hidden_offset == head.rows_offset &&
		       head.weights_offset ==
			       globals[V41_DATA_HEAD].stored.byte_offset &&
		       head.output_type == DENSE_FLOAT32 &&
		       head.flags == DENSE_STEP_ROWS &&
		       head.step_offset == arrays[V41_DATA_ARRAY_STEP] &&
		       head.output_offset == sampler.logits_offset &&
		       sampler.step_offset == arrays[V41_DATA_ARRAY_STEP] &&
		       sampler.result_offset == arrays[V41_DATA_ARRAY_RESULT] &&
		       sampler.session_offset ==
			       arrays[V41_DATA_ARRAY_SESSIONS] &&
		       sampler.session_capacity == SESSIONS &&
		       sampler.vocabulary_size == MODEL_VOCABULARY_VALUE,
	       "the head's input is the final norm, its float32 logits of the flagged rows are what the sampler reads");
    }
}

/* The stages of a placement: a card; its step segment (0 or 1), the embedding (0 or 1), its layers, the head. */
struct expected_stage {
    uint32_t card, has_step, embeds, first_layer, layer_count, has_head;
};

/* The program's segments are the stages', in order; each layer against the reference; every card's launches belong
 * to its segments in order; each segment runs through launch_run and the backend logs it in that order. */
static void expect_program(const struct program_case *made,
			   struct device_card *const *cards,
			   const struct expected_stage *stages,
			   uint32_t stage_count)
{
	const struct execution_program *const program = made->program;
	uint32_t segment_index = 0, next_launch[CARDS] = { 0 };
	uint64_t work_index = test_host_memory_backend_work_count();
	for (uint32_t stage = 0; stage < stage_count; stage++) {
		const struct expected_stage *const expected = &stages[stage];
		const uint32_t before_layers =
			expected->has_step + expected->embeds;
		/* layer 20's two segments: its front, then the rest */
		const bool splits =
			expected->first_layer <= MODEL_DECODER_FIRST_LAYER &&
			MODEL_DECODER_FIRST_LAYER - expected->first_layer <
				expected->layer_count;
		const uint32_t split_at =
			splits ? MODEL_DECODER_FIRST_LAYER -
					 expected->first_layer :
				 UINT32_MAX;
		const uint32_t layer_segments =
			expected->layer_count + (splits ? 1u : 0u);
		const uint32_t segment_count =
			before_layers + layer_segments + expected->has_head;
		for (uint32_t index = 0; index < segment_count &&
					 segment_index < program->segment_count;
		     index++) {
			const struct segment_record *const segment =
				&program->segments[segment_index++];
			if (segment->card >= CARDS ||
			    segment->first_launch >
				    program->cards[segment->card].launch_count ||
			    segment->launch_count >
				    program->cards[segment->card].launch_count -
					    segment->first_launch) {
				expect(false,
				       "a segment names a card of the program and launches it has");
				return;
			}
			const uint32_t kind =
				index < expected->has_step ? STEP_SEGMENT :
				index < before_layers	   ? EMBEDDING_SEGMENT :
				index < before_layers + layer_segments ?
							LAYER_SEGMENT :
							HEAD_SEGMENT;
			const uint32_t slot = index - before_layers;
			const uint32_t layer =
				expected->first_layer + slot -
				(slot > split_at && slot != UINT32_MAX ? 1u :
									 0u);
			const uint32_t part = !splits	       ? WHOLE_LAYER :
					      slot == split_at ? FRONT_PART :
					      slot == split_at + 1u ?
								 LATER_PART :
								 WHOLE_LAYER;
			const uint32_t outside_count =
				outside_launches[kind] +
				(kind == STEP_SEGMENT &&
						 made->plan->limits.row_capacity >=
							 MODEL_WIDE_LEAST_ROWS ?
					 1u :
					 0u);
			if (kind == LAYER_SEGMENT && part == FRONT_PART) {
				expect(program->front_segment_count ==
						       segment_index &&
					       program->cone_rows ==
						       MODEL_CONE_ROWS,
				       "the front ends behind layer 20's front, a cone of the model's rows");
			}
			expect(segment->card == expected->card &&
				       segment->stage == stage &&
				       segment->pause == 0 &&
				       segment->layer ==
					       (kind == LAYER_SEGMENT ?
							layer :
							SEGMENT_NO_LAYER) &&
				       segment->first_launch ==
					       next_launch[segment->card] &&
				       (kind == LAYER_SEGMENT ||
					segment->launch_count == outside_count),
			       "the segments are each stage's step segment, the embedding, its layers and the head, in order");
			if (kind == LAYER_SEGMENT && segment->layer == layer) {
				expect_layer(made, segment, part);
			} else if (segment->launch_count == outside_count) {
				expect_outside_layers(made, segment, kind);
			}
			const struct kernel_launch *const launches =
				program->cards[segment->card].launches;
			next_launch[segment->card] =
				segment->first_launch + segment->launch_count;
			expect_status(test_host_memory_backend.launch_run(
					      cards[segment->card],
					      launches + segment->first_launch,
					      segment->launch_count),
				      STATUS_OK,
				      "a segment's launches pass launch_run");
			for (uint32_t launch = 0;
			     launch < segment->launch_count; launch++) {
				static struct test_host_memory_backend_work work;
				expect(test_host_memory_backend_work_at(
					       work_index++, &work) &&
					       work.kind ==
						       TEST_HOST_MEMORY_BACKEND_LAUNCH &&
					       work.card_index ==
						       segment->card &&
					       memcmp(&work.launch,
						      &launches[segment->first_launch +
								launch],
						      sizeof(work.launch)) == 0,
				       "the backend runs the launches in the program's order");
			}
		}
    }
    expect(segment_index == program->segment_count && work_index == test_host_memory_backend_work_count(),
           "no segment and no work beyond the stages'");
    for (uint32_t card = 0; card < CARDS; card++) {
        expect(program->cards[card].launch_count == next_launch[card], "every launch of a card is in a segment");
    }
}

/* The program of rows rows against that of 1 row of its plan: the same launches, each word of a launch but its kernel
 * the same or rows times it (a grid's rows, a record's rows or pairs), one at least so: no launch covers fewer rows.
 * The kernels are checked by name above; the attention core's reference launches, whose blocks' heads grow with the
 * rows, are not held to it. */
static void expect_rows(const struct execution_program *one,
			const struct execution_program *program, uint32_t rows)
{
    static uint32_t one_words[LAUNCH_WORDS], words[LAUNCH_WORDS];
    for (uint32_t card = 0; card < CARDS; card++) {
        const uint32_t count = program->cards[card].launch_count;
        expect(one->cards[card].launch_count == count, "the programs of 1 and of 8 rows hold as many launches");
        for (uint32_t index = 0; index < count && index < one->cards[card].launch_count; index++) {
            uint32_t scaled = 0, otherwise = 0;
            char name[NAME_CAPACITY];
            name_of(card, &program->cards[card].launches[index], name);
	    if (strcmp(name, KERNEL_ATTENTION_CORE_SCORES) == 0 ||
		strcmp(name, KERNEL_ATTENTION_CORE_VALUES) == 0) {
		    continue; /* its blocks' heads follow the rows (attention_core_records.h) */
	    }
	    memcpy(one_words, &one->cards[card].launches[index], sizeof(one_words));
            memcpy(words, &program->cards[card].launches[index], sizeof(words));
	    const bool site_front = strstr(name, "site_front_") != NULL;
	    const bool ffn_front = strstr(name, "ffn_front_") != NULL;
	    if (site_front || ffn_front) {
		    const uint32_t kind = ffn_front ? SITE_FRONT_KIND_FFN :
						      SITE_FRONT_KIND_SITE;
		    expect(program->cards[card].launches[index].grid[0] ==
					   SITE_FRONT_BLOCKS(kind, rows) &&
				   one->cards[card].launches[index].grid[0] ==
					   SITE_FRONT_BLOCKS(kind, 1u),
			   "front grids have their specified mix, score and row blocks");
		    words[offsetof(struct kernel_launch, grid) /
			  sizeof(uint32_t)] = 0;
		    one_words[offsetof(struct kernel_launch, grid) /
			      sizeof(uint32_t)] = 0;
	    }
	    if (strstr(name, KERNEL_EXPERT_GATE_UP_QUANTIZE_STEM) != NULL) {
		    words[offsetof(struct kernel_launch, grid) /
				  sizeof(uint32_t) +
			  1u]--;
		    one_words[offsetof(struct kernel_launch, grid) /
				      sizeof(uint32_t) +
			      1u]--;
	    }
	    for (uint32_t word = KERNEL_WORDS; word < LAUNCH_WORDS; word++) {
                const bool is_same = words[word] == one_words[word];
                const bool is_scaled = !is_same && words[word] == (uint64_t)rows * one_words[word];
                scaled += is_scaled ? 1u : 0u;
                otherwise += !is_same && !is_scaled ? 1u : 0u;
            }
            if (scaled == 0 || otherwise != 0) {
                (void)printf("FAILED: %s: launch %" PRIu32 " of card %" PRIu32 " is not that of 1 row for %" PRIu32
                             " rows: %" PRIu32 " words %" PRIu32 " times, %" PRIu32 " otherwise\n", context_text, index,
                             card, rows, scaled, rows, otherwise);
                failures++;
            }
        }
    }
}

/* What each card's fields of the program say: its arena and the areas the step driver and the harness move. */
static void expect_cards(const struct program_case *made)
{
	const struct execution_program *const program = made->program;
	expect(program->card_count == CARDS &&
		       program->row_count == made->rows &&
		       program->result_card == made->plan->result_card &&
		       program->draft_segment_first == program->segment_count &&
		       memcmp(&program->limits, &made->plan->limits,
			      sizeof(program->limits)) == 0,
	       "the program's cards, rows, limits and the card that samples");
	for (uint32_t card_index = 0; card_index < CARDS; card_index++) {
		const struct v41_card *const card =
			&made->plan->cards[card_index];
		const struct program_card_record *const fields =
			&program->cards[card_index];
		const uint64_t *const arrays = card->array_offset;
		const bool runs_layers = card->layers != 0;
		expect(fields->arena_byte_count == card->arena_byte_count &&
			       fields->step_offset ==
				       arrays[V41_DATA_ARRAY_STEP] &&
			       fields->session_offset ==
				       arrays[V41_DATA_ARRAY_SESSIONS] &&
			       fields->result_offset ==
				       arrays[V41_DATA_ARRAY_RESULT] &&
			       fields->logits_offset ==
				       arrays[V41_DATA_ARRAY_LOGITS] &&
			       (card_index == made->plan->result_card) ==
				       (fields->result_offset !=
					PROGRAM_NOWHERE) &&
			       fields->streams_offset ==
				       (runs_layers ?
						arrays[V41_DATA_ARRAY_STREAMS] :
						PROGRAM_NOWHERE) &&
			       fields->carried_pre_offset ==
				       (runs_layers ?
						arrays[V41_DATA_ARRAY_CARRIED_PRE] :
						PROGRAM_NOWHERE) &&
			       fields->engram_rows_offset ==
				       arrays[V41_DATA_ARRAY_ENGRAM_ROWS] &&
			       fields->engram_tables == card->engram_tables &&
			       fields->round_offset == PROGRAM_NOWHERE,
		       "a card's areas are the plan's");
	}
}

/* ---- the placements ---- */

/* A hand-off as a placement must have it: behind the segment of a layer (or of the embedding), of an array of a
 * layer, between two cards. */
enum copied { STREAMS, CARRIED_PRE, SELECTION, POOL, KEYS, CANDIDATES };
struct expected_hand_off {
    uint32_t after_layer, copied, of_layer, from, to;
};
static const struct expected_hand_off hand_offs_of_1_3[] = {
    {7, STREAMS, 0, 0, 1}, {7, CARRIED_PRE, 0, 0, 1},                   /* 2.8: every boundary */
    {14, SELECTION, 14, 1, 2}, {14, POOL, 14, 1, 2},                    /* 1.3: layer 14's group and pool on C */
    {15, STREAMS, 0, 1, 2}, {15, CARRIED_PRE, 0, 1, 2},
    {20, POOL, 20, 2, 3}, {20, KEYS, 20, 2, 3}, {20, POOL, 20, 2, 4}, {20, KEYS, 20, 2, 4},  /* pool 20: D, the PRO */
    {20, CANDIDATES, 20, 2, 3}, {20, CANDIDATES, 20, 2, 4},             /* its candidates on both */
    {23, STREAMS, 0, 2, 3}, {23, CARRIED_PRE, 0, 2, 3},
    {31, STREAMS, 0, 3, 4}, {31, CARRIED_PRE, 0, 3, 4},
};
static const struct expected_hand_off hand_offs_moved[] = {         /* the embedding on card 4, the head on card 0 */
    {EMBEDDING, STREAMS, 0, 4, 0}, {EMBEDDING, CARRIED_PRE, 0, 4, 0},
    {7, STREAMS, 0, 0, 1}, {7, CARRIED_PRE, 0, 0, 1}, {14, SELECTION, 14, 1, 2}, {14, POOL, 14, 1, 2},
    {15, STREAMS, 0, 1, 2}, {15, CARRIED_PRE, 0, 1, 2},
    {20, POOL, 20, 2, 3}, {20, KEYS, 20, 2, 3}, {20, POOL, 20, 2, 4}, {20, KEYS, 20, 2, 4},  /* behind its front */
    {20, CANDIDATES, 20, 2, 3}, {20, CANDIDATES, 20, 2, 4},
    {23, STREAMS, 0, 2, 3}, {23, CARRIED_PRE, 0, 2, 3}, {31, STREAMS, 0, 3, 4}, {31, CARRIED_PRE, 0, 3, 4},
    {39, STREAMS, 0, 4, 0}, {39, CARRIED_PRE, 0, 4, 0},
};
static const struct expected_hand_off hand_offs_revisited[] = {     /* layer 14 on card 0, with layer 1 */
    {7, STREAMS, 0, 0, 1}, {7, CARRIED_PRE, 0, 0, 1}, {13, STREAMS, 0, 1, 0}, {13, CARRIED_PRE, 0, 1, 0},
    {14, SELECTION, 14, 0, 1}, {14, POOL, 14, 0, 1}, {14, SELECTION, 14, 0, 2}, {14, POOL, 14, 0, 2},
    {14, STREAMS, 0, 0, 1}, {14, CARRIED_PRE, 0, 0, 1}, {15, STREAMS, 0, 1, 2}, {15, CARRIED_PRE, 0, 1, 2},
    {20, POOL, 20, 2, 3}, {20, KEYS, 20, 2, 3}, {20, POOL, 20, 2, 4}, {20, KEYS, 20, 2, 4},  /* behind its front */
    {20, CANDIDATES, 20, 2, 3}, {20, CANDIDATES, 20, 2, 4},
    {23, STREAMS, 0, 2, 3}, {23, CARRIED_PRE, 0, 2, 3}, {31, STREAMS, 0, 3, 4}, {31, CARRIED_PRE, 0, 3, 4},
};
static const struct expected_hand_off hand_offs_bare[] = {          /* layers 32 to 39 on card 3: card 4 runs none */
    {EMBEDDING, STREAMS, 0, 4, 0}, {EMBEDDING, CARRIED_PRE, 0, 4, 0},
    {7, STREAMS, 0, 0, 1}, {7, CARRIED_PRE, 0, 0, 1}, {14, SELECTION, 14, 1, 2}, {14, POOL, 14, 1, 2},
    {15, STREAMS, 0, 1, 2}, {15, CARRIED_PRE, 0, 1, 2}, {20, POOL, 20, 2, 3}, {20, KEYS, 20, 2, 3},
    {20, CANDIDATES, 20, 2, 3},
    {23, STREAMS, 0, 2, 3}, {23, CARRIED_PRE, 0, 2, 3}, {39, STREAMS, 0, 3, 4}, {39, CARRIED_PRE, 0, 3, 4},
};
static const struct expected_stage stages_of_1_3[] = {
    {0, 1, 1, 0, LAYERS_PER_CARD, 0}, {1, 1, 0, 8, LAYERS_PER_CARD, 0}, {2, 1, 0, 16, LAYERS_PER_CARD, 0},
    {3, 1, 0, 24, LAYERS_PER_CARD, 0}, {4, 1, 0, 32, LAYERS_PER_CARD, 1}};
static const struct expected_stage stages_moved[] = {     /* card 4's first visit holds no layer: no step segment */
    {4, 0, 1, 0, 0, 0}, {0, 1, 0, 0, LAYERS_PER_CARD, 0}, {1, 1, 0, 8, LAYERS_PER_CARD, 0},
    {2, 1, 0, 16, LAYERS_PER_CARD, 0}, {3, 1, 0, 24, LAYERS_PER_CARD, 0}, {4, 1, 0, 32, LAYERS_PER_CARD, 0},
    {0, 0, 0, 0, 0, 1}};
static const struct expected_stage stages_revisited[] = {  /* a visit again: its own step segment */
    {0, 1, 1, 0, LAYERS_PER_CARD, 0}, {1, 1, 0, 8, 6, 0}, {0, 1, 0, 14, 1, 0}, {1, 1, 0, 15, 1, 0},
    {2, 1, 0, 16, LAYERS_PER_CARD, 0}, {3, 1, 0, 24, LAYERS_PER_CARD, 0}, {4, 1, 0, 32, LAYERS_PER_CARD, 1}};
static const struct expected_stage stages_bare[] = {
    {4, 0, 1, 0, 0, 0}, {0, 1, 0, 0, LAYERS_PER_CARD, 0}, {1, 1, 0, 8, LAYERS_PER_CARD, 0},
    {2, 1, 0, 16, LAYERS_PER_CARD, 0}, {3, 1, 0, 24, 2u * LAYERS_PER_CARD, 0}, {4, 0, 0, 0, 0, 1}};
/* 1.3, then layers (and their experts), the embedding and the head moved to other cards. */
static const struct placement_case {
    const char *what;
    const struct expected_stage *stages;
    const struct expected_hand_off *hand_offs;
    uint32_t stage_count, hand_off_count;
    uint32_t moved_first, moved_count, layer_card, embedding_card, head_card;
} placements[] = {
    {"1.3", stages_of_1_3, hand_offs_of_1_3, COUNT_OF(stages_of_1_3), COUNT_OF(hand_offs_of_1_3), 0, 0, 0, 0,
     PRO_CARD},
    {"the embedding on the last card, the head on the first", stages_moved, hand_offs_moved, COUNT_OF(stages_moved),
     COUNT_OF(hand_offs_moved), 0, 0, 0, PRO_CARD, 0},
    {"layer 14 on the card of layer 1", stages_revisited, hand_offs_revisited, COUNT_OF(stages_revisited),
     COUNT_OF(hand_offs_revisited), REVISITING_LAYER, 1, 0, 0, PRO_CARD},
    {"the embedding and the head on a card that runs no layer", stages_bare, hand_offs_bare, COUNT_OF(stages_bare),
     COUNT_OF(hand_offs_bare), LAST_CARD_LAYER, LAYERS_PER_CARD, LATE_CARD, PRO_CARD, PRO_CARD},
};

/* The segment of a layer, or of the embedding: the one that runs embed. */
static uint32_t segment_of_layer(const struct execution_program *program,
				 uint32_t layer)
{
    char name[NAME_CAPACITY];
    uint32_t found = UINT32_MAX;
    for (uint32_t index = 0; index < program->segment_count; index++) {
	    const struct segment_record *const segment =
		    &program->segments[index];
	    name_of(segment->card,
		    &program->cards[segment->card]
			     .launches[segment->first_launch],
		    name);
	    const bool is_embedding = segment->layer == SEGMENT_NO_LAYER &&
				      strcmp(name, KERNEL_HYPER_EMBED) == 0;
	    found = (layer == EMBEDDING ? is_embedding :
					  segment->layer == layer) ?
			    index :
			    found;
    }
    return found;
}

static void expect_hand_off(const struct program_case *made,
			    const struct hand_off_record *hand_off,
			    const struct expected_hand_off *expected)
{
	const struct model_plan_record *const plan = made->plan;
	const struct v41_card *const from = &plan->cards[expected->from],
				     *const to = &plan->cards[expected->to];
	const uint32_t layer = expected->of_layer;
	const uint32_t ratio = (kinds_of(layer) & POOLED) != 0 ?
				       MODEL_POOLED_RATIO :
				       MODEL_DIRECT_RATIO;
	const struct hand_off_record wanted[] = {
		[STREAMS] = { from->array_offset[V41_DATA_ARRAY_STREAMS],
			      to->array_offset[V41_DATA_ARRAY_STREAMS], 0,
			      MODEL_STREAMS_ROW_BYTES, 0, 0, 0, 0, 0 },
		[CARRIED_PRE] = { from->array_offset[V41_DATA_ARRAY_CARRIED_PRE],
				  to->array_offset[V41_DATA_ARRAY_CARRIED_PRE],
				  0, MODEL_SITE_ROW_BYTES, 0, 0, 0, 0, 0 },
		[SELECTION] = { from->selection_offset[layer],
				to->selection_offset[layer], 0,
				MODEL_SELECTION_ROW_BYTES, 0, 0, 0, 0, 0 },
		[POOL] = { from->pool_offset[layer], to->pool_offset[layer],
			   plan->pool_session_bytes[layer],
			   MODEL_STATE_ROW_BYTES, ratio, 0, 0, 0, 0 },
		[KEYS] = { from->key_tiles_offset[layer],
			   to->key_tiles_offset[layer],
			   plan->key_session_bytes[layer], MODEL_KEY_BYTES,
			   ratio, 0, 0, 0, 0 },
		[CANDIDATES] = { from->array_offset[V41_DATA_ARRAY_CANDIDATES],
				 to->array_offset[V41_DATA_ARRAY_CANDIDATES], 0,
				 MODEL_CANDIDATES_ROW_BYTES, 0, 0, 0, 0, 0 }
	};
	struct hand_off_record expected_hand_off = wanted[expected->copied];
	expected_hand_off.from_card = expected->from;
	expected_hand_off.to_card = expected->to;
	/* behind its last segment; the pool's and the keys' of layer 20 behind its front, the one before */
	expected_hand_off.after_segment =
		segment_of_layer(made->program, expected->after_layer) -
		(expected->after_layer == MODEL_DECODER_FIRST_LAYER &&
				 (expected->copied == POOL ||
				  expected->copied == KEYS) ?
			 1u :
			 0u);
	if (memcmp(hand_off, &expected_hand_off, sizeof(expected_hand_off)) !=
		    0 ||
	    expected_hand_off.source_offset == PROGRAM_NOWHERE ||
	    expected_hand_off.destination_offset == PROGRAM_NOWHERE) {
		(void)printf(
			"FAILED: %s: a hand-off of %" PRIu32
			" bytes from card %" PRIu32 " to %" PRIu32
			" behind segment %" PRIu32 ": expected %" PRIu32
			" bytes from card %" PRIu32 " to %" PRIu32
			" behind segment %" PRIu32
			", of the arrays the placement puts on two cards (2.8)\n",
			context_text, hand_off->byte_count, hand_off->from_card,
			hand_off->to_card, hand_off->after_segment,
			expected_hand_off.byte_count,
			expected_hand_off.from_card, expected_hand_off.to_card,
			expected_hand_off.after_segment);
		failures++;
	}
}

/* ---- the cases ---- */

static void cards_close(struct device_card **cards)
{
    for (uint32_t card = 0; card < CARDS; card++) {
        test_host_memory_backend.card_close(cards[card], 0);
        cards[card] = NULL;
    }
}

/* The backend afresh, its work log empty, with the five cards open: 0 to 3 of class 80, the last of class 120. */
static void cards_open(struct device_card **cards,
		       struct card_description_record *descriptions)
{
    cards_close(cards);
    test_host_memory_backend_reset(CARDS, HOST_CARD_BYTES);
    for (uint32_t card = 0; card < CARDS; card++) {
	    test_host_memory_backend_set_arithmetic_class(
		    card, card == PRO_CARD ? ARITHMETIC_CLASS_SM_120 :
					     ARITHMETIC_CLASS_SM_80);
	    expect_status(test_host_memory_backend.card_describe(
				  card, &descriptions[card]),
			  STATUS_OK, "a card is described");
	    expect_status(test_host_memory_backend.card_open(
				  card, HOST_CARD_BYTES, &cards[card]),
			  STATUS_OK, "a card is opened");
    }
}

static struct model_plan_record *plan_of(const struct pack_record *pack,
					 const struct placement_case *placement,
					 enum linear_order_code order,
					 uint32_t row_capacity)
{
    static uint32_t card_of_unit[UNIT_COUNT];
    struct load_layout *layout = NULL;
    struct model_plan_record *plan = NULL;
    const struct model_limits limits = { CONTEXT,
					 row_capacity,
					 SESSIONS,
					 STATE_WIDTH_16,
					 LINEAR_PROFILE_PRECISION,
					 order,
					 CMP_ORDER_REFERENCE,
					 0,
					 0,
					 0 };
    test_model_v41_five_cards(card_of_unit, 0, placement->embedding_card);
    card_of_unit[TEST_MODEL_V41_HEAD_UNIT] = placement->head_card;
    for (uint32_t layer = placement->moved_first; layer < placement->moved_first + placement->moved_count; layer++) {
        for (uint32_t unit = 0; unit < TEST_MODEL_V41_UNITS_PER_LAYER; unit++) {
            card_of_unit[TEST_MODEL_V41_LAYER_UNIT(layer) + unit] = placement->layer_card;
        }
    }
    expect_status(load_layout_create(pack, card_of_unit, UNIT_COUNT, CARDS,
				     &layout),
		  STATUS_OK, "the placement is laid out");
    expect_status(layout == NULL ?
			  STATUS_INVALID_ARGUMENT :
			  v41_plan_create(pack, layout, &limits, &plan),
		  STATUS_OK, "the plan is made");
    load_layout_destroy(layout);
    return plan;
}

/* Each placement in both orders, at 1 and at 8 rows, on cards opened afresh (the work log holds their launches); the
 * program of 8 rows against that of 1. */
static void test_programs(const struct model_operations *model,
			  const struct pack_record *pack,
			  struct device_card **cards,
			  struct card_description_record *descriptions)
{
    static const uint32_t row_counts[] = {1u, ROWS};
    for (uint32_t index = 0; index < COUNT_OF(placements) * ORDERS; index++) {
        const struct placement_case *const placement = &placements[index / ORDERS];
	const enum linear_order_code order =
		(enum linear_order_code)(index % ORDERS);
	struct model_plan_record *const plan =
		plan_of(pack, placement, order, ROWS);
	struct execution_program *programs[COUNT_OF(row_counts)] = { NULL };
	context_text = placement->what;
        cards_open(cards, descriptions);
        for (uint32_t row_index = 0; plan != NULL && row_index < COUNT_OF(row_counts); row_index++) {
		expect_status(model->program_prepare(
				      plan, &test_host_memory_backend, cards,
				      descriptions, PROGRAM_THIN,
				      row_counts[row_index],
				      &programs[row_index]),
			      STATUS_OK, "the program is made");
		const struct execution_program *const program =
			programs[row_index];
		if (program == NULL) {
			continue;
		}
            const struct program_case made = {plan, program, descriptions, row_counts[row_index]};
            expect_cards(&made);
            expect_program(&made, cards, placement->stages, placement->stage_count);
            expect(program->hand_off_count == placement->hand_off_count, "the placement's hand-offs, no more");
            for (uint32_t hand_off = 0; hand_off < placement->hand_off_count && hand_off < program->hand_off_count;
                 hand_off++) {
                expect_hand_off(&made, &program->hand_offs[hand_off], &placement->hand_offs[hand_off]);
            }
        }
        if (programs[0] != NULL && programs[1] != NULL) {
            expect_rows(programs[0], programs[1], ROWS);
        }
        for (uint32_t row_index = 0; row_index < COUNT_OF(row_counts); row_index++) {
            model->program_destroy(programs[row_index]);
        }
        model->plan_destroy(plan);
    }
}

/* ---- chunked programs ---- */

/* The first segment of a card (its step segment where it runs a layer), or of a layer: NO_SEGMENT for none. */
static uint32_t first_segment_of(const struct execution_program *program,
				 uint32_t card, uint32_t layer)
{
    for (uint32_t index = 0; index < program->segment_count; index++) {
	    const struct segment_record *const segment =
		    &program->segments[index];
	    if (layer == SEGMENT_NO_LAYER ? segment->card == card :
					    segment->layer == layer) {
		    return index;
	    }
    }
    return NO_SEGMENT;
}

/* A chunk's layer (model_v41_program.c's chunk_walk) in order: a step of the sequence taken over each slab of 8 rows in
 * turn (OVER_SLABS), over each span of SPAN_SLABS slabs (OVER_SPANS: in a run of slabs, ahead of the span's first
 * slab's), over each piece of at most PIECE_ROWS rows (OVER_PIECES: its launches over the chunk) or once over the
 * chunk (ONCE); a run of entries of one scope repeats for each slab or piece; NEW_RUN starts a run of the same scope;
 * layer 20's front ends at FRONT_END. NOT_LAYER_20's steps are every other layer's, LAYER_20_REST's layer 20's alone,
 * behind its front; a step the layer's kinds lack is not taken. */
enum chunk_scope { OVER_SLABS, OVER_SPANS, OVER_PIECES, ONCE, NEW_RUN, FRONT_END };
enum chunk_layers { ALL_LAYERS, NOT_LAYER_20, LAYER_20_REST };
static const struct chunk_entry {
    uint32_t step, scope, layers;
} chunk_layout[] = {
    {E_VALUES, OVER_PIECES, ALL_LAYERS}, {E_QUANTIZE, OVER_PIECES, ALL_LAYERS}, {E_PRODUCT, OVER_PIECES, ALL_LAYERS},
    {E_GATE, OVER_PIECES, ALL_LAYERS}, {A_MIX, OVER_PIECES, ALL_LAYERS}, {A_FINISH, OVER_PIECES, ALL_LAYERS},
    {A_QUANTIZE, OVER_PIECES, ALL_LAYERS}, {A_DOWN, OVER_PIECES, ALL_LAYERS}, {A_POOLED_KV, OVER_PIECES, ALL_LAYERS},
    {A_DIRECT_KV, OVER_PIECES, ALL_LAYERS}, {A_POOLED_SCORE, OVER_PIECES, ALL_LAYERS},
    {A_QUERY_NORM, OVER_PIECES, NOT_LAYER_20}, {A_QUANTIZE_NORMED, OVER_PIECES, NOT_LAYER_20},
    {A_UP, OVER_PIECES, NOT_LAYER_20}, {A_INDEX_UP, OVER_PIECES, NOT_LAYER_20},
    {A_INDEX_WEIGHTS, OVER_PIECES, NOT_LAYER_20}, {A_QUERIES, OVER_SPANS, NOT_LAYER_20},
    {A_PREPARE, OVER_SLABS, ALL_LAYERS}, {A_COMPRESSOR, OVER_SLABS, ALL_LAYERS},
    {A_KEY_PRODUCT, OVER_SLABS, ALL_LAYERS}, {A_KEY, OVER_SLABS, ALL_LAYERS}, {A_SCAN, OVER_SLABS, NOT_LAYER_20},
    {A_SELECT, OVER_SLABS, NOT_LAYER_20}, {A_CANDIDATES, OVER_SLABS, NOT_LAYER_20},
    {A_CORE_SCORES, OVER_SLABS, NOT_LAYER_20}, {A_CORE, OVER_SLABS, NOT_LAYER_20},
    {SEQUENCE_LENGTH, FRONT_END, ALL_LAYERS},
    {A_QUANTIZE_NORMED, OVER_PIECES, LAYER_20_REST}, {A_UP, OVER_PIECES, LAYER_20_REST},
    {A_INDEX_UP, OVER_PIECES, LAYER_20_REST}, {A_INDEX_WEIGHTS, OVER_PIECES, LAYER_20_REST},
    {A_QUERIES, OVER_SPANS, LAYER_20_REST}, {A_SCAN, OVER_SLABS, LAYER_20_REST}, {A_SELECT, OVER_SLABS, LAYER_20_REST},
    {A_CANDIDATES, OVER_SLABS, LAYER_20_REST}, {A_CORE_SCORES, OVER_SLABS, LAYER_20_REST},
    {A_CORE, OVER_SLABS, LAYER_20_REST},
    {A_GROUPS, OVER_PIECES, ALL_LAYERS}, {A_QUANTIZE_GROUPS, OVER_PIECES, ALL_LAYERS},
    {A_OUTPUT, OVER_PIECES, ALL_LAYERS}, {A_POST, OVER_PIECES, ALL_LAYERS}, {SEQUENCE_LENGTH, NEW_RUN, ALL_LAYERS},
    {F_MIX, OVER_PIECES, ALL_LAYERS}, {F_FINISH, OVER_PIECES, ALL_LAYERS}, {F_ROUTER, OVER_PIECES, ALL_LAYERS},
    {W_SELECT, ONCE, ALL_LAYERS}, {W_COUNT, ONCE, ALL_LAYERS}, {W_PLACE, ONCE, ALL_LAYERS}, {W_JOBS, ONCE, ALL_LAYERS},
    {F_QUANTIZE, ONCE, ALL_LAYERS}, {F_GATE_UP, ONCE, ALL_LAYERS}, {F_QUANTIZE_PAIRS, ONCE, ALL_LAYERS},
    {F_DOWN, ONCE, ALL_LAYERS}, {F_ROW_SUM, ONCE, ALL_LAYERS}, {F_SHARED_GATE_UP, OVER_PIECES, ALL_LAYERS},
    {F_SHARED_HIDDEN, OVER_PIECES, ALL_LAYERS}, {F_QUANTIZE_SHARED, OVER_PIECES, ALL_LAYERS},
    {F_SHARED_DOWN, OVER_PIECES, ALL_LAYERS}, {F_POST, OVER_PIECES, ALL_LAYERS},
};

/* A launch of a chunk's layer: its step, the first of its rows and their count, whether a slab's (through its view). */
struct mapped_launch {
    uint32_t step, first_row, rows, is_slab;
};
static struct mapped_launch layer_mapped[MOST_LAYER_LAUNCHES], control_mapped[MOST_LAYER_LAUNCHES];
static struct kernel_launch control_launches[MOST_LAYER_LAUNCHES];
/* What a launch of the layer writes that a flow reads: the bytes of a field, of its rows, row_bytes a row. */
static struct written_range {
    uint64_t low, high, row_bytes;
    uint32_t field_at, reserved_zero;
} layer_written[MOST_LAYER_LAUNCHES][MOST_WRITES];
static uint32_t layer_written_counts[MOST_LAYER_LAUNCHES];
_Static_assert(SEQUENCE_LENGTH <= sizeof(uint64_t) * CHAR_BIT, "a bit for each step of the sequence");

/* The two quantized records have the same byte count but different layouts; the prepared module identifies them. */
static bool is_carried_trunk(uint32_t card, uint32_t step,
			     const struct kernel_launch *launch)
{
    struct test_host_memory_backend_kernel kernel;
    return reference_sequence[step].naming ==
		   FAMILY + LINEAR_FAMILY_TRUNK_QUANTIZED &&
	   test_host_memory_backend_kernel_of(card, launch->kernel, &kernel) &&
	   strcmp(kernel.kernel_file_name, KERNEL_FILE_TRUNK_RELAXED) == 0;
}

static bool is_wide_expert(uint32_t card, uint32_t step,
			   const struct kernel_launch *launch)
{
    struct test_host_memory_backend_kernel kernel;
    return (step == F_GATE_UP || step == F_DOWN) &&
	   test_host_memory_backend_kernel_of(card, launch->kernel, &kernel) &&
	   strcmp(kernel.kernel_file_name, KERNEL_FILE_EXPERT_RELAXED) == 0;
}

/* Rows from product records or grid[1] for quantization, n-gram, hyper-site and query launches. */
static uint32_t rows_in_record(uint32_t card, uint32_t step,
			       const struct kernel_launch *launch)
{
	static struct trunk_linear_quantized_arguments trunk;
	static struct trunk_linear_bfloat16_arguments values;
	static struct dense_linear_arguments dense;
	static struct trunk_relaxed_quantized_arguments ordered_trunk;
	static struct trunk_relaxed_bfloat16_arguments ordered_values;
	const struct expected_launch *const expected =
		&reference_sequence[step];
	const bool has_grid_rows =
		strcmp(expected->function, KERNEL_ACTIVATION_QUANTIZE) == 0 ||
		step == E_VALUES || step == E_GATE || step == A_FINISH ||
		step == A_QUERIES || step == A_POST || step == F_FINISH ||
		step == F_POST;
	memcpy(&trunk, launch->argument, sizeof(trunk));
	memcpy(&values, launch->argument, sizeof(values));
	memcpy(&dense, launch->argument, sizeof(dense));
	if (is_carried_trunk(card, step, launch)) {
		memcpy(&ordered_trunk, launch->argument, sizeof(ordered_trunk));
		return ordered_trunk.row_count;
	}
	if (expected->naming == FAMILY + LINEAR_FAMILY_TRUNK_BFLOAT16 &&
	    launch->argument_byte_count == sizeof(ordered_values)) {
		memcpy(&ordered_values, launch->argument,
		       sizeof(ordered_values));
		return ordered_values.row_count;
	}
	return expected->naming == FAMILY + LINEAR_FAMILY_TRUNK_QUANTIZED ?
		       trunk.row_count :
	       expected->naming == FAMILY + LINEAR_FAMILY_TRUNK_BFLOAT16 ?
		       values.row_count :
	       expected->naming == ROWS_NAMED ? dense.row_count :
	       has_grid_rows		      ? launch->grid[1] :
						0u;
}

/* The bytes a row of a launch's rows takes at a field of its record (field_at, a byte of it), as its step's record
 * says; 0 for scratch of a slab read whole (the index and core scores, the job table). Of the routed experts' pairs, a
 * row's are its EXPERTS_PER_ROW pairs'. */
static uint64_t row_bytes_of(uint32_t card, uint32_t step, uint32_t field_at,
			     const struct kernel_launch *launch)
{
	static struct activation_quantize_arguments quantize;
	static struct trunk_linear_quantized_arguments trunk;
	static struct trunk_linear_bfloat16_arguments values;
	static struct dense_linear_arguments dense;
	static struct attention_compressor_arguments compressor;
	static struct expert_gate_up_arguments gate_up;
	static struct expert_down_arguments down;
	static struct expert_row_sum_arguments row_sum;
	static struct trunk_hidden_arguments hidden;
	static struct expert_relaxed_wide_gate_up_arguments wide_gate_up;
	static struct expert_relaxed_wide_down_arguments wide_down;
	static struct trunk_relaxed_quantized_arguments ordered_trunk;
	static struct trunk_relaxed_bfloat16_arguments ordered_values;
	const struct expected_launch *const expected =
		&reference_sequence[step];
	const uint64_t half = LINEAR_BFLOAT16_BYTES, word = LINEAR_SUM_BYTES;
	const uint64_t pairs = MODEL_EXPERTS_PER_ROW,
		       block = LINEAR_VALUES_PER_BLOCK;
	memcpy(&quantize, launch->argument, sizeof(quantize));
	memcpy(&trunk, launch->argument, sizeof(trunk));
	memcpy(&values, launch->argument, sizeof(values));
	memcpy(&dense, launch->argument, sizeof(dense));
	memcpy(&compressor, launch->argument, sizeof(compressor));
	memcpy(&gate_up, launch->argument, sizeof(gate_up));
	memcpy(&down, launch->argument, sizeof(down));
	memcpy(&row_sum, launch->argument, sizeof(row_sum));
	memcpy(&hidden, launch->argument, sizeof(hidden));
	if (is_wide_expert(card, step, launch)) {
		memcpy(&wide_gate_up, launch->argument, sizeof(wide_gate_up));
		memcpy(&wide_down, launch->argument, sizeof(wide_down));
		gate_up.input_size = wide_gate_up.input_size;
		gate_up.output_size = wide_gate_up.output_size;
		gate_up.pairs_per_row = wide_gate_up.pairs_per_row;
		down.input_size = wide_down.input_size;
		down.output_size = wide_down.output_size;
	}
    if (is_carried_trunk(card, step, launch)) {
        memcpy(&ordered_trunk, launch->argument, sizeof(ordered_trunk));
        trunk.input_size = ordered_trunk.input_size;
        memcpy(trunk.output_size, ordered_trunk.output_size, sizeof(trunk.output_size));
    }
    if (expected->naming == FAMILY + LINEAR_FAMILY_TRUNK_BFLOAT16 &&
	launch->argument_byte_count == sizeof(ordered_values)) {
	    memcpy(&ordered_values, launch->argument, sizeof(ordered_values));
	    values.input_size = ordered_values.input_size;
	    values.output_size = ordered_values.output_size;
	    values.segment_count = ordered_values.segment_count;
    }
    if (strcmp(expected->function, KERNEL_ACTIVATION_QUANTIZE) == 0) {
	    const uint64_t vectors = step == F_QUANTIZE_PAIRS ? pairs : 1u,
			   width = quantize.width;
	    return vectors * (field_at == AT(activation_quantize_arguments,
					     codes_offset) ?
				      width :
			      field_at == AT(activation_quantize_arguments,
					     exponents_offset) ?
				      width / block :
				      width * half);
    }
    if (expected->naming == FAMILY + LINEAR_FAMILY_TRUNK_QUANTIZED) {
	    const uint64_t inputs = trunk.input_size;
	    return field_at == AT(trunk_linear_quantized_arguments,
				  codes_offset) ?
			   inputs :
		   field_at == AT(trunk_linear_quantized_arguments,
				  exponents_offset) ?
			   inputs / block :
		   field_at == AT(trunk_linear_quantized_arguments,
				  operands_offset) ?
			   inputs * half :
		   field_at == AT(trunk_linear_quantized_arguments,
				  output_offset) ?
			   trunk.output_size[0] * half :
			   trunk.output_size[1] * half;
    }
    if (expected->naming == FAMILY + LINEAR_FAMILY_TRUNK_BFLOAT16) {
	    return field_at == AT(trunk_linear_bfloat16_arguments,
				  values_offset) ?
			   (uint64_t)values.segment_count * values.input_size *
				   half :
			   values.output_size * half;
    }
    if (expected->naming == ROWS_NAMED) {
	    return field_at == AT(dense_linear_arguments, rows_offset) ?
			   dense.input_size * half :
			   dense.output_size *
				   (dense.output_type == F32_OUT ? word : half);
    }
    const uint32_t finish_post = AT(hyper_finish_arguments, post_offset),
		   finish_comb = AT(hyper_finish_arguments, comb_offset);
    const uint32_t post_post = AT(hyper_post_arguments, post_offset),
		   post_comb = AT(hyper_post_arguments, comb_offset);
    const uint32_t heads = AT(attention_queries_arguments, heads_offset),
		   queries = AT(attention_queries_arguments, queries_offset);
    switch (step) {
    case A_FINISH:
    case F_FINISH:
	    return field_at == AT(hyper_finish_arguments, mixes_offset) ?
			   MODEL_MIXES_ROW_BYTES :
		   field_at == finish_post ? MODEL_SITE_ROW_BYTES :
		   field_at == finish_comb ? MODEL_COMB_ROW_BYTES :
					     MODEL_HIDDEN_ROW_BYTES;
    case A_POST:
    case F_POST:
	    return field_at == post_post ? MODEL_SITE_ROW_BYTES :
		   field_at == post_comb ? MODEL_COMB_ROW_BYTES :
		   field_at == AT(hyper_post_arguments, output_offset) ?
					   MODEL_HIDDEN_ROW_BYTES :
					   MODEL_HIDDEN * word;
    case E_VALUES:
	    return (uint64_t)V41_DATA_ENGRAM_WIDTH * half;
    case E_GATE:
	    return ENGRAM_ROWS_PROJECTION * half;
    case A_PREPARE:
    case A_QUERY_NORM:
	    return field_at == AT(attention_prepare_arguments,
				  window_input_offset) ?
			   MODEL_HEAD_DIMS * half :
			   MODEL_QUERY_RANK * half;
    case A_QUERIES:
	    return field_at == heads || field_at == queries ?
			   MODEL_HEADS_ROW_BYTES :
			   MODEL_INDEX_QUERIES_ROW_BYTES;
    case A_COMPRESSOR:
	    return field_at == AT(attention_compressor_arguments,
				  latent_offset) ?
			   MODEL_HEAD_DIMS * half :
		   field_at == AT(attention_compressor_arguments,
				  score_offset) ||
				   compressor.ratio == MODEL_POOLED_RATIO ?
			   MODEL_HEAD_DIMS * word :
			   MODEL_HEAD_DIMS * half;
    case A_KEY:
	    return MODEL_INDEX_HEAD_DIMS * half;
    case A_SCAN:
	    return field_at == AT(index_scan_arguments_record, queries_offset) ?
			   MODEL_INDEX_QUERIES_ROW_BYTES :
		   field_at == AT(index_scan_arguments_record, weights_offset) ?
			   MODEL_INDEX_WEIGHTS_ROW_BYTES :
			   0u;
    case A_CORE_SCORES:
    case A_CORE:
	    return field_at == AT(attention_core_arguments_record,
				  scores_offset) ?
			   0u :
			   MODEL_HEADS_ROW_BYTES;
    case W_SELECT:
	    return MODEL_ROUTER_ROW_BYTES;
    case F_GATE_UP:
	    return field_at == AT(expert_gate_up_arguments, codes_offset) ?
			   gate_up.input_size :
		   field_at == AT(expert_gate_up_arguments, exponents_offset) ?
			   gate_up.input_size / block :
		   field_at == AT(expert_gate_up_arguments, operands_offset) ?
			   gate_up.input_size * half :
		   field_at == AT(expert_gate_up_arguments, hidden_offset) ?
			   (uint64_t)gate_up.pairs_per_row *
				   gate_up.output_size * half :
			   0u;
    case F_DOWN:
	    return pairs *
		   (field_at == AT(expert_down_arguments, codes_offset) ?
			    down.input_size :
		    field_at == AT(expert_down_arguments, exponents_offset) ?
			    down.input_size / block :
		    field_at == AT(expert_down_arguments, operands_offset) ?
			    down.input_size * half :
		    field_at == AT(expert_down_arguments, output_offset) ?
			    down.output_size * half :
			    0u);
    case F_ROW_SUM:
	    return field_at == AT(expert_row_sum_arguments,
				  pair_output_offset) ?
			   (uint64_t)row_sum.pairs_per_row * row_sum.width *
				   half :
			   row_sum.width * word;
    case F_SHARED_HIDDEN:
        return hidden.width * half;
    default:
        return 0;
    }
}

/* Where a launch reads an area of its card: its rows of the streams, the carried pre and the rotation; the step area
 * (of a slab: its view's). */
static uint64_t area_of(const struct program_case *made, uint32_t card, uint32_t area, const struct mapped_launch *own)
{
	const struct program_card_record *const fields =
		&made->program->cards[card];
	const uint64_t first_row = own->first_row;
	return area == STREAMS_AREA ?
		       fields->streams_offset +
			       first_row * MODEL_STREAMS_ROW_BYTES :
	       area == PRE_AREA ?
		       fields->carried_pre_offset +
			       first_row * MODEL_SITE_ROW_BYTES :
	       area == ROTATION_AREA ?
		       made->plan->cards[card]
				       .array_offset[V41_DATA_ARRAY_ROTATION] +
			       first_row * MODEL_ROTATION_ROW_BYTES :
	       own->is_slab != 0 ?
		       fields->views[first_row / ROWS].block_offset :
		       fields->step_offset;
}

/* The scope of the run an entry of chunk_layout belongs to: a span's runs with the slabs'. */
static uint32_t run_scope_of(uint32_t scope)
{
    return scope == OVER_SPANS ? OVER_SLABS : scope;
}

/* The launches of a chunked program's layer on its card, launches[first_launch] to launches[end - 1], against
 * chunk_layout: each its step, rows and the first of them into mapped; part: WHOLE_LAYER, or layer 20's FRONT_PART
 * (its front alone, front_end its end) or LATER_PART (its front to front_end, then its rest). A launch runs the
 * function that a thin program of its slab's rows runs for its step (over a span, a piece or the chunk: of those rows,
 * a trunk or dense product of more than 8 its wide form's instance), leads with its layer's tensor and covers its rows
 * (rows_in_record); qr alone runs the prepare of one part. The launches mapped, or 0 behind a failure line where one is
 * not as it should be. */
static uint32_t chunk_layer_map(const struct program_case *made, uint32_t card, uint32_t layer, uint32_t first_launch,
                                uint32_t front_end, uint32_t end, uint32_t part, struct mapped_launch *mapped)
{
	const uint32_t kinds =
		kinds_of(layer) | EVERY | CHUNK |
		(made->plan->limits.linear_order == LINEAR_ORDER_REFERENCE ?
			 REFERENCE_ORDER :
			 0u);
	const bool is_layer_20 = layer == MODEL_DECODER_FIRST_LAYER;
	const struct kernel_launch *const launches =
		made->program->cards[card].launches;
	const struct v41_tensor *const tensors =
		made->plan->layer_tensors[layer];
	const uint32_t chunk_rows = made->program->row_count;
	char name[NAME_CAPACITY], expected_name[NAME_CAPACITY];
	uint32_t at = first_launch, count = 0, entry = 0;
	bool follows = true;
	while (entry < COUNT_OF(chunk_layout) && follows) {
		const uint32_t scope = run_scope_of(chunk_layout[entry].scope);
		uint32_t run_end = entry + 1u;
		while (run_end < COUNT_OF(chunk_layout) &&
		       run_scope_of(chunk_layout[run_end].scope) == scope) {
			run_end++;
		}
		const uint32_t unit_rows = scope == OVER_SLABS	? ROWS :
					   scope == OVER_PIECES ? PIECE_ROWS :
								  chunk_rows;
		const uint32_t units =
			scope == NEW_RUN || scope == FRONT_END ?
				0 :
				(chunk_rows + unit_rows - 1u) / unit_rows;
		follows = scope != FRONT_END || part == WHOLE_LAYER ||
			  at == front_end;
		for (uint32_t unit = 0; unit < units && follows; unit++) {
			const uint32_t first_row = unit * unit_rows,
				       left = chunk_rows - first_row;
			for (uint32_t index = entry; index < run_end && follows;
			     index++) {
				const struct chunk_entry *const taken =
					&chunk_layout[index];
				const struct expected_launch *const expected =
					&reference_sequence[taken->step];
				const uint32_t part_rows =
					taken->scope == OVER_SPANS ? SPAN_ROWS :
								     unit_rows;
				const struct program_case own = {
					made->plan, made->program,
					made->descriptions,
					left < part_rows ? left : part_rows
				};
				uint64_t leading[LINEAR_MERGED_MATRICES];
				if ((expected->kinds & kinds) == 0 ||
				    (taken->layers == NOT_LAYER_20 &&
				     is_layer_20) ||
				    (taken->layers == LAYER_20_REST &&
				     !is_layer_20) ||
				    (taken->scope == OVER_SPANS &&
				     unit % SPAN_SLABS != 0)) {
					continue;
				}
				follows = at < end &&
					  count < MOST_LAYER_LAUNCHES;
				const struct kernel_launch *const launch =
					&launches[follows ? at : first_launch];
				const uint32_t record_rows = rows_in_record(
					card, taken->step, launch);
				const uint32_t parts =
					taken->step == A_QUERY_NORM ?
						1u :
						ATTENTION_PREPARE_PARTS;
				name_of(card, launch, name);
				expected_name_of(&own, card, layer, expected,
						 expected_name);
				memcpy(leading, launch->argument,
				       sizeof(leading));
				follows =
					follows &&
					strcmp(name, expected_name) == 0 &&
					(expected->tensor == NO_TENSOR ||
					 (leading[0] ==
						  tensors[expected->tensor]
							  .stored.byte_offset &&
					  (expected->second_tensor ==
						   NO_TENSOR ||
					   leading[1] ==
						   tensors[expected->second_tensor]
							   .stored
							   .byte_offset))) &&
					(record_rows == 0 ||
					 record_rows ==
						 own.rows *
							 (taken->step == F_QUANTIZE_PAIRS ?
								  MODEL_EXPERTS_PER_ROW :
								  1u)) &&
					(strcmp(name,
						KERNEL_ATTENTION_PREPARE) !=
						 0 ||
					 launch->grid[1] == parts);
				if (!follows) {
					(void)printf(
						"FAILED: %s: launch %" PRIu32
						" of layer %" PRIu32
						" on card %" PRIu32
						" is %s of %" PRIu32
						" rows, the chunk's layer has %s of rows %" PRIu32
						" to %" PRIu32 " there\n",
						context_text, at, layer, card,
						name, record_rows,
						expected_name, first_row,
						first_row + own.rows - 1u);
					failures++;
				}
				mapped[count++] = (struct mapped_launch){
					taken->step, first_row, own.rows,
					taken->scope == OVER_SLABS
				};
				at++;
				if (follows &&
				    is_wide_expert(card, taken->step, launch)) {
					static struct expert_relaxed_wide_gate_up_arguments
						product;
					memcpy(&product, launch->argument,
					       sizeof(product));
					const uint64_t *const arrays =
						made->plan->cards[card]
							.array_offset;
					const bool is_gate_up = taken->step ==
								F_GATE_UP;
					const uint32_t output_size =
						is_gate_up ?
							V41_DATA_EXPERT_INNER :
							MODEL_HIDDEN;
					const uint32_t input_size =
						is_gate_up ?
							MODEL_HIDDEN :
							V41_DATA_EXPERT_INNER;
					follows =
						product.tiles_offset ==
							arrays[V41_DATA_ARRAY_ROUTER_TILES] &&
						product.pairs_offset ==
							arrays[V41_DATA_ARRAY_ROUTER_PAIRS] &&
						product.operands_offset ==
							arrays[is_gate_up ?
								       V41_DATA_ARRAY_ROW_OPERANDS :
								       V41_DATA_ARRAY_PAIR_OPERANDS] &&
						product.hidden_offset ==
							arrays[is_gate_up ?
								       V41_DATA_ARRAY_PAIR_HIDDEN :
								       V41_DATA_ARRAY_PAIR_OUTPUT] &&
						product.output_size ==
							output_size &&
						product.input_size ==
							input_size &&
						product.tile_count ==
							ROUTER_WIDE_TILES(
								chunk_rows) &&
						product.pair_count ==
							chunk_rows *
								MODEL_EXPERTS_PER_ROW;
					expect(follows,
					       "ordered experts read the router's tiles and sorted pairs at full chunk size");
					const char *const small_name =
						taken->step == F_GATE_UP ?
							KERNEL_EXPERT_ORDERED_WIDE_SMALL_GATE_UP :
							KERNEL_EXPERT_ORDERED_WIDE_SMALL_DOWN;
					const struct kernel_launch *const small =
						&launches[at < end ?
								  at :
								  first_launch];
					name_of(card, small, name);
					follows =
						follows && at < end &&
						count < MOST_LAYER_LAUNCHES &&
						strcmp(name, small_name) == 0 &&
						small->argument_byte_count ==
							launch->argument_byte_count &&
						memcmp(small->argument,
						       launch->argument,
						       launch->argument_byte_count) ==
							0 &&
						memcmp(small->grid,
						       launch->grid,
						       sizeof(small->grid)) ==
							0 &&
						memcmp(small->block,
						       launch->block,
						       sizeof(small->block)) ==
							0 &&
						small->dynamic_shared_byte_count ==
							EXPERT_RELAXED_WIDE_SMALL_SHARED_BYTES;
					expect(follows,
					       "each ordered expert product has its small companion with the same tile record");
					mapped[count] = mapped[count - 1u];
					count++;
					at++;
				}
			}
		}
		entry = scope == FRONT_END && part == FRONT_PART ?
				COUNT_OF(chunk_layout) :
				run_end;
	}
    follows = follows && at == end;
    expect(follows, "a chunk's layer runs chunk_layout's launches and nothing more");
    return follows ? count : 0;
}

/* The fields a step's launches write that a flow reads: each word of the producer of its flows, once. */
static uint32_t written_fields_of(uint32_t step, uint32_t *fields_out)
{
    uint32_t count = 0;
    for (uint32_t index = 0; index < COUNT_OF(data_flows); index++) {
        for (uint32_t word = 0; word < data_flows[index].words && data_flows[index].producer == step; word++) {
            const uint32_t field_at = data_flows[index].producer_at + word * (uint32_t)sizeof(uint64_t);
            bool is_new = count < MOST_WRITES;
            for (uint32_t known = 0; known < count && is_new; known++) {
                is_new = fields_out[known] != field_at;
            }
            fields_out[is_new ? count++ : 0] = is_new ? field_at : fields_out[0];
        }
    }
    return count;
}

/* Whether the bytes a launch (mapped[consumer]) reads at word `word` of a flow's field, row_bytes a row from offset,
 * were written in the layer before it by launches of the producers the flows name for that field, at the field the
 * flow names, each byte of its own row (row 0 at one place for both, rows of the same bytes), and not again since. */
static bool read_holds(const struct mapped_launch *mapped, uint32_t consumer, const struct data_flow *flow,
                       uint32_t word, uint64_t offset, uint64_t row_bytes)
{
    const struct mapped_launch *const reader = &mapped[consumer];
    struct {
        uint64_t low, high;
    } gaps[MOST_GAPS] = {{offset, offset + (row_bytes == 0 ? 1u : reader->rows * row_bytes)}};
    uint32_t gap_count = 1;
    bool holds = true;
    for (uint32_t back = consumer; back-- > 0 && gap_count != 0 && holds;) {
        const struct mapped_launch *const writer = &mapped[back];
        for (uint32_t index = 0; index < layer_written_counts[back] && holds; index++) {
            const struct written_range *const written = &layer_written[back][index];
            for (uint32_t gap = 0; gap < gap_count && holds; gap++) {
                const uint64_t low = gaps[gap].low > written->low ? gaps[gap].low : written->low;
                const uint64_t high = gaps[gap].high < written->high ? gaps[gap].high : written->high;
                bool is_named = false;
                if (low >= high) {
                    continue;
                }
                for (uint32_t named = 0; named < COUNT_OF(data_flows) && !is_named; named++) {
                    const struct data_flow *const other = &data_flows[named];
                    is_named = other->consumer == flow->consumer && other->consumer_at == flow->consumer_at &&
                               other->producer == writer->step &&
                               other->producer_at + word * (uint32_t)sizeof(uint64_t) == written->field_at;
                }
                holds = is_named && written->row_bytes == row_bytes && gap_count < MOST_GAPS &&
                        (row_bytes == 0 ? written->low == offset && writer->first_row == reader->first_row
                                        : written->low + reader->first_row * row_bytes ==
                                              offset + writer->first_row * row_bytes);
                if (holds) {                    /* what remains of the gap: below the bytes written, above them */
                    gaps[gap_count].low = high;
                    gaps[gap_count++].high = gaps[gap].high;
                    gaps[gap].high = low;
                }
            }
            for (uint32_t gap = 0; gap < gap_count;) {      /* the empty gaps go */
                const bool is_empty = gaps[gap].low >= gaps[gap].high;
                gaps[gap] = is_empty ? gaps[gap_count - 1u] : gaps[gap];
                gap_count -= is_empty ? 1u : 0u;
                gap += is_empty ? 0u : 1u;
            }
        }
    }
    return holds && gap_count == 0;
}

/* The reads of a chunk's layer (its launches from launches[0] on, mapped) against data_flows: a read at a field a flow
 * names, in a layer that runs a producer the flows name for it, must hold (read_holds); a read of an area must be of
 * the launch's rows of it. The reads that do not, counted. */
/* The ordered record reads carried operands only. Its output offsets keep the reference record's places. */
static uint32_t field_in_record(uint32_t card, uint32_t step,
				const struct kernel_launch *launch,
				uint32_t field_at)
{
    if (is_wide_expert(card, step, launch)) {
	    return field_at == AT(expert_gate_up_arguments, operands_offset) ?
			   AT(expert_relaxed_wide_gate_up_arguments,
			      operands_offset) :
		   field_at == AT(expert_gate_up_arguments, hidden_offset) ?
			   AT(expert_relaxed_wide_gate_up_arguments,
			      hidden_offset) :
			   field_at;
    }
    return is_carried_trunk(card, step, launch) &&
			   field_at == AT(trunk_linear_quantized_arguments,
					  operands_offset) ?
		   AT(trunk_relaxed_quantized_arguments, operands_offset) :
		   field_at;
}

static uint32_t chunk_flows_wrong(const struct program_case *made,
				  uint32_t card,
				  const struct kernel_launch *launches,
				  const struct mapped_launch *mapped,
				  uint32_t count)
{
    uint64_t present = 0;
    uint32_t wrong = 0;
    for (uint32_t index = 0; index < count; index++) {
        uint32_t fields[MOST_WRITES];
        present |= 1ull << mapped[index].step;
        layer_written_counts[index] = written_fields_of(mapped[index].step, fields);
        for (uint32_t written = 0; written < layer_written_counts[index]; written++) {
            uint64_t offset = 0;
            const uint64_t row_bytes = row_bytes_of(card, mapped[index].step, fields[written], &launches[index]);
            memcpy(&offset, launches[index].argument +
                                field_in_record(card, mapped[index].step, &launches[index], fields[written]),
                   sizeof(offset));
            layer_written[index][written] = (struct written_range){
                offset, offset + (row_bytes == 0 ? 1u : mapped[index].rows * row_bytes), row_bytes, fields[written], 0};
        }
    }
    for (uint32_t consumer = 0; consumer < count; consumer++) {
        for (uint32_t index = 0; index < COUNT_OF(data_flows); index++) {
            const struct data_flow *const flow = &data_flows[index];
            bool is_first = flow->consumer == mapped[consumer].step, is_produced = false;
            for (uint32_t other = 0; other < COUNT_OF(data_flows) && is_first; other++) {
                const struct data_flow *const same = &data_flows[other];
                is_first = other >= index || same->consumer != flow->consumer || same->consumer_at != flow->consumer_at;
                is_produced = is_produced || (same->consumer == flow->consumer &&
                                              same->consumer_at == flow->consumer_at &&
                                              (same->producer >= SEQUENCE_LENGTH || (present >> same->producer & 1u)));
            }
            for (uint32_t word = 0; word < flow->words && is_first && is_produced; word++) {
                const uint32_t field_at = flow->consumer_at + word * (uint32_t)sizeof(uint64_t);
		if (is_carried_trunk(card, mapped[consumer].step,
				     &launches[consumer]) &&
		    (field_at == AT(trunk_linear_quantized_arguments,
				    codes_offset) ||
		     field_at == AT(trunk_linear_quantized_arguments,
				    exponents_offset))) {
			continue;
		}
		if (is_wide_expert(card, mapped[consumer].step,
				   &launches[consumer]) &&
		    (field_at == AT(expert_gate_up_arguments, codes_offset) ||
		     field_at ==
			     AT(expert_gate_up_arguments, exponents_offset))) {
			continue;
		}
		uint64_t offset = 0;
                memcpy(&offset, launches[consumer].argument +
                                    field_in_record(card, mapped[consumer].step, &launches[consumer], field_at),
                       sizeof(offset));
                const bool holds = flow->producer >= SEQUENCE_LENGTH
                                       ? offset == area_of(made, card, flow->producer, &mapped[consumer])
                                       : read_holds(mapped, consumer, flow, word, offset,
                                                    row_bytes_of(card, mapped[consumer].step, field_at,
                                                                 &launches[consumer]));
                wrong += holds ? 0u : 1u;
            }
        }
    }
    return wrong;
}

/* Every layer of a chunked program mapped (chunk_layer_map) and its flows held (chunk_flows_wrong), layer 20 of a full
 * program across its front and its rest; the failures counted, the reads that do not hold behind a failure line each
 * layer. */
static uint32_t chunk_layers_wrong(const struct program_case *made)
{
	const struct execution_program *const program = made->program;
	uint32_t wrong = 0;
	for (uint32_t index = 0; index < program->segment_count; index++) {
		const struct segment_record *const segment =
			&program->segments[index];
		const bool is_layer_20 = segment->layer ==
					 MODEL_DECODER_FIRST_LAYER;
		const bool is_rest = is_layer_20 &&
				     index == program->front_segment_count;
		if (segment->layer == SEGMENT_NO_LAYER ||
		    (is_layer_20 && !is_rest &&
		     program->kind == PROGRAM_FULL)) {
			continue; /* layer 20's front is a full program's with its rest */
		}
		uint32_t first =
			is_rest ? program->segments[index - 1u].first_launch :
				  segment->first_launch;
		if (program->limits.draft_positions != 0 &&
		    segment->layer >= MODEL_DRAFT_FIRST_TAP_LAYER) {
			const struct kernel_launch *const tap =
				&program->cards[segment->card].launches[first++];
			struct test_host_memory_backend_kernel kernel;
			expect(test_host_memory_backend_kernel_of(
				       segment->card, tap->kernel, &kernel) &&
				       strcmp(kernel.function_name,
					      KERNEL_HYPER_DRAFT_INPUTS) == 0 &&
				       tap->grid[1] == program->row_count,
			       "a draft tap precedes the entire target layer's wide launches");
		}
		const uint32_t end =
			segment->first_launch + segment->launch_count;
		const uint32_t count = chunk_layer_map(
			made, segment->card, segment->layer, first,
			is_rest ? segment->first_launch : end, end,
			is_rest	    ? LATER_PART :
			is_layer_20 ? FRONT_PART :
				      WHOLE_LAYER,
			layer_mapped);
		const uint32_t flows =
			count == 0 ?
				0 :
				chunk_flows_wrong(
					made, segment->card,
					program->cards[segment->card].launches +
						first,
					layer_mapped, count);
		if (flows != 0) {
			(void)printf(
				"FAILED: %s: layer %" PRIu32 ": %" PRIu32
				" reads of its %" PRIu32
				" launches do not read "
				"what the flows name, written at their rows and not again\n",
				context_text, segment->layer, flows, count);
			failures++;
		}
		wrong += (count == 0 ? 1u : 0u) + flows;
	}
    return wrong;
}

/* The launches of a chunked program's layer (its one segment) mapped into layer_mapped (chunk_layer_map): their count,
 * 0 for none; *launches_out the first of them, *card_out their card. */
static uint32_t control_map(const struct program_case *made, uint32_t layer,
			    const struct kernel_launch **launches_out,
			    uint32_t *card_out)
{
	const struct execution_program *const program = made->program;
	const uint32_t segment_index = first_segment_of(program, 0, layer);
	if (segment_index == NO_SEGMENT) {
		return 0;
	}
	const struct segment_record *const segment =
		&program->segments[segment_index];
	*launches_out =
		program->cards[segment->card].launches + segment->first_launch;
	*card_out = segment->card;
	return chunk_layer_map(made, segment->card, layer,
			       segment->first_launch, 0,
			       segment->first_launch + segment->launch_count,
			       WHOLE_LAYER, layer_mapped);
}

/* The index of the mapped launch of step whose rows begin at first_row; count where there is none. */
static uint32_t mapped_index(uint32_t count, uint32_t step, uint32_t first_row)
{
    uint32_t index = 0;
    while (index < count && (layer_mapped[index].step != step || layer_mapped[index].first_row != first_row)) {
        index++;
    }
    return index;
}

/* The wrong reads (chunk_flows_wrong) of a copy of a mapped layer's count launches (control_launches and
 * control_mapped) with the launch at from moved behind the one at to (none moved where from is to), and the record
 * word at byte field_at of the copy's launch at shifted moved back by back bytes. */
static uint32_t control_flows_wrong(const struct program_case *made,
				    uint32_t card,
				    const struct kernel_launch *launches,
				    uint32_t count, uint32_t from, uint32_t to,
				    uint32_t shifted, uint32_t field_at,
				    uint64_t back)
{
    uint64_t offset = 0;
    memcpy(control_launches, launches, count * sizeof(control_launches[0]));
    memcpy(control_mapped, layer_mapped, count * sizeof(control_mapped[0]));
    memcpy(&control_launches[from], &launches[from + 1u], (to - from) * sizeof(control_launches[0]));
    memcpy(&control_mapped[from], &layer_mapped[from + 1u], (to - from) * sizeof(control_mapped[0]));
    control_launches[to] = launches[from];
    control_mapped[to] = layer_mapped[from];
    memcpy(&offset, control_launches[shifted].argument + field_at, sizeof(offset));
    offset -= back;
    memcpy(control_launches[shifted].argument + field_at, &offset, sizeof(offset));
    return chunk_flows_wrong(made, card, control_launches, control_mapped, count);
}

/* Shown failing once, on copies of the launches of a chunked program's layers: of CONTROL_LAYER, wo_a over the chunk
 * ahead of the last slab's core (it reads that slab's attention output before it is written), span 1's queries at
 * span 0's heads, and span 1's queries behind the core of its first slab (which reads span 0's queries, the rows of
 * slab 0, where its own are not yet written); of POOL_CONTROL_LAYER, the indexer's weights over the chunk behind slab
 * 0's scan (which reads them before they are written) and slab 1's compressor at slab 0's rows of wkv's product over
 * the chunk. The flows must count each, and nothing in either layer as made. */
static void chunk_flows_shown_failing(const struct program_case *made)
{
	const struct kernel_launch *launches = NULL;
	uint32_t card = 0,
		 count = control_map(made, CONTROL_LAYER, &launches, &card);
	const uint32_t groups = mapped_index(count, A_GROUPS, 0),
		       queries = mapped_index(count, A_QUERIES, SPAN_ROWS);
	const uint32_t core = mapped_index(count, A_CORE, SPAN_ROWS);
	const bool has_cores = count != 0 && groups != 0 && groups < count &&
			       layer_mapped[groups - 1u].step == A_CORE &&
			       queries < core && core < count;
	const uint32_t early =
		has_cores ? control_flows_wrong(made, card, launches, count,
						groups - 1u, groups, 0, 0, 0) :
			    0;
	const uint32_t misplaced =
		has_cores ?
			control_flows_wrong(
				made, card, launches, count, 0, 0, queries,
				AT(attention_queries_arguments, heads_offset),
				(uint64_t)SPAN_ROWS * MODEL_HEADS_ROW_BYTES) :
			0;
	const uint32_t stale =
		has_cores ? control_flows_wrong(made, card, launches, count,
						queries, core, 0, 0, 0) :
			    0;
	const uint32_t intact = has_cores ?
					chunk_flows_wrong(made, card, launches,
							  layer_mapped, count) :
					1u;
	expect(has_cores && early != 0 && misplaced != 0 && stale != 0 &&
		       intact == 0,
	       "the flows count wo_a ahead of the last core, queries at another span's heads and a core ahead of its "
	       "span's queries, and nothing in the layer");
	(void)printf(
		"  shown failing once, layer %" PRIu32 " of %" PRIu32
		" rows: wo_a ahead of the last slab's core, "
		"%" PRIu32
		" wrong reads; span 1's queries at span 0's heads, %" PRIu32
		"; slab 8's core ahead of "
		"span 1's queries, %" PRIu32 "; as made, %" PRIu32 "\n",
		CONTROL_LAYER, made->program->row_count, early, misplaced,
		stale, intact);
	count = control_map(made, POOL_CONTROL_LAYER, &launches, &card);
	const uint32_t weights = mapped_index(count, A_INDEX_WEIGHTS, 0),
		       scan = mapped_index(count, A_SCAN, 0);
	const uint32_t compressor = mapped_index(count, A_COMPRESSOR, ROWS);
	const bool has_pool = count != 0 && weights < scan && scan < count &&
			      compressor < count;
	const uint32_t late =
		has_pool ? control_flows_wrong(made, card, launches, count,
					       weights, scan, 0, 0, 0) :
			   0;
	const uint64_t kv_slab_bytes =
		(uint64_t)ROWS * MODEL_HEAD_DIMS * LINEAR_SUM_BYTES;
	const uint32_t shifted =
		has_pool ?
			control_flows_wrong(
				made, card, launches, count, 0, 0, compressor,
				AT(attention_compressor_arguments, kv_offset),
				kv_slab_bytes) :
			0;
	const uint32_t pool_intact =
		has_pool ? chunk_flows_wrong(made, card, launches, layer_mapped,
					     count) :
			   1u;
	expect(has_pool && late != 0 && shifted != 0 && pool_intact == 0,
	       "the flows count the indexer's weights behind a scan and a compressor at another slab's rows of wkv's "
	       "product, and nothing in the layer");
	(void)printf(
		"  shown failing once, layer %" PRIu32
		": the indexer's weights behind slab 0's scan, %" PRIu32
		" wrong reads; slab 1's compressor at slab 0's rows of wkv's product, %" PRIu32
		"; as made, %" PRIu32 "\n",
		POOL_CONTROL_LAYER, late, shifted, pool_intact);
}

/* The embedding of a chunked program: one launch over its rows (hyper_connection.h: the bytes of its slabs'), from the
 * step area into its card's streams and carried pre. */
static bool embedding_holds(const struct program_case *made)
{
	static struct hyper_embed_arguments embed; /* static: see expect_record */
	const struct execution_program *const program = made->program;
	const uint32_t index = segment_of_layer(program, EMBEDDING);
	if (index >= program->segment_count) {
		return false;
	}
	const struct segment_record *const segment = &program->segments[index];
	const struct kernel_launch *const launch =
		&program->cards[segment->card].launches[segment->first_launch];
	const uint64_t *const arrays =
		made->plan->cards[segment->card].array_offset;
	memcpy(&embed, launch->argument, sizeof(embed));
	return segment->launch_count == 1 && launch->grid[1] == made->rows &&
	       embed.step_offset == arrays[V41_DATA_ARRAY_STEP] &&
	       embed.embedding_offset == made->plan->globals[V41_DATA_EMBEDDING]
						 .stored.byte_offset &&
	       embed.streams_offset == arrays[V41_DATA_ARRAY_STREAMS] &&
	       embed.pre_offset == arrays[V41_DATA_ARRAY_CARRIED_PRE];
}

/* The wide programs of 64, 65 and WIDE_CAPACITY rows of 1.3 in a plan whose chunk holds WIDE_CAPACITY rows: a front
 * step's program (the step segments, the embedding, layers 0 to 19 and layer 20's front, no head, no pause); a view a
 * slab on each card that runs any of them, a view's block and rows a slab of 8 apart in the plan's views; the
 * embedding (embedding_holds); each layer as chunk_layout says, its data flow held (chunk_layers_wrong); the hand-offs
 * of the thin program's front. */
static void test_wide_programs(const struct model_operations *model,
			       const struct pack_record *pack,
			       struct device_card **cards,
			       struct card_description_record *descriptions)
{
	static const uint32_t wide_rows[] = { MODEL_WIDE_LEAST_ROWS,
					      MODEL_WIDE_LEAST_ROWS + 1u,
					      WIDE_CAPACITY };
	const uint64_t front_layers =
		(1ull << (MODEL_DECODER_FIRST_LAYER + 1u)) - 1u;
	struct model_plan_record *const plan = plan_of(
		pack, &placements[0], LINEAR_ORDER_FIXED, WIDE_CAPACITY);
	struct execution_program *thin = NULL;
	context_text = "wide programs of 1.3";
	cards_open(cards, descriptions);
	expect_status(plan == NULL ?
			      STATUS_INVALID_ARGUMENT :
			      model->program_prepare(
				      plan, &test_host_memory_backend, cards,
				      descriptions, PROGRAM_THIN, ROWS, &thin),
		      STATUS_OK, "a thin program of a wide plan is made");
	for (uint32_t index = 0; index < COUNT_OF(wide_rows) && thin != NULL;
	     index++) {
		const uint32_t rows = wide_rows[index],
			       slabs = (rows + ROWS - 1u) / ROWS;
		struct execution_program *wide = NULL;
		uint32_t wrong_views = 0;
		expect_status(model->program_prepare(
				      plan, &test_host_memory_backend, cards,
				      descriptions, PROGRAM_WIDE, rows, &wide),
			      STATUS_OK, "a wide program is made");
		if (wide == NULL) {
			continue;
		}
		for (uint32_t card = 0; card < CARDS; card++) {
			const struct program_card_record *const own =
				&wide->cards[card];
			const bool runs =
				card == plan->embedding_card ||
				(plan->cards[card].layers & front_layers) != 0;
			wrong_views += own->view_count == (runs ? slabs : 0) &&
						       (own->views != NULL) ==
							       runs ?
					       0u :
					       1u;
			for (uint32_t slab = 0; slab < own->view_count &&
						runs && own->views != NULL;
			     slab++) {
				const struct step_view expected = {
					plan->cards[card].array_offset
							[V41_DATA_ARRAY_VIEWS] +
						(uint64_t)slab *
							(STEP_BLOCK_BYTES_VALUE +
							 ROWS * STEP_ROW_BYTES_VALUE),
					slab * ROWS, ROWS
				};
				wrong_views += memcmp(&own->views[slab],
						      &expected,
						      sizeof(expected)) == 0 ?
						       0u :
						       1u;
			}
		}
		const struct program_case made = { plan, wide, descriptions,
						   rows };
		expect(embedding_holds(&made),
		       "the embedding is one launch over the chunk, from the step area into its "
		       "card's streams and carried pre");
		expect(chunk_layers_wrong(&made) == 0,
		       "each layer of a wide program runs chunk_layout's launches, each read "
		       "of them what its flows name, of its rows, written before it");
		const uint32_t layer_20 =
			first_segment_of(wide, 0, MODEL_DECODER_FIRST_LAYER);
		expect(wide->kind == PROGRAM_WIDE && wide->row_count == rows &&
			       wide->expert_pause_count == 0 &&
			       wide->front_segment_count ==
				       wide->segment_count &&
			       layer_20 + 1u == wide->segment_count,
		       "a wide program ends with layer 20's front, every segment a front step's");
		expect(wrong_views == 0,
		       "a view a slab on every card that runs the front, a slab of 8 apart");
		uint32_t front_hand_offs = 0, wrong_hand_offs = 0;
		for (uint32_t hand_off = 0; hand_off < thin->hand_off_count;
		     hand_off++) {
			const struct hand_off_record *const own =
				&thin->hand_offs[hand_off];
			const bool is_front = own->after_segment <
					      thin->front_segment_count;
			wrong_hand_offs +=
				is_front && (front_hand_offs >=
						     wide->hand_off_count ||
					     memcmp(own,
						    &wide->hand_offs
							     [front_hand_offs],
						    sizeof(*own)) != 0) ?
					1u :
					0u;
			front_hand_offs += is_front ? 1u : 0u;
		}
		expect(wide->hand_off_count == front_hand_offs &&
			       wrong_hand_offs == 0,
		       "the hand-offs of the thin program's front, those alone");
		(void)printf("  a wide program of %" PRIu32 " rows: %" PRIu32
			     " slabs, %" PRIu32 " segments, %" PRIu32
			     " launches on card 0\n",
			     rows, slabs, wide->segment_count,
			     wide->cards[0].launch_count);
		model->program_destroy(wide);
	}
    model->program_destroy(thin);
    model->plan_destroy(plan);
}

/* What a full program's records read across layers (model_v41.h): the candidates of the source (layer 20's rest, its
 * candidates launch) and of every reader (its scan), at the slab's rows of the chunk's CANDIDATES, a launch each
 * slab. */
static bool candidates_flow_holds(const struct model_plan_record *plan,
				  const struct execution_program *full,
				  const struct segment_record *segment,
				  uint32_t segment_index, uint32_t slabs)
{
	const struct kernel_launch *const launches =
		full->cards[segment->card].launches;
	const uint64_t *const arrays = plan->cards[segment->card].array_offset;
	const bool is_rest_of_20 = segment->layer ==
					   MODEL_DECODER_FIRST_LAYER &&
				   segment_index >= full->front_segment_count;
	const bool is_reader =
		(V41_DATA_CANDIDATE_READERS >> segment->layer & 1u) != 0;
	const bool scans_candidates = segment->layer == MODEL_CANDIDATE_SOURCE ?
					      is_rest_of_20 :
					      is_reader;
	const char *const candidates_launch =
		is_reader ? KERNEL_INDEX_SCAN : KERNEL_INDEX_CANDIDATES;
	char name[NAME_CAPACITY];
	uint32_t scans = 0;
	bool holds = true;
	for (uint32_t launch = segment->first_launch;
	     launch < segment->first_launch + segment->launch_count && holds &&
	     scans_candidates;
	     launch++) {
		name_of(segment->card, &launches[launch], name);
		if (strcmp(name, candidates_launch) == 0) {
			struct index_scan_arguments_record scan; /* a reader's */
			struct index_candidates_arguments
				source; /* the source's */
			memcpy(&scan, launches[launch].argument, sizeof(scan));
			memcpy(&source, launches[launch].argument,
			       sizeof(source));
			holds = (is_reader ? scan.candidates_offset :
					     source.candidates_offset) ==
				arrays[V41_DATA_ARRAY_CANDIDATES] +
					(uint64_t)scans * ROWS *
						MODEL_CANDIDATES_ROW_BYTES;
			scans++;
		}
    }
    return holds && (!scans_candidates || scans == slabs);
}

/* The full programs of 64, 65, WIDE_CAPACITY and PIECES_ROWS rows of 1.3 in a plan whose chunk holds FULL_CAPACITY
 * rows: all of the thin program but its head (its segments, in card, stage and layer, and its front; the hand-offs
 * behind them, byte for byte), no pause; a view a slab on every card that runs launches; the embedding
 * (embedding_holds); each layer as chunk_layout says, layer 20 a front segment and a segment of its rest and FFN, its
 * data flow held across both (chunk_layers_wrong); the candidates read across layers at each slab's rows; and, once,
 * the flows shown failing (chunk_flows_shown_failing). Then all of it again with the PRO (card 4: layers 32 to 39) in
 * class 120 native, whose wide counts are its default ones and whose dense products take class 120's wide forms. */
static void test_full_programs(const struct model_operations *model,
			       const struct pack_record *pack,
			       struct device_card **cards,
			       struct card_description_record *descriptions)
{
	static const uint32_t full_rows[] = { MODEL_WIDE_LEAST_ROWS,
					      MODEL_WIDE_LEAST_ROWS + 1u,
					      WIDE_CAPACITY, PIECES_ROWS };
	struct model_plan_record *const plan = plan_of(
		pack, &placements[0], LINEAR_ORDER_FIXED, FULL_CAPACITY);
	struct execution_program *first = NULL;
	for (uint32_t index = 0; index < COUNT_OF(full_rows) * PRO_CLASSES;
	     index++) {
		const uint32_t rows = full_rows[index % COUNT_OF(full_rows)],
			       slabs = (rows + ROWS - 1u) / ROWS;
		if (index % COUNT_OF(full_rows) ==
		    0) { /* the cards afresh, the PRO of each class: a program of 8 rows */
			context_text =
				index == 0 ?
					"full programs of 1.3" :
					"full programs of 1.3, the PRO in class 120 native";
			model->program_destroy(first);
			first = NULL;
			cards_open(
				cards,
				descriptions); /* class 120 native: an sm_120a card, its own forms asked */
			descriptions[PRO_CARD].arithmetic_class =
				index == 0 ?
					descriptions[PRO_CARD].arithmetic_class :
					ARITHMETIC_CLASS_SM_120_NATIVE;
			expect_status(
				plan == NULL ?
					STATUS_INVALID_ARGUMENT :
					model->program_prepare(
						plan, &test_host_memory_backend,
						cards, descriptions,
						PROGRAM_THIN, ROWS, &first),
				STATUS_OK,
				"a thin program of a wide plan is made");
		}
		if (first == NULL) {
			break;
		}
		struct execution_program *full = NULL;
		uint32_t wrong_views = 0, wrong_segments = 0,
			 wrong_candidates = 0, wrong_hand_offs = 0,
			 kept_hand_offs = 0;
		expect_status(model->program_prepare(
				      plan, &test_host_memory_backend, cards,
				      descriptions, PROGRAM_FULL, rows, &full),
			      STATUS_OK, "a full program is made");
		if (full == NULL) {
			continue;
		}
		expect(full->kind == PROGRAM_FULL && full->row_count == rows &&
			       full->expert_pause_count == 0 &&
			       full->segment_count + 1u ==
				       first->segment_count &&
			       full->front_segment_count ==
				       first->front_segment_count &&
			       first->segments[full->segment_count].layer ==
				       SEGMENT_NO_LAYER,
		       "a full program is the thin program's segments but its last, the head, with its front");
		for (uint32_t card = 0; card < CARDS; card++) {
			const struct program_card_record *const own =
				&full->cards[card];
			const bool runs = own->launch_count != 0;
			wrong_views += own->view_count == (runs ? slabs : 0) &&
						       (own->views != NULL) ==
							       runs ?
					       0u :
					       1u;
		}
		for (uint32_t segment = 0; segment < full->segment_count &&
					   segment < first->segment_count;
		     segment++) {
			const struct segment_record *const
				own = &full->segments[segment],
				*const thin_own = &first->segments[segment];
			wrong_segments +=
				own->card == thin_own->card &&
						own->stage == thin_own->stage &&
						own->layer == thin_own->layer &&
						own->pause == 0 ?
					0u :
					1u;
			wrong_candidates +=
				own->layer != SEGMENT_NO_LAYER &&
						!candidates_flow_holds(
							plan, full, own,
							segment, slabs) ?
					1u :
					0u;
		}
		for (uint32_t hand_off = 0; hand_off < first->hand_off_count;
		     hand_off++) {
			const struct hand_off_record *const own =
				&first->hand_offs[hand_off];
			const bool is_kept = own->after_segment <
					     full->segment_count;
			wrong_hand_offs +=
				is_kept && (kept_hand_offs >=
						    full->hand_off_count ||
					    memcmp(own,
						   &full->hand_offs
							    [kept_hand_offs],
						   sizeof(*own)) != 0) ?
					1u :
					0u;
			kept_hand_offs += is_kept ? 1u : 0u;
		}
		const struct program_case made = { plan, full, descriptions,
						   rows };
		expect(embedding_holds(&made),
		       "the embedding is one launch over the chunk, from the step area into its "
		       "card's streams and carried pre");
		expect(wrong_views == 0,
		       "a view a slab on every card that runs launches");
		expect(wrong_segments == 0,
		       "a full program's segments are the thin program's, in card, stage and layer");
		expect(chunk_layers_wrong(&made) == 0,
		       "each layer of a full program runs chunk_layout's launches, layer 20 in "
		       "its front and its rest, each read of them what its flows name, of its "
		       "rows, written before it");
		expect(wrong_candidates == 0,
		       "every candidate scan and the candidate source at their slab's rows");
		expect(full->hand_off_count == kept_hand_offs &&
			       wrong_hand_offs == 0,
		       "the hand-offs of the thin program behind the full program's segments, byte for byte");
		if (rows == WIDE_CAPACITY && index < COUNT_OF(full_rows)) {
			chunk_flows_shown_failing(&made);
		}
		(void)printf("  a full program of %" PRIu32 " rows: %" PRIu32
			     " slabs, %" PRIu32 " segments, %" PRIu32
			     " launches on card 0\n",
			     rows, slabs, full->segment_count,
			     full->cards[0].launch_count);
		model->program_destroy(full);
	}
    model->program_destroy(first);
    model->plan_destroy(plan);
}

/* Relaxed thin products unfold projection/shared joins; subthreshold chunks add each small expert tile. */
static uint32_t
extra_expert_launches(const struct execution_program *program,
		      const struct card_description_record *descriptions,
		      uint32_t card, uint32_t end)
{
	if (program->row_count >= EXPERT_ORDERED_MINIMUM_ROWS ||
	    descriptions[card].arithmetic_class != ARITHMETIC_CLASS_SM_80) {
		return 0;
	}
    uint32_t count = 0;
    for (uint32_t index = 0; index < end; index++) {
        char name[NAME_CAPACITY];
        name_of(card, &program->cards[card].launches[index], name);
	count +=
		program->row_count <= ROWS ?
			(strstr(name, KERNEL_ATTENTION_PROJECTION_STEM) !=
						 NULL ||
					 strstr(name,
						KERNEL_EXPERT_SHARED_JOIN_STEM) !=
						 NULL ?
				 1u :
				 0u) :
			(strstr(name, "expert_gate_up_") != NULL ||
					 strstr(name, "expert_down_") != NULL ?
				 1u :
				 0u);
    }
    return count;
}

static void expect_expert_tile_conversion(const struct kernel_launch *before,
					  const struct kernel_launch *after,
					  uint32_t rows, bool gate_up)
{
	struct expert_gate_up_arguments old_gate;
	struct expert_down_arguments old_down;
	struct expert_relaxed_wide_gate_up_arguments new_gate;
	struct expert_relaxed_wide_down_arguments new_down;
	memcpy(&old_gate, before->argument, sizeof(old_gate));
	memcpy(&old_down, before->argument, sizeof(old_down));
	memcpy(&new_gate, after->argument, sizeof(new_gate));
	memcpy(&new_down, after->argument, sizeof(new_down));
	expect(gate_up ?
		       old_gate.operands_offset == new_gate.operands_offset &&
			       old_gate.hidden_offset ==
				       new_gate.hidden_offset &&
			       old_gate.output_size == new_gate.output_size &&
			       old_gate.input_size == new_gate.input_size &&
			       old_gate.pair_count == new_gate.pair_count &&
			       old_gate.pairs_per_row ==
				       new_gate.pairs_per_row &&
			       old_gate.activation_limit ==
				       new_gate.activation_limit &&
			       new_gate.tile_count == ROUTER_WIDE_TILES(rows) :
		       old_down.operands_offset == new_down.operands_offset &&
			       old_down.output_offset ==
				       new_down.output_offset &&
			       old_down.output_size == new_down.output_size &&
			       old_down.input_size == new_down.input_size &&
			       old_down.pair_count == new_down.pair_count &&
			       new_down.tile_count == ROUTER_WIDE_TILES(rows),
	       "subthreshold relaxed tiles keep the original expert input, output and numerical shape");
}

/* The relaxed product and its separate post/join retain every read and write of the reference fusion. */
static void
expect_unfused_tail(const struct kernel_launch *fused,
		    const struct kernel_launch *tail, uint32_t card,
		    bool projection,
		    struct trunk_linear_quantized_arguments *product)
{
    char name[NAME_CAPACITY];
    name_of(card, tail, name);
    if (projection) {
	    struct attention_projection_arguments before;
	    memcpy(&before, fused->argument, sizeof(before));
	    *product = (struct trunk_linear_quantized_arguments){
		    { before.weights_offset, 0 },
		    { before.output_offset, 0 },
		    before.codes_offset,
		    before.exponents_offset,
		    before.operands_offset,
		    { before.output_size, 0 },
		    before.input_size,
		    before.row_count
	    };
	    const struct hyper_post_arguments post = { before.streams_offset,
						       before.post_offset,
						       before.comb_offset,
						       before.output_offset,
						       0,
						       0,
						       0 };
	    expect(strcmp(name, KERNEL_HYPER_POST) == 0 &&
			   tail->argument_byte_count == sizeof(post) &&
			   memcmp(tail->argument, &post, sizeof(post)) == 0 &&
			   tail->grid[1] == before.row_count,
		   "relaxed projection retains its separate attention post");
    } else {
	    struct expert_shared_join_arguments before;
	    memcpy(&before, fused->argument, sizeof(before));
	    *product = (struct trunk_linear_quantized_arguments){
		    { before.weights_offset, 0 },
		    { before.output_offset, 0 },
		    before.codes_offset,
		    before.exponents_offset,
		    before.operands_offset,
		    { MODEL_HIDDEN, 0 },
		    before.input_size,
		    before.row_count
	    };
	    const struct expert_join_arguments join = {
		    before.pair_output_offset, before.output_offset,
		    before.streams_offset,     before.post_offset,
		    before.comb_offset,	       before.row_sums_offset,
		    before.pairs_per_row,      0
	    };
	    expect(strcmp(name, KERNEL_EXPERT_JOIN) == 0 &&
			   tail->argument_byte_count == sizeof(join) &&
			   memcmp(tail->argument, &join, sizeof(join)) == 0 &&
			   tail->grid[1] == before.row_count,
		   "relaxed shared down retains a plain join with captured ordered row sums");
    }
}

/* Relaxed programs keep state/slab launches and every PRO launch, adding tile companions below the threshold. */
static void
expect_relaxed_program(const struct execution_program *reference,
		       const struct execution_program *relaxed,
		       const struct card_description_record *descriptions)
{
    expect(reference->segment_count == relaxed->segment_count &&
               reference->hand_off_count == relaxed->hand_off_count &&
               memcmp(reference->hand_offs, relaxed->hand_offs,
                      (size_t)reference->hand_off_count * sizeof(*reference->hand_offs)) == 0,
           "relaxed programs keep every segment and state hand-off in order");
    for (uint32_t index = 0; index < reference->segment_count && index < relaxed->segment_count; index++) {
	    struct segment_record expected = reference->segments[index];
	    const uint32_t first_extra =
		    extra_expert_launches(reference, descriptions,
					  expected.card, expected.first_launch);
	    expected.launch_count +=
		    extra_expert_launches(
			    reference, descriptions, expected.card,
			    expected.first_launch + expected.launch_count) -
		    first_extra;
	    expected.first_launch += first_extra;
	    expect(memcmp(&expected, &relaxed->segments[index],
			  sizeof(expected)) == 0,
		   "relaxed segments differ only by the added small expert tile launches");
    }
    uint32_t changed = 0;
    for (uint32_t card = 0; card < reference->card_count; card++) {
	    const struct program_card_record
		    *const before = &reference->cards[card],
			   *const after = &relaxed->cards[card];
	    expect(before->launch_count + extra_expert_launches(
						  reference, descriptions, card,
						  before->launch_count) ==
				   after->launch_count &&
			   before->view_count == after->view_count &&
			   (before->view_count == 0 ||
			    memcmp(before->views, after->views,
				   (size_t)before->view_count *
					   sizeof(*before->views)) == 0),
		   "relaxed programs keep their launch count and slab views");
	    uint32_t extra = 0;
	    for (uint32_t index = 0; index < before->launch_count &&
				     index + extra < after->launch_count;
		 index++) {
		    const struct kernel_launch *const old_launch =
			    &before->launches[index];
		    const struct kernel_launch *const new_launch =
			    &after->launches[index + extra];
		    struct test_host_memory_backend_kernel old_kernel,
			    new_kernel;
		    const bool found =
			    test_host_memory_backend_kernel_of(
				    card, old_launch->kernel, &old_kernel) &&
			    test_host_memory_backend_kernel_of(
				    card, new_launch->kernel, &new_kernel);
		    expect(found,
			   "both program launches name prepared kernels");
		    if (!found) {
			    continue;
		    }
		    const bool is_cmp = descriptions[card].arithmetic_class ==
					ARITHMETIC_CLASS_SM_80;
		    const bool is_expert =
			    strstr(old_kernel.function_name,
				   "expert_gate_up_") != NULL ||
			    strstr(old_kernel.function_name, "expert_down_") !=
				    NULL ||
			    strstr(old_kernel.function_name,
				   "expert_ordered_wide_") != NULL;
		    if (is_cmp && is_expert) {
			    const bool is_wide = reference->row_count > ROWS;
			    const bool gate_up =
				    strstr(old_kernel.function_name,
					   "gate_up") != NULL;
			    const bool is_small =
				    strstr(old_kernel.function_name, "small") !=
				    NULL;
			    const char *const expected =
				    !is_wide ?
					    (gate_up ?
						     KERNEL_EXPERT_RELAXED_GATE_UP :
						     KERNEL_EXPERT_RELAXED_DOWN) :
				    gate_up ?
					    (is_small ?
						     KERNEL_EXPERT_RELAXED_WIDE_SMALL_GATE_UP :
						     KERNEL_EXPERT_RELAXED_WIDE_GATE_UP) :
					    (is_small ?
						     KERNEL_EXPERT_RELAXED_WIDE_SMALL_DOWN :
						     KERNEL_EXPERT_RELAXED_WIDE_DOWN);
			    expect(strcmp(new_kernel.kernel_file_name,
					  KERNEL_FILE_EXPERT_RELAXED) == 0 &&
					   strcmp(new_kernel.function_name,
						  expected) == 0,
				   "every CMP routed expert product selects its relaxed form");
			    if (is_wide &&
				reference->row_count <
					EXPERT_ORDERED_MINIMUM_ROWS) {
				    expect_expert_tile_conversion(
					    old_launch, new_launch,
					    reference->row_count, gate_up);
				    extra++;
				    const struct kernel_launch *const small =
					    &after->launches[index + extra];
				    char small_name[NAME_CAPACITY];
				    name_of(card, small, small_name);
				    expect(strcmp(small_name,
						  gate_up ?
							  KERNEL_EXPERT_RELAXED_WIDE_SMALL_GATE_UP :
							  KERNEL_EXPERT_RELAXED_WIDE_SMALL_DOWN) ==
							   0 &&
						   small->argument_byte_count ==
							   new_launch
								   ->argument_byte_count &&
						   memcmp(small->argument,
							  new_launch->argument,
							  new_launch
								  ->argument_byte_count) ==
							   0,
					   "the relaxed product has its small companion with the same record");
			    } else {
				    expect(old_launch->argument_byte_count ==
							   new_launch
								   ->argument_byte_count &&
						   memcmp(old_launch->argument,
							  new_launch->argument,
							  old_launch
								  ->argument_byte_count) ==
							   0,
					   "same-geometry expert products keep every record byte");
			    }
			    changed++;
			    continue;
		    }
		    const bool projection =
			    strstr(old_kernel.function_name,
				   KERNEL_ATTENTION_PROJECTION_STEM) != NULL;
		    const bool shared_join =
			    strstr(old_kernel.function_name,
				   KERNEL_EXPERT_SHARED_JOIN_STEM) != NULL;
		    struct trunk_linear_quantized_arguments unfused_product = {
			    0
		    };
		    if (is_cmp && (projection || shared_join)) {
			    extra++;
			    expect(index + extra < after->launch_count,
				   "an unfolded product retains its post or join");
			    if (index + extra >= after->launch_count) {
				    break;
			    }
			    expect_unfused_tail(old_launch,
						&after->launches[index + extra],
						card, projection,
						&unfused_product);
		    }
		    const bool quantized =
			    projection || shared_join ||
			    strstr(old_kernel.function_name,
				   "trunk_linear_quantized") != NULL ||
			    strstr(old_kernel.function_name,
				   "trunk_ordered_wide_quantized") != NULL;
		    const bool bfloat16 =
			    strstr(old_kernel.function_name,
				   "trunk_linear_bfloat16") != NULL ||
			    strstr(old_kernel.function_name,
				   "trunk_ordered_wide_bfloat16") != NULL;
		    if (!is_cmp || (!quantized && !bfloat16)) {
			    expect(memcmp(old_launch, new_launch,
					  sizeof(*old_launch)) == 0,
				   "relaxed selection preserves every non-product and PRO launch byte");
			    continue;
		    }
		    const bool is_wide = reference->row_count > ROWS;
		    const char *const expected_name =
			    quantized ?
				    (is_wide ?
					     KERNEL_RELAXED_WIDE_QUANTIZED_SMALL :
					     KERNEL_RELAXED_DECODE_QUANTIZED) :
				    (is_wide ?
					     KERNEL_RELAXED_WIDE_BFLOAT16_SMALL :
					     KERNEL_RELAXED_DECODE_BFLOAT16);
		    expect(strcmp(new_kernel.kernel_file_name,
				  KERNEL_FILE_TRUNK_RELAXED) == 0 &&
				   strcmp(new_kernel.function_name,
					  expected_name) == 0,
			   "every CMP trunk product site selects the relaxed form");
		    if (quantized) {
			    static struct trunk_linear_quantized_arguments
				    old_record;
			    static struct trunk_relaxed_quantized_arguments
				    new_record;
			    memcpy(&new_record, new_launch->argument,
				   sizeof(new_record));
			    if (is_wide) {
				    expect(memcmp(old_launch->argument,
						  new_launch->argument,
						  sizeof(new_record)) == 0,
					   "wide quantized products retain every record byte");
			    } else {
				    if (projection || shared_join) {
					    old_record = unfused_product;
				    } else {
					    memcpy(&old_record,
						   old_launch->argument,
						   sizeof(old_record));
				    }
				    expect(memcmp(old_record.weights_offset,
						  new_record.weights_offset,
						  sizeof(old_record
								 .weights_offset)) ==
							   0 &&
						   memcmp(old_record
								  .output_offset,
							  new_record
								  .output_offset,
							  sizeof(old_record
									 .output_offset)) ==
							   0 &&
						   memcmp(old_record.output_size,
							  new_record.output_size,
							  sizeof(old_record
									 .output_size)) ==
							   0 &&
						   old_record.operands_offset ==
							   new_record
								   .operands_offset &&
						   old_record.input_size ==
							   new_record.input_size &&
						   old_record.row_count ==
							   new_record.row_count,
					   "relaxed decode reads the original quantizer's operands and writes the same rows");
			    }
		    } else {
			    static struct trunk_linear_bfloat16_arguments
				    old_record;
			    static struct trunk_relaxed_bfloat16_arguments
				    new_record,
				    wide_record;
			    memcpy(&new_record, new_launch->argument,
				   sizeof(new_record));
			    if (is_wide) {
				    memcpy(&wide_record, old_launch->argument,
					   sizeof(wide_record));
				    wide_record.fold_reference =
					    new_record.fold_reference;
				    expect(memcmp(&wide_record, &new_record,
						  sizeof(new_record)) == 0,
					   "wide grouped BF16 records differ only in their checked fold reference");
			    } else {
				    memcpy(&old_record, old_launch->argument,
					   sizeof(old_record));
				    expect(memcmp(&old_record, &new_record,
						  offsetof(
							  struct trunk_linear_bfloat16_arguments,
							  output_size)) == 0 &&
						   old_record.output_size ==
							   new_record
								   .output_size &&
						   old_record.input_size ==
							   new_record.input_size &&
						   old_record.row_count ==
							   new_record.row_count &&
						   old_record.segment_count ==
							   new_record
								   .segment_count,
					   "relaxed grouped BF16 keeps its weights, values, outputs and segmentation");
			    }
		    }
		    changed++;
	    }
    }
    expect(changed != 0, "the prepared relaxed program contains CMP trunk products");
}

static void test_relaxed_programs(const struct model_operations *model,
				  const struct pack_record *pack,
				  struct device_card **cards,
				  struct card_description_record *descriptions)
{
	static const uint32_t rows_tried[] = { 1,
					       ROWS,
					       64,
					       256,
					       EXPERT_ORDERED_MINIMUM_ROWS - 1u,
					       EXPERT_ORDERED_MINIMUM_ROWS,
					       WIDE_CAPACITY };
	context_text = "relaxed prepared program wiring";
	struct model_plan_record *const plan = plan_of(
		pack, &placements[0], LINEAR_ORDER_REFERENCE, WIDE_CAPACITY);
	if (plan == NULL) {
		return;
	}
    for (uint32_t layer = 0; layer < LAYERS; layer++) {
        plan->expert_folds_admitted[layer] = true;
	for (uint32_t tensor = 0; tensor < V41_DATA_LAYER_TENSOR_COUNT;
	     tensor++) {
		plan->layer_tensors[layer][tensor].fold_reference =
			RELAXED_FOLD_REFERENCE;
	}
    }
    cards_open(cards, descriptions);
    for (uint32_t row_index = 0; row_index < COUNT_OF(rows_tried); row_index++) {
        const uint32_t rows = rows_tried[row_index];
	const uint32_t first_kind = rows <= ROWS ? PROGRAM_THIN : PROGRAM_WIDE;
	const uint32_t last_kind = rows <= ROWS ? PROGRAM_THIN : PROGRAM_FULL;
	for (uint32_t kind = first_kind; kind <= last_kind; kind++) {
		if (kind == PROGRAM_CONE) {
			continue;
		}
		struct execution_program *reference = NULL, *relaxed = NULL;
		plan->limits.cmp_order = CMP_ORDER_REFERENCE;
		expect_status(model->program_prepare(
				      plan, &test_host_memory_backend, cards,
				      descriptions, (enum program_kind)kind,
				      rows, &reference),
			      STATUS_OK,
			      "the reference comparison program is prepared");
		plan->limits.cmp_order = CMP_ORDER_RELAXED;
		expect_status(model->program_prepare(
				      plan, &test_host_memory_backend, cards,
				      descriptions, (enum program_kind)kind,
				      rows, &relaxed),
			      STATUS_OK,
			      "the relaxed comparison program is prepared");
		if (reference != NULL && relaxed != NULL) {
			expect_relaxed_program(reference, relaxed,
					       descriptions);
		}
            model->program_destroy(reference);
            model->program_destroy(relaxed);
        }
    }
    model->plan_destroy(plan);
}

/* ---- interleaved layouts ---- */

/* Layouts that visit cards twice a step (tmp/rnd/20261026: codex_parallel_v1/REPORT_P1_pp_placement.md's W11, I12
 * and R11, codex_build_v1/REPORT_B7_dspark_layout.md's recommended layout), and the production layout D, the
 * embedding on card 0: each layer's card, the head's, the pool and key copies of layer 20 behind its front, its leases
 * (of thin and full programs) and the layers of the visit each lease begins at and of the one it last reaches. */
struct interleaved_layout {
    const char *what;
    uint8_t layer_cards[LAYERS];
    uint32_t head_card, front_copies, lease_count;
    uint32_t leased_layers[TEST_INTERLEAVED_LEASES][TEST_LEASE_ENDS];
};
static const struct interleaved_layout interleaved_layouts[] = {
    {"D: 0-5 | 6-11 | 24-31 | 32-39 head | PRO 12-23",
     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
      3, 3, 3}, 3, 4, 0, {{0, 0}, {0, 0}}},
    {"W11: 0-2 25-28 | 3-6 29-32 | 7-9 33-36 | 10-13 37-39 head | PRO 14-24",
     {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
      3, 3, 3}, 3, 7, 0, {{0, 0}, {0, 0}}},
    {"B7: 0-2 21-24 | 3-6 25-27 29 | 7-9 28 31-32 36 | 10-13 30 33-35 | PRO 14-20 37-39 head",
     {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 1, 1, 1, 2, 1, 3, 2, 2, 3, 3, 3, 2,
      4, 4, 4}, 4, 6, 2, {{28, 31}, {28, 36}}},
    {"I12: 0-2 24-27 | 3-5 28-30 head | 6-8 31-34 | 9-11 35-39 | PRO 12-23",
     {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3,
      3, 3, 3}, 1, 8, 0, {{0, 0}, {0, 0}}},
    {"R11: 0-3 27-29 | 4-7 30-33 | 8-11 34-36 | 12-15 37-39 head | PRO 16-26",
     {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
      3, 3, 3}, 3, 7, 0, {{0, 0}, {0, 0}}},
    {"P6: 0-2 24-27 | 3-5 28-31 | 6-8 32-35 | 9-11 36-39 head | PRO 12-23",
     {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3,
      3, 3, 3}, 3, 8, 0, {{0, 0}, {0, 0}}},
    /* B7 with 28 on card 0 and 24 on card 1: the candidates go up at card 0's visit of 21 to 23, which reads none */
    {"0-2 21-23 28 | 3-6 24-27 29 | 7-9 31-32 36 | 10-13 30 33-35 | PRO 14-20 37-39 head",
     {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 1, 1, 1, 1, 0, 1, 3, 2, 2, 3, 3, 3, 2,
      4, 4, 4}, 4, 7, 2, {{21, 28}, {32, 36}}},
};

static struct model_plan_record *
interleaved_plan_of(const struct pack_record *pack,
		    const struct interleaved_layout *placed,
		    uint32_t row_capacity, uint32_t context_limit)
{
    static uint32_t card_of_unit[UNIT_COUNT];
    struct load_layout *layout = NULL;
    struct model_plan_record *plan = NULL;
    const struct model_limits limits = { context_limit,
					 row_capacity,
					 SESSIONS,
					 STATE_WIDTH_16,
					 LINEAR_PROFILE_PRECISION,
					 LINEAR_ORDER_FIXED,
					 CMP_ORDER_REFERENCE,
					 0,
					 0,
					 0 };
    test_model_v41_five_cards(card_of_unit, UNIT_COUNT, 0);
    card_of_unit[TEST_MODEL_V41_HEAD_UNIT] = placed->head_card;
    for (uint32_t layer = 0; layer < LAYERS; layer++) {
        for (uint32_t unit = 0; unit < TEST_MODEL_V41_UNITS_PER_LAYER; unit++) {
            card_of_unit[TEST_MODEL_V41_LAYER_UNIT(layer) + unit] = placed->layer_cards[layer];
        }
    }
    expect_status(load_layout_create(pack, card_of_unit, UNIT_COUNT, CARDS,
				     &layout),
		  STATUS_OK, "the interleaved placement is laid out");
    expect_status(layout == NULL ?
			  STATUS_INVALID_ARGUMENT :
			  v41_plan_create(pack, layout, &limits, &plan),
		  STATUS_OK, "its plan is made");
    load_layout_destroy(layout);
    return plan;
}

/* Whether each visit of the program (model_seam.h) that holds a layer begins with its step segment, a rotation step,
 * and no other segment is one. */
static bool visits_begin_with_rotation(const struct execution_program *program)
{
	const struct segment_record *const segments = program->segments;
	char name[NAME_CAPACITY];
	bool holds = true;
	for (uint32_t first = 0, last = 0; first < program->segment_count;
	     first = ++last) {
		bool has_layer = segments[first].layer != SEGMENT_NO_LAYER;
		while (last + 1u < program->segment_count &&
		       segments[last + 1u].card == segments[first].card) {
			last++;
			has_layer = has_layer ||
				    segments[last].layer != SEGMENT_NO_LAYER;
		}
		for (uint32_t segment = first; segment <= last; segment++) {
			const struct segment_record *const own =
				&segments[segment];
			name_of(own->card,
				&program->cards[own->card]
					 .launches[own->first_launch],
				name);
			if (program->kind == PROGRAM_THIN &&
			    strcmp(name, KERNEL_FILL_WORDS) == 0) {
				holds = holds && own->launch_count > 1u;
				if (own->launch_count > 1u) {
					name_of(own->card,
						&program->cards[own->card].launches
							 [own->first_launch +
							  1u],
						name);
				}
			}
			const bool is_rotation =
				own->layer == SEGMENT_NO_LAYER &&
				strcmp(name, KERNEL_ATTENTION_ROTATION) == 0;
			holds = holds &&
				is_rotation == (segment == first && has_layer);
		}
    }
    return holds;
}

/* Whether two segments of a program lie in one visit (model_seam.h): every segment from the one to the other on one
 * card. */
static bool is_one_visit(const struct execution_program *program,
			 uint32_t segment, uint32_t other)
{
    const uint32_t first = segment < other ? segment : other, last = segment < other ? other : segment;
    bool holds = last < program->segment_count;
    for (uint32_t index = first; index < last && holds; index++) {
        holds = program->segments[index + 1u].card == program->segments[first].card;
    }
    return holds;
}

/* Each interleaved layout's programs: thin ones of 1 and ROWS rows, a wide and a full one of the least wide rows in a
 * plan of WIDE_CAPACITY rows: their visits, layer 20's copies behind its front and their leases; a step driver of them
 * all (see the top). */
static void
test_interleaved_programs(const struct model_operations *model,
			  const struct pack_record *pack,
			  struct device_card **cards,
			  struct card_description_record *descriptions,
			  const struct engram_record *engram)
{
    static const struct {
	    enum program_kind kind;
	    uint32_t rows;
    } made_programs[] = { { PROGRAM_THIN, 1u },
			  { PROGRAM_THIN, ROWS },
			  { PROGRAM_WIDE, MODEL_WIDE_LEAST_ROWS },
			  { PROGRAM_FULL, MODEL_WIDE_LEAST_ROWS } };
    const uint32_t per_context =
	    COUNT_OF(interleaved_layouts) * (CMP_ORDER_RELAXED + 1u);
    for (uint32_t index = 0; index < per_context * TEST_CONTEXTS; index++) {
        const struct interleaved_layout *const placed = &interleaved_layouts[index % COUNT_OF(interleaved_layouts)];
	const uint32_t context_limit = index < per_context ? CONTEXT :
							     MODEL_CONTEXT_MOST;
	struct model_plan_record *const plan =
		interleaved_plan_of(pack, placed, WIDE_CAPACITY, context_limit);
	struct execution_program *programs[COUNT_OF(made_programs)] = { NULL };
	bool is_made = plan != NULL;
        if (plan != NULL) {
		plan->limits.cmp_order = (enum cmp_order_code)(
			index % per_context / COUNT_OF(interleaved_layouts));
		for (uint32_t layer = 0; layer < LAYERS; layer++) {
			for (uint32_t tensor = 0;
			     tensor < V41_DATA_LAYER_TENSOR_COUNT; tensor++) {
				plan->layer_tensors[layer][tensor]
					.fold_reference =
					RELAXED_FOLD_REFERENCE;
			}
	    }
        }
        context_text = placed->what;
        cards_open(cards, descriptions);
        for (uint32_t made = 0; made < COUNT_OF(made_programs) && plan != NULL; made++) {
		expect_status(model->program_prepare(
				      plan, &test_host_memory_backend, cards,
				      descriptions, made_programs[made].kind,
				      made_programs[made].rows,
				      &programs[made]),
			      STATUS_OK, "the program is made");
		const struct execution_program *const program = programs[made];
		is_made = is_made && program != NULL;
		if (program == NULL) {
			continue;
		}
		const bool is_wide = program->kind == PROGRAM_WIDE;
		uint32_t front_copies = 0, wrong_leases = 0;
		for (uint32_t hand_off = 0; hand_off < program->hand_off_count;
		     hand_off++) {
			const struct hand_off_record *const own =
				&program->hand_offs[hand_off];
			front_copies +=
				own->entry_ratio != 0 &&
						own->from_card ==
							plan->layer_card
								[MODEL_DECODER_FIRST_LAYER] &&
						own->after_segment + 1u ==
							program->front_segment_count ?
					1u :
					0u;
		}
            for (uint32_t lease = 0; lease < program->lease_count && lease < TEST_INTERLEAVED_LEASES; lease++) {
		    const struct visit_lease *const own =
			    &program->leases[lease];
		    const uint32_t
			    from = placed->leased_layers[lease][0],
			    to = placed->leased_layers[lease]
						      [TEST_LEASE_ENDS - 1u];
		    wrong_leases +=
			    own->card == plan->layer_card[from] &&
					    is_one_visit(
						    program, own->first_segment,
						    segment_of_layer(program,
								     from)) &&
					    own->last_segment ==
						    segment_of_layer(program,
								     to) ?
				    0u :
				    1u;
            }
            expect(visits_begin_with_rotation(program),
                   "each visit that holds a layer begins with its step segment, and no other segment is one");
            expect(front_copies == placed->front_copies,
                   "layer 20's pool and keys go behind its front to every card that reads them");
            expect(program->lease_count == (is_wide ? 0 : placed->lease_count) && wrong_leases == 0,
                   "the leases of the rows a card's later visit reads, from the visit they come in at: card 2's of B7, "
                   "card 0's and card 2's of the last, none in a wide program");
        }
	struct step_driver *driver = NULL;
	expect_status(!is_made ?
			      STATUS_INVALID_ARGUMENT :
			      step_driver_create(
				      &test_host_memory_backend, cards,
				      (const struct execution_program *const *)
					      programs,
				      COUNT_OF(made_programs), engram,
				      WAIT_LIMIT, &driver),
		      STATUS_OK,
		      "a step driver takes the thin, wide and full programs");
	step_driver_destroy(driver);
	for (uint32_t made = 0; made < COUNT_OF(made_programs); made++) {
            model->program_destroy(programs[made]);
        }
        model->plan_destroy(plan);
    }
}

/* Isolate the busiest index layer with embedding and head on card 0, so another layer's launch slack cannot hide
 * an undersized allocation. The pack still contains every layer and expert; only their placement changes. */
static struct model_plan_record *deep_plan_of(const struct pack_record *pack,
					      uint32_t isolated_layer)
{
    uint32_t card_of_unit[UNIT_COUNT];
    struct load_layout *layout = NULL;
    struct model_plan_record *plan = NULL;
    const struct model_limits limits = { MODEL_CONTEXT_MOST,
					 MODEL_WIDE_MOST_ROWS,
					 SESSIONS,
					 STATE_WIDTH_16,
					 LINEAR_PROFILE_PRECISION,
					 LINEAR_ORDER_REFERENCE,
					 CMP_ORDER_REFERENCE,
					 0,
					 0,
					 0 };
    test_model_v41_five_cards(card_of_unit, 0, 0);
    card_of_unit[TEST_MODEL_V41_HEAD_UNIT] = 0;
    for (uint32_t layer = 0; layer < LAYERS; layer++) {
        const uint32_t card = layer == isolated_layer ? 0u : layer < LAYERS_PER_CARD ? 1u : layer / LAYERS_PER_CARD;
        for (uint32_t unit = 0; unit < TEST_MODEL_V41_UNITS_PER_LAYER; unit++) {
            card_of_unit[TEST_MODEL_V41_LAYER_UNIT(layer) + unit] = card;
        }
    }
    expect_status(load_layout_create(pack, card_of_unit, UNIT_COUNT, CARDS,
				     &layout),
		  STATUS_OK, "the isolated-layer placement is laid out");
    expect_status(layout == NULL ?
			  STATUS_INVALID_ARGUMENT :
			  v41_plan_create(pack, layout, &limits, &plan),
		  STATUS_OK, "the deep plan reserves index scratch");
    load_layout_destroy(layout);
    return plan;
}

static void expect_deep_split_program(const struct model_plan_record *plan,
				      const struct execution_program *program)
{
	static const char *const stages[] = { KERNEL_INDEX_SPLIT_HISTOGRAM_HIGH,
					      KERNEL_INDEX_SPLIT_CUT_HIGH,
					      KERNEL_INDEX_SPLIT_HISTOGRAM_LOW,
					      KERNEL_INDEX_SPLIT_CUT_LOW,
					      KERNEL_INDEX_SPLIT_OUTPUT };
	uint32_t ordinary = 0, candidates = 0;
	for (uint32_t card = 0; card < program->card_count; card++) {
		const struct program_card_record *const prepared =
			&program->cards[card];
		const struct v41_card *const planned = &plan->cards[card];
		const uint64_t scores =
			planned->array_offset[V41_DATA_ARRAY_INDEX_SCORES];
		uint64_t allocation_end = planned->arena_byte_count;
		for (uint32_t array = 0; array < V41_DATA_ARRAY_COUNT;
		     array++) {
			const uint64_t offset = planned->array_offset[array];
			allocation_end =
				offset > scores && offset < allocation_end ?
					offset :
					allocation_end;
		}
		for (uint32_t index = 0; index < prepared->launch_count;
		     index++) {
			char name[NAME_CAPACITY];
			name_of(card, &prepared->launches[index], name);
			if (strcmp(name, stages[0]) != 0) {
				continue;
			}
			const struct kernel_launch *const first =
				&prepared->launches[index];
			struct index_split_arguments record;
			memcpy(&record, first->argument, sizeof(record));
			const bool ratio_valid =
				record.ratio == MODEL_DIRECT_RATIO ||
				record.ratio == MODEL_POOLED_RATIO;
			expect(ratio_valid,
			       "the split record has a supported score ratio");
			if (!ratio_valid) {
				continue;
			}
			const uint32_t rows = first->grid[0];
			const uint64_t score_bytes =
				(uint64_t)rows * MODEL_CONTEXT_MOST /
				record.ratio * INDEX_SCORE_BYTES;
			const uint64_t scratch_bytes =
				(uint64_t)rows * INDEX_SPLIT_ROW_BYTES;
			expect(record.context_token_limit ==
					       MODEL_CONTEXT_MOST &&
				       record.scores_offset == scores &&
				       record.scratch_offset ==
					       scores + score_bytes &&
				       record.scratch_offset <=
					       allocation_end &&
				       scratch_bytes <=
					       allocation_end -
						       record.scratch_offset,
			       "each split chain's scratch fits the planned score allocation");
			uint32_t first_row = 0, layer = LAYERS;
			for (uint32_t view = 0; view < prepared->view_count;
			     view++) {
				if (prepared->views[view].block_offset ==
				    record.step_offset) {
					first_row =
						prepared->views[view].first_row;
				}
			}
			for (uint32_t segment = 0;
			     segment < program->segment_count; segment++) {
				const struct segment_record *const owner =
					&program->segments[segment];
				if (owner->card == card &&
				    index >= owner->first_launch &&
				    index - owner->first_launch <
					    owner->launch_count) {
					layer = owner->layer;
				}
			}
			expect(layer < LAYERS,
			       "the split chain belongs to a layer segment");
			if (layer < LAYERS) {
				const uint64_t output =
					record.is_block != 0 ?
						planned->array_offset
								[V41_DATA_ARRAY_CANDIDATES] +
							(uint64_t)first_row *
								MODEL_CANDIDATES_ROW_BYTES :
						planned->selection_offset[layer] +
							(uint64_t)first_row *
								MODEL_SELECTION_ROW_BYTES;
				expect(record.output_offset == output &&
					       record.candidates_offset ==
						       INDEX_NO_CANDIDATES,
				       "split output preserves layer and slab ownership");
			}
			expect(index + INDEX_SPLIT_LAUNCH_COUNT <=
				       prepared->launch_count,
			       "the complete split chain fits the program launch allocation");
			for (uint32_t pass = 0;
			     pass < INDEX_SPLIT_LAUNCH_COUNT &&
			     index + pass < prepared->launch_count;
			     pass++) {
				const struct kernel_launch *const launch =
					&prepared->launches[index + pass];
				name_of(card, launch, name);
				expect(strcmp(name, stages[pass]) == 0 &&
					       launch->argument_byte_count ==
						       sizeof(record) &&
					       memcmp(launch->argument, &record,
						      sizeof(record)) == 0,
				       "all five stages occur consecutively with the same record");
			}
			ordinary += record.is_block == 0 ? 1u : 0u;
			candidates += record.is_block != 0 ? 1u : 0u;
		}
    }
    const uint32_t slabs = (program->row_count + ROWS - 1u) / ROWS;
    expect(ordinary == FULL_INDEX_SOURCES * slabs && candidates == slabs,
           "every full source and candidate source is split once per slab; readers retain legacy");
}

static void
test_deep_split_programs(const struct model_operations *model,
			 const struct pack_record *pack,
			 struct device_card **cards,
			 struct card_description_record *descriptions)
{
	static const uint32_t isolated_layers[] = { REVISITING_LAYER,
						    MODEL_CANDIDATE_SOURCE };
	static const uint32_t row_counts[] = { 1, ROWS, MODEL_WIDE_MOST_ROWS };
	context_text = "deep split selection program integration";
	for (uint32_t which = 0; which < COUNT_OF(isolated_layers); which++) {
		struct model_plan_record *const plan =
			deep_plan_of(pack, isolated_layers[which]);
		cards_open(cards, descriptions);
		for (uint32_t row_index = 0;
		     plan != NULL && row_index < COUNT_OF(row_counts);
		     row_index++) {
			struct execution_program *program = NULL;
			const uint32_t rows = row_counts[row_index];
			const enum program_kind kind =
				rows <= ROWS ? PROGRAM_THIN : PROGRAM_FULL;
			expect_status(
				model->program_prepare(
					plan, &test_host_memory_backend, cards,
					descriptions, kind, rows, &program),
				STATUS_OK,
				"the deep thin or largest full program fits every card's launch capacity");
			if (program != NULL) {
				expect_deep_split_program(plan, program);
			}
			model->program_destroy(program);
		}
		model->plan_destroy(plan);
	}
}

/* Run the ordinary byte-range flow checker on draft-enabled FULL programs, including the second piece. */
static void
test_draft_full_programs(const struct model_operations *model,
			 struct device_card **cards,
			 struct card_description_record *descriptions)
{
    static uint32_t placement[TEST_MODEL_V41_DRAFT_UNIT_COUNT];
    static const uint32_t classes[] = { ARITHMETIC_CLASS_SM_80,
					ARITHMETIC_CLASS_SM_120,
					ARITHMETIC_CLASS_SM_120_NATIVE };
    static const uint32_t rows[] = { MODEL_WIDE_LEAST_ROWS,
				     MODEL_WIDE_LEAST_ROWS + 1u, WIDE_CAPACITY,
				     PIECES_ROWS };
    struct pack_record *pack = NULL;
    struct load_layout *layout = NULL;
    context_text = "draft-enabled full programs retain the wide-small data flow";
    expect_status(test_model_v41_draft_pack_open(
			  (struct test_model_v41_change){
				  TEST_MODEL_V41_CHANGE_NONE, NULL, 0 },
			  &pack),
		  STATUS_OK, "the draft pack opens");
    test_model_v41_five_cards(placement, 0, 0);
    test_model_v41_draft_place(placement, PRO_CARD);
    expect_status(load_layout_create(pack, placement, COUNT_OF(placement),
				     CARDS, &layout),
		  STATUS_OK, "the draft fill placement is laid out");
    for (uint32_t order = 0; model != NULL && layout != NULL && order < ORDERS; order++) {
	    struct model_plan_record *plan = NULL;
	    const struct model_limits limits = { CONTEXT,
						 FULL_CAPACITY,
						 SESSIONS,
						 STATE_WIDTH_16,
						 LINEAR_PROFILE_PRECISION,
						 (enum linear_order_code)order,
						 CMP_ORDER_REFERENCE,
						 MODEL_DRAFT_BLOCK,
						 0,
						 0 };
	    expect_status(model->plan_create(pack, layout, &limits, &plan),
			  STATUS_OK,
			  "the combined draft and wide plan is made");
	    for (uint32_t class_index = 0;
		 plan != NULL && class_index < COUNT_OF(classes);
		 class_index++) {
		    cards_open(cards, descriptions);
		    const uint32_t physical_class =
			    classes[class_index] == ARITHMETIC_CLASS_SM_80 ?
				    ARITHMETIC_CLASS_SM_80 :
				    ARITHMETIC_CLASS_SM_120;
		    test_host_memory_backend_set_arithmetic_class(
			    PRO_CARD, physical_class);
		    descriptions[PRO_CARD].arithmetic_class =
			    classes[class_index];
		    for (uint32_t row_index = 0; row_index < COUNT_OF(rows);
			 row_index++) {
			    struct execution_program *full = NULL;
			    expect_status(
				    model->program_prepare(
					    plan, &test_host_memory_backend,
					    cards, descriptions, PROGRAM_FULL,
					    rows[row_index], &full),
				    STATUS_OK,
				    "the combined full program is made");
			    if (full != NULL) {
				    const struct program_case made = {
					    plan, full, descriptions,
					    rows[row_index]
				    };
				    expect(embedding_holds(&made),
					   "draft-enabled full embedding covers the chunk");
				    expect(chunk_layers_wrong(&made) == 0,
					   "all target layers retain wide piece/span/slab launches and byte-range data flow");
			    }
			    model->program_destroy(full);
		    }
        }
        model->plan_destroy(plan);
    }
    load_layout_destroy(layout);
    pack_close(pack);
}

/* The opt-in must replace both fused fronts and retain the explicit FFN router.
 * This validates program construction on the host backend, not device numerics. */
static void test_official_hc(const struct model_operations *model,
			     const struct pack_record *pack,
			     struct device_card **cards,
			     struct card_description_record *descriptions)
{
	context_text = "official HC reference shape";
	struct model_plan_record *const plan =
		plan_of(pack, &placements[0], LINEAR_ORDER_REFERENCE, ROWS);
	if (plan == NULL)
		return;
	plan->limits.official_hc_rows = ROWS;
	cards_open(cards, descriptions);
	struct execution_program *program = NULL;
	const result_status status =
		model->program_prepare(plan, &test_host_memory_backend, cards,
				       descriptions, PROGRAM_THIN, 1, &program);
	expect_status(
		status, STATUS_OK,
		"serial program targets the eight-row reference HC order");
	uint32_t projections = 0, fronts = 0, routers = 0;
	for (uint32_t card = 0; program != NULL && card < plan->card_count;
	     card++) {
		const struct program_card_record *const prepared =
			&program->cards[card];
		for (uint32_t index = 0; index < prepared->launch_count;
		     index++) {
			const struct kernel_launch *const launch =
				&prepared->launches[index];
			char name[NAME_CAPACITY];
			name_of(card, launch, name);
			if (strcmp(name, KERNEL_OFFICIAL_HC_SPLIT) == 0) {
				struct dense_linear_arguments arguments;
				memcpy(&arguments, launch->argument,
				       sizeof(arguments));
				expect(arguments.row_count == ROWS &&
					       launch->grid[1] == 1,
				       "actual serial extent and reference shape preserved");
				projections++;
			}
			fronts += strncmp(name, "site_front",
					  strlen("site_front")) == 0;
			routers += strcmp(name, KERNEL_ROUTER_SELECT) == 0;
		}
	}
	expect(projections == MODEL_LAYERS * OFFICIAL_HC_SITES,
	       "both HC sites of every layer use the opt-in");
	expect(fronts == 0, "no fused legacy front bypasses the opt-in");
	expect(routers == MODEL_LAYERS,
	       "each FFN retains explicit router selection");
	model->program_destroy(program);
	model->plan_destroy(plan);
}

/* Exercise preparation with the real table-only pack: no adapter or sidecar can supply arithmetic. */
static void
test_official_reference(const struct model_operations *model,
			const struct pack_record *pack,
			struct device_card **cards,
			struct card_description_record *descriptions)
{
	context_text = "official serial reference program";
	uint32_t card_of_unit[UNIT_COUNT];
	struct load_layout *layout = NULL;
	struct model_plan_record *plan = NULL;
	const struct model_limits limits = {
		.context_token_limit = OFFICIAL_REFERENCE_CONTEXT_LIMIT,
		.row_capacity = 1,
		.session_capacity = 1,
		.state_width = STATE_WIDTH_16,
		.linear_profile = LINEAR_PROFILE_PRECISION,
		.linear_order = LINEAR_ORDER_REFERENCE,
		.cmp_order = CMP_ORDER_REFERENCE,
		.official_hc_rows = 1,
		.official_reference = 1
	};
	test_model_v41_five_cards(card_of_unit, 0, 0);
	expect_status(load_layout_create(pack, card_of_unit, UNIT_COUNT, CARDS,
					 &layout),
		      STATUS_OK, "reference layout is made");
	if (layout == NULL)
		return;
	expect_status(model->plan_create(pack, layout, &limits, &plan),
		      STATUS_OK, "reference plan owns its arena annex");
	load_layout_destroy(layout);
	if (plan == NULL)
		return;
	cards_open(cards, descriptions);
	descriptions[PRO_CARD].arithmetic_class =
		ARITHMETIC_CLASS_SM_120_NATIVE;
	struct execution_program *program = NULL;
	expect_status(model->program_prepare(plan, &test_host_memory_backend,
					     cards, descriptions, PROGRAM_THIN,
					     1, &program),
		      STATUS_OK, "reference program is prepared");
	struct test_program_view *view = NULL;
	if (program != NULL)
		expect_status(
			test_program_view_create(
				&test_host_memory_backend, cards, descriptions,
				program, MODEL_LAYERS,
				(1ull << MODEL_LAYERS) - 1u, &view),
			STATUS_OK,
			"the ordinary oracle view recognizes every production reference variant");
	expect(view != NULL && view->head_segment != TEST_PROGRAM_VIEW_NONE,
	       "kept final identifies the source head");
	test_program_view_destroy(view);
	uint32_t sites = 0, cores = 0, routers = 0, pools = 0, keys = 0,
		 engrams = 0, heads = 0;
	for (uint32_t card = 0; program != NULL && card < program->card_count;
	     card++) {
		expect(plan->cards[card].official_scratch_offset !=
			       PROGRAM_NOWHERE,
		       "each serial visit owns its fixed scratch range");
		for (uint32_t index = 0;
		     index < program->cards[card].launch_count; index++) {
			const struct kernel_launch *const launch =
				&program->cards[card].launches[index];
			char name[NAME_CAPACITY];
			name_of(card, launch, name);
			sites += strcmp(name, "official_site_finish") == 0;
			cores += strcmp(name, "official_attention") == 0;
			routers += strcmp(name, "official_router_order") == 0;
			pools += strcmp(name, "official_compressed_state") == 0;
			keys += strcmp(name, "official_key_state") == 0;
			engrams += strcmp(name, "official_engram_gate") == 0;
			heads += strcmp(name, KERNEL_OFFICIAL_HEAD) == 0;
			if (strcmp(name, KERNEL_OFFICIAL_HEAD) == 0) {
				/* No cast, temperature transform or alternate softmax may sit
				 * between the source F32 head and the ordinary sampler. */
				struct official_head_arguments head;
				memcpy(&head, launch->argument, sizeof(head));
				const bool has_sampler =
					index + 1u <
					program->cards[card].launch_count;
				expect(has_sampler,
				       "the exact head has a following sampler");
				if (has_sampler) {
					const struct kernel_launch *const
						sampled = launch + 1;
					struct sampler_arguments sampler;
					memcpy(&sampler, sampled->argument,
					       sizeof(sampler));
					name_of(card, sampled, name);
					expect(strcmp(name, KERNEL_SAMPLER) ==
							       0 &&
						       sampler.logits_offset ==
							       head.output_offset &&
						       sampler.vocabulary_size ==
							       head.output_size &&
						       sampler.weights_offset !=
							       sampler.logits_offset,
					       "ordinary sampler reads exact head F32 logits directly and uses separate weights scratch");
					expect(index + 1u ==
						       program->cards[card]
								       .launch_count -
							       1u,
					       "no exact-tier transform follows the ordinary sampler");
					struct sampler_kernels_record kernels = {
						sampled->kernel
					};
					const struct sampler_description ordinary = {
						{ sampler.step_offset,
						  STEP_BLOCK_BYTES_VALUE +
							  STEP_ROW_BYTES_VALUE },
						{ sampler.session_offset,
						  STEP_SESSION_BYTES },
						{ head.output_offset,
						  (uint64_t)head.output_size *
							  sizeof(float) },
						{ sampler.weights_offset,
						  (uint64_t)head.output_size *
							  sizeof(float) },
						{ sampler.result_offset,
						  STEP_RESULT_BYTES +
							  STEP_ROW_RESULT_BYTES },
						head.output_size,
						1,
						1,
						0
					};
					struct kernel_launch expected = { 0 };
					expect_status(
						sampler_prepare_entry(
							&ordinary,
							plan->cards[card]
								.arena_byte_count,
							&kernels, &expected),
						STATUS_OK,
						"ordinary sampler prepares on the exact F32 head range");
					expect(memcmp(&expected, sampled,
						      sizeof(expected)) == 0,
					       "exact and ordinary sampler launches are byte-identical, including reduction geometry");
				}
			}
			expect(strncmp(name, "r_", strlen("r_")) != 0,
			       "no packet kernel is prepared");
		}
	}
	expect(sites == MODEL_LAYERS * OFFICIAL_HC_SITES,
	       "every HC finish uses its source F1 form");
	expect(cores == MODEL_LAYERS,
	       "every core consumes a source BF16 query");
	expect(routers == MODEL_LAYERS,
	       "every router retains source normalization order");
	expect(pools == MODEL_STREAMS && keys == pools,
	       "only source owners write compressed caches");
	expect(engrams == (uint32_t)__builtin_popcountll(
				  V41_DATA_ENGRAM_LAYERS) &&
		       heads == 1,
	       "Engram and head use source shapes");
	model->program_destroy(program);
	model->plan_destroy(plan);
}

int main(void)
{
	struct device_card *cards[CARDS] = { NULL };
	struct card_description_record descriptions[CARDS];
	struct pack_record *pack = NULL;
	struct engram_record *engram = NULL;
	unsigned char *engram_pack = NULL;
	const struct model_operations *const model =
		model_operations_of(V41_DATA_MODEL_NAME);
	context_text = "the model's pack and the cards";
	expect(model != NULL, "the model's name finds its table");
	expect_status(test_model_v41_pack_open(
			      (struct test_model_v41_change){
				      TEST_MODEL_V41_CHANGE_NONE, NULL, 0 },
			      &pack),
		      STATUS_OK, "the model's pack opens from its tables");
	expect_status(test_engram_pack_open(&engram_pack, &engram), STATUS_OK,
		      "the n-gram tables open");
	if (pack != NULL && model != NULL && engram != NULL && failures == 0) {
		test_programs(model, pack, cards, descriptions);
		test_official_hc(model, pack, cards, descriptions);
		test_official_reference(model, pack, cards, descriptions);
		test_wide_programs(model, pack, cards, descriptions);
		test_full_programs(model, pack, cards, descriptions);
		test_relaxed_programs(model, pack, cards, descriptions);
		test_interleaved_programs(model, pack, cards, descriptions,
					  engram);
		test_deep_split_programs(model, pack, cards, descriptions);
	}
    cards_close(cards);
    engram_close(engram);
    free(engram_pack);
    pack_close(pack);
    test_memory_storage_reset();
    test_draft_full_programs(model, cards, descriptions);
    cards_close(cards);
    test_memory_storage_reset();
    if (failures != 0) {
        (void)printf("model_v41_program_test FAILED: %d failures\n", failures);
        return EXIT_FAILURE;
    }
    (void)printf("model_v41_program_test passed\n");
    return EXIT_SUCCESS;
}
