/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* split_overlap_program_test: model_v41_tp2_stage.h's split_overlap on an EXL3 TP4 plan of the BF16 recipe (head and
 * draft shards, TP draft layers, TC MoE, trunk and head MMA, owner sums) on the host memory backend. The wide and full
 * programs of 1,024 rows with split_overlap carry the two half views behind the slabs' and, on every card, three side
 * runs a split layer (each run's resume and join marks once); without it they carry no branch. The step driver takes
 * the thin, wide and full programs with split_overlap, with the owner sums and with the copy-engine sums, and with
 * its pipeline (mode 5: the attention in halves, a layer's last sums beside the next layer's first half, as a side
 * run in place of the copies' own runs on the main run; every side run resumed and joined once), and its 2,048-row
 * programs (halves of 1,024 rows, a wide launch each), and without TP draft layers (the window kernel then prepared
 * for chunk_core's window writes on every card). */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/exl3_tc_records.h"
#include "anemone/front_end_seam.h"
#include "anemone/step.h"
#include "host_memory_backend.h"
#include "model_v41_pack.h"
#include "model_v41_tp2_model.h"

#define SPLIT_CARDS TP4_RANKS
#define SPLIT_CARD_BYTES 4096u
#define SPLIT_REPORTED_BYTES (1ull << 40u)
#define SPLIT_SESSIONS 8u
#define SPLIT_THIN_ROWS 6u
#define SPLIT_WIDE_ROWS 1024u
#define SPLIT_LADDER_1536_ROWS 1536u /* --prefill_ladder_1536's programs */
#define SPLIT_WIDEST_ROWS 2048u /* V41_DATA_TP_SPLIT_MOST_ROWS */
#define SPLIT_PROGRAMS 3u
#define SPLIT_TAIL_ROWS 128u /* --prefill_tail_rows: a full program of two 64-row tail chunks */
#define SPLIT_LONG_TAIL_ROWS 320u /* --prefill_tail_rows=320: and one of five, unsplit */
#define SPLIT_WIDEST_TAIL_ROWS 448u /* --prefill_tail_rows=448: and one of seven, unsplit */
#define SPLIT_THIN_TAIL_ROWS 40u /* --prefill_thin_tail: a full program of 40 rows in a thin step's forms */
#define SPLIT_THIN_TAIL_LEAST 16u
#define SPLIT_WAIT_NS 1000000000ull
#define SPLIT_RUNS_A_LAYER 3u
#define SPLIT_MARKS (LAUNCH_SIDE_MORE + 1u) /* a count for each branch */

static uint32_t failures;
static struct device_card
	*cards[SPLIT_CARDS]; /* the host cards, opened once for every model */
static bool draft_layers = true; /* false: the draft on the result card alone (its window kernel there alone) */
static uint32_t rotation; /* --exl3_expert_rotation of the next programs_prepare */
static bool track_c_knobs; /* --tp_gather_balance=1 and --tp_pool_chunk=1 for the next programs_prepare */
static bool balance_alone; /* with track_c_knobs: --tp_gather_balance=1 alone */
static bool balance_wide; /* with track_c_knobs and balance_alone: --tp_gather_balance=2 (the wide programs') */
static uint32_t full_split; /* --tp_full_split_overlap of the next programs_prepare */
static uint32_t tail_rows; /* --prefill_tail_rows of the next programs_prepare */
static bool tail_overlay; /* --prefill_tail_overlay=1 for the next programs_prepare */
static bool tail_sets; /* --prefill_tail_sets=1 for the next programs_prepare */
static bool thin_tail; /* --prefill_thin_tail=1 for the next programs_prepare */
static uint32_t chunk_rows; /* 0, or the full program's rows below the model's wide_rows */
static bool pool_chunk_alone; /* --tp_pool_chunk=1 without gather_balance for the next programs_prepare */
static bool pool_gather; /* --tp_pool_gather=1 for the next programs_prepare */
static uint32_t cone_behind; /* --prefill_dead_halves=1's full_cone_behind for the next programs_prepare */
static bool router_first_whole; /* --tp_router_first_whole=1 for the next programs_prepare */
static bool rows_trimmed; /* --prefill_dead_halves=2 for the next programs_prepare */
static bool queries_piece; /* --tp_queries_piece=1 for the next programs_prepare */
static bool index_span; /* --tp_index_span=1 for the next programs_prepare */
static bool balance_weighted; /* --tp_gather_balance=3 for the next programs_prepare */
static uint32_t skinny_bf16; /* --dense_skinny_bf16 for the next programs_prepare */
static bool core_compact_halves; /* --tp_core_compact_halves=1 for the next programs_prepare */
/* --tp_core_compact_halves=1: a split layer's two halves' tp2_join copies of the core values into wo_a's rows less */
#define CORE_COMPACT_LAYER_LAUNCHES 2u
/* --tp_queries_piece=1: a 1,024-row half's span queries less its piece's, and wq_b's heads copy into the full rows */
#define QUERIES_PIECE_HALF_LAUNCHES \
	(SPLIT_WIDEST_ROWS / 2u / V41_DATA_SPAN_ROWS)
/* --tp_router_first_whole=1: a split layer's first half's router gather (copies to three peers, READY, DONE) less */
#define FIRST_WHOLE_LAUNCHES (SPLIT_CARDS - 1u + POOL_GATHER_BARRIERS)
#define DEAD_HALVES_MARGIN 127u /* --prefill_dead_halves=1 at --prefill_margin's default */
/* --tp_pool_gather=1: a pooled source layer's half (layers 2, 8, 14; each half's) adds two ranges' copies to three
 * peers, READY and DONE */
#define POOL_GATHER_POOLED_LAYERS 3u
#define POOL_GATHER_HALVES 2u
#define POOL_GATHER_RANGES 2u   /* wkv's products and wgate's scores */
#define POOL_GATHER_BARRIERS 2u /* READY and DONE */
#define POOL_GATHER_HALF_LAUNCHES (POOL_GATHER_RANGES * (SPLIT_CARDS - 1u) + POOL_GATHER_BARRIERS)

static void expect(bool condition, const char *what)
{
	if (!condition) {
		printf("split_overlap_program_test: %s\n", what);
		failures++;
	}
}

/* Each card's branch marks of a program: side launches, resumes, joins, more side runs. */
static void branches(const struct execution_program *program,
		     uint32_t marks[SPLIT_CARDS][SPLIT_MARKS])
{
	memset(marks, 0, sizeof(uint32_t) * SPLIT_CARDS * SPLIT_MARKS);
	for (uint32_t card = 0; card < program->card_count && card < SPLIT_CARDS; card++)
		for (uint32_t index = 0; index < program->cards[card].launch_count; index++)
			marks[card][program->cards[card].launches[index].branch]++;
}

static bool programs_prepare(uint32_t split, uint32_t exchange, bool chunk_core,
			     uint32_t wide_rows,
			     struct execution_program *programs[SPLIT_PROGRAMS])
{
	struct pack_record *pack = NULL;
	const struct test_model_v41_change change = { .kind = TEST_MODEL_V41_CHANGE_EXL3 };
	if (test_model_v41_draft_pack_open(change, &pack) != STATUS_OK) {
		printf("split_overlap_program_test: the fixture did not open\n");
		return false;
	}
	struct v41_tp2_model_description description = {
		.limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES, wide_rows,
			    SPLIT_SESSIONS, STATE_WIDTH_16,
			    LINEAR_PROFILE_PRECISION, LINEAR_ORDER_FIXED,
			    CMP_ORDER_REFERENCE, MODEL_DRAFT_BLOCK, 0, 0 },
		.tensor_layout = V41_DATA_TP_LAYOUT_QUAD_ALL,
		.wire_format = ORDERED_PUSH_BF16,
		.head_shards = true,
		.draft_shards = true,
		.draft_layers = draft_layers,
		.exl3_fast_products = true,
		.exl3_wide_entry = true,
		.replicated_embedding = true,
		.exl3_tc_moe = true,
		.trunk_mma = true,
		.head_mma = true,
		.ordered_wide = true,
		.wide_exchange = exchange,
		.split_overlap = split,
		.full_split_overlap = full_split,
		.tail_rows = tail_rows,
		.tail_overlay = tail_overlay,
		.tail_sets = tail_sets,
		.thin_tail = thin_tail,
		.chunk_core = chunk_core,
		.exl3_expert_rotation = rotation,
		.gather_balance = balance_weighted ?
					  V41_DATA_GATHER_BALANCE_WEIGHTED :
				  !track_c_knobs ? 0u :
				  balance_wide ? V41_DATA_GATHER_BALANCE_WIDE :
						 V41_DATA_GATHER_BALANCE_ALL,
		.pool_chunk = (track_c_knobs && !balance_alone) ||
			      pool_chunk_alone,
		.pool_gather = pool_gather,
		.full_cone_behind = cone_behind,
		.router_first_whole = router_first_whole,
		.dead_rows_trimmed = rows_trimmed,
		.queries_piece = queries_piece,
		.index_span = index_span,
		.dense_skinny_bf16 = skinny_bf16,
		.core_compact_halves = core_compact_halves
	};
	static struct card_description_record reported[SPLIT_CARDS];
	static bool opened;
	for (uint32_t card = 0; card < SPLIT_CARDS; card++) {
		reported[card].arithmetic_class = ARITHMETIC_CLASS_SM_80;
		reported[card].memory_total_bytes = reported[card].memory_free_bytes = SPLIT_REPORTED_BYTES;
		description.cards[card] = reported[card];
	}
	if (!opened) { /* the host cards once, for both models */
		test_host_memory_backend_reset(SPLIT_CARDS, SPLIT_CARD_BYTES);
		for (uint32_t card = 0; card < SPLIT_CARDS; card++) {
			test_host_memory_backend_set_arithmetic_class(
				card, ARITHMETIC_CLASS_SM_80);
			if (test_host_memory_backend.card_open(
				    card, SPLIT_CARD_BYTES, &cards[card]) !=
			    STATUS_OK) {
				printf("split_overlap_program_test: card %" PRIu32 " did not open\n", card);
				return false;
			}
		}
		opened = true;
	}
	struct v41_tp2_model *model = NULL;
	struct v41_tp2_program_description build = { 0 };
	const result_status created =
		v41_tp2_model_create(pack, &description, &model);
	const result_status described =
		created != STATUS_OK ?
			created :
			v41_tp2_model_program_description(model, &build);
	if (described != STATUS_OK) {
		printf("split_overlap_program_test: the model (split %" PRIu32
		       "): %s / %s\n",
		       split, status_text(created), status_text(described));
		return false;
	}
	const enum program_kind kinds[SPLIT_PROGRAMS] = { PROGRAM_THIN,
							  PROGRAM_WIDE,
							  PROGRAM_FULL };
	const uint32_t rows[SPLIT_PROGRAMS] = { SPLIT_THIN_ROWS, wide_rows, chunk_rows != 0 ? chunk_rows : wide_rows };
	for (uint32_t index = 0; index < SPLIT_PROGRAMS; index++) {
		struct step_companion_program companions = { 0 };
		const result_status prepared = v41_tp_program_prepare(
			&build, &test_host_memory_backend, cards, reported,
			kinds[index], rows[index], &programs[index],
			&companions);
		if (prepared != STATUS_OK) {
			printf("split_overlap_program_test: program %" PRIu32
			       " (split %" PRIu32 "): %s\n",
			       index, split, status_text(prepared));
			return false;
		}
		for (uint32_t card = 0; card < programs[index]->card_count; card++) {
			const uint32_t none = 0; /* no engram tables on the host backend */
			memcpy((void *)(uintptr_t)&programs[index]->cards[card].engram_tables, &none, sizeof(none));
		}
	}
	if (split != 0) {
		struct step_driver *driver = NULL;
		const result_status taken = step_driver_create(
			&test_host_memory_backend, cards,
			(const struct execution_program *const *)programs,
			SPLIT_PROGRAMS, NULL, SPLIT_WAIT_NS, &driver);
		expect(taken == STATUS_OK,
		       "the step driver takes the thin, wide and full programs with halves");
		step_driver_destroy(driver);
	}
	v41_tp2_model_destroy(model);
	pack_close(pack);
	return true;
}

int main(void)
{
	struct execution_program *whole[SPLIT_PROGRAMS] = { 0 },
				 *split[SPLIT_PROGRAMS] = { 0 },
				 *engines[SPLIT_PROGRAMS] = { 0 },
				 *cores[SPLIT_PROGRAMS] = { 0 },
				 *pipelined[SPLIT_PROGRAMS] = { 0 },
				 *widest[SPLIT_PROGRAMS] = { 0 },
				 *drafted[SPLIT_PROGRAMS] = { 0 },
				 *joint[SPLIT_PROGRAMS] = { 0 },
				 *joint_widest[SPLIT_PROGRAMS] = { 0 },
				 *four_way[SPLIT_PROGRAMS] = { 0 },
				 *three_way[SPLIT_PROGRAMS] = { 0 },
				 *balanced[SPLIT_PROGRAMS] = { 0 },
				 *balance_only[SPLIT_PROGRAMS] = { 0 },
				 *balance_wide_only[SPLIT_PROGRAMS] = { 0 },
				 *mixed[SPLIT_PROGRAMS] = { 0 },
				 *down_halves[SPLIT_PROGRAMS] = { 0 },
				 *tail_plain[SPLIT_PROGRAMS] = { 0 },
				 *tail_forms[SPLIT_PROGRAMS] = { 0 };
	/* the copy-engine sums (both sites) with split_overlap: prepared and taken by the driver */
	/* and chunk_core's core over the chunk */
	const uint32_t owners = V41_DATA_TP_EXCHANGE_OWNER_SUMS,
		       rings = V41_DATA_TP_EXCHANGE_CE_RINGS;
	if (!programs_prepare(0, owners, false, SPLIT_WIDE_ROWS, whole) ||
	    !programs_prepare(V41_DATA_SPLIT_OVERLAP, owners, false,
			      SPLIT_WIDE_ROWS, split) ||
	    !programs_prepare(V41_DATA_SPLIT_OVERLAP, rings, false,
			      SPLIT_WIDE_ROWS, engines) ||
	    !programs_prepare(V41_DATA_SPLIT_OVERLAP, rings, true,
			      SPLIT_WIDE_ROWS, cores) ||
	    !programs_prepare(V41_DATA_SPLIT_PIPELINE, rings, true,
			      SPLIT_WIDE_ROWS, pipelined) ||
	    !programs_prepare(V41_DATA_SPLIT_PIPELINE, rings, true,
			      SPLIT_WIDEST_ROWS, widest) ||
	    !programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
			      SPLIT_WIDE_ROWS, joint) ||
	    !programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
			      SPLIT_WIDEST_ROWS, joint_widest)) {
		printf("split_overlap_program_test: the programs did not prepare\n");
		return 1;
	}
	draft_layers = false;
	if (!programs_prepare(V41_DATA_SPLIT_PIPELINE, rings, true,
			      SPLIT_WIDE_ROWS, drafted)) {
		printf("split_overlap_program_test: the programs without TP draft layers did not prepare\n");
		return 1;
	}
	for (uint32_t index = 1; index < SPLIT_PROGRAMS; index++) {
		uint32_t off[SPLIT_CARDS][SPLIT_MARKS], on[SPLIT_CARDS][SPLIT_MARKS];
		branches(whole[index], off);
		branches(split[index], on);
		for (uint32_t card = 0; card < SPLIT_CARDS; card++) {
			const uint32_t slabs =
				SPLIT_WIDE_ROWS / LINEAR_MAXIMUM_ROWS;
			expect(whole[index]->cards[card].view_count == 0 || whole[index]->cards[card].view_count == slabs,
			       "without split_overlap a wide card's views are its slabs");
			expect(split[index]->cards[card].view_count == 0 ||
				       split[index]->cards[card].view_count == slabs + 2u,
			       "with split_overlap a wide card's views are its slabs and the two halves");
			expect(off[card][LAUNCH_SIDE] == 0 &&
				       off[card][LAUNCH_RESUME] == 0 &&
				       off[card][LAUNCH_JOIN] == 0,
			       "without split_overlap no branch");
			expect(on[card][LAUNCH_RESUME] ==
					       on[card][LAUNCH_JOIN] &&
				       on[card][LAUNCH_RESUME] %
						       SPLIT_RUNS_A_LAYER ==
					       0 &&
				       (on[card][LAUNCH_RESUME] == 0) ==
					       (on[card][LAUNCH_SIDE] == 0),
			       "with split_overlap three side runs a split layer, each resumed and joined once");
			printf("program %" PRIu32 " card %" PRIu32
			       ": launches %" PRIu32 " -> %" PRIu32
			       ", split layers %" PRIu32
			       ", side launches %" PRIu32 "\n",
			       index, card,
			       whole[index]->cards[card].launch_count,
			       split[index]->cards[card].launch_count,
			       on[card][LAUNCH_RESUME] / SPLIT_RUNS_A_LAYER,
			       on[card][LAUNCH_SIDE]);
		}
	}
	for (uint32_t index = 1; index < SPLIT_PROGRAMS; index++)
		printf("program %" PRIu32 " card 0 with chunk_core: launches %" PRIu32 " -> %" PRIu32 "\n", index,
		       engines[index]->cards[0].launch_count, cores[index]->cards[0].launch_count);
	for (uint32_t index = 1; index < SPLIT_PROGRAMS; index++) {
		uint32_t on[SPLIT_CARDS][SPLIT_MARKS], piped[SPLIT_CARDS][SPLIT_MARKS];
		branches(cores[index], on);
		branches(pipelined[index], piped);
		for (uint32_t card = 0; card < SPLIT_CARDS; card++) {
			const uint32_t runs = on[card][LAUNCH_RESUME];
			const uint32_t piped_runs = piped[card][LAUNCH_RESUME];
			expect(piped_runs == piped[card][LAUNCH_JOIN] &&
				       piped_runs != 0 &&
				       piped[card][LAUNCH_SIDE_MORE] <
					       on[card][LAUNCH_SIDE_MORE],
			       "the pipeline: each side run resumed and joined once, fewer FFN sums on the main run");
			printf("program %" PRIu32 " card %" PRIu32
			       " pipelined: launches %" PRIu32 " -> %" PRIu32
			       ", side runs %" PRIu32 " -> %" PRIu32
			       ", more side runs %" PRIu32 " -> %" PRIu32 "\n",
			       index, card,
			       cores[index]->cards[card].launch_count,
			       pipelined[index]->cards[card].launch_count, runs,
			       piped_runs, on[card][LAUNCH_SIDE_MORE],
			       piped[card][LAUNCH_SIDE_MORE]);
		}
	}
	for (uint32_t index = 1; index < SPLIT_PROGRAMS; index++) {
		uint32_t marks[SPLIT_CARDS][SPLIT_MARKS];
		branches(widest[index], marks);
		expect(widest[index]->cards[0].view_count ==
				       SPLIT_WIDEST_ROWS / LINEAR_MAXIMUM_ROWS +
					       2u &&
			       marks[0][LAUNCH_RESUME] ==
				       marks[0][LAUNCH_JOIN] &&
			       marks[0][LAUNCH_RESUME] != 0,
		       "2,048 rows: the slabs' views and two halves of 1,024, every side run resumed and joined once");
		printf("program %" PRIu32
		       " of 2,048 rows card 0: launches %" PRIu32
		       ", side runs %" PRIu32 "\n",
		       index, widest[index]->cards[0].launch_count,
		       marks[0][LAUNCH_RESUME]);
	}
	/* the joint mode: as many side runs as the pipeline (the FFN sums of A beside B's shared expert) */
	for (uint32_t index = 1; index < SPLIT_PROGRAMS; index++) {
		uint32_t piped[SPLIT_CARDS][SPLIT_MARKS], joined[SPLIT_CARDS][SPLIT_MARKS], wide[SPLIT_CARDS][SPLIT_MARKS];
		branches(pipelined[index], piped);
		branches(joint[index], joined);
		branches(joint_widest[index], wide);
		for (uint32_t card = 0; card < SPLIT_CARDS; card++) {
			expect(joined[card][LAUNCH_RESUME] ==
					       piped[card][LAUNCH_RESUME] &&
				       joined[card][LAUNCH_JOIN] ==
					       piped[card][LAUNCH_JOIN] &&
				       wide[card][LAUNCH_RESUME] ==
					       wide[card][LAUNCH_JOIN],
			       "the joint mode: the pipeline's side runs, each resumed and joined once");
			printf("program %" PRIu32 " card %" PRIu32
			       " joint: launches %" PRIu32 " -> %" PRIu32
			       ", 2,048 rows %" PRIu32 ", side runs %" PRIu32
			       "\n",
			       index, card,
			       pipelined[index]->cards[card].launch_count,
			       joint[index]->cards[card].launch_count,
			       joint_widest[index]->cards[card].launch_count,
			       joined[card][LAUNCH_RESUME]);
		}
	}
	/* --exl3_expert_rotation: the joint programs of 2,048 rows with both rotations, the same launches a card */
	draft_layers = true; /* as joint_widest's */
	rotation = EXL3_TC_ROTATION_FOUR_WAY;
	const bool four_ready = programs_prepare(
		V41_DATA_SPLIT_JOINT, rings, true, SPLIT_WIDEST_ROWS, four_way);
	rotation = EXL3_TC_ROTATION_THREE_WAY;
	const bool three_ready = programs_prepare(V41_DATA_SPLIT_JOINT, rings,
						  true, SPLIT_WIDEST_ROWS,
						  three_way);
	rotation = 0;
	expect(four_ready && three_ready, "the rotated joint programs prepare");
	for (uint32_t index = 1; index < SPLIT_PROGRAMS && four_ready && three_ready; index++)
		for (uint32_t card = 0; card < SPLIT_CARDS; card++)
			expect(four_way[index]->cards[card].launch_count ==
					       joint_widest[index]->cards[card].launch_count &&
				       three_way[index]->cards[card].launch_count ==
					       joint_widest[index]->cards[card].launch_count,
			       "the rotated programs: the unrotated launch count");
	/* --tp_gather_balance=1 and --tp_pool_chunk=1: the joint programs of 2,048 rows prepare; the 640-column ranks
	 * (1, 3) scan no slab, the 512-column ones scan theirs too; every card's pool writes a launch a piece */
	track_c_knobs = true;
	const bool balanced_ready = programs_prepare(
		V41_DATA_SPLIT_JOINT, rings, true, SPLIT_WIDEST_ROWS, balanced);
	balance_alone = true;
	const bool balance_ready = programs_prepare(V41_DATA_SPLIT_JOINT, rings,
						    true, SPLIT_WIDEST_ROWS,
						    balance_only);
	balance_wide = true;
	const bool wide_balance_ready =
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
				 SPLIT_WIDEST_ROWS, balance_wide_only);
	balance_wide = false;
	balance_alone = false;
	track_c_knobs = false;
	expect(balance_ready, "the joint programs with gather_balance alone prepare");
	for (uint32_t index = 1; index < SPLIT_PROGRAMS && balance_ready; index++)
		for (uint32_t card = 0; card < SPLIT_CARDS; card++)
			printf("program %" PRIu32 " card %" PRIu32 " gather_balance: launches %" PRIu32
			       " -> %" PRIu32 "\n",
			       index, card, joint_widest[index]->cards[card].launch_count,
			       balance_only[index]->cards[card].launch_count);
	/* --tp_gather_balance=2: the wide program's launches gather_balance's, the full program's the default's */
	expect(wide_balance_ready, "the joint programs with gather_balance of the wide programs prepare");
	for (uint32_t card = 0; card < SPLIT_CARDS && wide_balance_ready && balance_ready; card++)
		expect(balance_wide_only[1]->cards[card].launch_count == balance_only[1]->cards[card].launch_count &&
			       balance_wide_only[2]->cards[card].launch_count ==
				       joint_widest[2]->cards[card].launch_count,
		       "gather_balance 2: the wide program balanced, the full program as without it");
	/* --tp_full_split_overlap=5 with --tp_split_overlap=6: the 2,048-row wide program the joint mode's, the full
	 * program the pipeline's */
	full_split = V41_DATA_SPLIT_PIPELINE;
	const bool mixed_ready = programs_prepare(
		V41_DATA_SPLIT_JOINT, rings, true, SPLIT_WIDEST_ROWS, mixed);
	full_split = 0;
	expect(mixed_ready, "the joint programs with the pipeline's full program prepare");
	for (uint32_t card = 0; card < SPLIT_CARDS && mixed_ready; card++)
		expect(mixed[1]->cards[card].launch_count == joint_widest[1]->cards[card].launch_count &&
			       mixed[2]->cards[card].launch_count == widest[2]->cards[card].launch_count,
		       "full_split_overlap: the wide program the joint mode's, the full program the pipeline's");
	/* mode 7: the joint call's down GEMM and combine a half each (three launches more a split layer), the joint
	 * mode's side runs, each resumed and joined once */
	const bool down_ready = programs_prepare(V41_DATA_SPLIT_JOINT_DOWN,
						 rings, true, SPLIT_WIDEST_ROWS,
						 down_halves);
	expect(down_ready, "the joint programs with a half's down GEMMs prepare");
	for (uint32_t index = 1; index < SPLIT_PROGRAMS && down_ready; index++) {
		uint32_t joint_marks[SPLIT_CARDS][SPLIT_MARKS], down_marks[SPLIT_CARDS][SPLIT_MARKS];
		branches(joint_widest[index], joint_marks);
		branches(down_halves[index], down_marks);
		for (uint32_t card = 0; card < SPLIT_CARDS; card++) {
			const uint32_t joint_launches = joint_widest[index]->cards[card].launch_count;
			const uint32_t down_launches = down_halves[index]->cards[card].launch_count;
			printf("program %" PRIu32 " card %" PRIu32 " a half's down GEMMs: launches %" PRIu32 " -> %" PRIu32
			       "\n", index, card, joint_launches, down_launches);
			expect(down_launches > joint_launches &&
				       (down_launches - joint_launches) %
						       EXL3_TCP_PART_LAUNCHES ==
					       0 &&
				       down_marks[card][LAUNCH_RESUME] ==
					       joint_marks[card][LAUNCH_RESUME] &&
				       down_marks[card][LAUNCH_JOIN] ==
					       joint_marks[card][LAUNCH_JOIN],
			       "mode 7: a part of three launches more a split layer, the joint mode's side runs");
		}
	}
	/* --prefill_tail_rows=128: the 128-row full program takes its 64-row groups in the 64-row chunk's forms (the
	 * decode TC MoE's launches, trunk_mma's _64 form a group); a wide program of 128 rows too (main.c prepares one
	 * with --prefill_front_tail alone: no ladder has 128 rows then) */
	const bool plain_ready = programs_prepare(
		V41_DATA_SPLIT_JOINT, rings, true, SPLIT_TAIL_ROWS, tail_plain);
	tail_rows = SPLIT_TAIL_ROWS;
	const bool tail_ready = programs_prepare(
		V41_DATA_SPLIT_JOINT, rings, true, SPLIT_TAIL_ROWS, tail_forms);
	tail_rows = 0;
	expect(plain_ready && tail_ready, "the 128-row programs with and without the tail's forms prepare");
	/* --prefill_tail_rows=320: the 320-row full program in the tail's forms (five 64-row groups, one decode TC MoE
	 * call, no split halves) prepares, the ladder's 256-row one not a tail program */
	struct execution_program *long_tail[SPLIT_PROGRAMS] = { 0 },
				 *ladder_256[SPLIT_PROGRAMS] = { 0 },
				 *ladder_256_plain[SPLIT_PROGRAMS] = { 0 };
	tail_rows = SPLIT_LONG_TAIL_ROWS;
	const uint32_t ladder_rows = MODEL_FULL_CONE_LEAST_ROWS;
	const bool long_ready = programs_prepare(V41_DATA_SPLIT_JOINT, rings,
						 true, SPLIT_LONG_TAIL_ROWS,
						 long_tail) &&
				programs_prepare(V41_DATA_SPLIT_JOINT, rings,
						 true, ladder_rows, ladder_256);
	tail_rows = 0;
	const bool ladder_plain_ready =
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true, ladder_rows,
				 ladder_256_plain);
	expect(long_ready && ladder_plain_ready, "the 320-row tail program and the 256-row ladder programs prepare");
	for (uint32_t card = 0; card < SPLIT_CARDS && long_ready && ladder_plain_ready; card++) {
		uint32_t marks[SPLIT_CARDS][SPLIT_MARKS];
		branches(long_tail[2], marks);
		printf("card %" PRIu32
		       " 320-row tail program: launches %" PRIu32
		       ", side runs %" PRIu32 "\n",
		       card, long_tail[2]->cards[card].launch_count,
		       marks[card][LAUNCH_SIDE]);
		expect(ladder_256[2]->cards[card].launch_count == ladder_256_plain[2]->cards[card].launch_count,
		       "tail_rows 320: the 256-row full program as without it");
	}
	/* --prefill_tail_rows=448: the 448-row full program in the tail's forms (seven 64-row groups, one decode TC MoE
	 * call of 2,688 pairs) and the 320-row one beside it prepare */
	struct execution_program *widest_tail[SPLIT_PROGRAMS] = { 0 },
				 *widest_long[SPLIT_PROGRAMS] = { 0 };
	tail_rows = SPLIT_WIDEST_TAIL_ROWS;
	const bool widest_ready =
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
				 SPLIT_WIDEST_TAIL_ROWS, widest_tail) &&
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
				 SPLIT_LONG_TAIL_ROWS, widest_long);
	tail_rows = 0;
	expect(widest_ready, "the 448-row tail program and the 320-row one beside it prepare");
	for (uint32_t card = 0; card < SPLIT_CARDS && widest_ready && long_ready; card++) {
		printf("card %" PRIu32 " 448-row tail program: launches %" PRIu32 "\n", card,
		       widest_tail[2]->cards[card].launch_count);
		expect(widest_long[2]->cards[card].launch_count == long_tail[2]->cards[card].launch_count,
		       "tail_rows 448: the 320-row tail program as with tail_rows 320");
	}
	/* --prefill_tail_overlay=1: the tail programs' TC MoE workspace over the prefill form's: the same launches */
	struct execution_program *overlay_widest[SPLIT_PROGRAMS] = { 0 },
				 *overlay_long[SPLIT_PROGRAMS] = { 0 },
				 *overlay_short[SPLIT_PROGRAMS] = { 0 };
	tail_rows = SPLIT_WIDEST_TAIL_ROWS;
	tail_overlay = true;
	const bool overlay_ready =
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
				 SPLIT_WIDEST_TAIL_ROWS, overlay_widest) &&
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
				 SPLIT_LONG_TAIL_ROWS, overlay_long) &&
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
				 SPLIT_TAIL_ROWS, overlay_short);
	tail_overlay = false;
	tail_rows = 0;
	expect(overlay_ready, "tail_overlay: the 448-, 320- and 128-row tail programs prepare");
	/* --prefill_tail_sets=1: a tail program's trunk products one launch (and one reduce) over its 64-row groups */
	struct execution_program *sets_widest[SPLIT_PROGRAMS] = { 0 },
				 *sets_short[SPLIT_PROGRAMS] = { 0 };
	tail_rows = SPLIT_WIDEST_TAIL_ROWS;
	tail_overlay = tail_sets = true;
	const bool sets_ready =
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
				 SPLIT_WIDEST_TAIL_ROWS, sets_widest) &&
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
				 SPLIT_TAIL_ROWS, sets_short);
	tail_overlay = tail_sets = false;
	tail_rows = 0;
	expect(sets_ready, "tail_sets: the 448- and 128-row tail programs prepare");
	/* --prefill_thin_tail=1: full programs of 16 to 56 rows in a thin step's forms (without it, none below 64 rows:
	 * model_v41_program.c refuses them) */
	struct execution_program *thin_forms[SPLIT_PROGRAMS] = { 0 },
				 *thin_least[SPLIT_PROGRAMS] = { 0 };
	tail_rows = SPLIT_LONG_TAIL_ROWS;
	chunk_rows = SPLIT_THIN_TAIL_ROWS;
	thin_tail = true;
	const bool thin_ready = programs_prepare(V41_DATA_SPLIT_JOINT, rings,
						 true, SPLIT_LONG_TAIL_ROWS,
						 thin_forms);
	chunk_rows = SPLIT_THIN_TAIL_LEAST;
	const bool least_ready = programs_prepare(V41_DATA_SPLIT_JOINT, rings,
						  true, SPLIT_LONG_TAIL_ROWS,
						  thin_least);
	chunk_rows = 0;
	thin_tail = false;
	tail_rows = 0;
	expect(thin_ready && least_ready, "thin_tail: the 40- and 16-row full programs prepare");
	for (uint32_t card = 0; card < SPLIT_CARDS && thin_ready && least_ready; card++)
		printf("card %" PRIu32 " thin_tail: 40-row full program launches %" PRIu32 ", 16-row %" PRIu32 "\n",
		       card, thin_forms[2]->cards[card].launch_count, thin_least[2]->cards[card].launch_count);
	for (uint32_t card = 0; card < SPLIT_CARDS && sets_ready && widest_ready && tail_ready; card++) {
		printf("card %" PRIu32 " tail_sets: 448-row tail program launches %" PRIu32 " -> %" PRIu32
		       ", 128-row %" PRIu32 " -> %" PRIu32 "\n",
		       card, widest_tail[2]->cards[card].launch_count, sets_widest[2]->cards[card].launch_count,
		       tail_forms[2]->cards[card].launch_count, sets_short[2]->cards[card].launch_count);
		expect(sets_widest[2]->cards[card].launch_count < widest_tail[2]->cards[card].launch_count &&
			       sets_short[2]->cards[card].launch_count < tail_forms[2]->cards[card].launch_count,
		       "tail_sets: fewer launches in the tail programs");
	}
	for (uint32_t card = 0; card < SPLIT_CARDS && overlay_ready && widest_ready && tail_ready; card++)
		expect(overlay_widest[2]->cards[card].launch_count == widest_tail[2]->cards[card].launch_count &&
			       overlay_long[2]->cards[card].launch_count == long_tail[2]->cards[card].launch_count &&
			       overlay_short[2]->cards[card].launch_count == tail_forms[2]->cards[card].launch_count,
		       "tail_overlay: each tail program's launches as without it");
	for (uint32_t card = 0; card < SPLIT_CARDS && plain_ready && tail_ready; card++) {
		printf("card %" PRIu32 " 128-row full program, the tail's forms: launches %" PRIu32 " -> %" PRIu32 "\n",
		       card, tail_plain[2]->cards[card].launch_count, tail_forms[2]->cards[card].launch_count);
		expect(tail_forms[1]->cards[card].launch_count != tail_plain[1]->cards[card].launch_count &&
			       tail_forms[2]->cards[card].launch_count != tail_plain[2]->cards[card].launch_count,
		       "tail_rows: the full and the wide program of 128 rows in the tail's forms");
	}
	/* --tp_pool_gather=1 (with --tp_pool_chunk=1): the wide and the full program of 2,048 rows gather the pooled
	 * source layers' compressor products, the thin program as without it */
	struct execution_program *pool_plain[SPLIT_PROGRAMS] = { 0 },
				 *pool_gathered[SPLIT_PROGRAMS] = { 0 };
	pool_chunk_alone = true;
	const bool pool_plain_ready =
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
				 SPLIT_WIDEST_ROWS, pool_plain);
	pool_gather = true;
	const bool pool_gathered_ready =
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
				 SPLIT_WIDEST_ROWS, pool_gathered);
	pool_gather = pool_chunk_alone = false;
	expect(pool_plain_ready && pool_gathered_ready, "the joint programs with and without pool_gather prepare");
	for (uint32_t card = 0; card < SPLIT_CARDS && pool_plain_ready && pool_gathered_ready; card++) {
		printf("card %" PRIu32 " pool_gather: wide launches %" PRIu32 " -> %" PRIu32 ", full %" PRIu32
		       " -> %" PRIu32 "\n", card, pool_plain[1]->cards[card].launch_count,
		       pool_gathered[1]->cards[card].launch_count, pool_plain[2]->cards[card].launch_count,
		       pool_gathered[2]->cards[card].launch_count);
		const uint32_t more = POOL_GATHER_POOLED_LAYERS * POOL_GATHER_HALVES * POOL_GATHER_HALF_LAUNCHES;
		const uint32_t thin = pool_plain[0]->cards[card].launch_count;
		const uint32_t wide = pool_plain[1]->cards[card].launch_count;
		const uint32_t full = pool_plain[2]->cards[card].launch_count;
		expect(pool_gathered[0]->cards[card].launch_count == thin &&
			       pool_gathered[1]->cards[card].launch_count == wide + more &&
			       pool_gathered[2]->cards[card].launch_count == full + more,
		       "pool_gather: the gathers' launches more in the wide and full programs, the thin one unchanged");
	}
	/* --prefill_dead_halves=1: the full programs of 1,024 and 2,048 rows (joint mode 6, and 7 at 2,048) leave a dead
	 * half's sums and gathers (fewer launches, every side run resumed and joined once); the thin and wide ones the same */
	struct execution_program *dead_wide[SPLIT_PROGRAMS] = { 0 },
				 *dead_widest[SPLIT_PROGRAMS] = { 0 },
				 *dead_down[SPLIT_PROGRAMS] = { 0 };
	cone_behind =
		engine_full_cone_behind(1u, 1u, 0, DEAD_HALVES_MARGIN, 1u);
	expect(cone_behind == MODEL_FULL_CONE_LEAST_ROWS - DEAD_HALVES_MARGIN,
	       "the exact plan's bound: a granule less the margin");
	const bool dead_ready =
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
				 SPLIT_WIDE_ROWS, dead_wide) &&
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
				 SPLIT_WIDEST_ROWS, dead_widest) &&
		programs_prepare(V41_DATA_SPLIT_JOINT_DOWN, rings, true,
				 SPLIT_WIDEST_ROWS, dead_down);
	cone_behind = 0;
	expect(dead_ready, "the programs with dead halves prepare");
	/* --prefill_dead_halves=2: the 2,048-row full programs' sums copy each half's live rows alone (no more launches
	 * than 1's: a quarter of no live row copies nothing), every side run resumed and joined once */
	struct execution_program *trimmed[SPLIT_PROGRAMS] = { 0 };
	cone_behind =
		engine_full_cone_behind(1u, 1u, 0, DEAD_HALVES_MARGIN, 1u);
	rows_trimmed = true;
	const bool trimmed_ready = programs_prepare(
		V41_DATA_SPLIT_JOINT, rings, true, SPLIT_WIDEST_ROWS, trimmed);
	rows_trimmed = false;
	cone_behind = 0;
	expect(trimmed_ready, "the programs with trimmed dead rows prepare");
	for (uint32_t card = 0; card < SPLIT_CARDS && trimmed_ready && dead_ready; card++) {
		uint32_t marks[SPLIT_CARDS][SPLIT_MARKS];
		branches(trimmed[2], marks);
		printf("card %" PRIu32 " trimmed dead rows: full launches %" PRIu32 " -> %" PRIu32 "\n", card,
		       dead_widest[2]->cards[card].launch_count, trimmed[2]->cards[card].launch_count);
		expect(trimmed[2]->cards[card].launch_count <=
				       dead_widest[2]->cards[card].launch_count &&
			       trimmed[1]->cards[card].launch_count ==
				       dead_widest[1]->cards[card].launch_count &&
			       marks[card][LAUNCH_RESUME] ==
				       marks[card][LAUNCH_JOIN],
		       "trimmed dead rows: at most 1's full launches, the wide program the same, side runs whole");
	}
	struct execution_program **const dead_sets[] = { dead_wide, dead_widest,
							 dead_down };
	struct execution_program **const live_sets[] = { joint, joint_widest,
							 down_halves };
	for (uint32_t set = 0; set < sizeof(dead_sets) / sizeof(dead_sets[0]) && dead_ready; set++) {
		uint32_t marks[SPLIT_CARDS][SPLIT_MARKS];
		branches(dead_sets[set][2], marks);
		for (uint32_t card = 0; card < SPLIT_CARDS; card++) {
			printf("set %" PRIu32 " card %" PRIu32
			       " dead halves: full launches %" PRIu32
			       " -> %" PRIu32 ", side runs %" PRIu32 "\n",
			       set, card,
			       live_sets[set][2]->cards[card].launch_count,
			       dead_sets[set][2]->cards[card].launch_count,
			       marks[card][LAUNCH_RESUME]);
			for (uint32_t index = 0; index + 1u < SPLIT_PROGRAMS; index++)
				expect(dead_sets[set][index]->cards[card].launch_count ==
					       live_sets[set][index]->cards[card].launch_count,
				       "dead halves: the thin and wide programs as without them");
			expect(dead_sets[set][2]->cards[card].launch_count <
					       live_sets[set][2]
						       ->cards[card]
						       .launch_count &&
				       marks[card][LAUNCH_RESUME] ==
					       marks[card][LAUNCH_JOIN],
			       "dead halves: fewer full launches, every side run resumed and joined once");
		}
	}
	/* --tp_router_first_whole=1: the 2,048-row wide and full programs lose a gather of the first half's router a
	 * split layer: launches a multiple of the gather's fewer */
	struct execution_program *first_whole[SPLIT_PROGRAMS] = { 0 };
	router_first_whole = true;
	const bool first_whole_ready =
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
				 SPLIT_WIDEST_ROWS, first_whole);
	router_first_whole = false;
	expect(first_whole_ready, "the joint programs with router_first_whole prepare");
	for (uint32_t card = 0; card < SPLIT_CARDS && first_whole_ready; card++)
		for (uint32_t index = 1; index < SPLIT_PROGRAMS; index++) {
			const uint32_t fewer = joint_widest[index]->cards[card].launch_count -
					       first_whole[index]->cards[card].launch_count;
			printf("program %" PRIu32 " card %" PRIu32 " router_first_whole: %" PRIu32 " launches fewer\n",
			       index, card, fewer);
			expect(fewer != 0 && fewer % FIRST_WHOLE_LAUNCHES == 0,
			       "router_first_whole: whole gathers of the first half's router fewer");
		}
	/* --tp_queries_piece=1: the 2,048-row wide and full programs' queries a piece (a half's), with no heads copy */
	struct execution_program *pieces_queries[SPLIT_PROGRAMS] = { 0 };
	queries_piece = true;
	const bool queries_piece_ready =
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
				 SPLIT_WIDEST_ROWS, pieces_queries);
	queries_piece = false;
	expect(queries_piece_ready, "the joint programs with queries_piece prepare");
	for (uint32_t card = 0; card < SPLIT_CARDS && queries_piece_ready; card++)
		for (uint32_t index = 0; index < SPLIT_PROGRAMS; index++) {
			const uint32_t fewer = joint_widest[index]->cards[card].launch_count -
					       pieces_queries[index]->cards[card].launch_count;
			printf("program %" PRIu32 " card %" PRIu32 " queries_piece: %" PRIu32 " launches fewer\n", index,
			       card, fewer);
			expect(index == 0 ? fewer == 0 : fewer != 0 && fewer % QUERIES_PIECE_HALF_LAUNCHES == 0,
			       "queries_piece: the thin program the same, a half's span queries and copy fewer");
		}
	/* --tp_index_span=1 (with --tp_pool_chunk=1): the wide and full programs' index scans a span of 8 slabs a launch */
	struct execution_program *spans[SPLIT_PROGRAMS] = { 0 };
	index_span = pool_chunk_alone = true;
	const bool spans_ready = programs_prepare(
		V41_DATA_SPLIT_JOINT, rings, true, SPLIT_WIDEST_ROWS, spans);
	index_span = pool_chunk_alone = false;
	expect(spans_ready, "the joint programs with index_span prepare");
	for (uint32_t card = 0; card < SPLIT_CARDS && spans_ready && pool_plain_ready; card++)
		for (uint32_t index = 0; index < SPLIT_PROGRAMS; index++) {
			const uint32_t fewer = pool_plain[index]->cards[card].launch_count -
					       spans[index]->cards[card].launch_count;
			printf("program %" PRIu32 " card %" PRIu32 " index_span: %" PRIu32 " launches fewer\n", index, card,
			       fewer);
			expect(index == 0 ? fewer == 0 : fewer != 0 && fewer < pool_plain[index]->cards[card].launch_count,
			       "index_span: the thin program the same, fewer wide and full launches");
		}
	/* --tp_gather_balance=3 (with --tp_pool_chunk=1): every rank a part of the index scans, the wider ranks fewer
	 * slabs: the cards' launches as many in all as with quarters, not each card's */
	struct execution_program *weighted[SPLIT_PROGRAMS] = { 0 };
	balance_weighted = pool_chunk_alone = true;
	const bool weighted_ready = programs_prepare(
		V41_DATA_SPLIT_JOINT, rings, true, SPLIT_WIDEST_ROWS, weighted);
	balance_weighted = pool_chunk_alone = false;
	expect(weighted_ready, "the joint programs with the weighted gather balance prepare");
	for (uint32_t index = 0; index < SPLIT_PROGRAMS && weighted_ready && pool_plain_ready; index++) {
		uint32_t plain_all = 0, weighted_all = 0, moved = 0;
		for (uint32_t card = 0; card < SPLIT_CARDS; card++) {
			plain_all += pool_plain[index]->cards[card].launch_count;
			weighted_all += weighted[index]->cards[card].launch_count;
			moved += pool_plain[index]->cards[card].launch_count != weighted[index]->cards[card].launch_count;
			printf("program %" PRIu32 " card %" PRIu32 " weighted gather balance: launches %" PRIu32 " -> %" PRIu32
			       "\n", index, card, pool_plain[index]->cards[card].launch_count,
			       weighted[index]->cards[card].launch_count);
		}
		expect(plain_all == weighted_all && (index == 0) == (moved == 0),
		       "weighted gather balance: as many launches in all, the chunked programs' moved between cards");
	}
	/* the four prefill knobs of round 4 together (--tp_queries_piece, --tp_index_span, --dense_skinny_bf16=4,
	 * --tp_gather_balance=3 with --tp_pool_chunk=1): the programs prepare, the thin one as without them */
	struct execution_program *combined[SPLIT_PROGRAMS] = { 0 };
	queries_piece = index_span = balance_weighted = pool_chunk_alone = true;
	skinny_bf16 = 4u;
	const bool combined_ready = programs_prepare(
		V41_DATA_SPLIT_JOINT, rings, true, SPLIT_WIDEST_ROWS, combined);
	queries_piece = index_span = balance_weighted = pool_chunk_alone = false;
	skinny_bf16 = 0;
	expect(combined_ready, "the joint programs with the four prefill knobs prepare");
	for (uint32_t card = 0; card < SPLIT_CARDS && combined_ready && pool_plain_ready; card++) {
		printf("card %" PRIu32 " the four knobs: launches %" PRIu32 " / %" PRIu32 " / %" PRIu32 "\n", card,
		       combined[0]->cards[card].launch_count, combined[1]->cards[card].launch_count,
		       combined[2]->cards[card].launch_count);
		expect(combined[0]->cards[card].launch_count == pool_plain[0]->cards[card].launch_count &&
			       combined[2]->cards[card].launch_count < pool_plain[2]->cards[card].launch_count,
		       "the four knobs: the thin program as without them, the full one fewer launches");
	}
	/* --tp_core_compact_halves=1: the split halves' cores write wo_a's compact rows, two copies a split layer fewer */
	struct execution_program *halves_compact[SPLIT_PROGRAMS] = { 0 };
	core_compact_halves = true;
	const bool halves_compact_ready =
		programs_prepare(V41_DATA_SPLIT_JOINT, rings, true,
				 SPLIT_WIDEST_ROWS, halves_compact);
	/* and with the round's defaults beside it: mode 7, dead halves (trimmed), pool gather, at 2,048 and 1,024 rows */
	struct execution_program *compact_defaults[SPLIT_PROGRAMS] = { 0 },
				 *compact_wide[SPLIT_PROGRAMS] = { 0 };
	cone_behind = MODEL_FULL_CONE_LEAST_ROWS - DEAD_HALVES_MARGIN;
	rows_trimmed = pool_gather = true;
	const bool compact_defaults_ready =
		programs_prepare(V41_DATA_SPLIT_JOINT_DOWN, rings, true,
				 SPLIT_WIDEST_ROWS, compact_defaults) &&
		programs_prepare(V41_DATA_SPLIT_JOINT_DOWN, rings, true,
				 SPLIT_WIDE_ROWS, compact_wide);
	cone_behind = 0;
	rows_trimmed = pool_gather = false;
	core_compact_halves = false;
	expect(compact_defaults_ready, "core_compact_halves with mode 7, dead halves and pool gather prepares");
	/* --prefill_ladder_1536: a 1,536-row plan's programs (split halves of 768 rows) with mode 7, dead halves and pool
	 * gather prepare */
	struct execution_program *ladder_1536[SPLIT_PROGRAMS] = { 0 };
	cone_behind = MODEL_FULL_CONE_LEAST_ROWS - DEAD_HALVES_MARGIN;
	rows_trimmed = pool_gather = true;
	bool ladder_1536_ready =
		programs_prepare(V41_DATA_SPLIT_JOINT_DOWN, rings, true,
				 SPLIT_LADDER_1536_ROWS, ladder_1536);
	/* --prefill_ladder_fine: 1,792, 1,280 and 768-row plans alike (split halves of 896, 640 and 384 rows) */
	static const uint32_t fine_rows[] = { 1792u, 1280u, 768u };
	for (uint32_t fine = 0; fine < sizeof(fine_rows) / sizeof(fine_rows[0]); fine++) {
		struct execution_program *fine_set[SPLIT_PROGRAMS] = { 0 };
		ladder_1536_ready =
			programs_prepare(V41_DATA_SPLIT_JOINT_DOWN, rings, true,
					 fine_rows[fine], fine_set) &&
			ladder_1536_ready;
		for (uint32_t index = 0; index < SPLIT_PROGRAMS; index++)
			free(fine_set[index]);
	}
	cone_behind = 0;
	rows_trimmed = pool_gather = false;
	expect(ladder_1536_ready, "the 1,536, 1,792, 1,280 and 768-row programs prepare");
	expect(halves_compact_ready, "the joint programs with core_compact_halves prepare");
	for (uint32_t card = 0; card < SPLIT_CARDS && halves_compact_ready; card++)
		for (uint32_t index = 0; index < SPLIT_PROGRAMS; index++) {
			const uint32_t fewer = joint_widest[index]->cards[card].launch_count -
					       halves_compact[index]->cards[card].launch_count;
			printf("program %" PRIu32 " card %" PRIu32 " core_compact_halves: %" PRIu32 " launches fewer\n",
			       index, card, fewer);
			expect(index == 0 ? fewer == 0 : fewer != 0 && fewer % CORE_COMPACT_LAYER_LAUNCHES == 0,
			       "core_compact_halves: the thin program the same, two copies a split layer fewer");
		}
	expect(balanced_ready, "the joint programs with gather_balance and pool_chunk prepare");
	for (uint32_t index = 1; index < SPLIT_PROGRAMS && balanced_ready; index++)
		for (uint32_t card = 0; card < SPLIT_CARDS; card++) {
			printf("program %" PRIu32 " card %" PRIu32 " gather_balance + pool_chunk: launches %" PRIu32
			       " -> %" PRIu32 "\n", index, card, joint_widest[index]->cards[card].launch_count,
			       balanced[index]->cards[card].launch_count);
			expect(balanced[index]->cards[card].launch_count <
				       joint_widest[index]->cards[card].launch_count,
			       "gather_balance and pool_chunk: fewer launches on every card");
		}
	struct execution_program **const sets[] = {
		whole,		split,
		engines,	cores,
		pipelined,	widest,
		drafted,	joint,
		joint_widest,	four_way,
		three_way,	balanced,
		balance_only,	balance_wide_only,
		mixed,		down_halves,
		tail_plain,	tail_forms,
		pool_plain,	pool_gathered,
		dead_wide,	dead_widest,
		dead_down,	first_whole,
		trimmed,	long_tail,
		ladder_256,	ladder_256_plain,
		pieces_queries, spans,
		weighted,	combined,
		halves_compact, compact_defaults,
		compact_wide,	ladder_1536,
		widest_tail,	widest_long,
		overlay_widest, overlay_long,
		overlay_short,	sets_widest,
		sets_short,	thin_forms,
		thin_least
	};
	for (uint32_t set = 0; set < sizeof(sets) / sizeof(sets[0]); set++)
		for (uint32_t index = 0; index < SPLIT_PROGRAMS; index++)
			free(sets[set][index]);
	for (uint32_t card = 0; card < SPLIT_CARDS; card++)
		test_host_memory_backend.card_close(cards[card], SPLIT_WAIT_NS);
	printf("split_overlap_program_test %s\n", failures == 0 ? "passed" : "failed");
	return failures == 0 ? 0 : 1;
}
