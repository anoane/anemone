/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* ring_sequence_dump_test (gnmac_port diagnostics, track A): every card's exact-ring launches (the cut-through rings
 * whose per-block generation counters live in a bank of a site's channel, one bank a row count) of the programs one
 * driver takes, for the THIN recipe (thin 6 and 8, wide, full) and the BATCH recipe (BATCH 12 and 8, wide, full) at
 * 64 and 1,024-row chunks, in the order each card runs them, by segment range (front steps run the front, drafting
 * steps every segment, the others stop before the draft). Two cards whose rings on one bank differ in count or grid
 * fall out of step: a peer then waits on another epoch's slots until the ring's deadline. Prints the first
 * disagreement a range; with RING_DUMP_DIRECTORY set, every card's launches a program (the BATCH recipe's wide and
 * full programs are the THIN recipe's). Always passes; its output is the evidence. */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/dense_linear_records.h"
#include "anemone/exact_ring_records.h"
#include "anemone/exl3_tc_records.h"
#include "anemone/attention_state_records.h"
#include "anemone/relaxed_linear_records.h"
#include "anemone/sampler_shard_records.h"
#include "anemone/step.h"
#include "host_memory_backend.h"
#include "model_v41_pack.h"
#include "model_v41_tp2_model.h"

#define DUMP_CARDS TP4_RANKS
#define DUMP_CARD_BYTES 4096u
#define DUMP_REPORTED_BYTES (1ull << 40u)
#define DUMP_SESSIONS 8u
#define DUMP_PROGRAMS 4u
#define DUMP_WIDE_ROWS 1024u
#define DUMP_WAIT_NS 1000000000ull
#define DUMP_MOST_RINGS 8192u
#define DUMP_RANGES 3u
#define AUDIT_RECIPES 5u
#define DUMP_TILE_OUTPUTS 16u     /* trunk_mma's tile */
#define DUMP_THIN_SPLIT_TILES 128u /* the 1-8-row form: split 16 up to 128 tiles, 4 up to 512, else 2 */
#define DUMP_THIN_SPLIT_MORE_TILES 512u

struct ring_call {
	uint64_t bank; /* the bank's place in its channel's region: the same on every card */
	uint32_t packs, grid, mode, segment, launch;
	bool unequal;
};

static struct ring_call calls[DUMP_CARDS][DUMP_MOST_RINGS];
static uint32_t call_counts[DUMP_CARDS];
static bool dump_column_owners; /* the recipe with --tp_ring_column_owners (track G) */
static uint32_t dump_failures;
static uint32_t bank_audit_recipe;

static void rings_collect(const struct execution_program *program)
{
	for (uint32_t card = 0; card < DUMP_CARDS; card++) {
		call_counts[card] = 0;
		if (card >= program->card_count)
			continue;
		for (uint32_t segment = 0; segment < program->segment_count;
		     segment++) {
			const struct segment_record *const own =
				&program->segments[segment];
			if (own->card != card)
				continue;
			for (uint32_t index = own->first_launch;
			     index < own->first_launch + own->launch_count;
			     index++) {
				const struct kernel_launch *const launch =
					&program->cards[card].launches[index];
				struct test_host_memory_backend_kernel kernel;
				if (launch->kernel == BACKEND_COPY ||
				    !test_host_memory_backend_kernel_of(
					    card, launch->kernel, &kernel) ||
				    strncmp(kernel.function_name,
					    KERNEL_EXACT_RING,
					    strlen(KERNEL_EXACT_RING)) != 0 ||
				    call_counts[card] == DUMP_MOST_RINGS)
					continue;
				struct ring_call call = {
					.segment = segment,
					.launch = index,
					.grid = launch->grid[0]
				};
				call.unequal =
					strcmp(kernel.function_name,
					       KERNEL_EXACT_RING_UNEQUAL) == 0;
				if (!call.unequal) {
					struct exact_ring_arguments arguments;
					memcpy(&arguments, launch->argument,
					       sizeof(arguments));
					call.bank = arguments.peer_bank_delta;
					call.packs = arguments.packs_per_owner;
					call.mode = arguments.mode;
				} else {
					struct exact_ring_unequal_arguments
						arguments;
					memcpy(&arguments, launch->argument,
					       sizeof(arguments));
					call.bank = arguments.peer_bank_delta;
					call.packs = arguments.maximum_packs;
				}
				calls[card][call_counts[card]++] = call;
			}
		}
	}
}

/* The calls of card below segment `end` (its segments' order is the program's). */
static uint32_t calls_below(uint32_t card, uint32_t end)
{
	uint32_t count = 0;
	while (count < call_counts[card] && calls[card][count].segment < end)
		count++;
	return count;
}

/* Every card's launches (segment, name, grid, branch) to RING_DUMP_DIRECTORY/<title>_card<c>.txt, when it is set. */
static void launches_write(const char *title,
			   const struct execution_program *program)
{
	const char *const directory = getenv("RING_DUMP_DIRECTORY");
	if (directory == NULL)
		return;
	for (uint32_t card = 0; card < program->card_count && card < DUMP_CARDS;
	     card++) {
		char path[256];
		(void)snprintf(path, sizeof(path), "%s/%s_card%" PRIu32 ".txt",
			       directory, title, card);
		for (char *at = path + strlen(directory) + 1u; *at != 0; at++)
			*at = *at == ' ' ? '_' : *at;
		FILE *const file = fopen(path, "w");
		if (file == NULL)
			continue;
		for (uint32_t segment = 0; segment < program->segment_count;
		     segment++) {
			const struct segment_record *const own =
				&program->segments[segment];
			if (own->card != card)
				continue;
			for (uint32_t index = own->first_launch;
			     index < own->first_launch + own->launch_count;
			     index++) {
				const struct kernel_launch *const launch =
					&program->cards[card].launches[index];
				struct test_host_memory_backend_kernel kernel;
				const bool named =
					launch->kernel != BACKEND_COPY &&
					test_host_memory_backend_kernel_of(
						card, launch->kernel, &kernel);
				fprintf(file,
					"%3" PRIu32
					" %s grid %u,%u,%u branch %u\n",
					segment,
					launch->kernel == BACKEND_COPY ?
						"(copy)" :
					named ? kernel.function_name :
						"?",
					launch->grid[0], launch->grid[1],
					launch->grid[2], launch->branch);
			}
		}
		(void)fclose(file);
	}
	for (uint32_t index = 0; index < program->hand_off_count; index++)
		printf("  hand-off %" PRIu32 ": after segment %" PRIu32
		       " to card %" PRIu32 "\n",
		       index, program->hand_offs[index].after_segment,
		       program->hand_offs[index].to_card);
}

/* The step blocks the target head's samplers read on each card (the segments before the draft tail): a BATCH
 * program's slabs' views, or for one of 8 rows or fewer, which has no views, the step's own block. */
static void sampler_steps_report(const struct execution_program *program)
{
	for (uint32_t card = 0; card < program->card_count && card < DUMP_CARDS;
	     card++) {
		printf("  card %" PRIu32 " head sampler step blocks:", card);
		for (uint32_t segment = 0;
		     segment < program->draft_segment_first; segment++) {
			const struct segment_record *const own =
				&program->segments[segment];
			for (uint32_t index = own->first_launch;
			     own->card == card &&
			     index < own->first_launch + own->launch_count;
			     index++) {
				const struct kernel_launch *const launch =
					&program->cards[card].launches[index];
				struct test_host_memory_backend_kernel kernel;
				if (launch->kernel == BACKEND_COPY ||
				    !test_host_memory_backend_kernel_of(
					    card, launch->kernel, &kernel) ||
				    strcmp(kernel.function_name,
					   KERNEL_SAMPLER_SHARD) != 0)
					continue;
				struct sampler_shard_arguments arguments;
				memcpy(&arguments, launch->argument,
				       sizeof(arguments));
				printf(" %" PRIu64, arguments.step_offset);
			}
		}
		printf("\n");
	}
}

/* Optional preparation audit: all programs of one model share a namespace. A future protocol selector must
 * never assign two packet formats to the same bank, including full tails and draft segments. No GPU claim. */
static void bank_audit_report(const char *name,
			      const struct execution_program *program)
{
	if (getenv("RING_BANK_AUDIT") == NULL ||
	    (bank_audit_recipe > AUDIT_RECIPES &&
	     strstr(name, "column-owners") == NULL))
		return;
	for (uint32_t segment = 0; segment < program->segment_count;
	     segment++) {
		const struct segment_record *const own =
			&program->segments[segment];
		for (uint32_t index = own->first_launch;
		     index < own->first_launch + own->launch_count; index++) {
			const struct kernel_launch *const launch =
				&program->cards[own->card].launches[index];
			struct test_host_memory_backend_kernel kernel;
			if (launch->kernel == BACKEND_COPY ||
			    !test_host_memory_backend_kernel_of(
				    own->card, launch->kernel, &kernel) ||
			    strncmp(kernel.function_name, KERNEL_EXACT_RING,
				    strlen(KERNEL_EXACT_RING)) != 0)
				continue;
			struct exact_ring_arguments arguments;
			memcpy(&arguments, launch->argument, sizeof(arguments));
			const bool unequal =
				strcmp(kernel.function_name,
				       KERNEL_EXACT_RING_UNEQUAL) == 0;
			if (unequal) {
				struct exact_ring_unequal_arguments actual;
				memcpy(&actual, launch->argument,
				       sizeof(actual));
				arguments.packs_per_owner =
					actual.maximum_packs;
				arguments.vectors = actual.vectors;
				arguments.mode = EXACT_RING_GATHER;
			}
			printf("BANK,%u,%u,\"%s\",%u,%u,%s,%" PRIu64 ",%" PRIu64
			       ",%" PRIu64 ",%u,%u,%u,%u,%u\n",
			       bank_audit_recipe, own->card, name,
			       program->kind, program->limits.row_capacity,
			       segment < program->draft_segment_first ?
				       "target" :
				       "draft",
			       arguments.bank_offset,
			       arguments.peer_table_offset,
			       arguments.peer_bank_delta,
			       arguments.packs_per_owner, arguments.vectors,
			       launch->grid[0], arguments.mode,
			       (unsigned int)unequal);
		}
	}
}
static void program_report(const char *name,
			   const struct execution_program *program)
{
	bank_audit_report(name, program);
	rings_collect(program);
	launches_write(name, program);
	const uint32_t ends[DUMP_RANGES] = { program->front_segment_count,
					     program->draft_segment_first,
					     program->segment_count };
	const char *const range_names[DUMP_RANGES] = { "front", "no draft",
						       "drafting" };
	printf("%s: %" PRIu32 " segments, front %" PRIu32
	       ", draft from %" PRIu32 ", result card %" PRIu32
	       "; launches a card %" PRIu32 " %" PRIu32 " %" PRIu32 " %" PRIu32 "\n",
	       name, program->segment_count, program->front_segment_count,
	       program->draft_segment_first, program->result_card,
	       program->cards[0].launch_count, program->cards[1].launch_count,
	       program->cards[2].launch_count, program->cards[3].launch_count);
	sampler_steps_report(program);
	/* --tp_ring_column_owners: every card's BF16 sums with column owners (mode 2), none with row
	 * quarters (mode 1) */
	uint32_t row_owners = 0, column_owners = 0;
	for (uint32_t card = 0; card < DUMP_CARDS; card++)
		for (uint32_t index = 0; index < call_counts[card]; index++) {
			row_owners +=
				!calls[card][index].unequal &&
						calls[card][index].mode ==
							EXACT_RING_BF16_SUM ?
					1u :
					0u;
			column_owners +=
				!calls[card][index].unequal &&
						calls[card][index].mode ==
							EXACT_RING_BF16_SUM_COLUMNS ?
					1u :
					0u;
		}
	printf("  BF16 sum rings: %" PRIu32 " of row owners, %" PRIu32
	       " of column owners\n",
	       row_owners, column_owners);
	if (dump_column_owners && row_owners != 0) { /* chunk programs sum by other collectives: none at all */
		printf("  %s: FAILED: with --tp_ring_column_owners every BF16 sum has column owners\n",
		       name);
		dump_failures++;
	}
	for (uint32_t range = 0; range < DUMP_RANGES; range++) {
		uint32_t counts[DUMP_CARDS];
		for (uint32_t card = 0; card < DUMP_CARDS; card++)
			counts[card] = calls_below(card, ends[range]);
		printf("  %-8s rings a card: %" PRIu32 " %" PRIu32 " %" PRIu32
		       " %" PRIu32 "\n",
		       range_names[range], counts[0], counts[1], counts[2],
		       counts[3]);
		uint32_t most = 0;
		for (uint32_t card = 0; card < DUMP_CARDS; card++)
			most = counts[card] > most ? counts[card] : most;
		for (uint32_t index = 0; index < most; index++) {
			bool same = true;
			for (uint32_t card = 1; card < DUMP_CARDS && same;
			     card++) {
				same = index < counts[0] &&
				       index < counts[card] &&
				       calls[card][index].bank ==
					       calls[0][index].bank &&
				       calls[card][index].grid ==
					       calls[0][index].grid &&
				       calls[card][index].packs ==
					       calls[0][index].packs &&
				       calls[card][index].unequal ==
					       calls[0][index].unequal;
			}
			if (same)
				continue;
			printf("  %-8s first disagreement at ring %" PRIu32
			       ":\n",
			       range_names[range], index);
			for (uint32_t card = 0; card < DUMP_CARDS; card++) {
				for (uint32_t at = index > 2u ? index - 2u : 0;
				     at < index + 3u && at < counts[card];
				     at++) {
					const struct ring_call *const call =
						&calls[card][at];
					printf("    card %" PRIu32
					       " ring %4" PRIu32
					       " segment %3" PRIu32
					       " launch %5" PRIu32
					       " bank +%10" PRIu64
					       " packs %6" PRIu32
					       " grid %2" PRIu32
					       " mode %" PRIu32 "%s\n",
					       card, at, call->segment,
					       call->launch, call->bank,
					       call->packs, call->grid,
					       call->mode,
					       call->unequal ? " unequal" : "");
				}
			}
			break;
		}
	}
	/* the banks each card uses, with each bank's grids (a bank's generation counters are a block's) */
	for (uint32_t card = 0; card < 1u; card++) {
		printf("  card %" PRIu32 " banks (bank: calls, grids):", card);
		for (uint32_t index = 0; index < call_counts[card]; index++) {
			bool seen = false;
			for (uint32_t before = 0; before < index && !seen;
			     before++)
				seen = calls[card][before].bank ==
				       calls[card][index].bank;
			if (seen)
				continue;
			uint32_t count = 0, grid_mask = 0;
			for (uint32_t at = index; at < call_counts[card];
			     at++) {
				if (calls[card][at].bank ==
				    calls[card][index].bank) {
					count++;
					grid_mask |= calls[card][at].grid;
				}
			}
			printf(" [+%" PRIu64 " packs %" PRIu32 ": %" PRIu32
			       " x, grids 0x%" PRIx32 "]",
			       calls[card][index].bank,
			       calls[card][index].packs, count, grid_mask);
		}
		printf("\n");
	}
}

static struct device_card *cards[DUMP_CARDS];
static struct card_description_record reported[DUMP_CARDS];
static bool dump_thin_splits; /* the recipe with --tp_trunk_mma_thin_splits */
static bool dump_thin_bits; /* the recipe with --batch_thin_bits (track G) */
static uint32_t dump_grouped_rows; /* and --batch_grouped_rows */
static uint32_t dump_tc_paired; /* and --batch_tc_paired (track J) */
static uint32_t dump_row_sets; /* and --batch_row_sets (track J) */
static uint32_t dump_core_once; /* and --batch_core_once (track J) */
static bool
	dump_round_once; /* target THIN/BATCH owner exchange with FP32 sums of BF16 partials */
/* the core-once recipe's prepares of each multi-slab BATCH program in report order: level 2's windows replace them */
static uint32_t core_once_prepares[16], core_once_windows[16], core_once_ordinal;
static uint32_t dump_front_branch; /* and --tp_front_branch (track J) */
static bool dump_draft_bf16_rows; /* and --draft_bf16_rows (track J) */
static uint32_t dump_draft_wide; /* and --batch_draft_wide (track L) */
static bool dump_draft_operands; /* and --draft_bf16_operands (track L) */
/* the core-once 2 recipe's rings and posts on all cards of each BATCH program in report order: the draft-wide recipe
 * (the same flags and --batch_draft_wide=1) has 4 cards x 3 draft layers x 2 sites fewer a draft slab past the first */
#define DUMP_DRAFT_SITES 2u /* a draft layer's sites: attention and FFN, a ring and a post each */
#define DUMP_CORE_ONCE_BASELINE 2u /* retained level-2 launch-digest baseline */
#define DUMP_CORE_ONCE_POOLS \
	3u /* step-wide pools followed by deferred slot retention */
#define DUMP_DRAFT_WIDE_PRODUCTS 2u /* --batch_draft_wide=2: the draft's products over all its rows too */
#define DUMP_DRAFT_WIDE_CORES 3u /* --batch_draft_wide=3: and its window writes and cores once */
static uint32_t draft_wide_rings[sizeof(core_once_prepares) / sizeof(core_once_prepares[0])],
	draft_wide_posts[sizeof(core_once_prepares) / sizeof(core_once_prepares[0])], draft_wide_ordinal;
static uint32_t plain_quantized_trunks[2]; /* the THIN recipe's quantized trunk launches of THIN 6 and 8 */
static uint32_t one_slab_cores; /* the target cores of the recipe's one-slab BATCH program (BATCH 8) */
/* the plain BATCH recipe's WIDE and FULL programs at 64 rows (launch digests): the thin-bits recipe's equal them */
static uint64_t chunk_digests[2];

/* --tp_trunk_mma_thin_splits: of every card's _64 trunk_mma launches (the record of
 * trunk_relaxed_bfloat16_arguments), how many split the input groups as the 1-8-row form of the shape does
 * (trunk_mma_form_of in model_v41_program.c); with the knob all of them must. */
static void thin_splits_report(const char *title,
			       const struct execution_program *program,
			       uint32_t rows)
{
	uint32_t total = 0, thin = 0, grouped = 0, sets = 0;
	for (uint32_t card = 0; card < program->card_count && card < DUMP_CARDS;
	     card++) {
		for (uint32_t index = 0;
		     index < program->cards[card].launch_count; index++) {
			const struct kernel_launch *const launch =
				&program->cards[card].launches[index];
			struct test_host_memory_backend_kernel kernel;
			if (launch->kernel == BACKEND_COPY ||
			    !test_host_memory_backend_kernel_of(
				    card, launch->kernel, &kernel))
				continue;
			/* --batch_row_sets: a 1-8-row or grouped form over gridDim.y sets of the step's rows, each
			 * set of --batch_row_sets rows (the thin bits by construction) */
			struct trunk_linear_bfloat16_arguments
				rows_record; /* a pair's first record */
			memcpy(&rows_record, launch->argument, sizeof(rows_record));
			if (dump_thin_bits && dump_row_sets != 0 &&
			    rows > dump_grouped_rows &&
			    rows_record.row_count ==
				    rows && /* the target's, not a draft slab's */
			    strncmp(kernel.function_name, "trunk_mma_",
				    strlen("trunk_mma_")) == 0 &&
			    strncmp(kernel.function_name, "trunk_mma_64",
				    strlen("trunk_mma_64")) != 0) {
				sets += launch->grid[1] == (rows + dump_row_sets - 1u) /
								   (dump_row_sets != 0 ? dump_row_sets : 1u) ?
						1u :
						0u;
				grouped++;
				continue;
			}
			/* --batch_grouped_rows: the 1-8-row forms over the step's row groups (the thin bits by
			 * construction) */
			if (strncmp(kernel.function_name, "trunk_mma_16_",
				    strlen("trunk_mma_16_")) == 0 ||
			    strncmp(kernel.function_name, "trunk_mma_24_",
				    strlen("trunk_mma_24_")) == 0 ||
			    strncmp(kernel.function_name, "trunk_mma_32_",
				    strlen("trunk_mma_32_")) == 0 ||
			    strncmp(kernel.function_name, "trunk_mma_40_",
				    strlen("trunk_mma_40_")) == 0 ||
			    strncmp(kernel.function_name, "trunk_mma_48_",
				    strlen("trunk_mma_48_")) == 0) {
				grouped++;
				continue;
			}
			if (strncmp(kernel.function_name, "trunk_mma_64",
				    strlen("trunk_mma_64")) != 0)
				continue;
			struct trunk_relaxed_bfloat16_arguments record;
			memcpy(&record, launch->argument, sizeof(record));
			const uint32_t tiles =
				record.output_size / DUMP_TILE_OUTPUTS;
			const uint32_t split =
				tiles <= DUMP_THIN_SPLIT_TILES ? 16u :
				tiles <= DUMP_THIN_SPLIT_MORE_TILES ? 4u :
								      2u;
			total++;
			thin += record.k_split_count == split ? 1u : 0u;
		}
	}
	printf("  %s: %" PRIu32 " of %" PRIu32
	       " _64 trunk launches split as the thin form, %" PRIu32
	       " grouped trunk launches (%" PRIu32 " in row sets)\n",
	       title, thin, total, grouped, sets);
	const bool set_rows = dump_thin_bits && dump_row_sets != 0 && rows > dump_grouped_rows;
	if (set_rows && (total != 0 || sets == 0 || sets != grouped)) {
		printf("  %s: FAILED: with --batch_row_sets=%" PRIu32 " every trunk launch of %" PRIu32
		       " rows in row sets\n", title, dump_row_sets, rows);
		dump_failures++;
	}
	/* --batch_draft_wide=2: the draft's products over its own rows (verifiers x 5), _64 launches above 16 */
	const bool grouped_rows = dump_thin_bits && (rows <= dump_grouped_rows || set_rows) &&
				  dump_draft_wide < DUMP_DRAFT_WIDE_PRODUCTS;
	if (grouped_rows && (total != 0 || grouped == 0)) {
		printf("  %s: FAILED: with --batch_grouped_rows=%" PRIu32
		       " every trunk launch of %" PRIu32 " rows is grouped\n",
		       title, dump_grouped_rows, rows);
		dump_failures++;
	}
	if ((dump_thin_splits || dump_thin_bits) && !grouped_rows &&
	    (thin != total || (total == 0 && dump_draft_wide < DUMP_DRAFT_WIDE_PRODUCTS))) {
		printf("  %s: FAILED: with --tp_trunk_mma_thin_splits or --batch_thin_bits every _64 launch splits"
		       " as the thin form\n",
		       title);
		dump_failures++;
	}
}

static bool name_has(const char *name, const char *part)
{
	return strstr(name, part) != NULL;
}

/* --tp_front_branch, a thin program (THIN, or BATCH of one slab): on every card each input front of an attention site
 * followed by its coefficient front opening a side run of one launch, a resume behind it and a join (the site's post)
 * later in the segment; each FFN site's input front followed (behind the separate router and the shared expert's
 * side run) by its coefficient front as a side run of its own (SIDE_MORE), a resume behind it and a join later in the
 * segment; the remaining site fronts and FFN fronts counted. */
static void front_branch_report(const char *title,
				const struct execution_program *program)
{
	uint32_t inputs = 0, attention = 0, ffn = 0, site_fronts = 0, ffn_fronts = 0;
	for (uint32_t segment = 0; segment < program->segment_count; segment++) {
		const struct segment_record *const own =
			&program->segments[segment];
		const struct kernel_launch *const launches =
			program->cards[own->card].launches;
		const uint32_t end = own->first_launch + own->launch_count;
		for (uint32_t index = own->first_launch; index < end; index++) {
			struct test_host_memory_backend_kernel kernel, next;
			if (launches[index].kernel == BACKEND_COPY ||
			    !test_host_memory_backend_kernel_of(
				    own->card, launches[index].kernel, &kernel))
				continue;
			site_fronts +=
				name_has(kernel.function_name, "site_front_") ?
					1u :
					0u;
			ffn_fronts +=
				name_has(kernel.function_name, "ffn_front_") ?
					1u :
					0u;
			if (!name_has(kernel.function_name, "input_front_"))
				continue;
			inputs++;
			/* the coefficient front behind this input front: the next launch, or after the router and the
			 * shared expert's side run */
			uint32_t side = index + 1u;
			while (side < end &&
			       !(launches[side].kernel != BACKEND_COPY &&
				 test_host_memory_backend_kernel_of(
					 own->card, launches[side].kernel,
					 &next) &&
				 name_has(next.function_name,
					  "coefficient_front_")))
				side++;
			if (side + 1u >= end ||
			    launches[side + 1u].branch != LAUNCH_RESUME)
				continue;
			const bool first_run = side == index + 1u &&
					       launches[side].branch ==
						       LAUNCH_SIDE;
			const bool more_run =
				side > index + 1u &&
				launches[side].branch == LAUNCH_SIDE_MORE &&
				launches[side - 1u].branch == LAUNCH_SIDE;
			uint32_t at = side + 2u;
			while (at < end && launches[at].branch == LAUNCH_CHAIN)
				at++;
			if (at >= end || launches[at].branch != LAUNCH_JOIN)
				continue;
			attention += first_run ? 1u : 0u;
			ffn += more_run ? 1u : 0u;
		}
	}
	printf("  %s: input fronts %" PRIu32 ": attention side runs %" PRIu32 ", FFN side runs %" PRIu32
	       " (each joined in its segment); site fronts %" PRIu32 ", FFN fronts %" PRIu32 "\n", title, inputs,
	       attention, ffn, site_fronts, ffn_fronts);
	if (dump_front_branch != 0 &&
	    (inputs == 0 || attention + ffn != inputs || attention == 0 || (ffn == 0) != (dump_front_branch < 2u))) {
		printf("  %s: FAILED: with --tp_front_branch every input front a side run joined in its segment\n",
		       title);
		dump_failures++;
	}
}

/* The quantized trunk launches (trunk_linear_quantized_*: rows quantized to E4M3) of every card. */
static uint32_t quantized_trunks_of(const struct execution_program *program)
{
	uint32_t trunks = 0;
	for (uint32_t card = 0; card < program->card_count && card < DUMP_CARDS; card++)
		for (uint32_t index = 0; index < program->cards[card].launch_count; index++) {
			struct test_host_memory_backend_kernel kernel;
			const struct kernel_launch *const launch =
				&program->cards[card].launches[index];
			if (launch->kernel != BACKEND_COPY &&
			    test_host_memory_backend_kernel_of(
				    card, launch->kernel, &kernel) &&
			    name_has(kernel.function_name,
				     "trunk_linear_quantized"))
				trunks++;
		}
	return trunks;
}

/* The target attention cores of every card of a program (the draft's block cores left out). */
static uint32_t cores_of(const struct execution_program *program)
{
	uint32_t cores = 0;
	for (uint32_t card = 0; card < program->card_count && card < DUMP_CARDS; card++)
		for (uint32_t index = 0; index < program->cards[card].launch_count; index++) {
			struct test_host_memory_backend_kernel kernel;
			const struct kernel_launch *const launch =
				&program->cards[card].launches[index];
			if (launch->kernel != BACKEND_COPY &&
			    test_host_memory_backend_kernel_of(
				    card, launch->kernel, &kernel) &&
			    strstr(kernel.function_name,
				   "attention_core_entry") != NULL &&
			    strstr(kernel.function_name, "_block") == NULL)
				cores++;
		}
	return cores;
}

/* --batch_thin_bits, a BATCH program of more than 8 rows: no TC MoE long form (t40 products, s8 activation, s1
 * combine, long operands) and its target head (the head_mma_64 launch over the program's rows) at the 8-row form's
 * split (_s4), on every card; with --batch_tc_paired=R the paired products (t20x2 / t16x2) exactly from R rows on. */
static void thin_bits_report(const char *title,
			     const struct execution_program *program,
			     uint32_t rows)
{
	uint32_t long_forms = 0, heads = 0, thin_heads = 0, paired = 0, cores = 0, prepares = 0, windows = 0;
	uint32_t step_pools = 0; /* compressor launches over more than a slab's rows */
	uint32_t rings = 0, posts = 0, carries = 0;
	for (uint32_t card = 0; card < program->card_count && card < DUMP_CARDS;
	     card++) {
		for (uint32_t index = 0;
		     index < program->cards[card].launch_count; index++) {
			const struct kernel_launch *const launch =
				&program->cards[card].launches[index];
			struct test_host_memory_backend_kernel kernel;
			if (launch->kernel == BACKEND_COPY ||
			    !test_host_memory_backend_kernel_of(
				    card, launch->kernel, &kernel))
				continue;
			const char *const function = kernel.function_name;
			long_forms +=
				name_has(function, "exl3_tc_") &&
						(name_has(function, "_long") ||
						 name_has(function,
							  "products_t40") ||
						 name_has(function,
							  "activate_s8") ||
						 name_has(function,
							  "combine_s1")) ?
					1u :
					0u;
			cores += name_has(function, "attention_core_entry") &&
						 !name_has(function, "_block") ?
					 1u :
					 0u;
			prepares +=
				name_has(function, "attention_prepare_entry") ?
					1u :
					0u;
			windows += name_has(function, "attention_window") ? 1u :
									    0u;
			rings += name_has(function, "exact_ring") ? 1u : 0u;
			posts += name_has(function, "hyper_post_entry") ? 1u :
									  0u;
			carries += name_has(function, "activation_carry") ? 1u :
									    0u;
			step_pools +=
				name_has(function,
					 "attention_compressor_entry") &&
						launch->grid[0] >
							LINEAR_MAXIMUM_ROWS ?
					1u :
					0u;
			if (name_has(function, "attention_compressor_entry")) {
				struct attention_compressor_arguments record;
				memcpy(&record, launch->argument,
				       sizeof(record));
				if (record.slot_mode ==
				    ATTENTION_COMPRESSOR_SLOTS_DEFER) {
					struct attention_compressor_arguments
						following = { 0 };
					const bool has_following =
						index + 1u <
						program->cards[card]
							.launch_count;
					if (has_following)
						memcpy(&following,
						       program->cards[card]
							       .launches[index +
									 1u]
							       .argument,
						       sizeof(following));
					const bool keeps =
						following.slot_mode ==
						ATTENTION_COMPRESSOR_SLOTS_KEEP;
					following.slot_mode = record.slot_mode;
					if (!has_following || !keeps ||
					    memcmp(&record, &following,
						   sizeof(record)) != 0) {
						printf("  %s: FAILED: deferred pool has no matching following keep\n",
						       title);
						dump_failures++;
					}
				}
			}
			paired += name_has(function, "exl3_tc_products_") &&
						  name_has(function, "x2_k") ?
					  1u :
					  0u;
			if (!name_has(function, "head_mma_64"))
				continue;
			struct dense_linear_arguments record;
			memcpy(&record, launch->argument, sizeof(record));
			if (record.row_count != rows)
				continue; /* the draft head (its slot rows) */
			heads++;
			thin_heads += name_has(function, "_s4") ? 1u : 0u;
		}
	}
	printf("  %s: TC long-form launches %" PRIu32 ", paired products %" PRIu32 ", target heads at split 4 %" PRIu32
	       " of %" PRIu32 ", target cores %" PRIu32 "\n", title, long_forms, paired, thin_heads, heads, cores);
	/* --batch_core_once=2: each target layer's slab prepares (of the core-once recipe) one window launch */
	const uint32_t slabs = (rows + LINEAR_MAXIMUM_ROWS - 1u) /
			       LINEAR_MAXIMUM_ROWS,
		       ordinal = core_once_ordinal++ %
				 (uint32_t)(sizeof(core_once_prepares) /
					    sizeof(core_once_prepares[0]));
	printf("  %s: prepares %" PRIu32 ", window launches %" PRIu32 "\n", title, prepares, windows);
	if (step_pools != 0 && dump_core_once < DUMP_CORE_ONCE_POOLS) {
		printf("  %s: FAILED: %" PRIu32 " compressor launches over more than 8 rows\n", title, step_pools);
		dump_failures++;
	}
	const uint32_t wide_ordinal =
		draft_wide_ordinal++ % (uint32_t)(sizeof(draft_wide_rings) / sizeof(draft_wide_rings[0]));
	/* a draft slab's rings, or its posts */
	const uint32_t site_launches =
		DUMP_CARDS * MODEL_DRAFT_STAGES * DUMP_DRAFT_SITES;
	if (dump_core_once == DUMP_CORE_ONCE_BASELINE && !dump_draft_wide &&
	    !dump_round_once) {
		draft_wide_rings[wide_ordinal] = rings;
		draft_wide_posts[wide_ordinal] = posts;
	}
	if (dump_draft_wide) {
		const uint32_t fewer = draft_wide_rings[wide_ordinal] - rings;
		printf("  %s: rings %" PRIu32 " (%" PRIu32 " fewer), posts %" PRIu32 " (%" PRIu32 " fewer)\n", title,
		       rings, fewer, posts, draft_wide_posts[wide_ordinal] - posts);
		if (rings > draft_wide_rings[wide_ordinal] || fewer % site_launches != 0 ||
		    draft_wide_posts[wide_ordinal] - posts != fewer) {
			printf("  %s: FAILED: with --batch_draft_wide a draft slab past the first %" PRIu32
			       " rings and posts fewer\n", title, site_launches);
			dump_failures++;
		}
	}
	/* --draft_bf16_operands: the draft's quantizations by the carry kernel, on every card */
	if (dump_draft_operands != (carries != 0)) {
		printf("  %s: FAILED: %" PRIu32 " carry launches with --draft_bf16_operands=%u\n", title, carries,
		       dump_draft_operands ? 1u : 0u);
		dump_failures++;
	}
	if (dump_core_once == 1u) {
		core_once_prepares[ordinal] = prepares;
		core_once_windows[ordinal] = windows; /* the draft's own window launches */
	}
	/* --batch_draft_wide=2: the draft's qr alone over its rows is a prepare launch of its own a layer */
	if (dump_core_once == 2u && dump_draft_wide < DUMP_DRAFT_WIDE_PRODUCTS &&
	    (windows <= core_once_windows[ordinal] ||
	     prepares + (windows - core_once_windows[ordinal]) * slabs != core_once_prepares[ordinal])) {
		printf("  %s: FAILED: with --batch_core_once=2 each window launch in place of %" PRIu32
		       " slab prepares (%" PRIu32 ")\n", title, slabs, core_once_prepares[ordinal]);
		dump_failures++;
	}
	if (dump_core_once && rows > LINEAR_MAXIMUM_ROWS &&
	    one_slab_cores != 0 && cores != one_slab_cores) {
		printf("  %s: FAILED: with --batch_core_once the target cores of the one-slab program (%" PRIu32 ")\n",
		       title, one_slab_cores);
		dump_failures++;
	}
	if (dump_thin_bits && (paired != 0) != (dump_tc_paired != 0 && rows >= dump_tc_paired)) {
		printf("  %s: FAILED: paired TC products exactly with --batch_tc_paired from %" PRIu32 " rows\n",
		       title, dump_tc_paired);
		dump_failures++;
	}
	if (dump_thin_bits &&
	    (long_forms != 0 || heads == 0 || thin_heads != heads)) {
		printf("  %s: FAILED: with --batch_thin_bits no TC long form and every target head at split 4\n",
		       title);
		dump_failures++;
	}
}

/* FNV-1a over every card's launches: kernel names, grids, blocks and records. */
static uint64_t launches_digest(const struct execution_program *program)
{
	uint64_t digest = 14695981039346656037ull;
	for (uint32_t card = 0; card < program->card_count && card < DUMP_CARDS;
	     card++) {
		for (uint32_t index = 0;
		     index < program->cards[card].launch_count; index++) {
			const struct kernel_launch *const launch =
				&program->cards[card].launches[index];
			struct test_host_memory_backend_kernel kernel = { 0 };
			const bool named =
				launch->kernel != BACKEND_COPY &&
				test_host_memory_backend_kernel_of(
					card, launch->kernel, &kernel);
			const unsigned char *parts[] = {
				named ? (const unsigned char *)kernel.function_name :
					(const unsigned char *)"",
				(const unsigned char *)launch->grid,
				(const unsigned char *)launch->block,
				launch->argument
			};
			const size_t sizes[] = {
				named ? strlen(kernel.function_name) : 0u,
				sizeof(launch->grid), sizeof(launch->block),
				launch->argument_byte_count
			};
			for (uint32_t part = 0; part < 4u; part++)
				for (size_t byte = 0; byte < sizes[part]; byte++)
					digest = (digest ^ parts[part][byte]) *
						 1099511628211ull;
		}
	}
	return digest;
}

static void recipe_report(const char *name, bool batch, uint32_t wide_rows)
{
	bank_audit_recipe++;
	struct pack_record *pack = NULL;
	const struct test_model_v41_change change = {
		.kind = TEST_MODEL_V41_CHANGE_EXL3
	};
	if (test_model_v41_draft_pack_open(change, &pack) != STATUS_OK) {
		printf("ring_sequence_dump_test: the fixture did not open\n");
		return;
	}
	struct v41_tp2_model_description description = {
		.limits = { MODEL_CONTEXT_WITHOUT_CANDIDATES, wide_rows,
			    DUMP_SESSIONS, STATE_WIDTH_16,
			    LINEAR_PROFILE_PRECISION, LINEAR_ORDER_FIXED,
			    CMP_ORDER_REFERENCE, MODEL_DRAFT_BLOCK, 0, 0 },
		.tensor_layout = V41_DATA_TP_LAYOUT_QUAD_ALL,
		.wire_format = ORDERED_PUSH_BF16,
		.head_shards = true,
		.head_stream = true,
		.draft_shards = true,
		.draft_layers = true,
		.exl3_fast_products = true,
		.exl3_wide_entry = true,
		.replicated_embedding = true,
		.batch = batch,
		.batch_wide_head = batch,
		.batch_draft_rows = batch,
		.exl3_tc_moe = true,
		.trunk_mma = true,
		.trunk_mma_thin_splits = dump_thin_splits,
		.batch_thin_bits = dump_thin_bits,
		.batch_grouped_rows = dump_grouped_rows,
		.batch_tc_paired = dump_tc_paired,
		.batch_row_sets = dump_row_sets,
		.batch_core_once = dump_core_once,
		.decode_round_once = dump_round_once,
		.pool_chunk =
			dump_core_once !=
			0, /* main's tp4_exl3 default: BATCH keeps its slabs' pools */
		.front_branch = dump_front_branch,
		.draft_bf16_rows = dump_draft_bf16_rows,
		.batch_draft_wide = dump_draft_wide,
		.draft_bf16_operands = dump_draft_operands,
		.ring_column_owners = dump_column_owners,
		.head_mma = true,
		.graph_branches = 1u,
		.wide_exchange = V41_DATA_TP_EXCHANGE_CE_RINGS,
		.split_overlap = V41_DATA_SPLIT_PIPELINE,
		.chunk_core = true
	};
	for (uint32_t card = 0; card < DUMP_CARDS; card++)
		description.cards[card] = reported[card];
	struct v41_tp2_model *model = NULL;
	struct v41_tp2_program_description build = { 0 };
	const result_status created =
		v41_tp2_model_create(pack, &description, &model);
	const result_status described =
		created != STATUS_OK ?
			created :
			v41_tp2_model_program_description(model, &build);
	if (described != STATUS_OK) {
		printf("%s: the model did not prepare (%s)\n", name,
		       status_text(described));
		pack_close(pack);
		return;
	}
	const enum program_kind kinds[DUMP_PROGRAMS] = {
		batch ? PROGRAM_BATCH : PROGRAM_THIN,
		batch ? PROGRAM_BATCH : PROGRAM_THIN, PROGRAM_WIDE, PROGRAM_FULL
	};
	const uint32_t rows[DUMP_PROGRAMS] = { batch ? 12u : 6u,
					       LINEAR_MAXIMUM_ROWS, wide_rows,
					       wide_rows };
	const char *const names[DUMP_PROGRAMS] = { batch ? "BATCH" : "THIN",
						   batch ? "BATCH" : "THIN",
						   "WIDE", "FULL" };
	for (uint32_t index = 0; index < DUMP_PROGRAMS; index++) {
		struct execution_program *program = NULL;
		struct step_companion_program companions = { 0 };
		const result_status prepared = v41_tp_program_prepare(
			&build, &test_host_memory_backend, cards, reported,
			kinds[index], rows[index], &program, &companions);
		char title[64];
		(void)snprintf(title, sizeof(title), "%s %s %" PRIu32, name,
			       names[index], rows[index]);
		if (prepared != STATUS_OK || program == NULL) {
			printf("%s: did not prepare (%s)\n", title,
			       status_text(prepared));
			continue;
		}
		program_report(title, program);
		/* --draft_bf16_rows: the draft layers' attention products leave the quantized trunk (THIN 6 and 8) */
		if (!batch && index < 2u && !dump_thin_bits && !dump_column_owners && dump_front_branch == 0) {
			const uint32_t trunks = quantized_trunks_of(program);
			printf("  %s: quantized trunk launches %" PRIu32 "\n", title, trunks);
			if (!dump_draft_bf16_rows &&
			    wide_rows == MODEL_WIDE_LEAST_ROWS)
				plain_quantized_trunks[index] = trunks;
			if (dump_draft_bf16_rows && trunks >= plain_quantized_trunks[index]) {
				printf("  %s: FAILED: with --draft_bf16_rows fewer than %" PRIu32 "\n", title,
				       plain_quantized_trunks[index]);
				dump_failures++;
			}
		}
		if (dump_front_branch != 0 &&
		    rows[index] <= LINEAR_MAXIMUM_ROWS)
			front_branch_report(title, program);
		if (kinds[index] == PROGRAM_BATCH &&
		    rows[index] <= LINEAR_MAXIMUM_ROWS)
			one_slab_cores = cores_of(program); /* the BATCH programs behind it are held to it */
		if (kinds[index] == PROGRAM_BATCH &&
		    rows[index] > LINEAR_MAXIMUM_ROWS) {
			thin_splits_report(title, program, rows[index]);
			thin_bits_report(title, program, rows[index]);
		}
		/* WIDE and FULL at 64 rows: --batch_thin_bits leaves the chunk programs as they are */
		if (index >= 2u && wide_rows == MODEL_WIDE_LEAST_ROWS &&
		    !dump_thin_splits && !dump_column_owners) {
			const uint64_t digest = launches_digest(program);
			if (!dump_thin_bits) {
				chunk_digests[index - 2u] = digest;
			} else if (digest != chunk_digests[index - 2u]) {
				printf("  %s: FAILED: --batch_thin_bits changed a chunk program\n",
				       title);
				dump_failures++;
			} else {
				printf("  %s: the plain recipe's launches\n",
				       title);
			}
		}
		free(program);
	}
	/* --batch_verifiers: BATCH programs of 16 and 32 rows that verify every session (8 draft walks); track D's
	 * BATCH-8 of two (a thin walk drafting for two sessions) and BATCH-12 of four */
	const uint32_t decoupled[][2] = { { 16u, DUMP_SESSIONS },
					  { 32u, DUMP_SESSIONS },
					  { 8u, 2u },
					  { 12u, 4u },
					  { 48u, DUMP_SESSIONS } };
	for (uint32_t index = 0;
	     batch && index < sizeof(decoupled) / sizeof(decoupled[0]);
	     index++) {
		struct execution_program *program = NULL;
		struct step_companion_program companions = { 0 };
		build.batch_verifiers = decoupled[index][1];
		const result_status prepared = v41_tp_program_prepare(
			&build, &test_host_memory_backend, cards, reported,
			PROGRAM_BATCH, decoupled[index][0], &program,
			&companions);
		build.batch_verifiers = 0;
		char title[64];
		(void)snprintf(title, sizeof(title),
			       "%s BATCH %" PRIu32 " verifiers %" PRIu32, name,
			       decoupled[index][0], decoupled[index][1]);
		if (prepared != STATUS_OK || program == NULL) {
			printf("%s: did not prepare (%s)\n", title,
			       status_text(prepared));
			continue;
		}
		program_report(title, program);
		if (decoupled[index][0] > LINEAR_MAXIMUM_ROWS) {
			thin_splits_report(title, program,
					   decoupled[index][0]);
			thin_bits_report(title, program, decoupled[index][0]);
		}
		free(program);
	}
	v41_tp2_model_destroy(model);
	pack_close(pack);
}

int main(void)
{
	test_host_memory_backend_reset(DUMP_CARDS, DUMP_CARD_BYTES);
	for (uint32_t card = 0; card < DUMP_CARDS; card++) {
		reported[card].arithmetic_class = ARITHMETIC_CLASS_SM_80;
		reported[card].memory_total_bytes =
			reported[card].memory_free_bytes = DUMP_REPORTED_BYTES;
		test_host_memory_backend_set_arithmetic_class(
			card, ARITHMETIC_CLASS_SM_80);
		if (test_host_memory_backend.card_open(
			    card, DUMP_CARD_BYTES, &cards[card]) != STATUS_OK)
			printf("ring_sequence_dump_test: card %" PRIu32
			       " did not open\n",
			       card);
	}
	recipe_report("THIN recipe", false, MODEL_WIDE_LEAST_ROWS);
	dump_draft_bf16_rows = true;
	recipe_report("THIN draft-bf16 recipe", false, MODEL_WIDE_LEAST_ROWS);
	dump_draft_bf16_rows = false;
	recipe_report("BATCH recipe", true, MODEL_WIDE_LEAST_ROWS);
	recipe_report("THIN recipe", false, DUMP_WIDE_ROWS);
	recipe_report("BATCH recipe", true, DUMP_WIDE_ROWS);
	dump_thin_splits = true;
	recipe_report("BATCH thin-splits recipe", true, MODEL_WIDE_LEAST_ROWS);
	dump_thin_splits = false;
	dump_thin_bits = true;
	recipe_report("BATCH thin-bits recipe", true, MODEL_WIDE_LEAST_ROWS);
	dump_grouped_rows = 48u;
	recipe_report("BATCH thin-bits grouped recipe", true,
		      MODEL_WIDE_LEAST_ROWS);
	dump_column_owners = true;
	recipe_report("BATCH column-owners recipe", true,
		      MODEL_WIDE_LEAST_ROWS);
	dump_tc_paired = 27u; /* BATCH-32 and 48 */
	recipe_report("BATCH column-owners paired recipe", true,
		      MODEL_WIDE_LEAST_ROWS);
	dump_grouped_rows = 16u; /* the default */
	for (uint32_t set_rows = 8u; set_rows <= 24u; set_rows += 8u) {
		dump_row_sets = set_rows;
		char title[64];
		(void)snprintf(title, sizeof(title), "BATCH row-sets %" PRIu32 " recipe", set_rows);
		recipe_report(title, true, MODEL_WIDE_LEAST_ROWS);
	}
	dump_row_sets = 0;
	for (dump_core_once = 1u; dump_core_once <= DUMP_CORE_ONCE_POOLS;
	     dump_core_once++) {
		core_once_ordinal = 0;
		draft_wide_ordinal = 0;
		char title[64];
		(void)snprintf(title, sizeof(title), "BATCH core-once %" PRIu32 " recipe", dump_core_once);
		recipe_report(title, true, MODEL_WIDE_LEAST_ROWS);
	}
	dump_core_once = DUMP_CORE_ONCE_BASELINE;
	dump_round_once = true;
	core_once_ordinal = 0;
	draft_wide_ordinal = 0;
	recipe_report("BATCH round-once recipe", true, MODEL_WIDE_LEAST_ROWS);
	dump_round_once = false;
	dump_draft_wide = 1u;
	core_once_ordinal = 0;
	draft_wide_ordinal = 0;
	recipe_report("BATCH draft-wide recipe", true, MODEL_WIDE_LEAST_ROWS);
	dump_draft_wide = DUMP_DRAFT_WIDE_PRODUCTS;
	core_once_ordinal = 0;
	draft_wide_ordinal = 0;
	recipe_report("BATCH draft-wide 2 recipe", true, MODEL_WIDE_LEAST_ROWS);
	dump_draft_operands = true;
	core_once_ordinal = 0;
	draft_wide_ordinal = 0;
	recipe_report("BATCH draft-wide 2 operands recipe", true,
		      MODEL_WIDE_LEAST_ROWS);
	dump_draft_operands = false;
	dump_draft_wide = DUMP_DRAFT_WIDE_CORES;
	core_once_ordinal = 0;
	draft_wide_ordinal = 0;
	recipe_report("BATCH draft-wide 3 recipe", true, MODEL_WIDE_LEAST_ROWS);
	dump_draft_wide = 0;
	dump_core_once = 0;
	dump_front_branch = 1u;
	recipe_report("THIN front-branch 1 recipe", false,
		      MODEL_WIDE_LEAST_ROWS);
	dump_front_branch = 2u;
	recipe_report("THIN front-branch 2 recipe", false,
		      MODEL_WIDE_LEAST_ROWS);
	recipe_report("BATCH front-branch 2 recipe", true,
		      MODEL_WIDE_LEAST_ROWS);
	dump_front_branch = 0;
	dump_draft_bf16_rows = true; /* the serving recipe's BATCH programs (batched draft rows) with it */
	recipe_report("BATCH draft-bf16 recipe", true, MODEL_WIDE_LEAST_ROWS);
	dump_draft_bf16_rows = false;
	dump_tc_paired = 9u; /* every BATCH step above 8 rows */
	recipe_report("BATCH column-owners paired-9 recipe", true,
		      MODEL_WIDE_LEAST_ROWS);
	dump_tc_paired = 0;
	dump_column_owners = false;
	dump_grouped_rows = 0;
	dump_thin_bits = false;
	for (uint32_t card = 0; card < DUMP_CARDS; card++)
		if (cards[card] != NULL)
			test_host_memory_backend.card_close(cards[card],
							    DUMP_WAIT_NS);
	if (dump_failures != 0) {
		printf("ring_sequence_dump_test FAILED: %" PRIu32 " failures\n",
		       dump_failures);
		return 1;
	}
	printf("ring_sequence_dump_test passed\n");
	return 0;
}
