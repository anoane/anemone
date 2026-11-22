/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cli_engine_start.c: the serve start: the program list, the cards, each program's preparation and the engine (split
 * from commands.c by whole items).
 */
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <time.h>
#include <unistd.h>

#include "models/deepseek_v41_flash/prompt/ds41f_chat_template.h"
#include "common/hooks.h"
#include "engine/storage/load/loader.h"
#include "models/deepseek_v41_flash/ds41f_geometry.h"
#include "models/deepseek_v41_flash/official/ds41f_official_reference.h"
#include "engine/formats/digest/pack_digest.h"
#include "engine/runtime/placement/placement.h"
#include "engine/storage/replicas/replicas.h"
#include "engine/engram/engram_disk.h"
#include "engine/engram/engram_fit.h"
#include "engine/storage/io/local_device.h"
#include "engine/storage/profile/storage_profile.h"
#include "engine/storage/autotune/storage_autotune.h"
#include "engine/server/server.h"
#include "engine/runtime/placement/session_fit.h"
#include "cli_stats_client.h"
#include "cli_policy.h"
#include "engine/storage/io/storage_exl3.h"
#include "engine/library/exl3/exl3_tc_records.h"
#include "cli_model_folder.h"
#include "models/model_registry.h"
#include "engine/runtime/scheduler/serving_cursor.h"
#include "cli_axes.h"
#include "engine/runtime/step/step.h"
#include "engine/runtime/step/step_images.h"
#include "engine/input/tokenizer/tokenizer.h"
#include "engine/runtime/speculation/draft_lookup.h"
#include "engine/formats/pack/tokenizer_tables_pack.h"
#include "common/json.h"
#include "engine/engram/engram_step.h"
#include "engine/engram/engram_hot.h"
#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_model.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_peer.h"
#include "commands.h"
#include "cli_golden_capture.h"

#include "cli_internal.h"

#define CHUNK_ROWS_DIVISOR \
	4u /* wide and full programs of wide_rows, a quarter of it, ..., down to the least wide rows */
#define THIN_TAIL_KINDS \
	2u /* --prefill_thin_tail with FRONT_TAIL_ALL: full and wide thin-tail programs */
#define THIN_TAIL_LEAST_ROWS \
	16u /* --prefill_thin_tail: full programs of 16 to 56 rows, 8 apart (thin steps' forms) */
#define PREFILL_CHUNK_CAPS \
	2u /* global and decoder-only caps may each need another prepared chunk size */
#define LADDER_INSERTED_SIZES \
	2u /* --prefill_ladder_1024's and --prefill_ladder_1536's sizes, inserted behind wide_rows */
#define LADDER_SECOND 1u /* chunk_rows' place behind wide_rows */
#define LADDER_HALVING \
	2u /* above DS41F_CHUNK_LADDER_TOP: wide_rows, then 512 and 256 by halving */
static struct step_companion_program tensor_companions[MOST_PROGRAMS];

/* The exact reference recipe admits text even when the pack also contains a vision tower. */
bool model_images_are_enabled(const struct ds41f_plan *model_plan)
{
	return model_plan != NULL && model_plan->vision.enabled &&
	       model_plan->limits.official_reference == 0;
}

/* The programs a serve prepares, in engine_failed_to_start's order (the step driver's), of the parsed options alone: on
 * a TP split with --batch_rows its BATCH programs, else THIN ones of 1, 2 and the most rows; then the chunk ladder's
 * WIDE ones and with --full_steps its FULL ones; on a TP split the tails and thin tails. */
void serve_programs_choose(bool tensor, struct serve_programs *chosen)
{
	*chosen = (struct serve_programs){ 0 };
	uint32_t most_rows = step_rows < LINEAR_MAXIMUM_ROWS ?
				     step_rows :
				     LINEAR_MAXIMUM_ROWS;
	const bool prepares_pair = session_slots > 1 &&
				   most_rows > SERVING_PAIR_ROWS;
	const uint32_t program_rows[MOST_THIN_PROGRAMS] = {
		1, prepares_pair ? SERVING_PAIR_ROWS : most_rows, most_rows
	};
	const uint32_t programs_to_prepare = prepares_pair ?
						     MOST_THIN_PROGRAMS :
					     most_rows > 1 ? SERVING_PAIR_ROWS :
							     1;
	/* --batch_rows: BATCH programs of those rows in place of the thin ones (the engine's decode steps, any
	 * sessions) */
	chosen->batch = batch_rows[0] != 0 && tensor;
	chosen->verifiers_given = batch_verifiers[0] != 0;
	for (uint32_t index = 0;
	     index < MOST_BATCH_PROGRAMS && batch_rows[index] != 0; index++)
		for (uint32_t other = 0; other < index; other++)
			chosen->rows_repeat = chosen->rows_repeat ||
					      batch_rows[other] ==
						      batch_rows[index];
	for (uint32_t index = 0; chosen->batch && index < MOST_BATCH_PROGRAMS &&
				 batch_rows[index] != 0;
	     index++) {
		chosen->programs[chosen->count++] = (struct serve_program){
			PROGRAM_BATCH, batch_rows[index],
			batch_verifiers[index], SERVE_BATCH_ROWS
		};
		chosen->verifiers_given = chosen->verifiers_given &&
					  batch_verifiers[index] != 0;
		most_rows = batch_rows[index] > most_rows ? batch_rows[index] :
							    most_rows;
	}
	for (uint32_t thin = 0; !chosen->batch && thin < programs_to_prepare;
	     thin++)
		chosen->programs[chosen->count++] = (struct serve_program){
			PROGRAM_THIN, program_rows[thin], 0, SERVE_THIN_ROWS
		};
	chosen->first_count = chosen->count;
	chosen->most_rows = most_rows;
	/* behind the thin programs, chunks of wide_rows, a quarter of it, ..., at least the least wide rows, which ends
	 * the ladder (a prompt's rows below it go in thin steps): wide ones for the front rows, full ones for the rows
	 * behind the front (the engine picks a size: scheduler.c). Above DS41F_CHUNK_LADDER_TOP: wide_rows, 512, 256,
	 * 64 (the exact plan's whole multiples of 256 and its tail's 64; the fast plan's 2,048 + 512 + 64 of a 2.7K
	 * prompt). */
	uint32_t chunk_rows[MOST_CHUNK_PROGRAMS + LADDER_INSERTED_SIZES +
			    DS41F_LADDER_FINE_COUNT + PREFILL_CHUNK_CAPS],
		chunk_count = 0;
	for (uint32_t rows = wide_rows;
	     rows >= STEP_WIDE_LEAST_ROWS &&
			     chunk_count<MOST_CHUNK_PROGRAMS; rows = rows>
				     DS41F_CHUNK_LADDER_TOP ?
		     DS41F_CHUNK_LADDER_TOP / LADDER_HALVING :
	     rows == DS41F_CHUNK_LADDER_TOP / LADDER_HALVING &&
			     wide_rows > DS41F_CHUNK_LADDER_TOP ?
		     rows / LADDER_HALVING :
		     rows / CHUNK_ROWS_DIVISOR)
		chunk_rows[chunk_count++] = rows;
	if (chunk_count != 0 && chunk_count < MOST_CHUNK_PROGRAMS &&
	    chunk_rows[chunk_count - 1u] != STEP_WIDE_LEAST_ROWS)
		chunk_rows[chunk_count++] = STEP_WIDE_LEAST_ROWS;
	/* --prefill_ladder_1024: DS41F_CHUNK_LADDER_TOP's own programs behind wide_rows' (the exact plan's whole
	 * multiples of 256: a region of 1,024 to 2,047 whole rows in one step fewer; BATCH programs take the thin ones'
	 * place, so MOST_PROGRAMS holds them) */
	if (prefill_ladder_1024 != 0 && chunk_count > 1u &&
	    chunk_rows[0] > DS41F_CHUNK_LADDER_TOP) {
		memmove(&chunk_rows[LADDER_SECOND + 1u],
			&chunk_rows[LADDER_SECOND],
			(chunk_count - 1u) * sizeof(chunk_rows[0]));
		chunk_rows[LADDER_SECOND] = DS41F_CHUNK_LADDER_TOP;
		chunk_count++;
	}
	/* --prefill_ladder_1536: DS41F_LADDER_1536_ROWS' programs too (a region of 1,536 to 2,047 whole rows, e.g.
	 * Curtis's 2K cell's 1,024 + 512, in one step; each split half 768 rows) */
	if (prefill_ladder_1536 != 0 && chunk_count > 1u &&
	    chunk_rows[0] > DS41F_LADDER_1536_ROWS) {
		memmove(&chunk_rows[LADDER_SECOND + 1u],
			&chunk_rows[LADDER_SECOND],
			(chunk_count - 1u) * sizeof(chunk_rows[0]));
		chunk_rows[LADDER_SECOND] = DS41F_LADDER_1536_ROWS;
		chunk_count++;
	}
	/* Fine ladder and explicit caps, in the ladder's order. A single slot cannot decode beside another prompt, so
	 * its decoder-only cap prepares no extra programs. The global cap is also used to qualify its numerics. */
	uint32_t extra_rows[DS41F_LADDER_FINE_COUNT + PREFILL_CHUNK_CAPS] = {
		0
	};
	for (uint32_t fine = 0;
	     prefill_ladder_fine != 0 && fine < DS41F_LADDER_FINE_COUNT; fine++)
		extra_rows[fine] = ds41f_ladder_fine_rows[fine];
	extra_rows[DS41F_LADDER_FINE_COUNT] = prefill_chunk_rows;
	extra_rows[DS41F_LADDER_FINE_COUNT + 1u] =
		session_slots > 1u ? prefill_decode_rows : 0u;
	for (uint32_t extra = 0;
	     chunk_count > 1u &&
	     extra < DS41F_LADDER_FINE_COUNT + PREFILL_CHUNK_CAPS;
	     extra++) {
		if (extra_rows[extra] == 0)
			continue;
		uint32_t at = 0;
		while (at < chunk_count && chunk_rows[at] > extra_rows[extra])
			at++;
		if (at == 0 || at == chunk_count ||
		    chunk_rows[at] == extra_rows[extra])
			continue;
		memmove(&chunk_rows[at + 1u], &chunk_rows[at],
			(chunk_count - at) * sizeof(chunk_rows[0]));
		chunk_rows[at] = extra_rows[extra];
		chunk_count++;
	}
	const enum program_kind chunk_kinds[] = {
		PROGRAM_WIDE, full_steps != 0 ? PROGRAM_FULL : PROGRAM_WIDE
	};
	for (uint32_t kind = 0;
	     kind < sizeof(chunk_kinds) / sizeof(chunk_kinds[0]) &&
	     (kind == 0 || chunk_kinds[kind] != chunk_kinds[0]);
	     kind++)
		for (uint32_t step = 0;
		     step < chunk_count && chosen->count < MOST_PROGRAMS;
		     step++)
			chosen->programs[chosen->count++] =
				(struct serve_program){ chunk_kinds[kind],
							chunk_rows[step], 0,
							SERVE_CHUNK_LADDER };
	/* --prefill_tail_rows: a full program of the exact plan's tail, two 64-row chunks a step in their forms (and
	 * with 320 one of five, with 448 one of five and one of seven); --prefill_front_tail: a wide one of two for the
	 * front region's tail too */
	const struct {
		enum program_kind kind;
		uint32_t rows;
		bool wanted;
	} tails[] = { { PROGRAM_FULL, DS41F_TAIL_PROGRAM_ROWS,
			prefill_tail_rows != 0 },
		      { PROGRAM_FULL, DS41F_TAIL_LONG_ROWS,
			prefill_tail_rows >= DS41F_TAIL_LONG_ROWS },
		      { PROGRAM_FULL, DS41F_TAIL_WIDEST_ROWS,
			prefill_tail_rows == DS41F_TAIL_WIDEST_ROWS },
		      { PROGRAM_WIDE, DS41F_TAIL_PROGRAM_ROWS,
			prefill_tail_rows != 0 && prefill_front_tail != 0 },
		      { PROGRAM_WIDE, DS41F_TAIL_LONG_ROWS,
			prefill_tail_rows >= DS41F_TAIL_LONG_ROWS &&
				prefill_front_tail == FRONT_TAIL_ALL },
		      { PROGRAM_WIDE, DS41F_TAIL_WIDEST_ROWS,
			prefill_tail_rows == DS41F_TAIL_WIDEST_ROWS &&
				prefill_front_tail == FRONT_TAIL_ALL } };
	for (uint32_t tail = 0; tail < sizeof(tails) / sizeof(tails[0]) &&
				tensor && chosen->count < MOST_PROGRAMS;
	     tail++)
		if (tails[tail].wanted)
			chosen->programs[chosen->count++] =
				(struct serve_program){ tails[tail].kind,
							tails[tail].rows, 0,
							SERVE_TAIL };
	/* --prefill_thin_tail: full programs of 16 to 56 rows, 8 apart: the tail's rows short of a 64-row chunk but its
	 * row of logits in one step, each row in a thin step's forms; with --prefill_front_tail=2 wide ones too */
	for (uint32_t kind = 0;
	     kind <
	     (prefill_front_tail == FRONT_TAIL_ALL ? THIN_TAIL_KINDS : 1u);
	     kind++)
		for (uint32_t rows = THIN_TAIL_LEAST_ROWS;
		     prefill_thin_tail != 0 && rows < STEP_WIDE_LEAST_ROWS &&
		     tensor && chosen->count < MOST_PROGRAMS;
		     rows += LINEAR_MAXIMUM_ROWS)
			chosen->programs[chosen->count++] =
				(struct serve_program){
					kind == 0 ? PROGRAM_FULL : PROGRAM_WIDE,
					rows, 0, SERVE_THIN_TAIL
				};
}

/* The step line of a program of the serve's list */
static const char *serve_program_step(const struct serve_program *own)
{
	static const char *const steps[SERVE_PROGRAM_RULES] = {
		"prepare a BATCH program", "prepare a program",
		"prepare a wide program", "prepare a tail program",
		"prepare a thin-tail program"
	};
	return own->rule == SERVE_CHUNK_LADDER && own->kind == PROGRAM_FULL ?
		       "prepare a full program" :
		       steps[own->rule];
}

/* A program of the serve's list on target, a TP split's with its companions */
MUST_CHECK result_status serve_program_prepare(
	const struct serve_program_target *target,
	const struct serve_program *own, struct execution_program **program_out,
	struct step_companion_program *companions_out)
{
	if (target->tensor_description == NULL)
		return target->model->program_prepare(
			target->plan, target->backend, target->cards,
			target->descriptions, own->kind, own->rows,
			program_out);
	target->tensor_description->batch_verifiers = own->verifiers;
	return own->kind == PROGRAM_THIN ?
		       ds41f_tp2_program_prepare(target->tensor_description,
						 target->backend, target->cards,
						 target->descriptions,
						 own->rows, program_out,
						 companions_out) :
		       ds41f_tp_program_prepare(target->tensor_description,
						target->backend, target->cards,
						target->descriptions, own->kind,
						own->rows, program_out,
						companions_out);
}

/* The image program of a text program on target: of its kind and rows */
MUST_CHECK result_status
serve_image_program_prepare(const struct serve_program_target *target,
			    const struct execution_program *text,
			    struct execution_program **image_out)
{
	return ds41f_image_program_prepare(target->plan, target->backend,
					   target->cards, target->descriptions,
					   text->kind, text->row_count,
					   image_out);
}

/* --original_head_offload=2's flight banks on target, over count text programs and their image programs (or NULL) */
MUST_CHECK result_status serve_head_banks_create(
	const struct serve_program_target *target,
	const struct execution_program *const *texts,
	const struct execution_program *const *images, uint32_t count,
	struct ds41f_head_program_banks **banks_out)
{
	return ds41f_original_head_banks_create(target->plan, target->backend,
						target->cards, texts, images,
						count, banks_out);
}

/* Once the pack is loaded: the n-gram tables, the state, the programs, the driver and the engine over them. Every
 * step ends as step_failed says; true when one failed. */
MUST_CHECK bool
engine_failed_to_start(const struct engine_description_record *description)
{
	const bool image_programs_enabled = model_images_are_enabled(plan);
	struct serve_programs chosen;
	serve_programs_choose(tensor_model != NULL, &chosen);
	result_status status =
		engram != NULL ?
			STATUS_OK :
			engram_open(storage_operations, reader, pack, &engram);
	status = status == STATUS_UNSUPPORTED ?
			 STATUS_OK :
			 status; /* a model without n-grams */
	const struct engram_description *const tables =
		engram != NULL ? engram_description_of(engram) : NULL;
	const void *const host_arena = tensor_model != NULL ?
					       tensor_model->host_arena :
					       result.host_arena;
	const uint64_t *const host_offsets =
		tensor_model != NULL ? tensor_model->host_offsets :
				       layout->byte_offset_of_extent;
	for (uint32_t table = 0;
	     engram_disk == NULL && tables != NULL &&
	     table < tables->table_count && status == STATUS_OK;
	     table++) {
		const uint32_t extent =
			pack_units_of(pack)[tables->table_unit_index[table]]
				.first_extent_index;
		status = host_arena == NULL ?
				 STATUS_INVALID_ARGUMENT :
				 engram_attach_entry(
					 engram, table,
					 (const unsigned char *)host_arena +
						 host_offsets[extent]);
	}
	bool failed = step_failed("open the n-gram tables", status) ||
		      (tensor_model == NULL &&
		       step_failed("initialise the state",
				   instance.operations->arena_initialise(
					   plan, &backend_cuda_value, cards,
					   CARD_WAIT_LIMIT_NANOSECONDS)));
	failed = failed ||
		 (original_head_offload != 0 &&
		  step_failed("load and qualify original target head offload",
			      ds41f_original_head_initialise(
				      plan, pack, storage_operations, reader,
				      &backend_cuda_value, cards,
				      CARD_WAIT_LIMIT_NANOSECONDS)));
	if (!failed && original_head_offload == DS41F_HEAD_MODE_LEASED)
		(void)fprintf(
			stderr,
			"original target head offload: 4 x %u BF16 rows; CMP dense1/stream148; "
			"native PRO normalization/sampler/draft; shared planes under PRO visit lease\n",
			DS41F_HEAD_SHARD_ROWS);
	if (!failed && original_head_offload == DS41F_HEAD_MODE_BANKED)
		(void)fprintf(
			stderr,
			"original target head offload: %u immutable flight banks; CMP dense1/stream148; "
			"native PRO normalization/sampler/draft; owner-scoped PRO visit ordering\n",
			plan->original_head.bank_count);
	struct ds41f_tp2_program_description tensor_description = { 0 };
	failed = failed ||
		 (tensor_model != NULL &&
		  step_failed("describe TP programs",
			      ds41f_tp2_model_program_description(
				      tensor_model, &tensor_description)));
	const struct serve_program_target target = {
		.model = instance.operations,
		.plan = plan,
		.backend = &backend_cuda_value,
		.cards = cards,
		.descriptions = descriptions,
		.tensor_description =
			tensor_model != NULL ? &tensor_description : NULL
	};
	const uint32_t row_capacity = effective_row_capacity();
	program_count = 0;
	uint32_t verifiers[MOST_PROGRAMS] = {
		0
	}; /* --batch_verifiers: each BATCH program's, the others 0 */
	failed = failed ||
		 (chosen.batch && chosen.rows_repeat &&
		  batch_verifiers[0] == 0 &&
		  step_failed(
			  "a repeated BATCH row count needs --batch_verifiers",
			  STATUS_INVALID_ARGUMENT));
	/* the BATCH or THIN programs first, each THIN one admitted for the serve's rows and sessions with its experts
	 * resident; a failed THIN one counts, so the stop destroys what it made */
	uint32_t entry = 0;
	for (; entry < chosen.first_count && !failed; entry++) {
		const struct serve_program *const own = &chosen.programs[entry];
		verifiers[program_count] = own->verifiers;
		failed = step_failed(
			serve_program_step(own),
			serve_program_prepare(
				&target, own, &programs[program_count],
				&tensor_companions[program_count]));
		if (own->rule == SERVE_BATCH_ROWS) {
			program_count += failed ? 0u : 1u;
			continue;
		}
		if (!failed &&
		    !serving_program_matches(
			    programs[program_count], own->rows, row_capacity,
			    session_slots, description->context_token_limit)) {
			failed = step_failed(
				"require admitted row/session programs",
				STATUS_UNSUPPORTED);
		}
		if (!failed &&
		    (programs[program_count]->host_expert_count != 0 ||
		     programs[program_count]->expert_pause_count != 0)) {
			failed = step_failed("require resident experts",
					     STATUS_UNSUPPORTED);
		}
		program_count++;
	}
	failed = failed ||
		 (batch_verifiers[0] != 0 && !chosen.verifiers_given &&
		  step_failed("one verifier count a BATCH program",
			      STATUS_INVALID_ARGUMENT));
	for (; entry < chosen.count && !failed; entry++) {
		failed =
			step_failed(serve_program_step(&chosen.programs[entry]),
				    serve_program_prepare(
					    &target, &chosen.programs[entry],
					    &programs[program_count],
					    &tensor_companions[program_count]));
		program_count += failed ? 0u : 1u;
	}
	for (uint32_t index = 0;
	     !failed && image_programs_enabled && index < program_count;
	     index++) {
		failed = step_failed(
			"prepare an image program",
			serve_image_program_prepare(&target, programs[index],
						    &image_programs[index]));
	}
	if (!failed && original_head_offload == DS41F_HEAD_MODE_BANKED)
		failed = step_failed(
			"prepare immutable original head program banks",
			serve_head_banks_create(
				&target,
				(const struct execution_program *const *)
					programs,
				image_programs_enabled ?
					(const struct execution_program *const *)
						image_programs :
					NULL,
				program_count, &head_program_banks));
	if (head_program_banks != NULL) {
		const struct ds41f_head_bank_memory memory =
			ds41f_original_head_banks_memory(head_program_banks);
		(void)fprintf(
			stderr,
			"original head program banks: host_heap_bytes=%zu source_launch_bytes=%zu "
			"source_leases=%zu thin_programs=%u programs=%u banks=%u program_metadata_bytes=%zu\n",
			memory.heap_bytes, memory.source_launch_bytes,
			memory.source_lease_count, memory.thin_programs,
			memory.program_count, memory.bank_count,
			memory.program_metadata_bytes);
	}
	const struct step_flight_programs *const head_banks =
		ds41f_original_head_bank_tables(head_program_banks);
	return failed ||
	       step_failed(
		       "create the driver",
		       step_driver_create(
			       &backend_cuda_value, cards,
			       head_banks != NULL ?
				       head_banks[0].programs :
				       (const struct execution_program *const *)
					       programs,
			       program_count, engram,
			       CARD_WAIT_LIMIT_NANOSECONDS, &driver)) ||
	       (chosen.verifiers_given &&
		step_failed("attach BATCH verifiers",
			    step_batch_verifiers_attach(driver, verifiers,
							program_count))) ||
	       (tensor_model != NULL &&
		step_failed("attach TP groups",
			    step_groups_attach(
				    driver, tensor_companions, program_count,
				    tensor_model->stages[0]
					    ->description.rank_count))) ||
	       (engram_shared != 0 && step_failed("share TP n-gram gathers",
						  step_engram_share(driver))) ||
	       (tensor_model != NULL && tp_parallel_launch != 0 &&
		step_failed("start the TP launchers",
			    step_launchers_start(driver))) ||
	       (plan->limits.official_reference == 0 &&
		step_failed("attach image encoder",
			    step_images_attach(
				    driver, plan,
				    descriptions[0].arithmetic_class))) ||
	       (image_programs_enabled &&
		step_failed(
			"attach image programs",
			step_image_programs_attach(
				driver,
				head_banks != NULL ?
					head_banks[0].image_programs :
					(const struct execution_program *const *)
						image_programs,
				program_count))) ||
	       (head_banks != NULL &&
		step_failed("attach immutable original head flight banks",
			    step_flight_programs_attach(
				    driver, head_banks,
				    plan->original_head.bank_count,
				    program_count,
				    ds41f_original_head_owner_guard(
					    head_program_banks)))) ||
	       step_failed("page-lock host experts",
			   step_host_experts_attach(
				   driver, host_arena,
				   tensor_model != NULL ?
					   tensor_model->host_bytes :
					   layout->host_byte_count)) ||
	       (step_graphs != 0 && step_failed("prepare step graphs",
						step_graphs_prepare(driver))) ||
	       ((pipeline_steps > 1 || batch_rows[0] != 0) &&
		step_failed(
			"start the step pipeline",
			tp_pipeline_overlap != 0 ?
				step_pipeline_overlap_start(driver,
							    pipeline_steps) :
				step_pipeline_start(driver, pipeline_steps))) ||
	       step_failed("start the engine",
			   engine_create(driver, description, chosen.most_rows,
					 &engine));
}

/* The digest of the words that hold range on card (the backend digests whole words). */
MUST_CHECK result_status aligned_digest(uint32_t card,
					struct linear_range range,
					uint64_t *digest_out)
{
	const uint64_t word = PACK_DIGEST_WORD_BYTES,
		       first = range.byte_offset / word * word;
	const uint64_t end =
		(range.byte_offset + range.byte_count + word - 1u) / word *
		word;
	return backend_cuda_value.digest(cards[card], first, end - first,
					 CARD_WAIT_LIMIT_NANOSECONDS,
					 digest_out);
}
