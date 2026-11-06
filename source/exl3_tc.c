/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* exl3_tc.c: the five launches of exl3_tc.h. The record's ranges are checked against the arena and each other; the
 * instances follow the pair count (short or long slices), the rank's intermediate and the bit rate. */
#include "anemone/exl3_tc.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "anemone/linear_records.h"
#include "anemone/model_layout.h"
#include "anemone/router_records.h"
#include "anemone/step_records.h"

#define OK STATUS_OK
#define TC_TRANSFORM 128u
#define TC_K3_BITS_X2 6u
#define TC_K2_BITS_X2 4u
#define TC_GATE_UP_SHORT_STEPS 20u /* k16 steps of a gate/up K slice: short 16 slices, long 8 */
#define TC_GATE_UP_LONG_STEPS 40u
#define TC_COLUMN_TILES 8u
#define TC_RING_STAGES 4u
#define TC_PRODUCT_WARPS (EXL3_TC_PRODUCT_THREADS / 32u)
#define TC_MOST_ROTATION \
	(EXL3_TC_THREE_WAY_FIRST + EXL3_TC_THREE_WAY_RANKS - 1u) /* 0 to 5 */
#define TC_NAME_BYTES 64u

struct tc_region {
	uint64_t offset, bytes;
};

static bool tc_regions_valid(const struct tc_region *regions, uint32_t count, uint64_t arena_bytes)
{
	for (uint32_t region = 0; region < count; region++) {
		if (regions[region].offset % LINEAR_OFFSET_ALIGNMENT_BYTES !=
			    0 ||
		    regions[region].offset > arena_bytes ||
		    regions[region].bytes == 0 ||
		    regions[region].bytes >
			    arena_bytes - regions[region].offset)
			return false;
		for (uint32_t earlier = 0; earlier < region; earlier++)
			if (regions[region].offset < regions[earlier].offset + regions[earlier].bytes &&
			    regions[earlier].offset < regions[region].offset + regions[region].bytes)
				return false;
	}
	return true;
}

static uint32_t tc_blocks(uint64_t tasks)
{
	return tasks < EXL3_TC_MOST_BLOCKS ? (uint32_t)tasks :
					     EXL3_TC_MOST_BLOCKS;
}

MUST_CHECK static result_status
tc_launch_bytes(const void *record, uint32_t record_bytes, const char *file,
		const char *name, uint32_t grid, uint32_t block,
		uint32_t shared, const struct backend_operations *backend,
		struct device_card *card, struct kernel_launch *launch_out)
{
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, file, name, record_bytes, &kernel);
	if (status != OK)
		return status;
	if (kernel == BACKEND_NO_KERNEL)
		return STATUS_UNSUPPORTED;
	struct kernel_launch launch = { .kernel = kernel,
					.grid = { grid, 1, 1 },
					.block = { block, 1, 1 },
					.dynamic_shared_byte_count = shared,
					.argument_byte_count = record_bytes };
	memcpy(launch.argument, record, record_bytes);
	*launch_out = launch;
	return OK;
}

MUST_CHECK static result_status
tc_launch(const struct exl3_tc_record *record, const char *name, uint32_t grid,
	  uint32_t block, uint32_t shared,
	  const struct backend_operations *backend, struct device_card *card,
	  struct kernel_launch *launch_out)
{
	uint32_t kernel = BACKEND_NO_KERNEL;
	const result_status status = backend->kernel_prepare(
		card, EXL3_TC_KERNEL_FILE, name, sizeof(*record), &kernel);
	if (status != OK)
		return status;
	if (kernel == BACKEND_NO_KERNEL)
		return STATUS_UNSUPPORTED;
	struct kernel_launch launch = { .kernel = kernel,
					.grid = { grid, 1, 1 },
					.block = { block, 1, 1 },
					.dynamic_shared_byte_count = shared,
					.argument_byte_count =
						sizeof(*record) };
	memcpy(launch.argument, record, sizeof(*record));
	*launch_out = launch;
	return OK;
}

/* The forms of a call from EXL3_TC_LONG_PAIRS pairs on. */
enum tc_forms { TC_LONG_FORMS, TC_SHORT_FORMS, TC_PAIRED_FORMS };

/* The five launches of a call: the long forms from EXL3_TC_LONG_PAIRS pairs on, or the short ones (paired:
 * their products two slices an item at any pair count, without rotation). */
static result_status
tc_prepare(const struct exl3_tc_record *record, uint64_t arena_byte_count,
	   const struct backend_operations *backend, struct device_card *card,
	   struct kernel_launch launches_out[EXL3_TC_LAUNCHES],
	   enum tc_forms forms)
{
	const bool short_forms = forms != TC_LONG_FORMS;
	if (record == NULL || backend == NULL ||
	    backend->kernel_prepare == NULL || card == NULL ||
	    launches_out == NULL || record->row_count == 0 ||
	    (record->intermediate != 640u && record->intermediate != 512u) ||
	    (record->bits_x2 != TC_K3_BITS_X2 &&
	     record->bits_x2 != TC_K2_BITS_X2) ||
	    record->job_capacity == 0 || record->pass != 0 ||
	    !(record->activation_limit >= 0.0f) ||
	    record->rotation > TC_MOST_ROTATION ||
	    (record->rotation != 0 &&
	     record->intermediate != EXL3_TC_MIXED_WIDTH) ||
	    record->shared_sum > 1u)
		return STATUS_INVALID_ARGUMENT;
	const uint64_t pairs =
		(uint64_t)record->row_count * EXL3_TC_PAIRS_PER_ROW;
	if (pairs > EXL3_TC_MOST_PAIRS || record->job_capacity > pairs)
		return STATUS_INVALID_ARGUMENT;
	const struct tc_region regions[] = {
		{ record->jobs_offset,
		  (uint64_t)record->job_capacity *
			  sizeof(struct expert_job_record) },
		{ record->hidden_offset,
		  (uint64_t)record->row_count * EXL3_TC_HIDDEN * 2u },
		{ record->workspace_offset,
		  exl3_tc_workspace_bytes(pairs, record->intermediate) },
		{ record->output_offset,
		  (uint64_t)record->row_count * EXL3_TC_HIDDEN * 4u },
		{ record->shared_offset,
		  (uint64_t)record->row_count * EXL3_TC_HIDDEN * 2u }
		/* shared_sum's */
	};
	const uint32_t region_count =
		(uint32_t)(sizeof(regions) / sizeof(regions[0])) - (record->shared_sum != 0 ? 0u : 1u);
	if (!tc_regions_valid(regions, region_count, arena_byte_count))
		return STATUS_INVALID_ARGUMENT;
	const bool is_long = !short_forms && pairs >= EXL3_TC_LONG_PAIRS,
		   k3 = record->bits_x2 == TC_K3_BITS_X2;
	const bool mixed = record->rotation != 0;
	const bool paired = forms == TC_PAIRED_FORMS && !mixed;
	const uint32_t step_bytes = TC_COLUMN_TILES * 16u * record->bits_x2;
	const uint32_t products_shared_bytes =
		TC_RING_STAGES * step_bytes * TC_PRODUCT_WARPS +
		(paired ? EXL3_TC_STASH_BYTES * TC_PRODUCT_WARPS : 0u);
	/* gate/up: short 20 steps a slice, long 40; down: short 2 slices (20 steps at 640, 16 at 512), long the whole K;
	 * mixed (exl3_tc_records.h): the 640 forms, a narrow job's down at its own steps (t20n16, t40n32) */
	const uint32_t down_steps = is_long ? record->intermediate / 16u : record->intermediate / 32u;
	const uint32_t narrow_steps =
		(record->intermediate - EXL3_TC_NARROWER) /
		(is_long ? 16u : 32u);
	const char *const bits = k3 ? "k3" : "k2";
	char operands[TC_NAME_BYTES], gate_up[TC_NAME_BYTES], activation[TC_NAME_BYTES], down_name[TC_NAME_BYTES];
	(void)snprintf(operands, sizeof(operands), "exl3_tc_operands_%s%s",
		       is_long ? "long" : "short", mixed ? "_m" : "");
	(void)snprintf(gate_up, sizeof(gate_up), "exl3_tc_products_t%u%s_%s%s",
		       is_long ? 40u : 20u, paired ? "x2" : "", bits,
		       mixed ? "_m" : "");
	(void)snprintf(activation, sizeof(activation),
		       "exl3_tc_activate_s%u_%c%u", is_long ? 8u : 16u,
		       mixed ? 'm' : 'i', record->intermediate);
	if (mixed)
		(void)snprintf(down_name, sizeof(down_name),
			       "exl3_tc_products_t%un%u_%s", down_steps,
			       narrow_steps, bits);
	else
		(void)snprintf(down_name, sizeof(down_name),
			       "exl3_tc_products_t%u%s_%s", down_steps,
			       paired ? "x2" : "", bits);
	struct kernel_launch launches[EXL3_TC_LAUNCHES];
	struct exl3_tc_record down = *record;
	down.pass = 1u;
	/* GNmac's geometry: an operand block a (pair, projection) behind block 0, an activation block a pair */
	const uint64_t operand_items = pairs * 2u;
	result_status status = tc_launch(record, operands,
					 1u + (uint32_t)operand_items,
					 EXL3_TC_OPERAND_THREADS, 0, backend,
					 card, &launches[0]);
	if (status == OK)
		status = tc_launch(record, gate_up, EXL3_TC_PRODUCT_BLOCKS,
				   EXL3_TC_PRODUCT_THREADS,
				   products_shared_bytes, backend, card,
				   &launches[1]);
	if (status == OK)
		status = tc_launch(record, activation,
				   pairs < EXL3_TC_ACTIVATE_GRID ?
					   (uint32_t)pairs :
					   EXL3_TC_ACTIVATE_GRID,
				   32u * (record->intermediate / TC_TRANSFORM),
				   0, backend, card, &launches[2]);
	if (status == OK)
		status = tc_launch(&down, down_name, EXL3_TC_PRODUCT_BLOCKS,
				   EXL3_TC_PRODUCT_THREADS,
				   products_shared_bytes, backend, card,
				   &launches[3]);
	if (status == OK)
		status = tc_launch(record,
				   is_long ? "exl3_tc_combine_s1" :
					     "exl3_tc_combine_s2",
				   tc_blocks((uint64_t)record->row_count *
					     (EXL3_TC_HIDDEN / TC_TRANSFORM)),
				   EXL3_TC_COMBINE_THREADS,
				   (uint32_t)(EXL3_TC_PAIRS_PER_ROW *
					      (TC_TRANSFORM * sizeof(double) +
					       sizeof(float) + sizeof(int))),
				   backend, card, &launches[4]);
	if (status == OK)
		memcpy(launches_out, launches, sizeof(launches));
	return status;
}

result_status
exl3_tc_prepare(const struct exl3_tc_record *record, uint64_t arena_byte_count,
		const struct backend_operations *backend,
		struct device_card *card,
		struct kernel_launch launches_out[EXL3_TC_LAUNCHES])
{
	return tc_prepare(record, arena_byte_count, backend, card, launches_out, TC_LONG_FORMS);
}

result_status exl3_tc_prepare_short(
	const struct exl3_tc_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch launches_out[EXL3_TC_LAUNCHES])
{
	return tc_prepare(record, arena_byte_count, backend, card, launches_out, TC_SHORT_FORMS);
}

result_status exl3_tc_prepare_paired(
	const struct exl3_tc_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch launches_out[EXL3_TC_LAUNCHES])
{
	return tc_prepare(record, arena_byte_count, backend, card, launches_out, TC_PAIRED_FORMS);
}

/* The prefill form's GEMM shape bytes of exl3_tcp (kernel_exl3_tcp.cuh tcp_shape): stages of 64-row operand tiles and
 * the block's trellis tiles of four k16 steps, three deep. */
#define TCP_STAGES 3u
#define TCP_OPERAND_STAGE_BYTES (2u * 64u * 64u * 2u)
#define TCP_K_STEPS 4u
#define TCP_GEMM_WARPS 8u
#define TCP_ROUTE_ARRAYS 5u       /* tcp_route's shared int arrays of an expert each (and one more int) */
#define TCP_PAIRED_ROUTE_EXTRA 2u /* tcp_route_paired's two ints more */
static uint32_t tcp_shared_bytes(uint32_t tiles_per_warp, uint32_t bits_x2)
{
	return TCP_STAGES * (TCP_OPERAND_STAGE_BYTES + TCP_K_STEPS * TCP_GEMM_WARPS * tiles_per_warp * 16u * bits_x2);
}

/* A 640-column record's gate/up GEMMs: i640 (five 128-column blocks of a tile a warp) or h640 (two 256-column blocks
 * of two tiles a warp, then 128 columns of one): the same bits, each tile's operand rows read three times, not five.
 * exl3_tcp_forms_card_test (rc03_card): the h640 call 4.5% faster at 512 rows, 2.6% at 1,024 (K2 5.6%), even at
 * 2,048 (gate, then up: its tiles of more than 32 rows take two passes): the default takes it up to the joint rows. */
enum tcp_form { TCP_FORM_DEFAULT, TCP_FORM_BASE, TCP_FORM_HYBRID };
MUST_CHECK static result_status
tcp_prepare(const struct exl3_tcp_record *record, uint64_t arena_byte_count,
	    const struct backend_operations *backend, struct device_card *card,
	    struct kernel_launch launches_out[EXL3_TCP_MOST_LAUNCHES],
	    uint32_t *launch_count_out, enum tcp_form form)
{
	if (record == NULL || backend == NULL ||
	    backend->kernel_prepare == NULL || card == NULL ||
	    launches_out == NULL || launch_count_out == NULL ||
	    record->row_count == 0 || record->row_count > EXL3_TCP_CHUNK_ROWS ||
	    (record->intermediate != 640u && record->intermediate != 512u) ||
	    (record->bits_x2 != TC_K3_BITS_X2 &&
	     record->bits_x2 != TC_K2_BITS_X2) ||
	    record->pass != 0 || record->rotation > TC_MOST_ROTATION ||
	    (record->rotation != 0 &&
	     record->intermediate != EXL3_TC_MIXED_WIDTH) ||
	    !(record->activation_limit >= 0.0f) ||
	    record->half_rows >= record->row_count || record->part != 0 ||
	    record->two_blocks > 1u ||
	    (record->cone_layer != 0 &&
	     (record->cone_layer < MODEL_DECODER_FIRST_LAYER ||
	      record->cone_layer >= MODEL_LAYERS)))
		return STATUS_INVALID_ARGUMENT;
	const uint64_t pairs =
		(uint64_t)record->row_count * EXL3_TC_PAIRS_PER_ROW;
	const struct tc_region regions[] = {
		{ record->picks_offset, pairs * sizeof(struct router_pick) },
		{ record->expert_offsets_offset,
		  (uint64_t)EXL3_TCP_EXPERTS * 3u * sizeof(uint64_t) },
		{ record->hidden_offset,
		  (uint64_t)record->row_count * EXL3_TC_HIDDEN * 2u },
		{ record->workspace_offset,
		  exl3_tcp_workspace_bytes(record->row_count,
					   record->intermediate) },
		{ record->output_offset,
		  (uint64_t)record->row_count * EXL3_TC_HIDDEN * 4u }
	};
	/* cone_layer: the route reads the step's rows (a step area or a view's: no alignment beyond the area's) */
	const uint64_t step_bytes =
		STEP_BLOCK_BYTES_VALUE +
		(uint64_t)record->row_count * STEP_ROW_BYTES_VALUE;
	if (!tc_regions_valid(regions, sizeof(regions) / sizeof(regions[0]), arena_byte_count) ||
	    (record->cone_layer != 0 &&
	     (record->step_offset > arena_byte_count || step_bytes > arena_byte_count - record->step_offset)))
		return STATUS_INVALID_ARGUMENT;
	const bool joint = record->row_count <= EXL3_TCP_JOINT_ROWS,
		   k3 = record->bits_x2 == TC_K3_BITS_X2;
	const bool i640 = record->intermediate == 640u, mixed = record->rotation != 0;
	const bool h640 = i640 && !mixed &&
			  (form == TCP_FORM_HYBRID || (form == TCP_FORM_DEFAULT && joint));
	const uint32_t tiles = (uint32_t)exl3_tcp_tile_capacity(pairs);
	const uint32_t gate_up_tpw = record->intermediate % 256u == 0 || h640 ? 2u : 1u;
	/* h640: the 256-column blocks and one of 128 columns */
	const uint32_t gate_up_column_blocks =
		record->intermediate / (16u * TCP_GEMM_WARPS * gate_up_tpw) + (h640 ? 1u : 0u);
	const uint32_t down_column_blocks =
		EXL3_TC_HIDDEN / (16u * TCP_GEMM_WARPS * 2u);
	char name[64];
	const char *const suffix = mixed ? (k3 ? "m640_k3" : "m640_k2") :
				   i640	 ? (k3 ? "i640_k3" : "i640_k2") :
					   (k3 ? "i512_k3" : "i512_k2");
	struct kernel_launch launches[EXL3_TCP_MOST_LAUNCHES];
	uint32_t count = 0;
	struct exl3_tcp_record part = *record;
	/* two_blocks: the GEMMs' forms at two blocks a multiprocessor (kernel_exl3_tcp.cu _b2) */
	const char *const two_blocks = record->two_blocks != 0 ? "_b2" : "";
	/* half_rows: the paired route's and the retile's shared ints (tcp_route_paired), else tcp_route's */
	const bool halves = record->half_rows != 0;
	const uint32_t route_ints =
		halves ? TCP_ROUTE_ARRAYS * EXL3_TCP_ROUTE_EXPERTS +
				 TCP_PAIRED_ROUTE_EXTRA :
			 TCP_ROUTE_ARRAYS * EXL3_TCP_EXPERTS + 1u;
	result_status status = tc_launch_bytes(
		&part, sizeof(part), EXL3_TCP_KERNEL_FILE, "exl3_tcp_route", 1u,
		EXL3_TCP_ROUTE_THREADS, (uint32_t)(route_ints * sizeof(int)),
		backend, card, &launches[count++]);
	const uint32_t passes[2] = { joint ? EXL3_TCP_GATE_AND_UP :
					     EXL3_TCP_GATE,
				     EXL3_TCP_UP };
	for (uint32_t index = 0; index < (joint ? 1u : 2u) && status == OK; index++) {
		part.pass = passes[index];
		const uint32_t projections =
			part.pass == EXL3_TCP_GATE_AND_UP ? 2u : 1u;
		status = tc_launch_bytes(&part, sizeof(part),
					 EXL3_TCP_KERNEL_FILE,
					 "exl3_tcp_operands",
					 tc_blocks(pairs * projections),
					 EXL3_TCP_OPERAND_THREADS, 0, backend,
					 card, &launches[count++]);
		(void)snprintf(name, sizeof(name), "exl3_tcp_gemm_%s_%s%s",
			       part.pass == EXL3_TCP_GATE_AND_UP ? "gate_up" :
			       part.pass == EXL3_TCP_GATE	 ? "gate" :
								   "up",
			       h640 ? (k3 ? "h640_k3" : "h640_k2") : suffix,
			       two_blocks);
		if (status == OK)
			status = tc_launch_bytes(
				&part, sizeof(part), EXL3_TCP_KERNEL_FILE, name,
				tiles * projections * gate_up_column_blocks,
				EXL3_TCP_GEMM_THREADS,
				tcp_shared_bytes(gate_up_tpw, record->bits_x2),
				backend, card, &launches[count++]);
	}
	/* half_rows: each half's down tiles behind the gate/up GEMMs (the paired route's tiles span both halves) */
	part.pass = 0;
	if (halves && status == OK)
		status = tc_launch_bytes(&part, sizeof(part),
					 EXL3_TCP_KERNEL_FILE,
					 "exl3_tcp_retile", 1u,
					 EXL3_TCP_ROUTE_THREADS,
					 (uint32_t)(route_ints * sizeof(int)),
					 backend, card, &launches[count++]);
	/* the down operands, down GEMM and combine of the whole call, or (half_rows) the whole call's down operands (a
	 * half's positions are not adjacent), then each half's down GEMM and combine in turn */
	const uint32_t parts[2] = { halves ? EXL3_TCP_FIRST_HALF :
					     EXL3_TCP_WHOLE,
				    EXL3_TCP_SECOND_HALF };
	for (uint32_t index = 0; index < (halves ? 2u : 1u) && status == OK; index++) {
		part.part = parts[index];
		const uint64_t half = record->half_rows, rest = (uint64_t)record->row_count - half;
		const uint64_t rows =
			part.part == EXL3_TCP_FIRST_HALF  ? half :
			part.part == EXL3_TCP_SECOND_HALF ? rest :
							    record->row_count;
		part.pass = 0;
		(void)snprintf(name, sizeof(name), "exl3_tcp_down_%s",
			       mixed ? "m640" :
			       i640  ? "i640" :
				       "i512");
		if (index == 0) {
			struct exl3_tcp_record whole = part;
			whole.part = EXL3_TCP_WHOLE;
			status = tc_launch_bytes(&whole, sizeof(whole),
						 EXL3_TCP_KERNEL_FILE, name,
						 tc_blocks((pairs + 1u) / 2u),
						 EXL3_TCP_DOWN_THREADS, 0,
						 backend, card,
						 &launches[count++]);
		}
		part.pass = EXL3_TCP_DOWN;
		(void)snprintf(name, sizeof(name), "exl3_tcp_gemm_down_%s%s",
			       suffix, two_blocks);
		if (status == OK)
			status = tc_launch_bytes(
				&part, sizeof(part), EXL3_TCP_KERNEL_FILE, name,
				tiles * down_column_blocks,
				EXL3_TCP_GEMM_THREADS,
				tcp_shared_bytes(2u, record->bits_x2), backend,
				card, &launches[count++]);
		part.pass = 0;
		if (status == OK)
			status = tc_launch_bytes(
				&part, sizeof(part), EXL3_TCP_KERNEL_FILE,
				"exl3_tcp_combine",
				(uint32_t)((rows * (EXL3_TC_HIDDEN / 512u) +
					    7u) /
					   8u),
				EXL3_TCP_COMBINE_THREADS, 0, backend, card,
				&launches[count++]);
	}
	if (status == OK) {
		memcpy(launches_out, launches,
		       sizeof(struct kernel_launch) * count);
		*launch_count_out = count;
	}
	return status;
}

result_status exl3_tcp_prepare(
	const struct exl3_tcp_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch launches_out[EXL3_TCP_MOST_LAUNCHES],
	uint32_t *launch_count_out)
{
	return tcp_prepare(record, arena_byte_count, backend, card, launches_out, launch_count_out, TCP_FORM_DEFAULT);
}

result_status exl3_tcp_prepare_base(
	const struct exl3_tcp_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch launches_out[EXL3_TCP_MOST_LAUNCHES],
	uint32_t *launch_count_out)
{
	return tcp_prepare(record, arena_byte_count, backend, card, launches_out, launch_count_out, TCP_FORM_BASE);
}

result_status exl3_tcp_prepare_hybrid(
	const struct exl3_tcp_record *record, uint64_t arena_byte_count,
	const struct backend_operations *backend, struct device_card *card,
	struct kernel_launch launches_out[EXL3_TCP_MOST_LAUNCHES],
	uint32_t *launch_count_out)
{
	return tcp_prepare(record, arena_byte_count, backend, card, launches_out, launch_count_out, TCP_FORM_HYBRID);
}
