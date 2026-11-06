/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_v41_program_dump_test: canonical byte dumps of prepared programs for comparison across source epochs.
 * With --dump it prints every record byte, grid, block, shared byte count, segment, hand-off, view and state range.
 * Kernel handles and their module/function names are both retained: unchanged programs keep their table order too.
 * Thin programs cover every row count 1..8, both profiles and orders and every arithmetic class. Wide and full
 * programs cover all classes at 64, 256 and 1,024 rows. Both CMP orders cover layout D and the 1.3 fill.
 * No CUDA context or model extent is needed.
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/loader.h"
#include "anemone/relaxed_linear_records.h"
#include "host_memory_backend.h"
#include "memory_storage.h"
#include "model_v41.h"
#include "model_v41_pack.h"

#define CARDS TEST_MODEL_V41_CARDS
#define UNIT_COUNT TEST_MODEL_V41_UNIT_COUNT
#define CONTEXT 16384u
#define ROW_CAPACITY 1024u
#define SESSIONS 3u
#define HOST_CARD_BYTES 4096u
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define MOST_ARGUMENTS 6
#define LAYOUTS 2u
#define DEPLOYMENT_PRO_CARD 2u
#define ORDERED_CROSSOVER_ROWS 512u /* the selected boundary; also compiled against the frozen baseline */

static int failures;
static bool is_dumped;
static bool is_deep;
static bool is_thin_only;
static bool is_preserved;
static bool has_draft;
static bool text_with_vision;
static const uint32_t classes[] = { ARITHMETIC_CLASS_SM_80,
				    ARITHMETIC_CLASS_SM_120,
				    ARITHMETIC_CLASS_SM_120_NATIVE };
static const uint32_t chunk_rows[] = { 64, 256, ROW_CAPACITY };

static bool succeeded(result_status status, const char *what)
{
	if (status != STATUS_OK) {
		(void)fprintf(stderr, "FAILED: %s: %s\n", what,
			      status_text(status));
		failures++;
		return false;
	}
	return true;
}

static void dump_bytes(const char *what, const void *bytes, size_t count)
{
	if (!is_dumped) {
		return;
	}
	(void)printf("%s %zu ", what, count);
	if (count != 0 &&
	    (bytes == NULL || fwrite(bytes, 1, count, stdout) != count)) {
		(void)fprintf(stderr,
			      "FAILED: a program dump could not be written\n");
		failures++;
	}
	(void)putchar('\n');
}

static void program_dump(const struct execution_program *program)
{
	const uint32_t counts[] = { program->card_count,
				    program->segment_count,
				    program->hand_off_count,
				    program->row_count,
				    program->result_card,
				    program->expert_pause_count,
				    program->host_expert_count,
				    (uint32_t)program->kind,
				    program->session_range_count,
				    program->front_segment_count,
				    program->cone_rows };
	dump_bytes("program", counts, sizeof(counts));
	/* Keep the canonical draft-off epoch readable byte for byte after adding the draft limit. */
	const uint32_t limits[] = { program->limits.context_token_limit,
				    program->limits.row_capacity,
				    program->limits.session_capacity,
				    (uint32_t)program->limits.state_width,
				    (uint32_t)program->limits.linear_profile,
				    (uint32_t)program->limits.linear_order,
				    (uint32_t)program->limits.cmp_order };
	dump_bytes("limits", limits, sizeof(limits));
	if (program->limits.draft_positions != 0) {
		dump_bytes("draft_positions", &program->limits.draft_positions,
			   sizeof(program->limits.draft_positions));
		dump_bytes("draft_segment_first", &program->draft_segment_first,
			   sizeof(program->draft_segment_first));
	}
	dump_bytes("lease_count", &program->lease_count,
		   sizeof(program->lease_count));
	dump_bytes("leases", program->leases,
		   (size_t)program->lease_count * sizeof(*program->leases));
	dump_bytes("checkpoint_count", &program->checkpoint_count,
		   sizeof(program->checkpoint_count));
	dump_bytes("checkpoints", program->checkpoints,
		   (size_t)program->checkpoint_count *
			   sizeof(*program->checkpoints));
	dump_bytes("segments", program->segments,
		   (size_t)program->segment_count * sizeof(*program->segments));
	dump_bytes("hand_offs", program->hand_offs,
		   (size_t)program->hand_off_count *
			   sizeof(*program->hand_offs));
	dump_bytes("expert_pauses", program->expert_pauses,
		   (size_t)program->expert_pause_count *
			   sizeof(*program->expert_pauses));
	dump_bytes("host_experts", program->host_expert_offsets,
		   (size_t)program->host_expert_count *
			   sizeof(*program->host_expert_offsets));
	dump_bytes("session_ranges", program->session_ranges,
		   (size_t)program->session_range_count *
			   sizeof(*program->session_ranges));
	for (uint32_t card = 0; card < program->card_count; card++) {
		const struct program_card_record *const own =
			&program->cards[card];
		const uint64_t offsets[] = {
			own->arena_byte_count,	 own->step_offset,
			own->session_offset,	 own->result_offset,
			own->logits_offset,	 own->streams_offset,
			own->carried_pre_offset, own->engram_rows_offset,
			own->engram_tables,	 own->launch_count,
			own->view_count
		};
		dump_bytes("card", offsets, sizeof(offsets));
		dump_bytes("views", own->views,
			   (size_t)own->view_count * sizeof(*own->views));
		for (uint32_t index = 0; index < own->launch_count; index++) {
			const struct kernel_launch *const launch =
				&own->launches[index];
			struct test_host_memory_backend_kernel kernel;
			if (!test_host_memory_backend_kernel_of(
				    card, launch->kernel, &kernel)) {
				(void)fprintf(
					stderr,
					"FAILED: a program launch names no prepared kernel\n");
				failures++;
				continue;
			}
			if (is_dumped) {
				(void)printf("kernel %s %s\n",
					     kernel.kernel_file_name,
					     kernel.function_name);
			}
			dump_bytes("launch", launch, sizeof(*launch));
		}
	}
}

static void case_run(const struct model_operations *model,
		     const struct pack_record *pack, uint32_t arithmetic_class,
		     enum linear_profile_code profile,
		     enum linear_order_code order, enum program_kind kind,
		     uint32_t rows, enum cmp_order_code cmp_order,
		     uint32_t layout_index)
{
	/* Every thin program is fused. All-PRO chunks are untouched in either CMP mode; reference CMP chunks below
	 * the measured crossover keep the original chain. Keep every byte of these programs in the comparison. */
	const bool has_cmp = arithmetic_class == ARITHMETIC_CLASS_SM_80 ||
			     layout_index != 0;
	if (is_preserved && (kind == PROGRAM_THIN ||
			     (has_cmp && (cmp_order != CMP_ORDER_REFERENCE ||
					  rows >= ORDERED_CROSSOVER_ROWS)))) {
		return;
	}
	static uint32_t card_of_unit[TEST_MODEL_V41_DRAFT_UNIT_COUNT];
	const uint32_t units = has_draft ? TEST_MODEL_V41_DRAFT_UNIT_COUNT : UNIT_COUNT;
	struct device_card *cards[CARDS] = { NULL };
	struct card_description_record descriptions[CARDS];
	struct load_layout *layout = NULL;
	struct model_plan_record *plan = NULL;
	struct execution_program *program = NULL;
	const struct model_limits limits = {
		.context_token_limit = is_deep ? MODEL_CONTEXT_MOST : CONTEXT,
		.row_capacity = ROW_CAPACITY,
		.session_capacity = SESSIONS,
		.state_width = STATE_WIDTH_16,
		.linear_profile = profile,
		.linear_order = order,
		.cmp_order = cmp_order,
		.draft_positions = has_draft ? MODEL_DRAFT_BLOCK : 0
	};
	test_model_v41_five_cards(card_of_unit, UNIT_COUNT, 0);
	if (has_draft) {
		test_model_v41_draft_place(card_of_unit, CARDS - 1u);
	}
	if (layout_index != 0) {
		static const uint8_t layers[MODEL_LAYERS] = {
			0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2,
			2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
			3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4
		};
		for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
			for (uint32_t unit = 0;
			     unit < TEST_MODEL_V41_UNITS_PER_LAYER; unit++) {
				card_of_unit[TEST_MODEL_V41_LAYER_UNIT(layer) +
					     unit] = layers[layer];
			}
		}
	}
	test_host_memory_backend_reset(CARDS, HOST_CARD_BYTES);
	bool is_ready = true;
	for (uint32_t card = 0; card < CARDS && is_ready; card++) {
		const uint32_t card_class =
			layout_index != 0 && card != DEPLOYMENT_PRO_CARD ?
				ARITHMETIC_CLASS_SM_80 :
				arithmetic_class;
		test_host_memory_backend_set_arithmetic_class(
			card, card_class == ARITHMETIC_CLASS_SM_120_NATIVE ?
				      ARITHMETIC_CLASS_SM_120 :
				      card_class);
		is_ready =
			succeeded(test_host_memory_backend.card_describe(
					  card, &descriptions[card]),
				  "describe a card") &&
			succeeded(test_host_memory_backend.card_open(
					  card, HOST_CARD_BYTES, &cards[card]),
				  "open a card");
		descriptions[card].arithmetic_class = card_class;
	}
	is_ready = is_ready &&
		   succeeded(load_layout_create(pack, card_of_unit, units,
						CARDS, &layout),
			     "create the placement") &&
		   succeeded(model->plan_create(pack, layout, &limits, &plan),
			     "create the plan");
	uint64_t text_arena = 0;
	if (is_ready && text_with_vision) {
		text_arena = plan->cards[0].arena_byte_count;
		plan->vision.enabled = true;
		plan->vision.text_arena_bytes = text_arena;
		plan->cards[0].arena_byte_count +=
			V41_DATA_ARRAY_ALIGNMENT_BYTES;
	}
	/* Synthetic admitted folds exercise the selected relaxed kernels without reading model weights. */
	for (uint32_t layer = 0;
	     is_ready && cmp_order == CMP_ORDER_RELAXED && layer < MODEL_LAYERS;
	     layer++) {
		for (uint32_t tensor = 0; tensor < V41_DATA_LAYER_TENSOR_COUNT;
		     tensor++) {
			plan->layer_tensors[layer][tensor].fold_reference =
				RELAXED_FOLD_REFERENCE;
		}
	}
	is_ready = is_ready &&
		   succeeded(model->program_prepare(
				     plan, &test_host_memory_backend, cards,
				     descriptions, kind, rows, &program),
			     "prepare a program");
	if (is_ready && text_with_vision) {
		/* Serialize the logical text arena; the physical allocation also holds private image scratch. */
		plan->cards[0].arena_byte_count = text_arena;
		plan->vision.enabled = false;
	}
	if (is_ready) {
		if (is_dumped) {
			(void)printf("layout=%" PRIu32 " cmp_order=%u\n",
				     layout_index, (unsigned)cmp_order);
			(void)printf(
				"CASE class=%" PRIu32
				" profile=%u order=%u kind=%u rows=%" PRIu32
				"\n",
				arithmetic_class, (unsigned int)profile,
				(unsigned int)order, (unsigned int)kind, rows);
		}
		/* Named allocation starts account for scratch growth without discarding addresses. */
		for (uint32_t card = 0; card < CARDS; card++) {
			const struct v41_card *const own = &plan->cards[card];
			/* Serialize every original address, omitting only newly added draft members when disabled. */
#define DRAFT_ARRAY(name, row_bytes, fixed_bytes, holder, rows) \
	(holder) == V41_DRAFT,
			static const bool draft_arrays[] = { V41_DATA_ARRAYS(
				DRAFT_ARRAY) };
#undef DRAFT_ARRAY
			uint64_t arrays[V41_DATA_ARRAY_COUNT];
			uint32_t array_count = 0;
			for (uint32_t array = 0; array < V41_DATA_ARRAY_COUNT;
			     array++) {
				if (limits.draft_positions != 0 ||
				    !draft_arrays[array]) {
					arrays[array_count++] =
						own->array_offset[array];
				}
			}
			const size_t layer_bytes =
				sizeof(uint64_t) *
				(limits.draft_positions != 0 ?
					 V41_DATA_WALK_LAYERS :
					 MODEL_LAYERS);
			dump_bytes("arrays", arrays,
				   (size_t)array_count * sizeof(*arrays));
			dump_bytes("selections", own->selection_offset,
				   layer_bytes);
			dump_bytes("rings", own->ring_offset, layer_bytes);
			dump_bytes("saved_windows", own->checkpoint_offset,
				   layer_bytes);
			dump_bytes("compressors", own->compressor_offset,
				   layer_bytes);
			dump_bytes("pools", own->pool_offset, layer_bytes);
			dump_bytes("keys", own->key_tiles_offset, layer_bytes);
			dump_bytes("arena", &own->arena_byte_count,
				   sizeof(own->arena_byte_count));
		}
		program_dump(program);
	}
	model->program_destroy(program);
	model->plan_destroy(plan);
	load_layout_destroy(layout);
	for (uint32_t card = 0; card < CARDS; card++) {
		test_host_memory_backend.card_close(cards[card], 0);
	}
}

int main(int argument_count, char **arguments)
{
	bool arguments_valid = argument_count <= MOST_ARGUMENTS;
	for (int argument = 1; argument < argument_count; argument++) {
		if (strcmp(arguments[argument], "--dump") == 0) {
			is_dumped = true;
		} else if (strcmp(arguments[argument], "--deep") == 0) {
			is_deep = true;
		} else if (strcmp(arguments[argument], "--preserved") == 0) {
			is_preserved = true;
		} else if (strcmp(arguments[argument], "--text-with-vision") ==
			   0) {
			text_with_vision = true;
		} else if (strcmp(arguments[argument], "--draft") == 0) {
			has_draft = true;
		} else if (strcmp(arguments[argument], "--thin") == 0) {
			is_thin_only = true;
		} else {
			arguments_valid = false;
		}
	}
	if (!arguments_valid) {
		(void)fprintf(
			stderr,
			"usage: model_v41_program_dump_test [--dump] [--deep] "
            "[--thin|--preserved] [--draft] [--text-with-vision]\n");
		return EXIT_FAILURE;
	}
	struct pack_record *pack = NULL;
	const struct model_operations *const model =
		model_operations_of(V41_DATA_MODEL_NAME);
	const struct test_model_v41_change unchanged = { TEST_MODEL_V41_CHANGE_NONE, NULL, 0 };
	if (model == NULL ||
	    !succeeded(has_draft ? test_model_v41_draft_pack_open(unchanged, &pack) :
				 test_model_v41_pack_open(unchanged, &pack),
		       "open the model's tables")) {
		return EXIT_FAILURE;
	}
	for (uint32_t layout_index = 0; layout_index < LAYOUTS;
	     layout_index++) {
		for (uint32_t cmp_order = CMP_ORDER_REFERENCE;
		     cmp_order <= CMP_ORDER_RELAXED; cmp_order++) {
			for (uint32_t class_index = 0;
			     class_index < COUNT_OF(classes); class_index++) {
				for (uint32_t profile = 0;
				     profile <= LINEAR_PROFILE_SPEED;
				     profile++) {
					for (uint32_t order = 0;
					     order <= LINEAR_ORDER_REFERENCE;
					     order++) {
						for (uint32_t rows = 1;
						     rows <=
						     LINEAR_MAXIMUM_ROWS;
						     rows++) {
							case_run(
								model, pack,
								classes[class_index],
								(enum linear_profile_code)
									profile,
								(enum linear_order_code)
									order,
								PROGRAM_THIN,
								rows,
								(enum cmp_order_code)
									cmp_order,
								layout_index);
						}
						for (uint32_t kind =
							     PROGRAM_WIDE;
						     !is_thin_only &&
						     kind <= PROGRAM_FULL;
						     kind++) {
							if (kind ==
							    PROGRAM_CONE) {
								continue;
							}
							for (uint32_t row_index =
								     0;
							     row_index <
							     COUNT_OF(
								     chunk_rows);
							     row_index++) {
								case_run(
									model,
									pack,
									classes[class_index],
									(enum linear_profile_code)
										profile,
									(enum linear_order_code)
										order,
									(enum program_kind)
										kind,
									chunk_rows
										[row_index],
									(enum cmp_order_code)
										cmp_order,
									layout_index);
							}
						}
					}
				}
			}
		}
	}
	pack_close(pack);
	test_memory_storage_reset();
	if (failures != 0) {
		(void)fprintf(
			stderr,
			"model_v41_program_dump_test FAILED: %d failures\n",
			failures);
		return EXIT_FAILURE;
	}
	if (!is_dumped) {
		(void)printf("model_v41_program_dump_test passed\n");
	}
	return EXIT_SUCCESS;
}
