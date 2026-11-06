/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Complete image-program preparation over synthetic pack tables and 4 KiB host-memory cards. No weights or kernels
 * execute. Base programs are prepared before vision registration, then compared byte for byte after it. */
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/image_routing.h"
#include "anemone/loader.h"
#include "anemone/step.h"
#include "anemone/step_images.h"
#include "engram_pack.h"
#include "host_memory_backend.h"
#include "memory_storage.h"
#include "model_v41.h"
#include "model_v41_pack.h"
#include "step_validation.h"

#define TEST_CARDS TEST_MODEL_V41_CARDS
#define TEST_CONTEXT 16384u
#define TEST_CAPACITY 2048u
#define TEST_SESSIONS 3u
#define TEST_HOST_BYTES 4096u
#define TEST_WAIT_NS 1000000000ull
#define TEST_WIDE_ROWS 64u
#define TEST_FULL_ROWS 1100u
#define TEST_DRIVER_PROGRAMS 4u
#define TEST_RESULT_CARD (TEST_CARDS - 1u)
#define COUNT_OF(table) ((uint32_t)(sizeof(table) / sizeof((table)[0])))
#define VISION_STEP_MERGE_NAME "vision_step_merge"

static const struct {
	enum program_kind kind;
	uint32_t rows;
} program_forms[] = { { PROGRAM_THIN, 1u },
		      { PROGRAM_THIN, LINEAR_MAXIMUM_ROWS },
		      { PROGRAM_WIDE, TEST_WIDE_ROWS },
		      { PROGRAM_FULL, TEST_WIDE_ROWS },
		      { PROGRAM_FULL, TEST_FULL_ROWS } };
static uint32_t failures, maximum_kernels;
static uint32_t selected_class, selected_draft, selected_order;

static bool require(bool valid, const char *description)
{
	if (!valid) {
		(void)fprintf(stderr,
			      "FAILED: image program class=%" PRIu32
			      " draft=%" PRIu32 " order=%" PRIu32 ": %s\n",
			      selected_class, selected_draft, selected_order,
			      description);
		failures++;
	}
	return valid;
}

static bool same_bytes(const void *first, const void *second, size_t bytes)
{
	return bytes == 0 || (first != NULL && second != NULL &&
			      memcmp(first, second, bytes) == 0);
}

/* Compare every published scalar and every array's bytes; allocation addresses are intentionally excluded. */
static bool same_program(const struct execution_program *first,
			 const struct execution_program *second)
{
	if (first == NULL || second == NULL ||
	    !same_bytes(&first->limits, &second->limits,
			sizeof(first->limits)) ||
	    first->card_count != second->card_count ||
	    first->segment_count != second->segment_count ||
	    first->hand_off_count != second->hand_off_count ||
	    first->row_count != second->row_count ||
	    first->result_card != second->result_card ||
	    first->expert_pause_count != second->expert_pause_count ||
	    first->host_expert_count != second->host_expert_count ||
	    first->kind != second->kind ||
	    first->lease_count != second->lease_count ||
	    first->session_range_count != second->session_range_count ||
	    first->front_segment_count != second->front_segment_count ||
	    first->cone_rows != second->cone_rows ||
	    first->checkpoint_count != second->checkpoint_count ||
	    first->draft_segment_first != second->draft_segment_first) {
		return false;
	}
	for (uint32_t card = 0; card < first->card_count; card++) {
		const struct program_card_record
			*const left = &first->cards[card],
			       *const right = &second->cards[card];
		const uint64_t left_fields[] = {
			left->arena_byte_count,	  left->step_offset,
			left->session_offset,	  left->result_offset,
			left->logits_offset,	  left->streams_offset,
			left->carried_pre_offset, left->engram_rows_offset,
			left->engram_tables,	  left->launch_count,
			left->view_count,	  left->round_offset
		};
		const uint64_t right_fields[] = {
			right->arena_byte_count,   right->step_offset,
			right->session_offset,	   right->result_offset,
			right->logits_offset,	   right->streams_offset,
			right->carried_pre_offset, right->engram_rows_offset,
			right->engram_tables,	   right->launch_count,
			right->view_count,	   right->round_offset
		};
		if (!same_bytes(left_fields, right_fields,
				sizeof(left_fields)) ||
		    !same_bytes(left->launches, right->launches,
				(size_t)left->launch_count *
					sizeof(*left->launches)) ||
		    !same_bytes(left->views, right->views,
				(size_t)left->view_count *
					sizeof(*left->views))) {
			return false;
		}
	}
	return same_bytes(first->segments, second->segments,
			  (size_t)first->segment_count *
				  sizeof(*first->segments)) &&
	       same_bytes(first->hand_offs, second->hand_offs,
			  (size_t)first->hand_off_count *
				  sizeof(*first->hand_offs)) &&
	       same_bytes(first->expert_pauses, second->expert_pauses,
			  (size_t)first->expert_pause_count *
				  sizeof(*first->expert_pauses)) &&
	       same_bytes(first->host_expert_offsets,
			  second->host_expert_offsets,
			  (size_t)first->host_expert_count *
				  sizeof(*first->host_expert_offsets)) &&
	       same_bytes(first->session_ranges, second->session_ranges,
			  (size_t)first->session_range_count *
				  sizeof(*first->session_ranges)) &&
	       same_bytes(first->checkpoints, second->checkpoints,
			  (size_t)first->checkpoint_count *
				  sizeof(*first->checkpoints)) &&
	       same_bytes(first->leases, second->leases,
			  (size_t)first->lease_count * sizeof(*first->leases));
}

static uint32_t kernel_count(uint32_t card)
{
	uint32_t count = 0;
	struct test_host_memory_backend_kernel kernel;
	while (count < BACKEND_MAXIMUM_KERNELS &&
	       test_host_memory_backend_kernel_of(card, count, &kernel)) {
		count++;
	}
	return count;
}

static bool image_launches_present(const struct execution_program *program)
{
	uint32_t merges = 0, routes = 0, gates = 0;
	for (uint32_t card = 0; card < program->card_count; card++) {
		for (uint32_t index = 0;
		     index < program->cards[card].launch_count; index++) {
			const struct kernel_launch *const launch =
				&program->cards[card].launches[index];
			struct test_host_memory_backend_kernel kernel;
			if (!require(
				    test_host_memory_backend_kernel_of(
					    card, launch->kernel, &kernel),
				    "every image launch names a registered kernel")) {
				return false;
			}
			merges += strcmp(kernel.function_name,
					 VISION_STEP_MERGE_NAME) == 0 ?
					  1u :
					  0u;
			routes += strcmp(kernel.kernel_file_name,
					 KERNEL_FILE_IMAGE_ROUTING) == 0 ?
					  1u :
					  0u;
			gates += strcmp(kernel.kernel_file_name,
					KERNEL_FILE_IMAGE_ENGRAM) == 0 ?
					 1u :
					 0u;
		}
	}
	return require(
		merges != 0 && routes != 0 && gates != 0,
		"image programs include embedding replacement, VL routing and masked Engram gates");
}

static struct model_plan_record *make_plan(const struct model_operations *model,
					   const struct pack_record *pack,
					   bool draft,
					   enum linear_order_code order)
{
	uint32_t placement[TEST_MODEL_V41_DRAFT_UNIT_COUNT] = { 0 };
	const uint32_t units = draft ? TEST_MODEL_V41_DRAFT_UNIT_COUNT :
				       TEST_MODEL_V41_UNIT_COUNT;
	test_model_v41_five_cards(placement, 0, 0);
	if (draft) {
		test_model_v41_draft_place(placement, TEST_RESULT_CARD);
	}
	const struct model_limits limits = { TEST_CONTEXT,
					     TEST_CAPACITY,
					     TEST_SESSIONS,
					     STATE_WIDTH_16,
					     LINEAR_PROFILE_PRECISION,
					     order,
					     CMP_ORDER_REFERENCE,
					     draft ? MODEL_DRAFT_BLOCK : 0,
					     0,
					     0 };
	struct load_layout *layout = NULL;
	struct model_plan_record *plan = NULL;
	result_status status =
		load_layout_create(pack, placement, units, TEST_CARDS, &layout);
	if (status == STATUS_OK) {
		status = model->plan_create(pack, layout, &limits, &plan);
	}
	(void)require(status == STATUS_OK && plan != NULL,
		      "make table-only base model plan");
	load_layout_destroy(layout);
	return plan;
}

static void run_case(const struct model_operations *model,
		     const struct pack_record *pack,
		     const struct engram_record *engram,
		     uint32_t arithmetic_class, bool draft,
		     enum linear_order_code order)
{
	selected_class = arithmetic_class;
	selected_draft = draft ? 1u : 0u;
	selected_order = (uint32_t)order;
	struct model_plan_record *const plan =
		make_plan(model, pack, draft, order);
	if (plan == NULL) {
		return;
	}
	struct device_card *cards[TEST_CARDS] = { NULL };
	struct card_description_record descriptions[TEST_CARDS] = { 0 };
	struct execution_program *base[COUNT_OF(program_forms)] = { NULL };
	struct execution_program *images[COUNT_OF(program_forms)] = { NULL };
	const struct execution_program *base_driver[TEST_DRIVER_PROGRAMS] = {
		NULL
	};
	const struct execution_program *image_driver[TEST_DRIVER_PROGRAMS] = {
		NULL
	};
	test_host_memory_backend_reset(TEST_CARDS, TEST_HOST_BYTES);
	bool ready = true;
	for (uint32_t card = 0; card < TEST_CARDS; card++) {
		test_host_memory_backend_set_arithmetic_class(
			card, arithmetic_class == ARITHMETIC_CLASS_SM_80 ?
				      ARITHMETIC_CLASS_SM_80 :
				      ARITHMETIC_CLASS_SM_120);
		const result_status described =
			test_host_memory_backend.card_describe(
				card, &descriptions[card]);
		const result_status opened = test_host_memory_backend.card_open(
			card, TEST_HOST_BYTES, &cards[card]);
		ready = require(described == STATUS_OK && opened == STATUS_OK,
				"open 4 KiB host-memory cards") &&
			ready;
		descriptions[card].arithmetic_class = arithmetic_class;
	}
	/* Preparing every base shape first pins all legacy handles before image-only functions are registered. */
	for (uint32_t index = 0; ready && index < COUNT_OF(program_forms);
	     index++) {
		const result_status status = model->program_prepare(
			plan, &test_host_memory_backend, cards, descriptions,
			program_forms[index].kind, program_forms[index].rows,
			&base[index]);
		ready = require(status == STATUS_OK && base[index] != NULL,
				"prepare every legacy program first");
	}
	const uint64_t old_arena = plan->cards[0].arena_byte_count;
	plan->vision.enabled = true;
	uint64_t arena_end = 0;
	const result_status reserved = model_vision_arena(
		&plan->vision, TEST_SESSIONS, old_arena, &arena_end);
	ready = require(reserved == STATUS_OK && arena_end > old_arena &&
				plan->vision.text_arena_bytes == old_arena,
			"append virtual vision ranges while preserving the text arena") &&
		ready;
	plan->cards[0].arena_byte_count = arena_end;
	for (uint32_t layer = 0; layer < MODEL_LAYERS; layer++) {
		/* Immutable dummy bytes from a different F32 tensor; no range aliases scores, text bias or experts. */
		const uint64_t offset =
			plan->layer_tensors[layer][V41_DATA_ATTENTION_MIX]
				.stored.byte_offset;
		const uint64_t aligned = (offset + VISION_ALIGNMENT - 1u) /
					 VISION_ALIGNMENT * VISION_ALIGNMENT;
		plan->vision.router_bias[layer] =
			(struct linear_range){ aligned,
					       MODEL_ROUTER_ROW_BYTES };
	}
	for (uint32_t index = 0; ready && index < COUNT_OF(program_forms);
	     index++) {
		result_status status = v41_image_program_prepare(
			plan, &test_host_memory_backend, cards, descriptions,
			program_forms[index].kind, program_forms[index].rows,
			&images[index]);
		ready = require(
			status == STATUS_OK && images[index] != NULL,
			"prepare the complete image-capable model program");
		if (!ready) {
			break;
		}
		(void)image_launches_present(images[index]);
		bool unsupported = false;
		const char *const problem = step_program_problem(
			images[index], base[0], cards, engram, &unsupported);
		if (!require(
			    problem == NULL && !unsupported,
			    "image program retains base topology and state metadata") &&
		    problem != NULL) {
			(void)fprintf(stderr, "image program metadata: %s\n",
				      problem);
		}
		struct execution_program *again = NULL;
		status = model->program_prepare(plan, &test_host_memory_backend,
						cards, descriptions,
						program_forms[index].kind,
						program_forms[index].rows,
						&again);
		(void)require(
			status == STATUS_OK && same_program(base[index], again),
			"vision-enabled plan preserves every text program scalar and published byte");
		model->program_destroy(again);
		if (index < TEST_DRIVER_PROGRAMS) {
			base_driver[index] = base[index];
			image_driver[index] = images[index];
		}
	}
	struct step_driver *driver = NULL;
	if (ready) {
		const result_status created = step_driver_create(
			&test_host_memory_backend, cards, base_driver,
			TEST_DRIVER_PROGRAMS, engram, TEST_WAIT_NS, &driver);
		(void)require(
			created == STATUS_OK && driver != NULL,
			"base programs create a driver without vision snapshot arrays");
		if (driver != NULL) {
			(void)require(
				step_images_attach(
					driver, plan,
					descriptions[0].arithmetic_class) ==
					STATUS_OK,
				"complete vision encoder registers within the card kernel bound");
			(void)require(
				step_image_programs_attach(
					driver, image_driver,
					TEST_DRIVER_PROGRAMS) == STATUS_OK,
				"complete image alternatives attach to the base driver");
		}
	}
	step_driver_destroy(driver);
	uint32_t case_maximum = 0;
	for (uint32_t card = 0; card < TEST_CARDS; card++) {
		const uint32_t count = kernel_count(card);
		case_maximum = count > case_maximum ? count : case_maximum;
		maximum_kernels = count > maximum_kernels ? count :
							    maximum_kernels;
		(void)require(count <= BACKEND_MAXIMUM_KERNELS,
			      "prepared kernel count remains within the bound");
		for (uint32_t index = 0; index < COUNT_OF(program_forms);
		     index++) {
			if (card == 0) {
				model->program_destroy(base[index]);
				model->program_destroy(images[index]);
			}
		}
		test_host_memory_backend.card_close(cards[card], TEST_WAIT_NS);
	}
	(void)require(
		test_host_memory_backend_work_count() == 0,
		"preparation and metadata validation submit no kernels or transfers");
	(void)printf("image programs: class=%" PRIu32 " draft=%" PRIu32
		     " order=%" PRIu32 " maximum kernels=%" PRIu32 "/%u\n",
		     arithmetic_class, selected_draft, selected_order,
		     case_maximum, (uint32_t)BACKEND_MAXIMUM_KERNELS);
	model->plan_destroy(plan);
}

int main(void)
{
	const struct model_operations *const model =
		model_operations_of(V41_DATA_MODEL_NAME);
	if (!require(model != NULL, "model operations available")) {
		return EXIT_FAILURE;
	}
	const uint32_t classes[] = { ARITHMETIC_CLASS_SM_80,
				     ARITHMETIC_CLASS_SM_120_NATIVE,
				     ARITHMETIC_CLASS_SM_120 };
	for (uint32_t draft = 0; draft <= 1u; draft++) {
		struct pack_record *pack = NULL;
		struct engram_record *engram = NULL;
		unsigned char *engram_bytes = NULL;
		test_memory_storage_reset();
		const struct test_model_v41_change change = {
			TEST_MODEL_V41_CHANGE_NONE, NULL, 0
		};
		const result_status opened =
			draft != 0 ?
				test_model_v41_draft_pack_open(change, &pack) :
				test_model_v41_pack_open(change, &pack);
		const result_status tables =
			test_engram_pack_open(&engram_bytes, &engram);
		if (require(opened == STATUS_OK && tables == STATUS_OK,
			    "open synthetic checkpoint and Engram table metadata")) {
			for (uint32_t class_index = 0;
			     class_index < COUNT_OF(classes); class_index++) {
				for (uint32_t order = LINEAR_ORDER_FIXED;
				     order <= LINEAR_ORDER_REFERENCE; order++) {
					run_case(model, pack, engram,
						 classes[class_index],
						 draft != 0,
						 (enum linear_order_code)order);
				}
			}
		}
		engram_close(engram);
		free(engram_bytes);
		pack_close(pack);
		test_memory_storage_reset();
	}
	(void)printf("model_vision_program_test: maximum kernels=%" PRIu32
		     ", failures=%" PRIu32 "\n",
		     maximum_kernels, failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
