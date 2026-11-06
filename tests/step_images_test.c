/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Image scheduling and ownership against a host-memory backend with no kernel
 * stand-ins: real official-shape launch records, no tensor arithmetic or GPU.
 * Read-only dummy weight ranges overlap deliberately. Only metadata/patch bytes
 * are accessed in the arena; the image encoder executes bounded no-op launches. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/chat_images.h"
#include "anemone/step_images.h"
#include "anemone/vision.h"
#include "host_memory_backend.h"
#include "model_v41.h"
#include "step_driver.h"

#define TEST_SESSIONS 2u
#define TEST_CONTEXT 32u
#define TEST_ARENA_BYTES (300ull * 1024ull * 1024ull)
#define TEST_WEIGHT_BYTES (96ull * 1024ull * 1024ull)
#define TEST_WAIT_NS 1000000000ull
#define TEST_IMAGE_COUNT 2u
#define TEST_FIRST_BEGIN 2u
#define TEST_SECOND_BEGIN 10u
#define TEST_SPAN_TOKENS 4u
#define TEST_BFLOAT_VALUE UINT16_C(0x3f00)
#define TEST_PATCH_BYTES (IMAGE_PATCH_ELEMENTS * sizeof(uint16_t))
#define TEST_LAUNCHES_PER_IMAGE (VISION_MAXIMUM_LAUNCHES + 1u)
#define TEST_ADVANCE_LIMIT (TEST_LAUNCHES_PER_IMAGE * TEST_IMAGE_COUNT)
#define TEST_TWO 2u
#define TEST_STAGING_PROBE_BYTES 256u

static unsigned failures;
static void image_runtime_require(bool condition, const char *name)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: step_images %s\n", name);
		++failures;
	}
}
static uint64_t image_runtime_align(uint64_t bytes)
{
	return (bytes + VISION_ALIGNMENT - 1u) / VISION_ALIGNMENT *
	       VISION_ALIGNMENT;
}
static struct linear_range image_runtime_weight(uint64_t elements,
						uint32_t bytes)
{
	return (struct linear_range){ .byte_offset = 0,
				      .byte_count = elements * bytes };
}
static void image_runtime_weights(struct vision_weights *weights)
{
	const uint64_t width = VISION_MAXIMUM_WIDTH;
	const uint64_t language = VISION_MAXIMUM_LANGUAGE_WIDTH;
	const uint64_t intermediate = VISION_MAXIMUM_INTERMEDIATE;
	const uint32_t bf16 = VISION_BFLOAT16_BYTES;
	const uint32_t f32 = VISION_FLOAT_BYTES;
	weights->patch =
		image_runtime_weight(width * IMAGE_PATCH_ELEMENTS, bf16);
	weights->patch_bias = image_runtime_weight(width, bf16);
	weights->final_norm = image_runtime_weight(width, f32);
	weights->projector1 = image_runtime_weight(
		language * width * IMAGE_DOWNSAMPLE * IMAGE_DOWNSAMPLE, bf16);
	weights->projector1_bias = image_runtime_weight(language, bf16);
	weights->projector2 = image_runtime_weight(language * language, bf16);
	weights->projector2_bias = image_runtime_weight(language, bf16);
	weights->image_start = image_runtime_weight(language, bf16);
	weights->image_newline = image_runtime_weight(language, bf16);
	weights->image_end = image_runtime_weight(language, bf16);
	for (uint32_t layer = 0; layer < VISION_MAXIMUM_LAYERS; ++layer) {
		struct vision_layer_weights *const current =
			&weights->layers[layer];
		current->norm1 = image_runtime_weight(width, f32);
		current->qkv = image_runtime_weight(
			VISION_QKV_COMPONENTS * width * width, bf16);
		current->qkv_bias = image_runtime_weight(
			VISION_QKV_COMPONENTS * width, bf16);
		current->attention_output =
			image_runtime_weight(width * width, bf16);
		current->attention_output_bias =
			image_runtime_weight(width, bf16);
		current->norm2 = image_runtime_weight(width, f32);
		current->gate_up = image_runtime_weight(
			VISION_GATE_COMPONENTS * intermediate * width, bf16);
		current->down =
			image_runtime_weight(width * intermediate, bf16);
	}
}
static bool image_runtime_plan(struct model_plan_record *plan)
{
	plan->limits.session_capacity = TEST_SESSIONS;
	plan->limits.context_token_limit = TEST_CONTEXT;
	plan->limits.row_capacity = 1;
	plan->cards[0].arena_byte_count = TEST_ARENA_BYTES;
	plan->vision.enabled = true;
	image_runtime_weights(&plan->vision.weights);
	const struct vision_config config = vision_official_config();
	struct vision_scratch scratch;
	const result_status status =
		vision_scratch_plan(&config, 1, 1, TEST_WEIGHT_BYTES, &scratch);
	if (status != STATUS_OK)
		return false;
	plan->vision.scratch =
		(struct linear_range){ .byte_offset = TEST_WEIGHT_BYTES,
				       .byte_count = scratch.byte_count };
	uint64_t next =
		image_runtime_align(TEST_WEIGHT_BYTES + scratch.byte_count);
	plan->vision.patches =
		(struct linear_range){ .byte_offset = next,
				       .byte_count = TEST_PATCH_BYTES };
	next = image_runtime_align(next + TEST_PATCH_BYTES);
	const uint64_t output_bytes =
		(uint64_t)VISION_MAXIMUM_LANGUAGE_WIDTH * VISION_BFLOAT16_BYTES;
	plan->vision.encoded =
		(struct linear_range){ .byte_offset = next,
				       .byte_count = output_bytes };
	next = image_runtime_align(next + output_bytes);
	plan->vision.session_embedding_stride =
		output_bytes * IMAGE_MAX_TOKENS * IMAGE_MAX_IMAGES;
	plan->vision.session_embeddings = (struct linear_range){
		.byte_offset = next,
		.byte_count =
			plan->vision.session_embedding_stride * TEST_SESSIONS
	};
	next += plan->vision.session_embeddings.byte_count;
	plan->vision.session_metadata =
		(struct linear_range){ .byte_offset = next,
				       .byte_count =
					       (uint64_t)TEST_SESSIONS *
					       VISION_SESSION_METADATA_BYTES };
	return next + plan->vision.session_metadata.byte_count <=
	       TEST_ARENA_BYTES;
}
static struct chat_images image_runtime_inputs(uint8_t *bytes,
					       uint16_t *patches)
{
	struct chat_images images = { .count = TEST_IMAGE_COUNT };
	for (uint32_t index = 0; index < TEST_IMAGE_COUNT; ++index) {
		images.images[index] = (struct chat_image_input){
			.bytes = bytes,
			.patches = patches,
			.byte_count = 1,
			.plan = { .width = IMAGE_PATCH_SIZE,
				  .height = IMAGE_PATCH_SIZE,
				  .vit_height = 1,
				  .vit_width = 1,
				  .llm_height = 1,
				  .llm_width = 1,
				  .token_count = TEST_SPAN_TOKENS,
				  .patch_elements = IMAGE_PATCH_ELEMENTS },
			.token_begin = index == 0 ? TEST_FIRST_BEGIN :
						    TEST_SECOND_BEGIN
		};
	}
	return images;
}
static void image_runtime_refused(struct step_driver *driver,
				  struct chat_images *images, const char *name)
{
	const uint64_t before = test_host_memory_backend_work_count();
	const uint32_t staging = test_host_memory_backend_staging_block_count();
	const result_status status = step_images_begin(driver, 0, images);
	image_runtime_require(status == STATUS_INVALID_ARGUMENT, name);
	image_runtime_require(
		test_host_memory_backend_work_count() == before &&
			test_host_memory_backend_staging_block_count() ==
				staging,
		"refusal changes no backend work");
	image_runtime_require(
		step_image_position(driver, 0, TEST_FIRST_BEGIN) &&
			step_image_position(driver, 0, TEST_SECOND_BEGIN),
		"failed begin preserves typed spans");
}
static void image_runtime_preflight(struct step_driver *driver,
				    const struct chat_images *valid)
{
	struct chat_images bad = *valid;
	bad.count = IMAGE_MAX_IMAGES + 1u;
	image_runtime_refused(driver, &bad, "request image count");
	bad = *valid;
	bad.images[1].patches = NULL;
	image_runtime_refused(driver, &bad,
			      "invalid second image before any upload");
	bad = *valid;
	bad.images[1].bytes = NULL;
	image_runtime_refused(driver, &bad, "missing original bytes");
	bad = *valid;
	bad.images[1].byte_count = 0;
	image_runtime_refused(driver, &bad, "empty original bytes");
	bad = *valid;
	bad.images[1].byte_count = IMAGE_MAX_BYTES + 1u;
	image_runtime_refused(driver, &bad, "original byte bound");
	bad = *valid;
	bad.images[1].token_begin = TEST_FIRST_BEGIN + TEST_SPAN_TOKENS - 1u;
	image_runtime_refused(driver, &bad, "overlapping spans");
	bad = *valid;
	bad.images[1].token_begin = TEST_CONTEXT - 1u;
	image_runtime_refused(driver, &bad, "span crosses context end");
	bad = *valid;
	bad.images[1].token_begin = TEST_CONTEXT;
	image_runtime_refused(driver, &bad, "span starts at context end");
	bad = *valid;
	bad.images[1].plan.vit_width = 0;
	image_runtime_refused(driver, &bad, "empty patch width");
	bad = *valid;
	bad.images[1].plan.vit_height = IMAGE_MAX_PATCHES + 1u;
	image_runtime_refused(driver, &bad, "patch count bound");
	bad = *valid;
	bad.images[1].plan.patch_elements++;
	image_runtime_refused(driver, &bad, "patch storage count");
	bad = *valid;
	bad.images[1].plan.llm_width++;
	image_runtime_refused(driver, &bad, "projector grid");
	bad = *valid;
	bad.images[1].plan.token_count = IMAGE_MAX_TOKENS + 1u;
	image_runtime_refused(driver, &bad, "token count bound");
	bad = *valid;
	bad.images[1].plan.width = 0;
	image_runtime_refused(driver, &bad, "empty pixel width");
}
/* This fixture owns a synthetic driver directly; both immutable program tables name its no-op program. */
static void image_runtime_typed_rows(struct step_driver *driver)
{
	static uint32_t next[TEST_SESSIONS] = { 0 };
	static uint32_t counts[TEST_SESSIONS] = { TEST_CONTEXT, TEST_CONTEXT };
	driver->next_positions = next;
	driver->token_counts = counts;
	driver->program_count = 1;
	driver->largest_row_count = 1;
	driver->image_programs = driver->programs;
	const uint64_t work = test_host_memory_backend_work_count();
	struct step_row_record row = { .position = TEST_FIRST_BEGIN,
				       .token_id = IMAGE_TOKEN_ID,
				       .flags = STEP_ROW_IMAGE };
	image_runtime_require(
		step_rows_check(driver, &row, 1, true) == STATUS_OK,
		"typed image row with reference token is admitted");
	row.flags = 0;
	image_runtime_require(
		step_rows_check(driver, &row, 1, true) ==
			STATUS_INVALID_ARGUMENT,
		"typed image position without IMAGE flag is refused");
	row.flags = STEP_ROW_IMAGE;
	row.token_id = 1;
	image_runtime_require(
		step_rows_check(driver, &row, 1, true) ==
			STATUS_INVALID_ARGUMENT,
		"typed image position with wrong token ID is refused");
	row.position = 0;
	row.token_id = IMAGE_TOKEN_ID;
	image_runtime_require(
		step_rows_check(driver, &row, 1, true) ==
			STATUS_INVALID_ARGUMENT,
		"IMAGE flag outside a typed image position is refused");
	row.flags = 0;
	row.token_id = 1;
	image_runtime_require(
		step_rows_check(driver, &row, 1, true) == STATUS_OK,
		"ordinary text position retains base row admission");
	image_runtime_require(test_host_memory_backend_work_count() == work,
			      "typed row validation submits no backend work");
	driver->next_positions = NULL;
	driver->token_counts = NULL;
	driver->image_programs = NULL;
}
static uint32_t image_runtime_quantum(uint64_t first_work)
{
	uint32_t launches = 0;
	const uint64_t end = test_host_memory_backend_work_count();
	for (uint64_t index = first_work; index < end; ++index) {
		struct test_host_memory_backend_work work;
		const bool present =
			test_host_memory_backend_work_at(index, &work);
		image_runtime_require(present, "work receipt exists");
		if (present && work.kind == TEST_HOST_MEMORY_BACKEND_LAUNCH) {
			image_runtime_require(work.card_index == 0,
					      "encoder card zero");
			++launches;
		}
	}
	image_runtime_require(launches != 0 &&
				      launches <= VISION_PASSES_PER_LAYER,
			      "bounded layer quantum");
	return launches;
}
static void image_runtime_run(struct step_driver *driver,
			      const struct model_plan_record *plan,
			      const struct chat_images *inputs)
{
	bool finished = true;
	const uint64_t before = test_host_memory_backend_work_count();
	result_status status = step_images_advance(driver, 0, &finished);
	image_runtime_require(status == STATUS_OK && !finished,
			      "first encoder quantum stays pending");
	(void)image_runtime_quantum(before);
	const unsigned char *const arena =
		test_host_memory_backend_arena_of(driver->cards[0]);
	image_runtime_require(
		arena != NULL &&
			memcmp(arena + plan->vision.patches.byte_offset,
			       inputs->images[0].patches,
			       TEST_PATCH_BYTES) == 0,
		"original patch upload bytes");
	const uint64_t owner_work = test_host_memory_backend_work_count();
	status = step_images_begin(driver, 0, inputs);
	image_runtime_require(status == STATUS_BUSY_VALUE,
			      "owning slot cannot restart partial image");
	status = step_images_advance(driver, 1, &finished);
	image_runtime_require(status == STATUS_BUSY_VALUE && !finished,
			      "other session waits for scratch owner");
	image_runtime_require(test_host_memory_backend_work_count() ==
				      owner_work,
			      "busy changes no work");
	step_images_cancel(driver, 0);
	status = step_images_advance(driver, 0, &finished);
	image_runtime_require(status == STATUS_OK && finished,
			      "cancel then advance is finished");
	image_runtime_require(
		test_host_memory_backend_work_count() == owner_work &&
			step_image_position(driver, 0, TEST_FIRST_BEGIN),
		"cancel keeps masks and starts no work");
	const uint32_t staging = test_host_memory_backend_staging_block_count();
	test_host_memory_backend_fail_staging_allocation(1);
	uint32_t total_launches = 0, advances = 0;
	finished = false;
	while (!finished && advances < TEST_ADVANCE_LIMIT) {
		const uint64_t first_work =
			test_host_memory_backend_work_count();
		status = step_images_advance(driver, 1, &finished);
		image_runtime_require(status == STATUS_OK,
				      "next owner advances");
		if (status != STATUS_OK)
			break;
		total_launches += image_runtime_quantum(first_work);
		++advances;
	}
	image_runtime_require(finished && total_launches ==
						  TEST_LAUNCHES_PER_IMAGE *
							  inputs->count,
			      "every image and merge executes exactly once");
	image_runtime_require(test_host_memory_backend_staging_block_count() ==
				      staging,
			      "advancement keeps fixed staging");
	void *probe = NULL;
	status = test_host_memory_backend.staging_allocate(
		TEST_STAGING_PROBE_BYTES, &probe);
	image_runtime_require(
		status == STATUS_OUT_OF_MEMORY && probe == NULL,
		"no hidden staging allocation consumed the armed failure");
	if (status == STATUS_OK)
		test_host_memory_backend.staging_free(probe);
	const uint64_t ended_work = test_host_memory_backend_work_count();
	status = step_images_advance(driver, 1, &finished);
	image_runtime_require(status == STATUS_OK && finished &&
				      test_host_memory_backend_work_count() ==
					      ended_work,
			      "finished advancement has no work");
}
int main(void)
{
	struct model_plan_record *const plan = calloc(1, sizeof(*plan));
	if (plan == NULL)
		return EXIT_FAILURE;
	struct execution_program program = {
		.limits = { .session_capacity = TEST_SESSIONS,
			    .context_token_limit = TEST_CONTEXT,
			    .row_capacity = 1 },
		.card_count = 1,
		.row_count = 1
	};
	const struct execution_program *programs[] = { &program };
	struct step_driver driver = { .backend = &test_host_memory_backend,
				      .programs = programs,
				      .wait_limit_nanoseconds = TEST_WAIT_NS };
	bool finished = false;
	result_status status =
		step_images_attach(&driver, plan, ARITHMETIC_CLASS_SM_80);
	image_runtime_require(status == STATUS_OK &&
				      !step_images_supported(&driver),
			      "absent tower attach no-op");
	status = step_images_advance(&driver, 0, &finished);
	image_runtime_require(status == STATUS_OK && finished,
			      "absent tower advance no-op");
	status = step_images_begin(&driver, 0, NULL);
	image_runtime_require(status == STATUS_OK, "absent tower text begin");
	uint8_t original = 0;
	uint16_t patches[IMAGE_PATCH_ELEMENTS];
	for (uint32_t index = 0; index < IMAGE_PATCH_ELEMENTS; ++index)
		patches[index] = (uint16_t)(TEST_BFLOAT_VALUE + index);
	const struct chat_images inputs =
		image_runtime_inputs(&original, patches);
	status = step_images_begin(&driver, 0, &inputs);
	image_runtime_require(status == STATUS_INVALID_ARGUMENT,
			      "absent tower refuses images");
	image_runtime_require(!step_image_position(NULL, 0, 0),
			      "null driver is not an image");
	if (!image_runtime_plan(plan)) {
		free(plan);
		return EXIT_FAILURE;
	}
	test_host_memory_backend_reset(1, TEST_ARENA_BYTES);
	test_host_memory_backend_set_arithmetic_class(0,
						      ARITHMETIC_CLASS_SM_80);
	status = test_host_memory_backend.card_open(0, TEST_ARENA_BYTES,
						    &driver.cards[0]);
	image_runtime_require(status == STATUS_OK, "open host-memory arena");
	if (status != STATUS_OK) {
		free(plan);
		return EXIT_FAILURE;
	}
	test_host_memory_backend_fail_staging_allocation(1);
	status = step_images_attach(&driver, plan, ARITHMETIC_CLASS_SM_80);
	image_runtime_require(
		status == STATUS_OUT_OF_MEMORY &&
			!step_images_supported(&driver) &&
			test_host_memory_backend_staging_block_count() == 0,
		"failed attach owns no staging");
	status = step_images_attach(&driver, plan, ARITHMETIC_CLASS_SM_80);
	image_runtime_require(status == STATUS_OK &&
				      step_images_supported(&driver),
			      "attach official tower");
	if (status != STATUS_OK)
		goto cleanup;
	status = step_images_attach(&driver, plan, ARITHMETIC_CLASS_SM_80);
	image_runtime_require(status == STATUS_INVALID_ARGUMENT,
			      "duplicate attach refused");
	status = step_images_begin(&driver, 0, &inputs);
	image_runtime_require(status == STATUS_OK, "begin first image request");
	image_runtime_preflight(&driver, &inputs);
	status = step_images_begin(&driver, 1, &inputs);
	image_runtime_require(status == STATUS_OK,
			      "begin second image request");
	for (uint32_t position = 0; position < TEST_CONTEXT; ++position) {
		const bool expected =
			(position >= TEST_FIRST_BEGIN &&
			 position < TEST_FIRST_BEGIN + TEST_SPAN_TOKENS) ||
			(position >= TEST_SECOND_BEGIN &&
			 position < TEST_SECOND_BEGIN + TEST_SPAN_TOKENS);
		image_runtime_require(step_image_position(&driver, 0,
							  position) == expected,
				      "exact typed span masks");
	}
	image_runtime_typed_rows(&driver);
	image_runtime_run(&driver, plan, &inputs);
	status = step_images_begin(&driver, 0, NULL);
	image_runtime_require(status == STATUS_OK &&
				      !step_image_position(&driver, 0,
							   TEST_FIRST_BEGIN),
			      "new text request clears old image mask");
	finished = false;
	status = step_images_advance(&driver, 0, &finished);
	image_runtime_require(status == STATUS_OK && finished,
			      "text request finishes without encoder");
	finished = false;
	status = step_images_advance(&driver, TEST_SESSIONS, &finished);
	image_runtime_require(status == STATUS_INVALID_ARGUMENT && !finished,
			      "invalid slot leaves scalar untouched");
	test_host_memory_backend_fail_upload(0, driver.last_ticket[0] + 1u);
	status = step_images_begin(&driver, 0, &inputs);
	image_runtime_require(status == STATUS_DEVICE_ERROR &&
				      driver.failure == STATUS_DEVICE_ERROR,
			      "failed metadata upload poisons image admission");
	const uint64_t failed_work = test_host_memory_backend_work_count();
	finished = false;
	image_runtime_require(
		step_images_begin(&driver, 0, &inputs) == STATUS_DEVICE_ERROR &&
			step_images_advance(&driver, 0, &finished) ==
				STATUS_DEVICE_ERROR &&
			!finished &&
			test_host_memory_backend_work_count() == failed_work,
		"poisoned image work never reuses staging or changes the output");
	/* Closing this host-memory card retires its failed copy before staging is released. */
	test_host_memory_backend.card_close(driver.cards[0], TEST_WAIT_NS);
	driver.cards[0] = NULL;
cleanup:
	step_images_free(&driver, false);
	image_runtime_require(
		!step_images_supported(&driver) &&
			test_host_memory_backend_staging_block_count() == 0,
		"free releases fixed staging");
	if (driver.cards[0] != NULL) {
		test_host_memory_backend.card_close(driver.cards[0],
						    TEST_WAIT_NS);
	}
	image_runtime_require(test_host_memory_backend_open_card_count() == 0,
			      "host arena closed");
	free(plan);
	(void)printf("step_images_test: %u failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
