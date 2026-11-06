/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* The worker's serialized image scratch and per-session results. No card is opened here. */
#include <stdlib.h>
#include <string.h>

#include "anemone/chat_images.h"
#include "anemone/hooks.h"
#include "anemone/step_images.h"
#include "model_v41.h"
#include "step_driver.h"
#include "serving_cursor.h"

#define IMAGE_NO_OWNER UINT32_MAX
#define IMAGE_PATCH_BYTES \
	((uint64_t)IMAGE_MAX_PATCHES * IMAGE_PATCH_ELEMENTS * sizeof(uint16_t))
#define IMAGE_QUANTUM VISION_PASSES_PER_LAYER
#define IMAGE_LAUNCH_CAPACITY (VISION_MAXIMUM_LAUNCHES + 1u)

struct image_slot {
	const struct chat_images *inputs;
	struct vision_session metadata;
	uint32_t next_image;
};

struct step_images {
	const struct model_plan_record *plan;
	struct vision_kernels kernels;
	struct kernel_launch launches[IMAGE_LAUNCH_CAPACITY];
	struct image_slot slots[SERVING_SESSION_CAPACITY];
	unsigned char *staging;
	uint32_t owner, launch_count, next_launch;
};

bool step_images_supported(const struct step_driver *driver)
{
	return driver != NULL && driver->images != NULL;
}

result_status step_images_attach(struct step_driver *driver,
				 const struct model_plan_record *plan,
				 uint32_t arithmetic_class)
{
	if (driver == NULL || plan == NULL || driver->images != NULL ||
	    driver->sequence != 0 || driver->pipeline != NULL ||
	    plan->limits.session_capacity > SERVING_SESSION_CAPACITY) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (!plan->vision.enabled) {
		return STATUS_OK;
	}
	struct step_images *const images = FAULT_POINT(step_driver_memory) ?
						   NULL :
						   calloc(1, sizeof(*images));
	if (images == NULL) {
		return STATUS_OUT_OF_MEMORY;
	}
	images->plan = plan;
	images->owner = IMAGE_NO_OWNER;
	result_status status =
		vision_kernels_prepare(driver->backend, driver->cards[0],
				       arithmetic_class, &images->kernels);
	if (status == STATUS_OK) {
		status = driver->backend->staging_allocate(
			IMAGE_PATCH_BYTES, (void **)&images->staging);
	}
	if (status != STATUS_OK) {
		free(images);
		return status;
	}
	driver->images = images;
	return STATUS_OK;
}

MUST_CHECK static result_status image_upload(struct step_driver *driver,
					     uint64_t offset, const void *bytes,
					     uint64_t byte_count)
{
	struct step_images *const images = driver->images;
	if (byte_count > IMAGE_PATCH_BYTES || bytes == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	memcpy(images->staging, bytes, (size_t)byte_count);
	result_status status = step_copy_begin(
		driver, 0, offset, images->staging, byte_count, false);
	if (status == STATUS_OK) {
		status = card_wait(driver, 0);
	}
	if (status != STATUS_OK) {
		driver->failure = status;
	}
	return status;
}

result_status step_images_begin(struct step_driver *driver, uint32_t slot,
				const struct chat_images *inputs)
{
	if (driver == NULL ||
	    slot >= driver->programs[0]->limits.session_capacity ||
	    !step_pipeline_is_idle(driver) ||
	    (inputs != NULL &&
	     (inputs->count > IMAGE_MAX_IMAGES || driver->images == NULL))) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (driver->images == NULL) {
		return STATUS_OK;
	}
	if (driver->failure != STATUS_OK) {
		return driver->failure;
	}
	struct step_images *const images = driver->images;
	if (images->owner == slot) {
		return STATUS_BUSY_VALUE;
	}
	struct image_slot prepared = { .inputs = inputs };
	prepared.metadata.image_count = inputs == NULL ? 0 : inputs->count;
	uint32_t previous_end = 0;
	for (uint32_t index = 0; index < prepared.metadata.image_count;
	     index++) {
		const struct chat_image_input *const image =
			&inputs->images[index];
		if (image->patches == NULL || image->bytes == NULL ||
		    image->byte_count == 0 ||
		    image->byte_count > IMAGE_MAX_BYTES ||
		    image->token_begin < previous_end ||
		    image->token_begin >=
			    images->plan->limits.context_token_limit ||
		    image->plan.token_count == 0 ||
		    image->plan.token_count >
			    images->plan->limits.context_token_limit -
				    image->token_begin ||
		    image->plan.width == 0 || image->plan.height == 0 ||
		    image->plan.vit_height == 0 || image->plan.vit_width == 0 ||
		    image->plan.vit_height >
			    IMAGE_MAX_PATCHES / image->plan.vit_width ||
		    image->plan.llm_height !=
			    (image->plan.vit_height + IMAGE_DOWNSAMPLE - 1u) /
				    IMAGE_DOWNSAMPLE ||
		    image->plan.llm_width !=
			    (image->plan.vit_width + IMAGE_DOWNSAMPLE - 1u) /
				    IMAGE_DOWNSAMPLE ||
		    image->plan.patch_elements !=
			    (size_t)image->plan.vit_height *
				    image->plan.vit_width *
				    IMAGE_PATCH_ELEMENTS ||
		    image->plan.token_count !=
			    image->plan.llm_height *
					    (image->plan.llm_width + 1u) +
				    VISION_HALF_DIVISOR ||
		    image->plan.token_count > IMAGE_MAX_TOKENS) {
			return STATUS_INVALID_ARGUMENT;
		}
		prepared.metadata.spans[index] = (struct vision_span){
			.token_begin = image->token_begin,
			.token_count = image->plan.token_count,
			.embedding_row = index * IMAGE_MAX_TOKENS
		};
		previous_end = image->token_begin + image->plan.token_count;
	}
	unsigned char metadata[VISION_SESSION_METADATA_BYTES] = { 0 };
	memcpy(metadata, &prepared.metadata, sizeof(prepared.metadata));
	const uint64_t offset =
		images->plan->vision.session_metadata.byte_offset +
		(uint64_t)slot * VISION_SESSION_METADATA_BYTES;
	const result_status status =
		image_upload(driver, offset, metadata, sizeof(metadata));
	if (status == STATUS_OK) {
		images->slots[slot] = prepared;
	}
	return status;
}

/* Prepare one image into shared scratch, with its delimiter span retained in this slot's own output. */
MUST_CHECK static result_status image_started(struct step_driver *driver,
					      uint32_t slot)
{
	struct step_images *const images = driver->images;
	const struct image_slot *const session = &images->slots[slot];
	const struct chat_image_input *const image =
		&session->inputs->images[session->next_image];
	const struct model_vision *const planned = &images->plan->vision;
	const uint64_t arena_bytes = images->plan->cards[0].arena_byte_count;
	const struct vision_description description = {
		.config = vision_official_config(),
		.weights = &planned->weights,
		.patches = { planned->patches.byte_offset,
			     image->plan.patch_elements * sizeof(uint16_t) },
		.output = planned->encoded,
		.scratch = planned->scratch,
		.grid_height = image->plan.vit_height,
		.grid_width = image->plan.vit_width
	};
	result_status status = vision_prepare(
		&description, arena_bytes, &images->kernels, images->launches,
		VISION_MAXIMUM_LAUNCHES, &images->launch_count);
	const struct linear_range output = {
		.byte_offset =
			planned->session_embeddings.byte_offset +
			(uint64_t)slot * planned->session_embedding_stride +
			(uint64_t)session->next_image * IMAGE_MAX_TOKENS *
				MODEL_HIDDEN_ROW_BYTES,
		.byte_count = (uint64_t)image->plan.token_count *
			      MODEL_HIDDEN_ROW_BYTES
	};
	if (status == STATUS_OK) {
		status = vision_merge_prepare(
			&planned->weights, &planned->encoded, &output,
			MODEL_HIDDEN, image->plan.llm_height,
			image->plan.llm_width, 1, arena_bytes, &images->kernels,
			&images->launches[images->launch_count]);
	}
	if (status == STATUS_OK) {
		images->launch_count++;
		TIMER_BEGIN(image_patch_upload);
		status = image_upload(driver, planned->patches.byte_offset,
				      image->patches,
				      (uint64_t)image->plan.patch_elements *
					      sizeof(uint16_t));
		TIMER_END(image_patch_upload);
	}
	if (status == STATUS_OK) {
		images->next_launch = 0;
		images->owner = slot;
	}
	return status;
}

result_status step_images_advance(struct step_driver *driver, uint32_t slot,
				  bool *finished_out)
{
	if (driver == NULL || finished_out == NULL ||
	    slot >= driver->programs[0]->limits.session_capacity ||
	    !step_pipeline_is_idle(driver)) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (driver->failure != STATUS_OK) {
		return driver->failure;
	}
	struct step_images *const images = driver->images;
	if (images == NULL ||
	    images->slots[slot].next_image ==
		    images->slots[slot].metadata.image_count) {
		*finished_out = true;
		return STATUS_OK;
	}
	if (images->owner != IMAGE_NO_OWNER && images->owner != slot) {
		return STATUS_BUSY_VALUE;
	}
	result_status status = images->owner == IMAGE_NO_OWNER ?
				       image_started(driver, slot) :
				       STATUS_OK;
	const uint32_t remaining = images->launch_count - images->next_launch;
	const uint32_t count = remaining > IMAGE_QUANTUM ? IMAGE_QUANTUM :
							   remaining;
	if (status == STATUS_OK) {
		/* Summing these completed quanta gives encoder occupancy without intervening session turns. */
		TIMER_BEGIN(image_encoder_quantum);
		driver->uncovered_work[0] = true;
		status = driver->backend->launch_run(
			driver->cards[0],
			images->launches + images->next_launch, count);
		if (status == STATUS_OK) {
			status = card_wait(driver, 0);
		}
		TIMER_END(image_encoder_quantum);
	}
	if (status != STATUS_OK && driver->uncovered_work[0]) {
		driver->failure = status;
	}
	if (status == STATUS_OK) {
		images->next_launch += count;
		if (images->next_launch == images->launch_count) {
			images->slots[slot].next_image++;
			images->owner = IMAGE_NO_OWNER;
			COUNTER_ADD(image_encoder_completed, 1u);
		}
		*finished_out = images->slots[slot].next_image ==
				images->slots[slot].metadata.image_count;
	}
	return status;
}

void step_images_cancel(struct step_driver *driver, uint32_t slot)
{
	if (driver != NULL && driver->images != NULL &&
	    slot < SERVING_SESSION_CAPACITY) {
		struct step_images *const images = driver->images;
		images->owner = images->owner == slot ? IMAGE_NO_OWNER :
							images->owner;
		images->slots[slot].inputs = NULL;
		images->slots[slot].next_image =
			images->slots[slot].metadata.image_count;
	}
}

bool step_image_position(const struct step_driver *driver, uint32_t slot,
			 uint32_t position)
{
	if (driver == NULL || driver->images == NULL ||
	    slot >= SERVING_SESSION_CAPACITY) {
		return false;
	}
	const struct vision_session *const metadata =
		&driver->images->slots[slot].metadata;
	for (uint32_t index = 0; index < metadata->image_count; index++) {
		const struct vision_span *const span = &metadata->spans[index];
		if (position >= span->token_begin &&
		    position - span->token_begin < span->token_count) {
			return true;
		}
	}
	return false;
}

void step_images_free(struct step_driver *driver, bool staging_is_read)
{
	if (driver != NULL && driver->images != NULL) {
		struct step_images *const images = driver->images;
		if (!staging_is_read) {
			driver->backend->staging_free(images->staging);
		} else if (driver->backend->staging_retire != NULL) {
			driver->backend->staging_retire(images->staging);
		}
		free(images);
		driver->images = NULL;
	}
}
