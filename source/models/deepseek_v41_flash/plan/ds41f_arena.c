/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * ds41f_arena.c: state and arena bytes, initialisation, folds, the embedding copy (split from model_v41.c by whole
 * items).
 */
#include "model_v41.h"
#include "models/deepseek_v41_flash/official/ds41f_official_hc.h"
#include "models/deepseek_v41_flash/official/ds41f_official_reference.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_layer.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_stage.h"
#include "engine/library/exl3/exl3_tc_records.h"
#include "engine/library/rings/exact_ring_records.h"
#include "engine/library/exl3/exl3.h"
#include "models/deepseek_v41_flash/draft/ds41f_dspark_records.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_draft_share_records.h"
#include "engine/library/sampling/sampler_shard_records.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine/backend/fill_records.h"
#include "engine/library/block_scaled_linear/expert_relaxed.h"
#include "common/hooks.h"
#include "models/deepseek_v41_flash/indexer/ds41f_index_scan.h"
#include "engine/formats/digest/pack_digest.h"
#include "engine/library/block_scaled_linear/relaxed_linear.h"
#include "engine/library/prepare_checks.h"

#include "ds41f_plan.h"
#include "models/deepseek_v41_flash/tensors/ds41f_tensors.h"

#define E2M1_VALUES_PER_BYTE 2u

result_status ds41f_state_byte_count(const struct pack_record *pack,
				     const struct placement_record *placement,
				     const struct program_limits *limits,
				     uint32_t card_index,
				     uint64_t *byte_count_out)
{
	struct load_layout *layout = NULL;
	struct ds41f_plan *plan = NULL;
	if (pack == NULL || placement == NULL || limits == NULL ||
	    byte_count_out == NULL ||
	    (card_index >= placement->card_count &&
	     card_index != PLACEMENT_IN_HOST_MEMORY)) {
		LOG_ERROR(
			"state_byte_count: a missing argument, or no card %" PRIu32
			" in the placement",
			card_index);
		return STATUS_INVALID_ARGUMENT;
	}
	result_status status = ds41f_model_check(pack);
	status = status == STATUS_OK ? ds41f_limits_check(limits) : status;
	status = status == STATUS_OK ?
			 load_layout_create(pack, placement->card_of_unit,
					    placement->unit_count,
					    placement->card_count, &layout) :
			 status;
	status = status == STATUS_OK ?
			 ds41f_plan_create(pack, layout, limits, &plan) :
			 status;
	if (status ==
	    STATUS_OK) { /* in host memory: the page-locked copies of its experts there */
		const struct ds41f_card *const card =
			&plan->cards[card_index < placement->card_count ?
					     card_index :
					     0];
		uint64_t host_bytes = 0;
		for (uint32_t layer = 0; layer < DS41F_LAYERS; layer++) {
			host_bytes = ds41f_sum(
				host_bytes,
				ds41f_product(plan->host_expert_count[layer],
					      plan->expert_byte_count));
		}
		*byte_count_out =
			card_index < placement->card_count ?
				card->arena_byte_count - card->state_offset :
				host_bytes;
	}
	ds41f_plan_destroy(plan);
	load_layout_destroy(layout);
	return status;
}

result_status ds41f_arena_byte_count(const struct ds41f_plan *plan,
				     uint32_t card_index,
				     uint64_t *byte_count_out)
{
	if (plan == NULL || byte_count_out == NULL ||
	    card_index >= plan->card_count) {
		LOG_ERROR(
			"arena_byte_count: a missing argument, or no card %" PRIu32
			" in the plan",
			card_index);
		return STATUS_INVALID_ARGUMENT;
	}
	*byte_count_out = plan->cards[card_index].arena_byte_count;
	return STATUS_OK;
}

/* The state of one card behind its slots. The backend's digest of the plan's last word proves the arena holds the
 * state before the card writes the zeros behind the image (fill_words_entry); the image goes up through staging;
 * the card's digest of that state is held to the host's, worked out while the card works. staging_is_read: an upload
 * began that was not seen to finish, so staging may still be read. */
MUST_CHECK static result_status
card_initialise(const struct ds41f_card *plan_card, uint32_t card_index,
		const struct backend_operations *backend,
		struct device_card *card, unsigned char *staging,
		uint64_t wait_limit_nanoseconds, bool *staging_is_read)
{
	const uint64_t image_bytes = plan_card->image_byte_count;
	const uint64_t image_offset = ds41f_image_offset_of(plan_card);
	const uint64_t state_bytes = plan_card->arena_byte_count - image_offset;
	const uint64_t word_bytes = PACK_DIGEST_WORD_BYTES;
	const struct fill_record zeros = { image_offset + image_bytes,
					   (state_bytes - image_bytes) /
						   FILL_WORD_BYTES,
					   0, 0 };
	const uint64_t blocks =
		(zeros.word_count + FILL_THREADS_PER_BLOCK_VALUE - 1u) /
		FILL_THREADS_PER_BLOCK_VALUE;
	const uint32_t grid = blocks == 0		? 1u :
			      blocks < FILL_MOST_BLOCKS ? (uint32_t)blocks :
							  FILL_MOST_BLOCKS;
	struct kernel_launch fill;
	uint64_t ticket = 0, expected = 0, digest = 0;
	uint32_t kernel = BACKEND_NO_KERNEL;
	result_status status = backend->kernel_prepare(card, KERNEL_FILE_FILL,
						       KERNEL_FILL_WORDS,
						       FILL_RECORD_BYTES,
						       &kernel);
	status = status == STATUS_OK ?
			 backend->digest(
				 card, plan_card->arena_byte_count - word_bytes,
				 word_bytes, wait_limit_nanoseconds, &digest) :
			 status;
	prepare_launch_fill(&fill, kernel, grid, 1,
			    FILL_THREADS_PER_BLOCK_VALUE, 0, &zeros,
			    sizeof(zeros));
	status = status == STATUS_OK ? backend->launch_run(card, &fill, 1) :
				       status;
	if (status == STATUS_OK && image_bytes != 0) {
		memcpy(staging, plan_card->image, (size_t)image_bytes);
		status = backend->upload_begin(card, image_offset, staging,
					       image_bytes, &ticket);
		*staging_is_read = status == STATUS_OK;
		status = status == STATUS_OK ?
				 pack_digest_of_host_bytes(plan_card->image,
							   image_bytes,
							   &expected) :
				 status;
	}
	for (uint64_t word = image_bytes / word_bytes;
	     word < state_bytes / word_bytes; word++) {
		expected += pack_digest_term(0, word);
	}
	if (status == STATUS_OK && image_bytes != 0) {
		status = backend->upload_wait(card, ticket,
					      wait_limit_nanoseconds);
		*staging_is_read = status != STATUS_OK;
	}
	status = status == STATUS_OK ?
			 backend->launch_wait(card, wait_limit_nanoseconds) :
			 status;
	status = status == STATUS_OK ?
			 backend->digest(card, image_offset, state_bytes,
					 wait_limit_nanoseconds, &digest) :
			 status;
	const bool differs = status == STATUS_OK && digest != expected;
	status = differs ? STATUS_DEVICE_ERROR : status;
	if (status != STATUS_OK) {
		LOG_ERROR(
			"card %" PRIu32 ": its state, arena bytes %" PRIu64
			" to %" PRIu64 ", was not set%s: %s",
			card_index, image_offset, plan_card->arena_byte_count,
			differs ?
				", the card's digest of it differs from the host's" :
				"",
			status_text(status));
	}
	return status;
}

/* The scales follow the weight bytes (weight_format_seam.h). A failed fold is an arithmetic fallback, while a
 * failed download still fails load. Only the CMP selection reads these folds; the PRO keeps its own arithmetic. */
MUST_CHECK static result_status
expert_folds_prepare(struct ds41f_plan *plan,
		     const struct backend_operations *backend,
		     struct device_card *const *cards, void *staging,
		     uint64_t wait_limit_nanoseconds, bool *staging_is_read)
{
	const uint64_t values = (uint64_t)DS41F_HIDDEN * DS41F_EXPERT_INNER;
	const uint64_t weight_bytes = values / E2M1_VALUES_PER_BYTE;
	const uint64_t scale_bytes = values / LINEAR_VALUES_PER_BLOCK;
	for (uint32_t layer = 0; layer < DS41F_WALK_LAYERS; layer++) {
		plan->expert_folds_admitted[layer] = false;
		if (plan->exl3_bits[layer] != 0 ||
		    plan->experts[layer].byte_count == 0 ||
		    plan->host_expert_count[layer] != 0) {
			continue;
		}
		const uint32_t card = plan->layer_card[layer];
		result_status status = STATUS_OK;
		bool admitted = true;
		const uint32_t expert_count = layer < DS41F_LAYERS ?
						      DS41F_EXPERTS :
						      DS41F_DRAFT_EXPERTS;
		for (uint32_t expert = 0; expert < expert_count && admitted;
		     expert++) {
			for (uint32_t matrix = 0;
			     matrix < DS41F_EXPERT_MATRICES && admitted;
			     matrix++) {
				const uint64_t offset =
					plan->expert_tables[layer][expert]
							   [matrix];
				uint64_t ticket = 0;
				status = backend->download_begin(
					cards[card], offset + weight_bytes,
					staging, scale_bytes, &ticket);
				*staging_is_read = status == STATUS_OK;
				status =
					status == STATUS_OK ?
						backend->upload_wait(
							cards[card], ticket,
							wait_limit_nanoseconds) :
						status;
				if (status != STATUS_OK) {
					return status;
				}
				*staging_is_read = false;
				status = expert_relaxed_fold_check(staging,
								   scale_bytes);
				if (status == STATUS_INVALID_ARGUMENT) {
					LOG_WARNING_ENTRY(
						"layers.%" PRIu32
						".ffn.experts.%" PRIu32
						".%s: scales refuse the fold; "
						"reference CMP order",
						layer, expert,
						ds41f_expert_rules[matrix].name);
					admitted = false;
				} else if (status != STATUS_OK) {
					return status;
				}
			}
		}
		plan->expert_folds_admitted[layer] = admitted;
	}
	return STATUS_OK;
}

MUST_CHECK static result_status
folds_prepare(struct ds41f_plan *plan, const struct backend_operations *backend,
	      struct device_card *const *cards, void *staging,
	      uint64_t wait_limit_nanoseconds, bool *staging_is_read)
{
	result_status status = STATUS_OK;
	for (uint32_t layer = 0; layer < DS41F_LAYERS && status == STATUS_OK;
	     layer++) {
		for (uint32_t index = 0;
		     index < DS41F_TENSOR_LAYER_COUNT && status == STATUS_OK;
		     index++) {
			struct ds41f_tensor *const tensor =
				&plan->layer_tensors[layer][index];
			tensor->fold_reference = 0;
			if (tensor->weight_format !=
			    WEIGHT_FORMAT_E4M3_SCALED) {
				continue;
			}
			const uint64_t weight_bytes =
				(uint64_t)tensor->shape.output_size *
				tensor->shape.input_size;
			const uint64_t scale_bytes =
				tensor->stored.byte_count - weight_bytes;
			uint64_t ticket = 0;
			status = backend->download_begin(
				cards[tensor->card],
				tensor->stored.byte_offset + weight_bytes,
				staging, scale_bytes, &ticket);
			*staging_is_read = status == STATUS_OK;
			status = status == STATUS_OK ?
					 backend->upload_wait(
						 cards[tensor->card], ticket,
						 wait_limit_nanoseconds) :
					 status;
			if (status != STATUS_OK) {
				break;
			}
			*staging_is_read = false;
			status = relaxed_fold_check(
				index == DS41F_TENSOR_OUTPUT_GROUPS ?
					RELAXED_BFLOAT16_ROWS :
					RELAXED_QUANTIZED_ROWS,
				staging, scale_bytes, &tensor->fold_reference);
			if (status == STATUS_INVALID_ARGUMENT) {
				LOG_WARNING_ENTRY(
					"layers.%" PRIu32
					".%s: scales refuse the exact fold; reference CMP order",
					layer, ds41f_layer_rules[index].name);
				status = STATUS_OK;
			}
		}
	}
	return status;
}

/* The draft's copy of embed.weight (model_v41.h): from the embedding card's weights into DRAFT_EMBEDDING of the result
 * card through staging, a piece at a time, each piece seen to arrive before the next; then the cards' digests of both
 * ranges, held equal. staging_is_read as card_initialise sets it. */
MUST_CHECK static result_status
embedding_copy(const struct ds41f_plan *plan,
	       const struct backend_operations *backend,
	       struct device_card *const *cards, unsigned char *staging,
	       uint64_t wait_limit_nanoseconds, bool *staging_is_read)
{
	const struct linear_range source =
		plan->globals[DS41F_TENSOR_EMBEDDING].stored;
	const uint64_t copy =
		plan->cards[plan->result_card]
			.array_offset[DS41F_ARRAY_DRAFT_EMBEDDING];
	const uint64_t piece = DS41F_EMBEDDING_COPY_PIECE_BYTES;
	struct device_card *const from = cards[plan->embedding_card],
				  *const to = cards[plan->result_card];
	uint64_t source_digest = 0, copy_digest = 0;
	result_status status = STATUS_OK;
	for (uint64_t done = 0; done < source.byte_count && status == STATUS_OK;
	     done += piece) {
		const uint64_t bytes = source.byte_count - done < piece ?
					       source.byte_count - done :
					       piece;
		uint64_t ticket = 0;
		status = backend->download_begin(from,
						 source.byte_offset + done,
						 staging, bytes, &ticket);
		*staging_is_read = status == STATUS_OK;
		status = status == STATUS_OK ?
				 backend->upload_wait(from, ticket,
						      wait_limit_nanoseconds) :
				 status;
		*staging_is_read = *staging_is_read && status != STATUS_OK;
		if (status == STATUS_OK) {
			status = backend->upload_begin(to, copy + done, staging,
						       bytes, &ticket);
			*staging_is_read = status == STATUS_OK;
			status = status == STATUS_OK ?
					 backend->upload_wait(
						 to, ticket,
						 wait_limit_nanoseconds) :
					 status;
			*staging_is_read = *staging_is_read &&
					   status != STATUS_OK;
		}
	}
	status = status == STATUS_OK ? backend->digest(from, source.byte_offset,
						       source.byte_count,
						       wait_limit_nanoseconds,
						       &source_digest) :
				       status;
	status = status == STATUS_OK ?
			 backend->digest(to, copy, source.byte_count,
					 wait_limit_nanoseconds, &copy_digest) :
			 status;
	const bool differs = status == STATUS_OK &&
			     source_digest != copy_digest;
	status = differs ? STATUS_DEVICE_ERROR : status;
	if (status != STATUS_OK) {
		LOG_ERROR(
			"the draft's copy of embed.weight, %" PRIu64
			" bytes from card %" PRIu32 " to byte %" PRIu64
			" of card %" PRIu32 ", was not made%s: %s",
			source.byte_count, plan->embedding_card, copy,
			plan->result_card,
			differs ?
				", the cards' digests of the two ranges differ" :
				"",
			status_text(status));
	}
	return status;
}

/* A plan's copy of embed.weight lies whole in the embedding card's weights and in the result card's state behind its
 * image, in whole digest words, on cards given, with a backend that downloads. */
static bool copy_is_planned(const struct ds41f_plan *plan,
			    const struct backend_operations *backend,
			    struct device_card *const *cards)
{
	if (plan->embedding_card >= plan->card_count ||
	    plan->result_card >= plan->card_count) {
		return false;
	}
	const struct linear_range source =
		plan->globals[DS41F_TENSOR_EMBEDDING].stored;
	const struct ds41f_card *const from =
		&plan->cards[plan->embedding_card];
	const struct ds41f_card *const to = &plan->cards[plan->result_card];
	const uint64_t copy = to->array_offset[DS41F_ARRAY_DRAFT_EMBEDDING];
	return cards[plan->embedding_card] != NULL &&
	       cards[plan->result_card] != NULL && backend != NULL &&
	       backend->download_begin != NULL && source.byte_count != 0 &&
	       source.byte_count <= DS41F_DRAFT_EMBEDDING_BYTES &&
	       source.byte_offset <= from->state_offset &&
	       source.byte_count <= from->state_offset - source.byte_offset &&
	       copy >= ds41f_image_offset_of(to) + to->image_byte_count &&
	       copy <= to->arena_byte_count &&
	       source.byte_count <= to->arena_byte_count - copy &&
	       (source.byte_offset | source.byte_count | copy) %
			       PACK_DIGEST_WORD_BYTES ==
		       0;
}

result_status ds41f_arena_initialise(struct ds41f_plan *plan,
				     const struct backend_operations *backend,
				     struct device_card *const *cards,
				     uint64_t wait_limit_nanoseconds)
{
	void *staging = NULL;
	bool staging_is_read = false;
	uint64_t largest_image = 0;
	bool cards_are_given = plan != NULL && cards != NULL &&
			       plan->card_count - 1u < BACKEND_MAXIMUM_CARDS;
	const bool has_draft = cards_are_given &&
			       plan->limits.draft_positions != 0;
	for (uint32_t card = 0; cards_are_given && card < plan->card_count;
	     card++) {
		const struct ds41f_card *const planned = &plan->cards[card];
		cards_are_given =
			(cards[card] != NULL ||
			 planned->arena_byte_count == planned->state_offset) &&
			planned->state_offset <= planned->arena_byte_count &&
			planned->slots_byte_count <=
				planned->arena_byte_count -
					planned->state_offset &&
			planned->image_byte_count <=
				planned->arena_byte_count -
					ds41f_image_offset_of(planned) &&
			(planned->image != NULL ||
			 planned->image_byte_count == 0) &&
			(planned->state_offset | planned->slots_byte_count |
			 planned->arena_byte_count |
			 planned->image_byte_count) %
					PACK_DIGEST_WORD_BYTES ==
				0;
		largest_image = planned->image_byte_count > largest_image ?
					planned->image_byte_count :
					largest_image;
	}
	const bool check_folds = plan != NULL &&
				 plan->limits.cmp_order == CMP_ORDER_RELAXED;
	if (check_folds) {
		memset(plan->expert_folds_admitted, 0,
		       sizeof(plan->expert_folds_admitted));
	}
	for (uint32_t layer = 0;
	     check_folds && cards_are_given && layer < DS41F_LAYERS; layer++) {
		for (uint32_t index = 0; index < DS41F_TENSOR_LAYER_COUNT;
		     index++) {
			const struct ds41f_tensor *const tensor =
				&plan->layer_tensors[layer][index];
			if (tensor->weight_format ==
			    WEIGHT_FORMAT_E4M3_SCALED) {
				const uint64_t weight_bytes =
					(uint64_t)tensor->shape.output_size *
					tensor->shape.input_size;
				cards_are_given = cards_are_given &&
						  tensor->card <
							  plan->card_count &&
						  tensor->stored.byte_count >
							  weight_bytes;
				const uint64_t scales =
					tensor->stored.byte_count >
							weight_bytes ?
						tensor->stored.byte_count -
							weight_bytes :
						0;
				largest_image = scales > largest_image ?
							scales :
							largest_image;
			}
		}
	}
	for (uint32_t layer = 0;
	     check_folds && cards_are_given && layer < DS41F_WALK_LAYERS;
	     layer++) {
		if (plan->experts[layer].byte_count != 0 &&
		    plan->host_expert_count[layer] == 0) {
			const uint64_t values =
				(uint64_t)DS41F_HIDDEN * DS41F_EXPERT_INNER;
			const uint64_t scale_bytes =
				values / LINEAR_VALUES_PER_BLOCK;
			const uint64_t matrix_bytes =
				values / E2M1_VALUES_PER_BYTE + scale_bytes;
			const struct linear_range *const range =
				&plan->experts[layer];
			cards_are_given =
				cards_are_given &&
				plan->layer_card[layer] < plan->card_count &&
				range->byte_offset <=
					UINT64_MAX - range->byte_count;
			if (cards_are_given) {
				cards_are_given =
					range->byte_offset +
						range->byte_count <=
					plan->cards[plan->layer_card[layer]]
						.state_offset;
			}
			const uint32_t expert_count =
				layer < DS41F_LAYERS ? DS41F_EXPERTS :
						       DS41F_DRAFT_EXPERTS;
			for (uint32_t expert = 0; expert < expert_count;
			     expert++) {
				for (uint32_t matrix = 0;
				     matrix < DS41F_EXPERT_MATRICES; matrix++) {
					const uint64_t offset =
						plan->expert_tables[layer]
								   [expert]
								   [matrix];
					cards_are_given =
						cards_are_given &&
						offset >= range->byte_offset &&
						offset - range->byte_offset <=
							range->byte_count &&
						matrix_bytes <=
							range->byte_count -
								(offset -
								 range->byte_offset);
				}
			}
			largest_image = scale_bytes > largest_image ?
						scale_bytes :
						largest_image;
		}
	}
	if (!cards_are_given || backend == NULL ||
	    backend->staging_allocate == NULL ||
	    backend->staging_free == NULL || backend->upload_begin == NULL ||
	    backend->upload_wait == NULL || backend->digest == NULL ||
	    backend->kernel_prepare == NULL || backend->launch_run == NULL ||
	    backend->launch_wait == NULL ||
	    (check_folds && backend->download_begin == NULL) ||
	    (has_draft && !copy_is_planned(plan, backend, cards))) {
		LOG_ERROR(
			"arena_initialise: a missing argument, card or function of the backend, or not a plan");
		return STATUS_INVALID_ARGUMENT;
	}
	const uint64_t staging_bytes =
		has_draft && largest_image < DS41F_EMBEDDING_COPY_PIECE_BYTES ?
			DS41F_EMBEDDING_COPY_PIECE_BYTES :
			largest_image;
	result_status status =
		staging_bytes == 0 ?
			STATUS_OK :
			backend->staging_allocate(staging_bytes, &staging);
	for (uint32_t card = 0; card < plan->card_count && status == STATUS_OK;
	     card++) {
		status = plan->cards[card].arena_byte_count ==
					 plan->cards[card].state_offset ?
				 STATUS_OK :
				 card_initialise(&plan->cards[card], card,
						 backend, cards[card], staging,
						 wait_limit_nanoseconds,
						 &staging_is_read);
	}
	status = check_folds && status == STATUS_OK ?
			 folds_prepare(plan, backend, cards, staging,
				       wait_limit_nanoseconds,
				       &staging_is_read) :
			 status;
	status = check_folds && status == STATUS_OK ?
			 expert_folds_prepare(plan, backend, cards, staging,
					      wait_limit_nanoseconds,
					      &staging_is_read) :
			 status;
	status = status == STATUS_OK && has_draft ?
			 embedding_copy(plan, backend, cards, staging,
					wait_limit_nanoseconds,
					&staging_is_read) :
			 status;
	if (staging_is_read) {
		LOG_ERROR(
			"arena_initialise: staging memory stays allocated: an upload may still read it");
	} else if (staging != NULL) {
		backend->staging_free(staging);
	}
	return status;
}
