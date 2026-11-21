/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* The program builder and immutable banks share these target-head descriptions. Startup only. */
#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "engine/library/bf16_linear/dense_linear.h"
#include "engine/library/bf16_linear/dense_stream_records.h"
#include "engine/library/tensor_parallel/peer_row_copy.h"
#include "engine/library/sampling/sampler.h"
#include "engine/runtime/step/step.h"
#define OK STATUS_OK
#define THEN(call) (status != OK ? status : (call))

static struct linear_range hidden_of(const struct ds41f_plan *plan,
				     uint32_t rank, uint32_t bank)
{
	const struct ds41f_original_head_card *const head =
		&plan->original_head.cards[rank];
	return bank == 0 ? head->hidden : head->bank_hidden[bank - 1u];
}

static struct linear_range logits_of(const struct ds41f_plan *plan,
				     uint32_t card, uint32_t bank)
{
	const struct ds41f_original_head_card *const head =
		&plan->original_head.cards[card];
	return bank != 0 ? head->bank_logits[bank - 1u] :
	       card != DS41F_HEAD_RANKS ?
			   head->logits :
			   (struct linear_range){
				   plan->cards[card]
					   .array_offset[DS41F_ARRAY_LOGITS],
				   (uint64_t)LINEAR_MAXIMUM_ROWS *
					   DS41F_LOGITS_ROW_BYTES
			   };
}

MUST_CHECK static result_status
copy_launch(const struct ds41f_plan *plan,
	    const struct backend_operations *backend,
	    struct device_card *const *cards, uint32_t rank, uint32_t bank,
	    uint32_t rows, bool broadcast, struct kernel_launch *launch)
{
	const uint32_t pro = DS41F_HEAD_RANKS, card = broadcast ? pro : rank;
	const struct ds41f_original_head_card *const head =
		&plan->original_head.cards[card];
	const struct linear_range step = {
		plan->cards[card].array_offset[DS41F_ARRAY_STEP],
		STEP_BLOCK_BYTES_VALUE + (uint64_t)rows * STEP_ROW_BYTES_VALUE
	};
	const struct linear_range source =
		broadcast ?
			(struct linear_range){
				plan->cards[pro]
					.array_offset[DS41F_ARRAY_HIDDEN],
				(uint64_t)rows * DS41F_HIDDEN_ROW_BYTES
			} :
			logits_of(plan, rank, bank);
	const struct head_offload_description copy = {
		source,
		head->peer_words[broadcast ? rank : 0u],
		step,
		broadcast ? hidden_of(plan, rank, bank) :
			    logits_of(plan, pro, bank),
		plan->cards[broadcast ? rank : pro].arena_byte_count,
		broadcast ? DS41F_HIDDEN_ROW_BYTES :
			    (uint64_t)DS41F_HEAD_SHARD_ROWS * sizeof(float),
		broadcast ? DS41F_HIDDEN_ROW_BYTES : DS41F_LOGITS_ROW_BYTES,
		broadcast ? DS41F_HIDDEN_ROW_BYTES :
			    DS41F_HEAD_SHARD_ROWS * sizeof(float),
		rows,
		broadcast ?
			0u :
			(size_t)(rank * DS41F_HEAD_SHARD_ROWS) * sizeof(float),
		0
	};
	uint32_t kernel = BACKEND_NO_KERNEL;
	result_status status = backend->kernel_prepare(
		cards[card], KERNEL_FILE_HEAD_OFFLOAD, KERNEL_HEAD_OFFLOAD_COPY,
		HEAD_OFFLOAD_ARGUMENT_BYTES, &kernel);
	return THEN(head_offload_prepare(
		&copy, plan->cards[card].arena_byte_count, kernel, launch));
}

MUST_CHECK static result_status
dense_launch(const struct ds41f_plan *plan,
	     const struct backend_operations *backend,
	     struct device_card *const *cards, uint32_t rank, uint32_t bank,
	     uint32_t rows, struct kernel_launch *launch)
{
	const struct dense_linear_description product = {
		plan->original_head.cards[rank].weights,
		hidden_of(plan, rank, bank),
		logits_of(plan, rank, bank),
		{ plan->cards[rank].array_offset[DS41F_ARRAY_STEP],
		  STEP_BLOCK_BYTES_VALUE +
			  (uint64_t)rows * STEP_ROW_BYTES_VALUE },
		DS41F_HEAD_SHARD_ROWS,
		DS41F_HIDDEN,
		rows,
		DENSE_BFLOAT16,
		DENSE_FLOAT32,
		DENSE_STEP_ROWS,
		1.0f,
		0.0f,
		0
	};
	struct dense_linear_kernels kernels;
	result_status status =
		dense_linear_kernels_prepare(backend, cards[rank], &kernels);
	status = THEN(dense_linear_prepare(&product,
					   plan->cards[rank].arena_byte_count,
					   &kernels, launch));
	if (status == OK && rows > 1u) {
		status = backend->kernel_prepare(cards[rank],
						 KERNEL_FILE_DENSE_STREAM,
						 KERNEL_DENSE_STREAM,
						 DENSE_ARGUMENT_BYTES,
						 &launch->kernel);
		launch->grid[0] = DENSE_STREAM_BLOCKS;
	}
	return status;
}

MUST_CHECK static result_status
sample_launch(const struct ds41f_plan *plan,
	      const struct backend_operations *backend,
	      struct device_card *const *cards, uint32_t bank, uint32_t rows,
	      struct kernel_launch *launch)
{
	const uint32_t card = DS41F_HEAD_RANKS;
	struct linear_range logits = logits_of(plan, card, bank);
	logits.byte_count = (uint64_t)rows * DS41F_LOGITS_ROW_BYTES;
	const struct sampler_description sample = {
		{ plan->cards[card].array_offset[DS41F_ARRAY_STEP],
		  STEP_BLOCK_BYTES_VALUE +
			  (uint64_t)rows * STEP_ROW_BYTES_VALUE },
		{ plan->cards[card].array_offset[DS41F_ARRAY_SESSIONS],
		  (uint64_t)plan->limits.session_capacity *
			  STEP_SESSION_BYTES },
		logits,
		{ plan->cards[card].array_offset[DS41F_ARRAY_SAMPLER_WEIGHTS],
		  (uint64_t)rows * DS41F_LOGITS_ROW_BYTES },
		{ plan->cards[card].array_offset[DS41F_ARRAY_RESULT],
		  STEP_RESULT_BYTES + (uint64_t)rows * STEP_ROW_RESULT_BYTES },
		DS41F_VOCABULARY_VALUE,
		plan->limits.session_capacity,
		rows,
		0
	};
	struct sampler_kernels_record kernels;
	result_status status =
		sampler_kernels_prepare(backend, cards[card], &kernels);
	return THEN(sampler_prepare_entry(
		&sample, plan->cards[card].arena_byte_count, &kernels, launch));
}

result_status ds41f_original_head_launch(
	const struct ds41f_plan *plan, const struct backend_operations *backend,
	struct device_card *const *cards, enum ds41f_head_launch_kind kind,
	uint32_t rank, uint32_t bank, uint32_t rows,
	struct kernel_launch *launch_out)
{
	if (plan == NULL || backend == NULL ||
	    backend->kernel_prepare == NULL || cards == NULL ||
	    launch_out == NULL || !plan->original_head.enabled ||
	    plan->card_count != DS41F_HEAD_CARDS || rank >= DS41F_HEAD_RANKS ||
	    rows == 0 || rows > LINEAR_MAXIMUM_ROWS ||
	    (bank != 0 && bank >= plan->original_head.bank_count))
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t card = 0; card < DS41F_HEAD_CARDS; card++)
		if (cards[card] == NULL)
			return STATUS_INVALID_ARGUMENT;
	struct kernel_launch launch;
	result_status status = STATUS_INVALID_ARGUMENT;
	switch (kind) {
	case DS41F_HEAD_BROADCAST:
	case DS41F_HEAD_GATHER:
		status = copy_launch(plan, backend, cards, rank, bank, rows,
				     kind == DS41F_HEAD_BROADCAST, &launch);
		break;
	case DS41F_HEAD_DENSE:
		status = dense_launch(plan, backend, cards, rank, bank, rows,
				      &launch);
		break;
	case DS41F_HEAD_SAMPLE:
		status = sample_launch(plan, backend, cards, bank, rows,
				       &launch);
		break;
	case DS41F_HEAD_LAUNCH_KINDS:
	default:
		break;
	}
	if (status == OK)
		*launch_out = launch;
	return status;
}
