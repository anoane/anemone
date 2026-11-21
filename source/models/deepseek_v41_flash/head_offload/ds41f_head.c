/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Default-off original PP5 target-head storage, checked load and real-allocation peer qualification.
 * Decoder weights/state retain their offsets. All five arenas stay alive through a proven group drain. */
#include "models/deepseek_v41_flash/plan/model_v41.h"

#include <string.h>

#include "engine/library/tensor_parallel/peer_row_copy.h"
#include "common/hooks.h"
#include "engine/storage/load/tp_load.h"

#define HEAD_PEER_BYTES 64u
#define HEAD_PILOT_BYTES 64u
#define HEAD_PILOT_MARK 0u
#define HEAD_PILOT_SENTINEL 0xa5u
/* a pilot copy from the PRO card to a rank, then back */
#define HEAD_PILOT_DIRECTIONS 2u
/* the pilot pattern: two marks around the rank and the direction */
#define HEAD_PILOT_WORDS 4u
#define HEAD_PILOT_FIRST_MARK 0x12345678u
#define HEAD_PILOT_LAST_MARK 0x9abcdef0u
#define OK STATUS_OK
#define THEN(call) (status != OK ? status : (call))

static bool append_range(uint64_t *end, uint64_t bytes,
			 struct linear_range *range)
{
	const uint64_t alignment = DS41F_ARRAY_ALIGNMENT_BYTES;
	if (*end > UINT64_MAX - (alignment - 1u))
		return false;
	const uint64_t offset = (*end + alignment - 1u) & ~(alignment - 1u);
	if (bytes > UINT64_MAX - offset - (alignment - 1u))
		return false;
	*range = (struct linear_range){ offset, bytes };
	*end = (offset + bytes + alignment - 1u) & ~(alignment - 1u);
	return true;
}

result_status
ds41f_original_head_plan(struct ds41f_plan *plan,
			 const struct card_description_record *descriptions)
{
	if (plan == NULL || descriptions == NULL)
		return STATUS_INVALID_ARGUMENT;
	if (plan->original_head.enabled ||
	    plan->card_count != DS41F_HEAD_CARDS ||
	    plan->result_card != DS41F_HEAD_RANKS ||
	    plan->embedding_card != 0 || plan->limits.official_reference != 0 ||
	    plan->limits.official_hc_rows != 0)
		return STATUS_UNSUPPORTED;
	static const uint32_t last_layer[DS41F_HEAD_CARDS] = { 8, 15, 22, 29,
							       40 };
	for (uint32_t layer = 0, card = 0; layer < DS41F_LAYERS; layer++) {
		card += layer == last_layer[card] ? 1u : 0u;
		if (plan->layer_card[layer] != card ||
		    plan->host_expert_count[layer] != 0 ||
		    plan->exl3_bits[layer] != 0)
			return STATUS_UNSUPPORTED;
	}
	for (uint32_t card = 0; card < DS41F_HEAD_CARDS; card++) {
		const bool pro = card == DS41F_HEAD_RANKS;
		if (descriptions[card].arithmetic_class !=
			    (pro ? ARITHMETIC_CLASS_SM_120_NATIVE :
				   ARITHMETIC_CLASS_SM_80) ||
		    strcmp(descriptions[card].name,
			   pro ? "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" :
				 "NVIDIA CMP 170HX") != 0)
			return STATUS_UNSUPPORTED;
	}
	struct ds41f_original_head prepared = { 0 };
	uint64_t arena[DS41F_HEAD_CARDS];
	const uint64_t controls = (uint64_t)plan->limits.row_capacity *
				  DS41F_HEAD_CONTROL_ROW_BYTES;
	for (uint32_t card = 0; card < DS41F_HEAD_CARDS; card++) {
		struct ds41f_original_head_card *const h =
			&prepared.cards[card];
		const bool pro = card == DS41F_HEAD_RANKS;
		arena[card] = plan->cards[card].arena_byte_count;
		if (!pro) {
			if (!append_range(&arena[card],
					  (uint64_t)DS41F_HEAD_SHARD_ROWS *
						  DS41F_HIDDEN_ROW_BYTES,
					  &h->weights))
				return STATUS_INVALID_ARGUMENT;
			if (!append_range(&arena[card],
					  (uint64_t)(LINEAR_MAXIMUM_ROWS *
						     DS41F_HIDDEN_ROW_BYTES),
					  &h->hidden))
				return STATUS_INVALID_ARGUMENT;
			if (!append_range(&arena[card],
					  (uint64_t)LINEAR_MAXIMUM_ROWS *
						  DS41F_HEAD_SHARD_ROWS *
						  sizeof(float),
					  &h->logits))
				return STATUS_INVALID_ARGUMENT;
		}
		for (uint32_t rank = 0; rank < (pro ? DS41F_HEAD_RANKS : 1u);
		     rank++) {
			if (!append_range(&arena[card], HEAD_PEER_BYTES,
					  &h->peer_words[rank]))
				return STATUS_INVALID_ARGUMENT;
			if (!append_range(&arena[card], controls,
					  &h->control_input[rank]))
				return STATUS_INVALID_ARGUMENT;
		}
		if (!append_range(&arena[card], controls, &h->control_source))
			return STATUS_INVALID_ARGUMENT;
		if (!append_range(&arena[card],
				  STEP_BLOCK_BYTES_VALUE +
					  LINEAR_MAXIMUM_ROWS *
						  STEP_ROW_BYTES_VALUE,
				  &h->pilot_step))
			return STATUS_INVALID_ARGUMENT;
		if (!append_range(&arena[card], HEAD_PILOT_BYTES,
				  &h->pilot_source))
			return STATUS_INVALID_ARGUMENT;
		if (!append_range(&arena[card], HEAD_PILOT_BYTES,
				  &h->pilot_result))
			return STATUS_INVALID_ARGUMENT;
	}
	prepared.enabled = true;
	for (uint32_t card = 0; card < DS41F_HEAD_CARDS; card++)
		plan->cards[card].arena_byte_count = arena[card];
	plan->original_head = prepared;
	return OK;
}

result_status ds41f_original_head_banks_plan(struct ds41f_plan *plan,
					     uint32_t bank_count)
{
	if (plan == NULL || !plan->original_head.enabled ||
	    plan->original_head.connected || plan->original_head.initialised ||
	    plan->original_head.bank_count != 0 || bank_count == 0 ||
	    bank_count > DS41F_HEAD_BANKS)
		return STATUS_INVALID_ARGUMENT;
	struct ds41f_original_head prepared = plan->original_head;
	uint64_t arena[DS41F_HEAD_CARDS];
	for (uint32_t card = 0; card < DS41F_HEAD_CARDS; card++) {
		arena[card] = plan->cards[card].arena_byte_count;
		for (uint32_t bank = 1; bank < bank_count; bank++) {
			struct ds41f_original_head_card *const head =
				&prepared.cards[card];
			if (card != DS41F_HEAD_RANKS) {
				if (!append_range(
					    &arena[card],
					    (uint64_t)(LINEAR_MAXIMUM_ROWS *
						       DS41F_HIDDEN_ROW_BYTES),
					    &head->bank_hidden[bank - 1u]))
					return STATUS_INVALID_ARGUMENT;
			}
			const uint64_t logits_bytes =
				card == DS41F_HEAD_RANKS ?
					(uint64_t)LINEAR_MAXIMUM_ROWS *
						DS41F_LOGITS_ROW_BYTES :
					(uint64_t)LINEAR_MAXIMUM_ROWS *
						DS41F_HEAD_SHARD_ROWS *
						sizeof(float);
			if (!append_range(&arena[card], logits_bytes,
					  &head->bank_logits[bank - 1u]))
				return STATUS_INVALID_ARGUMENT;
		}
	}
	prepared.bank_count = bank_count;
	for (uint32_t card = 0; card < DS41F_HEAD_CARDS; card++)
		plan->cards[card].arena_byte_count = arena[card];
	plan->original_head = prepared;
	return OK;
}

result_status
ds41f_original_head_connect(struct ds41f_plan *plan,
			    const struct backend_operations *backend,
			    struct device_card *const *cards)
{
	if (plan == NULL || backend == NULL || cards == NULL ||
	    !plan->original_head.enabled || plan->original_head.connected ||
	    plan->original_head.initialised)
		return STATUS_INVALID_ARGUMENT;
	if (backend->peer_connect == NULL)
		return STATUS_UNSUPPORTED;
	for (uint32_t card = 0; card < DS41F_HEAD_CARDS; card++)
		if (cards[card] == NULL)
			return STATUS_INVALID_ARGUMENT;
	result_status status = OK;
	for (uint32_t rank = 0; rank < DS41F_HEAD_RANKS && status == OK;
	     rank++) {
		/* A failed call may have attached an address: do not wait for success to acquire ownership. */
		plan->original_head.connected = true;
		status = backend->peer_connect(
			cards[DS41F_HEAD_RANKS], cards[rank], 0,
			plan->original_head.cards[DS41F_HEAD_RANKS]
				.peer_words[rank]
				.byte_offset);
		status = THEN(backend->peer_connect(
			cards[rank], cards[DS41F_HEAD_RANKS], 0,
			plan->original_head.cards[rank]
				.peer_words[0]
				.byte_offset));
	}
	return status;
}

result_status
ds41f_original_head_drain(const struct backend_operations *backend,
			  struct device_card *const *cards,
			  uint64_t wait_limit_nanoseconds,
			  uint32_t *retained_out)
{
	if (backend == NULL || cards == NULL || retained_out == NULL ||
	    wait_limit_nanoseconds == 0)
		return STATUS_INVALID_ARGUMENT;
	uint32_t opened = 0;
	result_status status = OK;
	for (uint32_t card = 0; card < DS41F_HEAD_CARDS; card++) {
		if (cards[card] == NULL)
			continue;
		opened |= 1u << card;
		const result_status waited =
			backend->launch_wait == NULL ?
				STATUS_UNSUPPORTED :
				backend->launch_wait(cards[card],
						     wait_limit_nanoseconds);
		status = status == OK ? waited : status;
	}
	*retained_out = status == OK ? 0 : opened;
	return status;
}

/* One registered host block is reused only after the actual upload/download wait. */
MUST_CHECK static result_status
transfer(const struct backend_operations *backend, struct device_card *card,
	 uint64_t offset, void *staging, uint64_t bytes, bool download,
	 uint64_t wait)
{
	uint64_t ticket = 0;
	result_status status =
		download ? backend->download_begin(card, offset, staging, bytes,
						   &ticket) :
			   backend->upload_begin(card, offset, staging, bytes,
						 &ticket);
	return THEN(backend->upload_wait(card, ticket, wait));
}

MUST_CHECK static result_status pilot(const struct ds41f_plan *plan,
				      const struct backend_operations *backend,
				      struct device_card *const *cards,
				      void *staging, uint64_t wait)
{
	const struct {
		struct step_block block;
		struct step_row_record row;
	} step = { { 1, 1, 0 }, { 0, 0, 0, STEP_ROW_LOGITS } };
	uint32_t kernels[DS41F_HEAD_CARDS];
	result_status status = OK;
	for (uint32_t card = 0; card < DS41F_HEAD_CARDS && status == OK;
	     card++) {
		memcpy(staging, &step, sizeof(step));
		status = transfer(
			backend, cards[card],
			plan->original_head.cards[card].pilot_step.byte_offset,
			staging, sizeof(step), false, wait);
		status = THEN(backend->kernel_prepare(
			cards[card], KERNEL_FILE_HEAD_OFFLOAD,
			KERNEL_HEAD_OFFLOAD_COPY, HEAD_OFFLOAD_ARGUMENT_BYTES,
			&kernels[card]));
	}
	for (uint32_t rank = 0; rank < DS41F_HEAD_RANKS && status == OK;
	     rank++) {
		for (uint32_t direction = 0;
		     direction < HEAD_PILOT_DIRECTIONS && status == OK;
		     direction++) {
			const uint32_t from =
				direction == 0 ? DS41F_HEAD_RANKS : rank;
			const uint32_t to = direction == 0 ? rank :
							     DS41F_HEAD_RANKS;
			const struct ds41f_original_head_card *const src =
				&plan->original_head.cards[from];
			const struct ds41f_original_head_card *const dst =
				&plan->original_head.cards[to];
			const uint32_t pattern[HEAD_PILOT_WORDS] = {
				HEAD_PILOT_FIRST_MARK, rank, direction,
				HEAD_PILOT_LAST_MARK
			};
			memset(staging, HEAD_PILOT_SENTINEL, HEAD_PILOT_BYTES);
			status = transfer(backend, cards[to],
					  dst->pilot_result.byte_offset,
					  staging, HEAD_PILOT_BYTES, false,
					  wait);
			if (status == OK)
				memcpy(staging, pattern, sizeof(pattern));
			status = THEN(transfer(backend, cards[from],
					       src->pilot_source.byte_offset,
					       staging, sizeof(pattern), false,
					       wait));
			const struct head_offload_description description = {
				src->pilot_source,
				src->peer_words[direction == 0 ? rank : 0],
				src->pilot_step,
				dst->pilot_result,
				plan->cards[to].arena_byte_count,
				sizeof(pattern),
				sizeof(pattern),
				sizeof(pattern),
				1,
				0,
				0
			};
			struct kernel_launch launch;
			status = THEN(head_offload_prepare(
				&description,
				plan->cards[from].arena_byte_count,
				kernels[from], &launch));
			status = THEN(
				backend->launch_run(cards[from], &launch, 1));
			status = THEN(backend->mark_record(cards[from],
							   HEAD_PILOT_MARK));
			status = THEN(backend->mark_follow(
				cards[to], cards[from], HEAD_PILOT_MARK));
			status = THEN(transfer(backend, cards[to],
					       dst->pilot_result.byte_offset,
					       staging, HEAD_PILOT_BYTES, true,
					       wait));
			if (status == OK &&
			    memcmp(staging, pattern, sizeof(pattern)) != 0)
				status = STATUS_DEVICE_ERROR;
			for (uint32_t byte = sizeof(pattern);
			     byte < HEAD_PILOT_BYTES && status == OK; byte++)
				if (((unsigned char *)staging)[byte] !=
				    HEAD_PILOT_SENTINEL)
					status = STATUS_DEVICE_ERROR;
		}
	}
	return status;
}

result_status
ds41f_original_head_initialise(struct ds41f_plan *plan,
			       const struct pack_record *pack,
			       const struct storage_operations_record *storage,
			       const struct storage_reader *reader,
			       const struct backend_operations *backend,
			       struct device_card *const *cards, uint64_t wait)
{
	if (plan == NULL || pack == NULL || storage == NULL || reader == NULL ||
	    backend == NULL || cards == NULL || wait == 0 ||
	    !plan->original_head.enabled || plan->original_head.initialised ||
	    plan->original_head.connected)
		return STATUS_INVALID_ARGUMENT;
	if (backend->peer_connect == NULL || backend->mark_record == NULL ||
	    backend->mark_follow == NULL || backend->launch_wait == NULL ||
	    backend->staging_allocate == NULL ||
	    backend->staging_free == NULL || backend->upload_begin == NULL ||
	    backend->download_begin == NULL || backend->upload_wait == NULL ||
	    backend->kernel_prepare == NULL || backend->launch_run == NULL ||
	    backend->digest == NULL)
		return STATUS_UNSUPPORTED;
	const uint32_t tensor = pack_find_tensor(pack, "head.weight",
						 sizeof("head.weight") - 1u);
	if (tensor == PACK_NOT_FOUND)
		return STATUS_CORRUPT_PACK;
	struct tp_load_job jobs[DS41F_HEAD_RANKS] = { 0 };
	struct tp_load_description description = {
		.pack = pack,
		.jobs = jobs,
		.job_count = DS41F_HEAD_RANKS,
		.card_count = DS41F_HEAD_CARDS,
		.maximum_extent_byte_count = TP_LOAD_MAXIMUM_EXTENT_BYTES
	};
	result_status status = OK;
	for (uint32_t card = 0; card < DS41F_HEAD_CARDS; card++)
		description.card_arena_byte_count[card] =
			plan->cards[card].arena_byte_count;
	for (uint32_t rank = 0; rank < DS41F_HEAD_RANKS && status == OK;
	     rank++) {
		jobs[rank] = (struct tp_load_job){
			.tensor_index = tensor,
			.card = rank,
			.destination_offset = plan->original_head.cards[rank]
						      .weights.byte_offset,
			.mode = TP_LOAD_SHARD
		};
		const struct tensor_shard_description cut = {
			WEIGHT_FORMAT_NONE,
			{ DS41F_VOCABULARY_VALUE, DS41F_HIDDEN,
			  TENSOR_SHARD_PLAIN_BITS, 0 },
			rank * DS41F_HEAD_SHARD_ROWS,
			DS41F_HEAD_SHARD_ROWS,
			0,
			DS41F_HIDDEN
		};
		status = tensor_shard_prepare(&cut, &jobs[rank].shard);
	}
	struct tp_load_plan load;
	status = THEN(tp_load_prepare(&description, &load));
	void *source = NULL, *output = NULL;
	status = THEN(backend->staging_allocate(
		load.source_workspace_byte_count, &source));
	status = THEN(backend->staging_allocate(
		load.output_workspace_byte_count, &output));
	if (status == OK) {
		const struct tp_load_request request = {
			.plan = &load,
			.storage = storage,
			.reader = reader,
			.backend = backend,
			.cards = cards,
			.source_workspace = source,
			.output_workspace = output,
			.source_workspace_byte_count =
				load.source_workspace_byte_count,
			.output_workspace_byte_count =
				load.output_workspace_byte_count,
			.wait_limit_nanoseconds = wait
		};
		struct tp_load_result loaded;
		status = tp_load_run(&request, &loaded);
	}
	status = THEN(ds41f_original_head_connect(plan, backend, cards));
	status = THEN(pilot(plan, backend, cards, source, wait));
	uint32_t retained = 0;
	const result_status drained =
		ds41f_original_head_drain(backend, cards, wait, &retained);
	status = status == OK ? drained : status;
	if (retained == 0) {
		if (source != NULL)
			backend->staging_free(source);
		if (output != NULL)
			backend->staging_free(output);
	} else {
		LOG_ERROR(
			"original head: staging retained after unproven five-card drain");
	}
	plan->original_head.initialised = status == OK;
	return status;
}
