/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "model_v41_tp2_peer.h"
#include "ordered_push_internal.h"
#include "anemone/tp_exchange.h"

#include <float.h>
#include <stddef.h>
#include <string.h>

#define PEER_MAXIMUM_RANKS TP4_RANKS
#define PEER_SIZES 2u
#define PEER_FIRST_SEQUENCE 2u
#define PEER_LAST_SEQUENCE 5u
#define PEER_DESCRIPTION_OFFSET 128u
#define PEER_METADATA_BYTES 256u
#define PEER_INTEGER_TIE 16777216.0f
#define PEER_PATTERN_MODULUS 257u
#define PEER_PATTERN_CENTER 128
#define PEER_PATTERN_DIVISOR 256.0f
#define PEER_LEFT_MULTIPLIER 73u
#define PEER_RIGHT_MULTIPLIER 101u
#define PEER_SEQUENCE_MULTIPLIER 17u
#define PEER_PAIR_MULTIPLIER 19u
#define PEER_SPECIAL_PERIOD 16u
#define PEER_TIE_COLUMN 1u
#define PEER_CANCEL_COLUMN 2u
#define PEER_ZERO_COLUMN 3u
#define PEER_VALUE_BUFFERS 2u
#define PEER_BFLOAT16_MAXIMUM 0x1.fep127f
#define PEER_BFLOAT16_SHIFT 16u
#define PEER_BFLOAT16_ROUND_BIAS 0x7FFFu
#define PEER_BFLOAT16_MASK 0xFFFF0000u
#define PEER_ROUNDING_TAIL 0x1p-12f
#define PEER_INACTIVE_CANARY 0x7FC12345u
#define PEER_PHASE_SAMPLES 3u

struct peer_scratch {
	unsigned char *zero;
	float *input[PEER_MAXIMUM_RANKS];
	uint32_t *output[PEER_MAXIMUM_RANKS];
	unsigned char *metadata[PEER_MAXIMUM_RANKS];
};

static float peer_value(uint32_t pair, uint32_t rank, uint32_t column,
			uint64_t sequence, uint32_t format)
{
	switch (column % PEER_SPECIAL_PERIOD) {
	case PEER_TIE_COLUMN:
		return rank == 0 ? PEER_INTEGER_TIE : 1.0f;
	case PEER_CANCEL_COLUMN:
		if (rank >= TP2_RANKS)
			return 0.0f;
		return (rank == 0 ? 1.0f : -1.0f) *
		       (format == ORDERED_PUSH_BF16 ? PEER_BFLOAT16_MAXIMUM :
						      FLT_MAX);
	case PEER_ZERO_COLUMN:
		return -0.0f;
	default: {
		const uint64_t mixed =
			(uint64_t)column * (rank == 0 ? PEER_LEFT_MULTIPLIER :
							PEER_RIGHT_MULTIPLIER) +
			sequence * PEER_SEQUENCE_MULTIPLIER +
			(uint64_t)pair * PEER_PAIR_MULTIPLIER + rank;
		const int32_t centered =
			(int32_t)(mixed % PEER_PATTERN_MODULUS) -
			PEER_PATTERN_CENTER;
		return (float)centered / PEER_PATTERN_DIVISOR +
		       (format == ORDERED_PUSH_BF16 ? PEER_ROUNDING_TAIL :
						      0.0f);
	}
	}
}

static float peer_wire_value(float value, uint32_t format)
{
	if (format == ORDERED_PUSH_BF16) {
		uint32_t bits;
		(void)memcpy(&bits, &value, sizeof(bits));
		bits = (bits + PEER_BFLOAT16_ROUND_BIAS +
			((bits >> PEER_BFLOAT16_SHIFT) & 1u)) &
		       PEER_BFLOAT16_MASK;
		(void)memcpy(&value, &bits, sizeof(value));
	}
	return value;
}

static uint32_t peer_expected(uint32_t pair, uint32_t column, uint64_t sequence,
			      uint32_t format, uint32_t ranks)
{
	volatile float sum = 0.0f;
	for (uint32_t rank = 0; rank < ranks; rank++)
		sum = sum + peer_wire_value(peer_value(pair, rank, column,
						       sequence, format),
					    format);
	const float value = sum;
	uint32_t bits;
	(void)memcpy(&bits, &value, sizeof(bits));
	return bits;
}

MUST_CHECK static result_status
peer_wait_pair(const struct backend_operations *backend,
	       struct device_card *const *cards, uint32_t first, uint32_t ranks,
	       uint64_t wait, const uint64_t tickets[PEER_MAXIMUM_RANKS],
	       result_status previous)
{
	for (uint32_t rank = 0; rank < ranks; rank++) {
		const result_status status =
			backend->launch_wait(cards[first + rank], wait);
		if (previous == STATUS_OK)
			previous = status;
	}
	/* launch_wait does not retire the bounded copy-ticket ring. */
	for (uint32_t rank = 0; rank < ranks; rank++) {
		if (tickets[rank] == 0)
			continue;
		const result_status status = backend->upload_wait(
			cards[first + rank], tickets[rank], wait);
		if (previous == STATUS_OK)
			previous = status;
	}
	return previous;
}

static bool peer_arguments_valid(const struct v41_tp2_model *model,
				 const struct backend_operations *backend,
				 struct device_card *const *cards,
				 uint64_t wait, uint32_t *output,
				 uint64_t *region_bytes)
{
	if (model == NULL || model->canonical == NULL || backend == NULL ||
	    cards == NULL || output == NULL || wait == 0 ||
	    model->load_attempted || model->ready ||
	    model->canonical->limits.row_capacity == 0 ||
	    model->canonical->limits.row_capacity > V41_DATA_TP_SPLIT_MOST_ROWS)
		return false;
	const bool quad_all = model->description.tensor_layout ==
			      V41_DATA_TP_LAYOUT_QUAD_ALL;
	const uint32_t card_count = quad_all ? TP4_RANKS :
					       V41_DATA_TP2_PROGRAM_CARDS;
	if (model->canonical->card_count != card_count)
		return false;
	*region_bytes = 0;
	const uint64_t values =
		(uint64_t)model->canonical->limits.row_capacity * MODEL_HIDDEN *
		sizeof(float);
	const uint64_t signals = (uint64_t)PEER_MAXIMUM_RANKS *
				 ORDERED_PUSH_PHASES *
				 ORDERED_COLLECTIVE_LINE_BYTES;
	const uint64_t largest_region =
		ORDERED_COLLECTIVE_SLOTS *
		(signals + (PEER_MAXIMUM_RANKS + PEER_VALUE_BUFFERS) * values +
		 V41_DATA_ARRAY_ALIGNMENT_BYTES);
	for (uint32_t card = 0; card < card_count; card++) {
		if (cards[card] == NULL)
			return false;
		if (card == V41_DATA_TP2_PRO_CARD)
			continue;
		const struct v41_tp2_stage_plan *const stage =
			model->stages[card];
		if (stage == NULL)
			return false;
		const uint32_t ranks = stage->description.rank_count;
		if (quad_all && ranks != TP4_RANKS)
			return false;
		if (ranks == 1u)
			continue;
		if (stage == NULL || stage->channels == NULL ||
		    stage->channel_count == 0 ||
		    stage->channel_rows <
			    model->canonical->limits.row_capacity ||
		    stage->description.card != card ||
		    (ranks != TP2_RANKS && ranks != TP4_RANKS) ||
		    stage->description.rank != card % ranks ||
		    card - stage->description.rank + ranks >
			    V41_DATA_TP2_PRO_CARD ||
		    model->description.cards[card].arithmetic_class !=
			    ARITHMETIC_CLASS_SM_80)
			return false;
		const struct ordered_push_description *const channel =
			&stage->channels[0].collective;
		const struct ordered_push_arguments *const args =
			&channel->arguments;
		struct kernel_launch unused;
		const result_status prepared =
			stage->channel_rows > LINEAR_MAXIMUM_ROWS ?
				tp_exchange_prepare(
					channel,
					TP_EXCHANGE_LAYER_PHASES *
						stage->description
							.first_target_layer,
					TP_EXCHANGE_PHASES, 0, &unused) :
				ordered_push_prepare(channel, 0, &unused);
		if (args->rank_count != ranks ||
		    args->rank != stage->description.rank ||
		    args->payload_format != model->description.wire_format ||
		    args->payload_format > ORDERED_PUSH_BF16 ||
		    args->element_count !=
			    model->canonical->limits.row_capacity *
				    MODEL_HIDDEN ||
		    args->incoming_host_mask != 0 ||
		    args->step_offset !=
			    model->canonical->cards[card]
				    .array_offset[V41_DATA_ARRAY_STEP] ||
		    channel->arena_byte_count !=
			    model->canonical->cards[card].arena_byte_count ||
		    prepared != STATUS_OK)
			return false;
		const uint64_t bytes =
			args->parity_stride * ORDERED_COLLECTIVE_SLOTS;
		if (bytes > largest_region)
			return false;
		if (stage->description.rank != 0) {
			if (stage->channels[0].layer !=
				    model->stages[card - 1u]->channels[0].layer ||
			    stage->channels[0].site !=
				    model->stages[card - 1u]->channels[0].site)
				return false;
			const struct ordered_push_arguments *const other =
				&model->stages[card - 1u]
					 ->channels[0]
					 .collective.arguments;
			if (args->input_offset != other->input_offset ||
			    args->rank_count != other->rank_count ||
			    args->payload_format != other->payload_format ||
			    args->output_offset != other->output_offset ||
			    args->inbox_offset != other->inbox_offset ||
			    args->parity_stride != other->parity_stride ||
			    args->inbox_rank_stride != other->inbox_rank_stride)
				return false;
		}
		*region_bytes = bytes > *region_bytes ? bytes : *region_bytes;
	}
	return true;
}

MUST_CHECK static result_status
peer_initialize(const struct v41_tp2_model *model,
		const struct backend_operations *backend,
		struct device_card *const *cards, uint32_t first,
		struct peer_scratch *scratch, uint64_t wait)
{
	result_status status = STATUS_OK;
	uint64_t tickets[PEER_MAXIMUM_RANKS] = { 0 };
	const uint32_t ranks = model->stages[first]->description.rank_count;
	for (uint32_t rank = 0; rank < ranks && status == STATUS_OK; rank++) {
		const uint32_t card = first + rank;
		const struct ordered_push_arguments *const args =
			&model->stages[card]->channels[0].collective.arguments;
		struct ordered_collective_peer *const peers =
			(void *)scratch->metadata[rank];
		const uint64_t table_bytes = ranks * sizeof(*peers);
		(void)memset(scratch->metadata[rank], 0, PEER_METADATA_BYTES);
		for (uint32_t remote = 0; remote < ranks; remote++) {
			const struct v41_tp2_stage_channel *const other =
				&model->stages[first + remote]->channels[0];
			peers[remote].region_offset =
				other->collective.arguments.region_offset;
			peers[remote].route = ORDERED_COLLECTIVE_DIRECT_ROUTE;
		}
		status = backend->upload_begin(
			cards[card], args->region_offset, scratch->zero,
			args->parity_stride * ORDERED_COLLECTIVE_SLOTS,
			&tickets[rank]);
		if (status == STATUS_OK)
			status = backend->upload_begin(cards[card],
						       args->peer_table_offset,
						       scratch->metadata[rank],
						       table_bytes,
						       &tickets[rank]);
		if (status == STATUS_OK)
			status = backend->upload_begin(
				cards[card], args->step_offset,
				scratch->metadata[rank] + table_bytes,
				STEP_BLOCK_BYTES_VALUE, &tickets[rank]);
		for (uint32_t remote = 0; remote < ranks && status == STATUS_OK;
		     remote++)
			status = backend->peer_connect(
				cards[card], cards[first + remote], 0,
				args->peer_table_offset +
					remote * sizeof(peers[0]) +
					offsetof(struct ordered_collective_peer,
						 arena_address));
	}
	return peer_wait_pair(backend, cards, first, ranks, wait, tickets,
			      status);
}

MUST_CHECK static result_status
peer_round(const struct v41_tp2_model *model,
	   const struct backend_operations *backend,
	   struct device_card *const *cards, uint32_t first, uint32_t count,
	   uint64_t sequence, uint32_t layer,
	   const uint32_t kernels[PEER_MAXIMUM_RANKS],
	   struct peer_scratch *scratch, uint64_t wait)
{
	struct kernel_launch launches[PEER_MAXIMUM_RANKS];
	uint64_t tickets[PEER_MAXIMUM_RANKS] = { 0 };
	const bool shared = model->canonical->limits.row_capacity >
			    LINEAR_MAXIMUM_ROWS;
	const uint32_t capacity =
		shared ? model->canonical->limits.row_capacity * MODEL_HIDDEN :
			 count;
	const uint64_t bytes = (uint64_t)count * sizeof(float);
	const uint64_t output_bytes = (uint64_t)capacity * sizeof(float);
	const uint64_t pattern = sequence * MODEL_LAYERS + layer;
	result_status status = STATUS_OK;
	const uint32_t ranks = model->stages[first]->description.rank_count;
	if (ranks != TP2_RANKS && ranks != TP4_RANKS)
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t rank = 0; rank < ranks && status == STATUS_OK; rank++) {
		/* Keep this cold mutable description in the owned aligned block. GCC's znver4 debug build can emit
		 * an aligned64 store for an ordinary stack aggregate copy whose ASAN fake frame is only aligned32. */
		struct ordered_push_description *const description =
			(void *)(scratch->metadata[rank] +
				 PEER_DESCRIPTION_OFFSET);
		(void)memcpy(
			description,
			&model->stages[first + rank]->channels[0].collective,
			sizeof(*description));
		if (model->canonical->limits.row_capacity >
		    LINEAR_MAXIMUM_ROWS) {
			status = tp_exchange_prepare(
				description, TP_EXCHANGE_LAYER_PHASES * layer,
				TP_EXCHANGE_PHASES, kernels[rank],
				&launches[rank]);
		} else {
			description->arguments.element_count = count;
			status = ordered_push_prepare(
				description, kernels[rank], &launches[rank]);
		}
		for (uint32_t column = 0; column < count; column++)
			scratch->input[rank][column] = peer_value(
				first / ranks, rank, column, pattern,
				description->arguments.payload_format);
		for (uint32_t column = 0; column < capacity; column++)
			scratch->output[rank][column] = PEER_INACTIVE_CANARY;
		const struct step_block step = { sequence, count / MODEL_HIDDEN,
						 0 };
		(void)memcpy(scratch->metadata[rank], &step, sizeof(step));
		const uint64_t input =
			description->arguments.region_offset +
			description->arguments.input_offset +
			(sequence & 1u) * description->arguments.parity_stride;
		if (status == STATUS_OK && shared)
			status = backend->upload_begin(
				cards[first + rank],
				description->arguments.region_offset +
					description->arguments.output_offset +
					(sequence & 1u) *
						description->arguments
							.parity_stride,
				scratch->output[rank], output_bytes,
				&tickets[rank]);
		if (status == STATUS_OK)
			status = backend->upload_begin(cards[first + rank],
						       input,
						       scratch->input[rank],
						       bytes, &tickets[rank]);
		if (status == STATUS_OK)
			status = backend->upload_begin(
				cards[first + rank],
				description->arguments.step_offset,
				scratch->metadata[rank], sizeof(step),
				&tickets[rank]);
	}
	/* Preparation failed before any peer-waiting kernel; drain queued uploads before returning. */
	if (status != STATUS_OK)
		return peer_wait_pair(backend, cards, first, ranks, wait,
				      tickets, status);
	for (uint32_t rank = 0; rank < ranks; rank++) {
		const result_status member = backend->launch_run(
			cards[first + rank], &launches[rank], 1);
		status = status == STATUS_OK ? member : status;
	}
	status = peer_wait_pair(backend, cards, first, ranks, wait, tickets,
				status);
	if (status != STATUS_OK)
		return status;
	for (uint32_t rank = 0; rank < ranks && status == STATUS_OK; rank++) {
		const struct ordered_push_arguments *const args =
			&model->stages[first + rank]
				 ->channels[0]
				 .collective.arguments;
		const uint64_t output = args->region_offset +
					args->output_offset +
					(sequence & 1u) * args->parity_stride;
		status = backend->download_begin(cards[first + rank], output,
						 scratch->output[rank],
						 output_bytes, &tickets[rank]);
	}
	status = peer_wait_pair(backend, cards, first, ranks, wait, tickets,
				status);
	for (uint32_t column = 0; column < count && status == STATUS_OK;
	     column++) {
		const uint32_t expected =
			peer_expected(first / ranks, column, pattern,
				      model->description.wire_format, ranks);
		for (uint32_t rank = 0; rank < ranks; rank++) {
			if (scratch->output[rank][column] != expected)
				status = STATUS_DEVICE_ERROR;
		}
	}
	for (uint32_t column = count; column < capacity && status == STATUS_OK;
	     column++) {
		for (uint32_t rank = 0; rank < ranks; rank++) {
			if (scratch->output[rank][column] !=
			    PEER_INACTIVE_CANARY)
				status = STATUS_DEVICE_ERROR;
		}
	}
	return status;
}

result_status
v41_tp2_peer_qualify(const struct v41_tp2_model *model,
		     const struct backend_operations *backend,
		     struct device_card *const *cards,
		     uint64_t wait_limit_nanoseconds,
		     uint32_t writer_masks_out[V41_DATA_TP2_PROGRAM_CARDS])
{
	uint64_t region_bytes = 0;
	if (!peer_arguments_valid(model, backend, cards, wait_limit_nanoseconds,
				  writer_masks_out, &region_bytes))
		return STATUS_INVALID_ARGUMENT;
	if (backend->staging_allocate == NULL ||
	    backend->staging_retire == NULL || backend->upload_begin == NULL ||
	    backend->download_begin == NULL || backend->launch_run == NULL ||
	    backend->launch_wait == NULL || backend->peer_connect == NULL ||
	    backend->kernel_prepare == NULL || backend->upload_wait == NULL)
		return STATUS_UNSUPPORTED;
	const uint32_t counts[PEER_SIZES] = {
		MODEL_HIDDEN,
		model->canonical->limits.row_capacity * MODEL_HIDDEN
	};
	const uint64_t value_bytes = (uint64_t)counts[1] * sizeof(float);
	const uint64_t scratch_bytes =
		region_bytes +
		PEER_MAXIMUM_RANKS * (PEER_VALUE_BUFFERS * value_bytes +
				      PEER_METADATA_BYTES);
	void *allocation = NULL;
	result_status status =
		backend->staging_allocate(scratch_bytes, &allocation);
	if (status != STATUS_OK)
		return status;
	struct peer_scratch scratch = { .zero = allocation };
	(void)memset(scratch.zero, 0, (size_t)region_bytes);
	unsigned char *next = scratch.zero + region_bytes;
	for (uint32_t rank = 0; rank < PEER_MAXIMUM_RANKS; rank++) {
		void *input_memory = next;
		scratch.input[rank] = input_memory;
		next += value_bytes;
		void *output_memory = next;
		scratch.output[rank] = output_memory;
		next += value_bytes;
		scratch.metadata[rank] = next;
		next += PEER_METADATA_BYTES;
	}
	/* Fixed-size public output keeps an inactive QUAD_ALL slot4 explicitly zero. */
	uint32_t masks[V41_DATA_TP2_PROGRAM_CARDS] = { 0 };
	for (uint32_t card = 0; card < model->canonical->card_count; card++)
		masks[card] = 1u << card;
	for (uint32_t first = 0;
	     first < V41_DATA_TP2_PRO_CARD && status == STATUS_OK;) {
		const uint32_t ranks =
			model->stages[first]->description.rank_count;
		if (ranks == 1u) {
			first++;
			continue;
		}
		uint32_t kernels[PEER_MAXIMUM_RANKS];
		for (uint32_t rank = 0; rank < ranks && status == STATUS_OK;
		     rank++)
			status = model->canonical->limits.row_capacity >
						 LINEAR_MAXIMUM_ROWS ?
					 tp_exchange_kernel_prepare(
						 backend, cards[first + rank],
						 &kernels[rank]) :
					 ordered_push_kernel_prepare(
						 backend, cards[first + rank],
						 &kernels[rank]);
		for (uint32_t size = 0;
		     size < PEER_SIZES && status == STATUS_OK; size++) {
			status = peer_initialize(model, backend, cards, first,
						 &scratch,
						 wait_limit_nanoseconds);
			for (uint64_t sequence = PEER_FIRST_SEQUENCE;
			     sequence <= PEER_LAST_SEQUENCE &&
			     status == STATUS_OK;
			     sequence++) {
				const struct v41_tp2_stage_description *const
					geometry = &model->stages[first]
							    ->description;
				const uint32_t samples =
					model->canonical->limits.row_capacity >
							LINEAR_MAXIMUM_ROWS ?
						PEER_PHASE_SAMPLES :
						1u;
				for (uint32_t sample = 0;
				     sample < samples && status == STATUS_OK;
				     sample++) {
					const uint32_t layer =
						geometry->first_target_layer +
						sample *
							(geometry->target_layer_count -
							 1u) /
							(PEER_PHASE_SAMPLES -
							 1u);
					const uint32_t count =
						samples > 1u && sample == 1u ?
							MODEL_HIDDEN :
							counts[size];
					status = peer_round(
						model, backend, cards, first,
						count, sequence, layer, kernels,
						&scratch,
						wait_limit_nanoseconds);
				}
			}
		}
		if (status == STATUS_OK) {
			uint32_t tested[PEER_MAXIMUM_RANKS],
				host[PEER_MAXIMUM_RANKS];
			for (uint32_t rank = 0; rank < PEER_MAXIMUM_RANKS;
			     rank++) {
				tested[rank] = (1u << ranks) - 1u;
				host[rank] = UINT32_MAX;
			}
			const struct ordered_collective_route_description
				route = { .cards = &model->description
							    .cards[first],
					  .direct_reader_masks = tested,
					  .rank_count = ranks,
					  .element_count = counts[1],
					  .operation = ORDERED_ALL_REDUCE };
			status = ordered_push_routes_prepare_capacity(
				&route, TP_EXCHANGE_MAXIMUM_ELEMENTS, host);
			for (uint32_t rank = 0;
			     rank < ranks && status == STATUS_OK; rank++) {
				if (host[rank] != 0)
					status = STATUS_UNSUPPORTED;
			}
			if (status == STATUS_OK) {
				for (uint32_t rank = 0; rank < ranks; rank++)
					masks[first + rank] =
						((1u << ranks) - 1u) << first;
			}
		}
		first += ranks;
	}
	backend->staging_retire(allocation);
	if (status == STATUS_OK)
		(void)memcpy(writer_masks_out, masks, sizeof(masks));
	return status;
}

result_status
v41_tp_group_drain(const struct backend_operations *backend,
		   struct device_card *const cards[V41_DATA_TP2_PROGRAM_CARDS],
		   uint64_t wait_limit_nanoseconds, uint32_t members,
		   uint32_t *retained_out)
{
	if (backend == NULL || cards == NULL || retained_out == NULL ||
	    wait_limit_nanoseconds == 0 ||
	    (members != TP2_RANKS && members != TP4_RANKS))
		return STATUS_INVALID_ARGUMENT;
	uint32_t retained = 0;
	result_status first_failure = STATUS_OK;
	for (uint32_t card = 0; card < V41_DATA_TP2_PROGRAM_CARDS; card++) {
		if (cards[card] == NULL)
			continue;
		const result_status status =
			backend->launch_wait == NULL ?
				STATUS_UNSUPPORTED :
				backend->launch_wait(cards[card],
						     wait_limit_nanoseconds);
		if (status == STATUS_OK)
			continue;
		if (first_failure == STATUS_OK)
			first_failure = status;
		retained |= card == V41_DATA_TP2_PRO_CARD ?
				    1u << card :
				    ((1u << members) - 1u)
					    << (card / members * members);
	}
	*retained_out = retained;
	return first_failure;
}

result_status
v41_tp2_peer_drain(const struct backend_operations *backend,
		   struct device_card *const cards[V41_DATA_TP2_PROGRAM_CARDS],
		   uint64_t wait_limit_nanoseconds, uint32_t *retained_out)
{
	return v41_tp_group_drain(backend, cards, wait_limit_nanoseconds,
				  TP2_RANKS, retained_out);
}
