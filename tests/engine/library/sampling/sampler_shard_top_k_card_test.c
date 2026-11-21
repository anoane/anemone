/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Prepared GPU gate: global truncation, empty local winners and session push versus the ordinary sampler. */
#include "engine/library/sampling/sampler.h"
#include "engine/library/sampling/sampler_shard_records.h"
#include "engine/library/block_scaled_linear/linear_card_fixture.h"
#include "support/byte_identity.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define VOCABULARY 32u
#define WORK_BYTES 262144u
#define DEADLINE_SECONDS 120u
#define TOP_FIVE 5u
#define NUCLEUS 0.6f
#define COOL 0.5f
#define SKIPPED_TOP_K 5u
#define SKIPPED_CALL 32u
#define SEED 0x123456789ABCDEF0ull
#define SETTINGS_SLOTS 2u
#define ACTIVE_SLOT 1u
#define LOGIT_BASE 9.0f
#define LOGIT_GROUP 3u
#define SHARD_SIZE (VOCABULARY / TP_GROUP_MOST_RANKS)
#define SETTINGS_COPY_BYTES \
	(SAMPLER_SETTINGS_SLOTS_OFFSET + SETTINGS_SLOTS * STEP_SESSION_BYTES)

struct shard_case {
	struct test_linear_card card;
	struct linear_range step, sessions, copy, exchange, entry, peers,
		settings_peers;
	struct linear_range logits, result, full_logits, weights,
		ordinary_result;
	struct kernel_launch shard, push, ordinary;
};

static bool prepare(struct shard_case *self, uint32_t rank)
{
	self->step = test_linear_card_work(
		&self->card, STEP_BLOCK_BYTES_VALUE + STEP_ROW_BYTES_VALUE);
	self->sessions = test_linear_card_work(
		&self->card, (uint64_t)(SETTINGS_SLOTS * STEP_SESSION_BYTES));
	self->copy = test_linear_card_work(&self->card, SETTINGS_COPY_BYTES);
	self->exchange = test_linear_card_work(
		&self->card, (uint64_t)SAMPLER_SHARD_DEFAULT_EXCHANGE_BYTES);
	self->peers = test_linear_card_work(&self->card,
					    SAMPLER_SHARD_PEER_TABLE_BYTES);
	self->settings_peers = test_linear_card_work(
		&self->card, TP_GROUP_MOST_RANKS * sizeof(uint64_t));
	self->logits =
		test_linear_card_work(&self->card, SHARD_SIZE * sizeof(float));
	self->result = test_linear_card_work(
		&self->card, STEP_RESULT_BYTES + STEP_ROW_RESULT_BYTES);
	self->full_logits =
		test_linear_card_work(&self->card, VOCABULARY * sizeof(float));
	self->weights =
		test_linear_card_work(&self->card, VOCABULARY * sizeof(float));
	self->ordinary_result = test_linear_card_work(
		&self->card, STEP_RESULT_BYTES + STEP_ROW_RESULT_BYTES);
	self->entry = test_linear_card_work(
		&self->card, (uint64_t)SAMPLER_SHARD_ENTRY_BYTES);
	if (!test_linear_card_has_room(&self->card))
		return false;
	self->shard =
		(struct kernel_launch){ .grid = { 1, 1, 1 },
					.block = { SAMPLER_THREADS, 1, 1 },
					.argument_byte_count =
						SAMPLER_SHARD_ARGUMENT_BYTES };
	self->push = (struct kernel_launch){
		.grid = { 1, 1, 1 },
		.block = { SAMPLER_SETTINGS_PUSH_THREADS, 1, 1 },
		.argument_byte_count = SAMPLER_SETTINGS_PUSH_ARGUMENT_BYTES
	};
	const struct sampler_shard_arguments shard = {
		.step_offset = self->step.byte_offset,
		.session_offset = self->copy.byte_offset,
		.logits_offset = self->logits.byte_offset,
		.result_offset = self->result.byte_offset,
		.exchange_offset = self->exchange.byte_offset,
		.peer_table_offset = self->peers.byte_offset,
		.deadline_cycles = SAMPLER_SHARD_DEADLINE_CYCLES,
		.first_token = rank * SHARD_SIZE,
		.shard_size = SHARD_SIZE,
		.vocabulary_size = VOCABULARY,
		.session_capacity = SETTINGS_SLOTS,
		.rank = rank
	};
	const struct sampler_settings_push_arguments push = {
		.step_offset = self->step.byte_offset,
		.sessions_offset = self->sessions.byte_offset,
		.peer_table_offset = self->settings_peers.byte_offset,
		.session_capacity = SETTINGS_SLOTS
	};
	memcpy(self->shard.argument, &shard, sizeof(shard));
	memcpy(self->push.argument, &push, sizeof(push));
	struct sampler_kernels_record kernels = { 0 };
	const struct sampler_description ordinary = {
		.step = self->step,
		.sessions = self->sessions,
		.logits = self->full_logits,
		.weights = self->weights,
		.result = self->ordinary_result,
		.vocabulary_size = VOCABULARY,
		.session_capacity = SETTINGS_SLOTS,
		.row_count = 1
	};
	return test_linear_expect_status(backend_cuda_value.kernel_prepare(
						 self->card.card,
						 KERNEL_FILE_SAMPLER_SHARD,
						 KERNEL_SAMPLER_SHARD,
						 SAMPLER_SHARD_ARGUMENT_BYTES,
						 &self->shard.kernel),
					 STATUS_OK, "prepare shard kernel") &&
	       test_linear_expect_status(
		       backend_cuda_value.kernel_prepare(
			       self->card.card, KERNEL_FILE_SAMPLER_SHARD,
			       KERNEL_SAMPLER_SETTINGS_PUSH,
			       SAMPLER_SETTINGS_PUSH_ARGUMENT_BYTES,
			       &self->push.kernel),
		       STATUS_OK, "prepare settings push") &&
	       test_linear_expect_status(
		       sampler_kernels_prepare(&backend_cuda_value,
					       self->card.card, &kernels),
		       STATUS_OK, "ordinary kernels") &&
	       test_linear_expect_status(
		       sampler_prepare_entry(&ordinary,
					     self->card.arena_byte_count,
					     &kernels, &self->ordinary),
		       STATUS_OK, "prepare ordinary sampler") &&
	       test_linear_card_load(&self->card);
}

static bool connect(struct shard_case *group)
{
	for (uint32_t accessor = 0; accessor < TP_GROUP_MOST_RANKS; accessor++)
		for (uint32_t owner = 0; owner < TP_GROUP_MOST_RANKS; owner++) {
			struct shard_case *const self = &group[accessor],
						 *const other = &group[owner];
			if (!test_linear_expect_status(
				    backend_cuda_value.peer_connect(
					    self->card.card, other->card.card,
					    other->exchange.byte_offset,
					    self->peers.byte_offset +
						    owner * sizeof(uint64_t)),
				    STATUS_OK, "exchange peer") ||
			    !test_linear_expect_status(
				    backend_cuda_value.peer_connect(
					    self->card.card, other->card.card,
					    other->entry.byte_offset,
					    self->peers.byte_offset +
						    (TP_GROUP_MOST_RANKS +
						     owner) *
							    sizeof(uint64_t)),
				    STATUS_OK, "entry peer") ||
			    !test_linear_expect_status(
				    backend_cuda_value.peer_connect(
					    self->card.card, other->card.card,
					    other->copy.byte_offset,
					    self->settings_peers.byte_offset +
						    owner * sizeof(uint64_t)),
				    STATUS_OK, "settings peer"))
				return false;
		}
	for (uint32_t rank = 0; rank < TP_GROUP_MOST_RANKS; rank++)
		if (!test_linear_expect_status(
			    backend_cuda_value.launch_wait(
				    group[rank].card.card,
				    TEST_LINEAR_WAIT_LIMIT_NANOSECONDS),
			    STATUS_OK, "peer words ready"))
			return false;
	return true;
}

static bool draw(struct shard_case *group, uint64_t sequence, uint32_t call,
		 uint32_t top_k, float temperature, float top_p)
{
	float logits[VOCABULARY];
	for (uint32_t token = 0; token < VOCABULARY; token++) {
		const uint32_t logit_group = token / LOGIT_GROUP;
		logits[token] = LOGIT_BASE - (float)logit_group;
	}
	const struct step_block block = { .sequence = sequence,
					  .row_count = 1 };
	const struct step_row_record row = { .flags = STEP_ROW_LOGITS,
					     .session_slot = ACTIVE_SLOT };
	const struct step_session settings[SETTINGS_SLOTS] = {
		{ .seed = { 0 }, .temperature = 0.0f, .top_p = 1.0f },
		{ .seed = { (uint32_t)SEED,
			    (uint32_t)(SEED >> STEP_SESSION_SEED_BITS) },
		  .temperature = temperature,
		  .top_p = top_p,
		  .top_k = top_k }
	};
	for (uint32_t rank = 0; rank < TP_GROUP_MOST_RANKS; rank++) {
		struct shard_case *const self = &group[rank];
		struct sampler_shard_arguments arguments = { 0 };
		memcpy(&arguments, self->shard.argument, sizeof(arguments));
		arguments.call = call;
		memcpy(self->shard.argument, &arguments, sizeof(arguments));
		if (!test_linear_card_put(&self->card, self->step, 0, &block,
					  sizeof(block)) ||
		    !test_linear_card_put(&self->card, self->step,
					  sizeof(block), &row, sizeof(row)) ||
		    !test_linear_card_put(&self->card, self->sessions, 0,
					  &settings, sizeof(settings)) ||
		    !test_linear_card_put(&self->card, self->logits, 0,
					  logits + (size_t)(rank * SHARD_SIZE),
					  SHARD_SIZE * sizeof(float)) ||
		    !test_linear_card_put(&self->card, self->full_logits, 0,
					  logits, sizeof(logits)) ||
		    !test_linear_card_upload(&self->card, self->step, 0,
					     self->step.byte_count) ||
		    !test_linear_card_upload(&self->card, self->sessions, 0,
					     self->sessions.byte_count) ||
		    !test_linear_card_upload(&self->card, self->logits, 0,
					     self->logits.byte_count) ||
		    !test_linear_card_upload(&self->card, self->full_logits, 0,
					     self->full_logits.byte_count))
			return false;
	}
	if (!test_linear_expect_status(
		    backend_cuda_value.launch_run(group[0].card.card,
						  &group[0].push, 1),
		    STATUS_OK, "push settings"))
		return false;
	/* Queue every communicating rank before waiting for any. */
	for (uint32_t rank = 0; rank < TP_GROUP_MOST_RANKS; rank++)
		if (!test_linear_expect_status(backend_cuda_value.launch_run(
						       group[rank].card.card,
						       &group[rank].shard, 1),
					       STATUS_OK, "queue shard"))
			return false;
	if (!test_linear_expect_status(
		    backend_cuda_value.launch_run(group[0].card.card,
						  &group[0].ordinary, 1),
		    STATUS_OK, "queue ordinary reference"))
		return false;
	for (uint32_t rank = 0; rank < TP_GROUP_MOST_RANKS; rank++)
		if (!test_linear_card_fetch(&group[rank].card))
			return false;
	struct step_row_result expected = { 0 };
	if (!test_linear_card_get(&group[0].card, group[0].ordinary_result,
				  STEP_RESULT_BYTES, &expected,
				  sizeof(expected)))
		return false;
	for (uint32_t rank = 0; rank < TP_GROUP_MOST_RANKS; rank++) {
		struct step_row_result actual = { 0 };
		if (!test_linear_card_get(&group[rank].card, group[rank].result,
					  STEP_RESULT_BYTES, &actual,
					  sizeof(actual)))
			return false;
		if (!test_linear_expect(
			    bytes_identical(&actual, &expected, sizeof(actual)),
			    "sharded top-k/nucleus winner and logit match ordinary sampler"))
			return false;
		test_linear_card_expect_untouched(
			&group[rank].card, group[rank].step, "step unchanged");
	}
	return true;
}

int main(void)
{
	(void)alarm(DEADLINE_SECONDS);
	static const uint32_t indices[TP_GROUP_MOST_RANKS] = { 0, 1, 3, 4 };
	static const uint32_t counts[] = { 0, 1, TOP_FIVE, VOCABULARY, 0 };
	static const float temperatures[] = { 0.0f, COOL, 1.0f };
	static const float nuclei[] = { NUCLEUS, 1.0f };
	struct shard_case group[TP_GROUP_MOST_RANKS] = { 0 };
	bool ready = true;
	for (uint32_t rank = 0; rank < TP_GROUP_MOST_RANKS && ready; rank++)
		ready = test_linear_card_open(&group[rank].card, indices[rank],
					      WORK_BYTES, 0) &&
			prepare(&group[rank], rank);
	if (ready)
		ready = connect(group);
	uint64_t sequence = 0;
	for (size_t temperature = 0;
	     temperature < sizeof(temperatures) / sizeof(temperatures[0]) &&
	     ready;
	     temperature++)
		for (size_t nucleus = 0;
		     nucleus < sizeof(nuclei) / sizeof(nuclei[0]) && ready;
		     nucleus++)
			for (size_t count = 0;
			     count < sizeof(counts) / sizeof(counts[0]) &&
			     ready;
			     count++) {
				sequence++;
				ready = draw(group, sequence, 0, counts[count],
					     temperatures[temperature],
					     nuclei[nucleus]);
			}
	/* The same active row returns at call32, on the same step and settings. */
	if (ready) {
		sequence++;
		ready = draw(group, sequence, 0, SKIPPED_TOP_K, 1.0f, 1.0f);
		if (ready)
			ready = draw(group, sequence, SKIPPED_CALL,
				     SKIPPED_TOP_K, 1.0f, 1.0f);
	}
	/* Drain the complete group before any peer allocation disappears. */
	for (uint32_t rank = 0; rank < TP_GROUP_MOST_RANKS; rank++)
		if (group[rank].card.card != NULL)
			test_linear_expect_status(
				backend_cuda_value.launch_wait(
					group[rank].card.card,
					TEST_LINEAR_WAIT_LIMIT_NANOSECONDS),
				STATUS_OK, "group drained");
	for (uint32_t rank = 0; rank < TP_GROUP_MOST_RANKS; rank++)
		test_linear_card_close(&group[rank].card);
	(void)printf(
		"sharded top-k CUDA gate: %llu step sequences plus a skipped-call draw, %d failures\n",
		(unsigned long long)sequence, test_linear_failures());
	return ready && test_linear_failures() == 0 ? EXIT_SUCCESS :
						      EXIT_FAILURE;
}
