/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* A row can skip calls. Consumed optional records must never answer a later default exchange. */
#include "engine/library/sampling/sampler_shard_records.h"
#include <stdio.h>
#include <stdlib.h>
#define FIRST_USE 100u
#define LATER_USE 132u
#define TOP_K_FINAL_PHASE 65u
#define DEFAULT_FINAL_PHASE 1u
#define OLD_DEFAULT_TAG 531u
#define OPTIONAL 1u
#define MAIN_PATH 0u
static int failures;
static void expect(int holds, const char *message)
{
	if (!holds) {
		(void)printf("FAILED: %s\n", message);
		failures++;
	}
}
#define TEST_MODEL_ARMS 4u
#define TEST_MODEL_STEPS 100000u
#define TEST_MODEL_VALUE_STRIDE 128u
#define TEST_MODEL_RANDOM_MULTIPLIER 1664525u
#define TEST_MODEL_RANDOM_INCREMENT 1013904223u
#define TEST_MODEL_RANDOM_SHIFT 16u
struct model_record {
	uint32_t tag, value;
};
struct model_rank {
	uint32_t arm, phase, published;
};
static void interleaved(void)
{
	static const uint32_t uses[TEST_MODEL_ARMS] = { FIRST_USE, LATER_USE,
							LATER_USE + 2u,
							LATER_USE + 4u };
	static const uint32_t lengths[TEST_MODEL_ARMS] = { 66u, 1u, 99u, 2u };
	static const uint32_t extended[TEST_MODEL_ARMS] = { OPTIONAL, MAIN_PATH,
							    OPTIONAL,
							    MAIN_PATH };
	struct model_record records[TP_GROUP_MOST_RANKS]
				   [SAMPLER_SHARD_EXCHANGE_SLOTS]
				   [TP_GROUP_MOST_RANKS] = { 0 };
	struct model_record entry_records[TP_GROUP_MOST_RANKS]
					 [TP_GROUP_MOST_RANKS] = { 0 };
	struct model_rank ranks[TP_GROUP_MOST_RANKS] = { 0 };
	uint32_t random = 1, finished = 0;
	for (uint32_t step = 0;
	     step < TEST_MODEL_STEPS && finished < TP_GROUP_MOST_RANKS;
	     step++) {
		random = random * TEST_MODEL_RANDOM_MULTIPLIER +
			 TEST_MODEL_RANDOM_INCREMENT;
		const uint32_t rank = (random >> TEST_MODEL_RANDOM_SHIFT) %
				      TP_GROUP_MOST_RANKS;
		struct model_rank *const self = &ranks[rank];
		if (self->arm == TEST_MODEL_ARMS)
			continue;
		const uint32_t use = uses[self->arm],
			       optional = extended[self->arm];
		const uint32_t logical =
			optional != 0 && self->phase == 0 ?
				SAMPLER_SHARD_ENTRY_PHASE :
				self->phase - (optional != 0 ? 1u : 0u);
		const uint32_t slot = sampler_shard_slot(use, logical);
		const uint32_t tag = sampler_shard_tag(use, logical, optional),
			       value = use * TEST_MODEL_VALUE_STRIDE +
				       self->phase;
		if (self->published == 0) {
			for (uint32_t destination = 0;
			     destination < TP_GROUP_MOST_RANKS; destination++) {
				struct model_record *const area =
					logical == SAMPLER_SHARD_ENTRY_PHASE ?
						entry_records[destination] :
						records[destination][slot];
				area[rank] =
					(struct model_record){ tag, value };
			}
			self->published = 1;
			continue;
		}
		struct model_record *const own =
			logical == SAMPLER_SHARD_ENTRY_PHASE ?
				entry_records[rank] :
				records[rank][slot];
		uint32_t ready = 1;
		for (uint32_t source = 0; source < TP_GROUP_MOST_RANKS;
		     source++)
			if (own[source].tag != tag)
				ready = 0;
		if (ready == 0)
			continue;
		for (uint32_t source = 0; source < TP_GROUP_MOST_RANKS;
		     source++) {
			struct model_record *const record = &own[source];
			if (record->value != value) {
				expect(0,
				       "an interleaved exchange consumed stale data");
				return;
			}
			if (optional != 0 && SAMPLER_SHARD_CLEAR_EXTENDED != 0)
				record->tag = 0;
		}
		self->published = 0;
		self->phase++;
		if (self->phase == (optional != 0 ?
					    1u + sampler_shard_complete_phases(
							 lengths[self->arm]) :
					    lengths[self->arm])) {
			self->phase = 0;
			self->arm++;
			if (self->arm == TEST_MODEL_ARMS)
				finished++;
		}
	}
	expect(finished == TP_GROUP_MOST_RANKS,
	       "all skewed ranks complete without deadlock");
}
int main(void)
{
	const uint32_t held = SAMPLER_SHARD_CLEAR_EXTENDED != 0 ?
				      0 :
				      sampler_shard_tag(FIRST_USE,
							TOP_K_FINAL_PHASE,
							OPTIONAL);
	const uint32_t wanted =
		sampler_shard_tag(LATER_USE, DEFAULT_FINAL_PHASE, MAIN_PATH);
	expect(wanted == OLD_DEFAULT_TAG, "default tag bytes are unchanged");
	expect(held != wanted,
	       "a skipped row cannot take a stale optional record as default data");
	expect(sampler_shard_tag(FIRST_USE, 0, OPTIONAL) !=
		       sampler_shard_tag(FIRST_USE, 0, MAIN_PATH),
	       "optional records have a distinct tag domain");
	expect(SAMPLER_SHARD_CLEAR_EXTENDED != 0,
	       "optional records are cleared before the next publication barrier");
	expect(sampler_shard_slot(FIRST_USE, SAMPLER_SHARD_ENTRY_PHASE) == 0,
	       "entry slots are relative to their separate range");
	interleaved();
	(void)printf("shard exchange lifetime: %d failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
