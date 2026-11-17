/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * model_registry.h: the models this build carries, an entry each, and the instance of one that a run binds. Private
 * to the engine's sources.
 *
 * Contract:
 *   - An entry is a model module's: its model_seam table and its routed experts as EXL3 maps and the tensor-core MoE
 *     read them (exl3_descriptor.h), NULL for a model with no EXL3 map. model_registry_find gives the entry whose
 *     table's name a pack's header names, by model_operations_of's rules (model_seam.h), which is the registry's.
 *   - An instance is a run's model: the entry its pack names (NULL: none this build carries), that entry's table once
 *     the run has described the model (generate and serve; NULL before, and for load), and the EXL3 map as the run
 *     names it (its descriptor's place). A run binds one when its pack has opened, and every later choice of the
 *     model reads it. A swap would drain and free an instance; the run's pack, plan and cards join it later.
 *   - model_instance_map_check is the step "check the map's descriptor": an instance of no entry with EXL3 maps is
 *     STATUS_UNSUPPORTED with one line naming the model the pack names; else exl3_descriptor_check of the map against
 *     the entry's routed experts, then, with tc_ranks (0: the tensor-core MoE is not selected), the entry's
 *     exl3_descriptor_tc_check at that many ranks and the rotation.
 */
#ifndef MODEL_REGISTRY_H
#define MODEL_REGISTRY_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/model_seam.h"
#include "anemone/pack.h"
#include "anemone/status.h"
#include "exl3_descriptor.h"

struct model_entry {
	const struct model_operations *operations;
	const struct exl3_descriptor_model *exl3;
};

/* The one entry: DeepSeek-V4.1-Flash's (model_v41_program.c) */
extern const struct model_entry model_v41_entry;

const struct model_entry *model_registry_find(const char *model_name);

struct model_instance {
	const struct model_entry *model;
	const struct model_operations *operations;
	const char *map_path;
};

/* The instance of the model the open pack names, its table not yet taken */
void model_instance_bind(const struct pack_record *pack, const char *map_path,
			 struct model_instance *instance_out);

MUST_CHECK result_status model_instance_map_check(
	const struct model_instance *instance, const struct pack_record *pack,
	const char *tables, uint64_t table_byte_count, bool admits_unrated,
	uint32_t tc_ranks, uint32_t rotation);

#endif
