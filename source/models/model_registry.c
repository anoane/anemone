/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* model_registry.c: the entries of model_registry.h, their lookup by a pack's model name, and a run's instance. */
#include "model_registry.h"

#include <stdio.h>
#include <string.h>

static const struct model_entry *const entries[] = { &ds41f_model_entry };

const struct model_entry *model_registry_find(const char *model_name)
{
	if (model_name == NULL ||
	    memchr(model_name, '\0', PACK_MODEL_NAME_CAPACITY) == NULL)
		return NULL;
	for (size_t index = 0; index < sizeof(entries) / sizeof(entries[0]);
	     index++)
		if (strcmp(model_name, entries[index]->operations->name) == 0)
			return entries[index];
	return NULL;
}

const struct model_program_operations *
model_operations_of(const char *model_name)
{
	const struct model_entry *const entry = model_registry_find(model_name);
	return entry != NULL ? entry->operations : NULL;
}

void model_instance_bind(const struct pack_record *pack, const char *map_path,
			 struct model_instance *instance_out)
{
	*instance_out = (struct model_instance){
		.model = pack != NULL ?
				 model_registry_find(
					 pack_header_of(pack)->model_name) :
				 NULL,
		.operations = NULL,
		.map_path = map_path
	};
}

result_status model_instance_map_check(const struct model_instance *instance,
				       const struct pack_record *pack,
				       const char *tables,
				       uint64_t table_byte_count,
				       bool admits_unrated, uint32_t tc_ranks,
				       uint32_t rotation)
{
	if (instance == NULL || instance->map_path == NULL || pack == NULL)
		return STATUS_INVALID_ARGUMENT;
	if (instance->model == NULL || instance->model->exl3 == NULL) {
		/* the pack's open holds its model name to a NUL within PACK_MODEL_NAME_CAPACITY (pack.c) */
		(void)fprintf(
			stderr,
			"exl3 descriptor: %s: the pack names the model \"%s\", of which this build serves no "
			"EXL3 map\n",
			instance->map_path, pack_header_of(pack)->model_name);
		return STATUS_UNSUPPORTED;
	}
	const result_status status = exl3_descriptor_check(
		instance->map_path, pack, instance->model->exl3, tables,
		table_byte_count, admits_unrated);
	return status != STATUS_OK || tc_ranks == 0 ?
		       status :
		       exl3_descriptor_tc_check(
			       instance->model->operations->name,
			       instance->model->exl3, tc_ranks, rotation);
}
