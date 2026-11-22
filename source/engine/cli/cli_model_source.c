/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cli_model_source.c: the model source: its folder, its pack, the pack's storage and family, the descriptor check
 * (split from commands.c by whole items).
 */
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <time.h>
#include <unistd.h>

#include "models/deepseek_v41_flash/prompt/ds41f_chat_template.h"
#include "common/hooks.h"
#include "engine/storage/load/loader.h"
#include "models/deepseek_v41_flash/ds41f_geometry.h"
#include "models/deepseek_v41_flash/official/ds41f_official_reference.h"
#include "engine/formats/digest/pack_digest.h"
#include "engine/runtime/placement/placement.h"
#include "engine/storage/replicas/replicas.h"
#include "engine/engram/engram_disk.h"
#include "engine/engram/engram_fit.h"
#include "engine/storage/io/local_device.h"
#include "engine/storage/profile/storage_profile.h"
#include "engine/storage/autotune/storage_autotune.h"
#include "engine/server/server.h"
#include "engine/runtime/placement/session_fit.h"
#include "cli_stats_client.h"
#include "cli_policy.h"
#include "engine/storage/io/storage_exl3.h"
#include "engine/library/exl3/exl3_tc_records.h"
#include "cli_model_folder.h"
#include "models/model_registry.h"
#include "engine/runtime/scheduler/serving_cursor.h"
#include "cli_axes.h"
#include "engine/runtime/step/step.h"
#include "engine/runtime/step/step_images.h"
#include "engine/input/tokenizer/tokenizer.h"
#include "engine/runtime/speculation/draft_lookup.h"
#include "engine/formats/pack/tokenizer_tables_pack.h"
#include "common/json.h"
#include "engine/engram/engram_step.h"
#include "engine/engram/engram_hot.h"
#include "models/deepseek_v41_flash/plan/model_v41.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_model.h"
#include "models/deepseek_v41_flash/tensor_parallel/ds41f_tp_peer.h"
#include "commands.h"
#include "cli_golden_capture.h"

#include "cli_internal.h"

static char exl3_map_file[CLI_MODEL_FOLDER_NAME_BYTES] =
	CLI_MODEL_FOLDER_DEFAULT_MAP
		CLI_MODEL_FOLDER_MAP_SUFFIX; /* its file name in the folder */

/* --weights=original|exl3, --model_dir=FOLDER and --exl3_map=NAME, as every startup option: exact case, no empty value,
 * no repeat. */
bool take_model_source(const char *argument)
{
	static const char weights_option[] = "--weights=",
			  folder_option[] = "--model_dir=",
			  map_option[] = "--exl3_map=";
	if (strncmp(argument, weights_option, strlen(weights_option)) == 0 &&
	    weights_family == WEIGHTS_UNSET) {
		const char *const weights = argument + strlen(weights_option);
		weights_family = strcmp(weights, "original") == 0 ?
					 WEIGHTS_ORIGINAL :
				 strcmp(weights, "exl3") == 0 ? WEIGHTS_EXL3 :
								WEIGHTS_UNSET;
		return weights_family != WEIGHTS_UNSET;
	}
	if (strncmp(argument, folder_option, strlen(folder_option)) == 0 &&
	    model_dir == NULL && argument[strlen(folder_option)] != '\0') {
		model_dir = argument + strlen(folder_option);
		return true;
	}
	/* an empty name is model_folder_map_file's refusal */
	if (strncmp(argument, map_option, strlen(map_option)) == 0 &&
	    exl3_map_name == NULL) {
		exl3_map_name = argument + strlen(map_option);
		return model_folder_map_file(exl3_map_name, exl3_map_file) ==
		       STATUS_OK;
	}
	return false;
}

/* The folder's pack: the EXL3 map for --weights=exl3 or GPU split TP4_EXL3, otherwise the original pack. */
enum model_file folder_pack_file(void)
{
	return weights_family == WEIGHTS_EXL3 ||
			       (weights_family == WEIGHTS_UNSET &&
				execution_placement == GPU_SPLIT_TP4_EXL3) ?
		       CLI_MODEL_FILE_EXL3 :
		       CLI_MODEL_FILE_ORIGINAL;
}

/* A model file's name in the folder: the EXL3 map's is --exl3_map's */
static const char *folder_file_name(enum model_file file)
{
	return file == CLI_MODEL_FILE_EXL3 ? exl3_map_file :
					     model_file_names[file];
}

/* --model_dir, once the arguments are parsed (the parser opens no file): the tables, when the command reads them, and
 * the pack from the folder (folder_pack_file, by its name there). */
MUST_CHECK result_status model_folder_files(bool reads_tables)
{
	static const char *pack_paths[1];
	const enum model_file pack_file = folder_pack_file();
	result_status status =
		reads_tables ? model_folder_resolve(
				       model_dir,
				       model_file_names[CLI_MODEL_FILE_TABLES],
				       &model_files[CLI_MODEL_FILE_TABLES]) :
			       STATUS_OK;
	status = status == STATUS_OK ?
			 model_folder_resolve(model_dir,
					      folder_file_name(pack_file),
					      &model_files[pack_file]) :
			 status;
	if (status != STATUS_OK)
		return status;
	if (reads_tables)
		front_end_paths[TABLES_FILE] =
			model_files[CLI_MODEL_FILE_TABLES].path;
	model_pack = &model_files[pack_file];
	pack_paths[0] = model_pack->path;
	copy_paths = pack_paths;
	copy_count = 1;
	return STATUS_OK;
}

/* The whole of a file, not empty, in memory of malloc (aligned for any type) that the caller frees; else NULL. */
/* The folder's pack as the reader opened it must be the file resolved: its device and inode, then its identity. */
MUST_CHECK result_status
model_pack_opened(const struct storage_operations_record *storage,
		  const struct storage_reader *opened_reader,
		  const struct pack_header *header)
{
	struct storage_description opened = { 0 };
	const char *const name =
		folder_file_name((enum model_file)(model_pack - model_files));
	if (storage->reader_describe(opened_reader, &opened) != STATUS_OK ||
	    !model_file_is(model_pack, opened.first_device,
			   opened.first_inode)) {
		(void)fprintf(
			stderr,
			"model_dir: %s changed between its check and its open\n",
			model_pack->path);
		return STATUS_INPUT_OUTPUT_ERROR;
	}
	(void)fprintf(
		stderr,
		"model_dir: %s -> %s (device %u:%u); pack %s, header digest %016" PRIx64
		", tables digest %016" PRIx64 "\n",
		name, model_pack->path, major(model_pack->device),
		minor(model_pack->device), header->model_name,
		header->header_digest, header->tables_digest);
	return STATUS_OK;
}

/* The pack's reader: in the folder form its fixed name sets it, never a link target's suffix. */
const struct storage_operations_record *pack_storage(void)
{
	return model_pack == NULL ? storage_for_path(copy_paths[0]) :
	       model_pack == &model_files[CLI_MODEL_FILE_EXL3] ?
				    &storage_exl3 :
				    &storage_direct;
}

bool pack_holds_exl3(const struct pack_record *opened_pack)
{
	const struct pack_header *const header = pack_header_of(opened_pack);
	const struct pack_tensor *const tensors = pack_tensors_of(opened_pack);
	bool holds_exl3 = false;
	for (uint32_t tensor = 0; tensor < header->tensor_count && !holds_exl3;
	     tensor++)
		holds_exl3 = tensors[tensor].weight_format ==
			     WEIGHT_FORMAT_EXL3;
	return holds_exl3;
}

/* --weights against the pack: an EXL3 map's pack holds EXL3 matrices, an original pack none. */
MUST_CHECK result_status
pack_family_status(const struct pack_record *opened_pack)
{
	const struct pack_header *const header = pack_header_of(opened_pack);
	const bool holds_exl3 = pack_holds_exl3(opened_pack);
	if (holds_exl3 == (weights_family == WEIGHTS_EXL3))
		return STATUS_OK;
	(void)fprintf(stderr, "--weights=%s, but the pack %s holds %s\n",
		      weights_family == WEIGHTS_EXL3 ? "exl3" : "original",
		      header->model_name,
		      holds_exl3 ? "EXL3 matrices" : "no EXL3 matrix");
	return STATUS_UNSUPPORTED;
}

/* --exl3_tc_moe=1 takes effect: TP4_EXL3 with the BF16 wire (the exact recipe ignores it), in the plan and in the
 * descriptor's check alike */
bool tc_moe_selected(void)
{
	return execution_placement == GPU_SPLIT_TP4_EXL3 && tp_wire_bf16 != 0 &&
	       exl3_tc_moe != 0;
}

/* A pack of EXL3 matrices against its map's descriptor and its model (model_registry.h), beside the folder's map or
 * the first pack argument, or without one against the model's established packs: the tables when the command read
 * them, the quality opt-in, and with the tensor-core MoE the model against its kernels. */
MUST_CHECK result_status map_descriptor_check(const char *tables,
					      uint64_t table_byte_count)
{
	return model_instance_map_check(
		&instance, pack, tables, table_byte_count,
		exl3_unrated_map != 0, tc_moe_selected() ? DS41F_TP4_RANKS : 0u,
		exl3_expert_rotation);
}

/* A whole file; with expected (a --model_dir file), only the resolved file itself. */
char *file_read(const char *path, const struct model_file_record *expected,
		uint64_t *byte_count_out)
{
	/* a --model_dir file opens without blocking (a FIFO swapped in refuses below), then reads blocking */
	const int descriptor =
		expected == NULL ? -1 :
				   open(path, O_RDONLY | O_CLOEXEC |
						      O_NONBLOCK | O_NOCTTY);
	if (descriptor >= 0 &&
	    fcntl(descriptor, F_SETFL,
		  fcntl(descriptor, F_GETFL) & ~O_NONBLOCK) != 0) {
		(void)close(descriptor);
		*byte_count_out = 0;
		return NULL;
	}
	FILE *const file = expected == NULL ? fopen(path, "rb") :
			   descriptor >= 0  ? fdopen(descriptor, "rb") :
					      NULL;
	if (expected != NULL && descriptor >= 0 && file == NULL)
		(void)close(descriptor);
	long byte_count = -1;
	char *bytes = NULL;
	struct stat metadata;
	if (file != NULL && expected != NULL &&
	    (fstat(fileno(file), &metadata) != 0 ||
	     !model_file_is(expected, metadata.st_dev, metadata.st_ino))) {
		(void)fprintf(
			stderr,
			"model_dir: %s changed between its check and its open\n",
			path);
		(void)fclose(file);
		*byte_count_out = 0;
		return NULL;
	}
	if (file != NULL && expected != NULL)
		(void)fprintf(stderr, "model_dir: %s -> %s (device %u:%u)\n",
			      model_file_names[CLI_MODEL_FILE_TABLES], path,
			      major(metadata.st_dev), minor(metadata.st_dev));
	if (file != NULL) {
		byte_count = fseek(file, 0, SEEK_END) == 0 ? ftell(file) : -1;
		bytes = byte_count > 0 && fseek(file, 0, SEEK_SET) == 0 ?
				malloc((size_t)byte_count) :
				NULL;
		if (bytes != NULL && fread(bytes, 1, (size_t)byte_count,
					   file) != (size_t)byte_count) {
			free(bytes);
			bytes = NULL;
		}
		(void)fclose(file);
	}
	*byte_count_out = bytes != NULL ? (uint64_t)byte_count : 0;
	return bytes;
}
