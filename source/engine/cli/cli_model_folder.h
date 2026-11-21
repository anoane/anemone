/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* --model_dir: the files of one model under fixed names, the EXL3 map's chosen by --exl3_map (one folder holds several
 * maps of one model, each <NAME>.exl3map with its descriptor). Each name is resolved once to its real path (a link is
 * followed, so that a pack linked in from another disk is found), must be a regular file, and keeps the device and
 * inode it had then, so that its open can show it met the same file. */
#ifndef CLI_MODEL_FOLDER_H
#define CLI_MODEL_FOLDER_H

#include "common/status.h"

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>

#define CLI_MODEL_FOLDER_PATH_BYTES \
	1024u /* the folder as typed, with its NUL */
#define CLI_MODEL_FOLDER_NAME_BYTES \
	(NAME_MAX + 1u) /* a file name in the folder, with its NUL */
#define CLI_MODEL_FOLDER_MAP_SUFFIX ".exl3map"
#define CLI_MODEL_FOLDER_DEFAULT_MAP \
	"model" /* --exl3_map's default: model.exl3map */

enum model_file {
	CLI_MODEL_FILE_TABLES,
	CLI_MODEL_FILE_ORIGINAL,
	CLI_MODEL_FILE_EXL3,
	CLI_MODEL_FILES
};

struct model_file_record {
	char path[PATH_MAX]; /* real */
	uint64_t device, inode;
};

/* Each file's name in the folder; the EXL3 map's is the default of model_folder_map_file's */
extern const char *const model_file_names[CLI_MODEL_FILES];

/* --exl3_map=NAME's file name in the folder, NAME.exl3map: NAME not empty, without '/', the name at most NAME_MAX
 * bytes. STATUS_INVALID_ARGUMENT otherwise, logged, file_out untouched. */
MUST_CHECK result_status model_folder_map_file(
	const char *map_name, char file_out[CLI_MODEL_FOLDER_NAME_BYTES]);

/* name: a file name in the folder (model_file_names, model_folder_map_file). STATUS_INVALID_ARGUMENT: the folder path
 * is empty, too long or no folder, or the name is no file name; STATUS_INPUT_OUTPUT_ERROR: the file is absent, no
 * regular file, or changed while it was resolved. Each refusal is logged with its path. */
MUST_CHECK result_status
model_folder_resolve(const char *folder, const char *name,
		     struct model_file_record *record_out);

/* Whether an opened file (its device and inode) is the resolved one. */
bool model_file_is(const struct model_file_record *record, uint64_t device,
		   uint64_t inode);

#endif
