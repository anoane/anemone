/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* --model_dir: the files of one model under fixed names. Each name is resolved once to its real path (a link is
 * followed, so that a pack linked in from another disk is found), must be a regular file, and keeps the device and
 * inode it had then, so that its open can show it met the same file. */
#ifndef MODEL_FOLDER_H
#define MODEL_FOLDER_H

#include "anemone/status.h"

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>

#define MODEL_FOLDER_PATH_BYTES 1024u /* the folder as typed, with its NUL */

enum model_file {
	MODEL_FILE_TABLES,
	MODEL_FILE_ORIGINAL,
	MODEL_FILE_EXL3,
	MODEL_FILES
};

struct model_file_record {
	char path[PATH_MAX]; /* real */
	uint64_t device, inode;
};

extern const char *const model_file_names[MODEL_FILES];

/* STATUS_INVALID_ARGUMENT: the folder path is empty, too long or no folder; STATUS_INPUT_OUTPUT_ERROR: the file is
 * absent, no regular file, or changed while it was resolved. Each refusal is logged with its path. */
MUST_CHECK result_status
model_folder_resolve(const char *folder, enum model_file file,
		     struct model_file_record *record_out);

/* Whether an opened file (its device and inode) is the resolved one. */
bool model_file_is(const struct model_file_record *record, uint64_t device,
		   uint64_t inode);

#endif
