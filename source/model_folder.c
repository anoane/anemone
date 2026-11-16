/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "model_folder.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

const char *const model_file_names[MODEL_FILES] = { "tokenizer.bin",
						    "original.anemone",
						    "model.exl3map" };

MUST_CHECK result_status
model_folder_resolve(const char *folder, enum model_file file,
		     struct model_file_record *record_out)
{
	const size_t folder_bytes =
		folder != NULL ? strnlen(folder, MODEL_FOLDER_PATH_BYTES) : 0;
	if (folder_bytes == 0 || folder_bytes == MODEL_FOLDER_PATH_BYTES ||
	    file >= MODEL_FILES || record_out == NULL) {
		(void)fprintf(
			stderr,
			"model_dir: the folder path is empty or longer than %u bytes\n",
			MODEL_FOLDER_PATH_BYTES - 1u);
		return STATUS_INVALID_ARGUMENT;
	}
	const int directory = open(folder, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
	if (directory < 0) {
		(void)fprintf(stderr, "model_dir: %s is no folder: %s\n",
			      folder, strerror(errno));
		return STATUS_INVALID_ARGUMENT;
	}
	const char *const name = model_file_names[file];
	char joined[MODEL_FOLDER_PATH_BYTES + PATH_MAX];
	struct stat found, resolved;
	errno = 0;
	/* the name as found in the folder (a link followed), then its real path, which must be that same file */
	const bool is_file =
		fstatat(directory, name, &found, 0) == 0 &&
		S_ISREG(found.st_mode) &&
		snprintf(joined, sizeof(joined), "%s/%s", folder, name) > 0 &&
		realpath(joined, record_out->path) != NULL &&
		stat(record_out->path, &resolved) == 0 &&
		resolved.st_dev == found.st_dev &&
		resolved.st_ino == found.st_ino;
	const int error = errno;
	(void)close(directory);
	if (!is_file) {
		(void)fprintf(
			stderr,
			"model_dir: %s/%s is absent, no regular file or changed while it was resolved%s%s\n",
			folder, name, error != 0 ? ": " : "",
			error != 0 ? strerror(error) : "");
		return STATUS_INPUT_OUTPUT_ERROR;
	}
	record_out->device = found.st_dev;
	record_out->inode = found.st_ino;
	return STATUS_OK;
}

bool model_file_is(const struct model_file_record *record, uint64_t device,
		   uint64_t inode)
{
	return record != NULL && record->device == device &&
	       record->inode == inode;
}
