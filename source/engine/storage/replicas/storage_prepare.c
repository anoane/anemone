/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "storage_prepare.h"
#include "engine/input/tokenizer/tokenizer.h"
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <inttypes.h>
#define JSON_CONTROL_LIMIT 32u
#define JSON_HEX_WIDTH 4

bool prepare_utf8_text(const char *text)
{
	const size_t end = strlen(text);
	bool valid = true;
	for (size_t position = 0; valid && position < end;)
		position += tokenizer_utf8_sequence_bytes(text, position, end,
							  &valid);
	return valid;
}

void prepare_json_text(FILE *output, const char *text)
{
	(void)fputc('"', output);
	const size_t end = strlen(text);
	for (size_t position = 0; position < end;) {
		bool valid = false;
		const uint32_t bytes = tokenizer_utf8_sequence_bytes(
			text, position, end, &valid);
		const unsigned char byte = (unsigned char)text[position];
		/* JSON text is UTF-8 (json_read_document refuses a whole document otherwise): what is no UTF-8 is
		   written as U+FFFD, so one raw path never makes a profile or manifest unreadable. */
		if (!valid)
			(void)fputs("\\ufffd", output);
		else if (byte == '"' || byte == '\\')
			(void)fprintf(output, "\\%c", byte);
		else if (byte < JSON_CONTROL_LIMIT)
			(void)fprintf(output, "\\u%0*x", JSON_HEX_WIDTH,
				      (unsigned)byte);
		else
			(void)fwrite(text + position, 1, bytes, output);
		position += bytes;
	}
	(void)fputc('"', output);
}

void prepare_file_abort(struct prepare_file *file)
{
	if (file->stream != NULL)
		(void)fclose(file->stream);
	file->stream = NULL;
	if (file->temporary[0] != '\0')
		(void)unlink(file->temporary);
	file->temporary[0] = '\0';
}

result_status prepare_file_open(const char *path, struct prepare_file *file)
{
	if (path == NULL || file == NULL || path[0] != '/' ||
	    strlen(path) >= sizeof(file->final))
		return STATUS_INVALID_ARGUMENT;
	struct prepare_file value = { 0 };
	const int length = snprintf(value.temporary, sizeof(value.temporary),
				    "%s.tmp.XXXXXX", path);
	if (length < 0 || (size_t)length >= sizeof(value.temporary))
		return STATUS_INVALID_ARGUMENT;
	(void)snprintf(value.final, sizeof(value.final), "%s", path);
	const int descriptor = mkostemp(value.temporary, O_CLOEXEC);
	if (descriptor < 0)
		return STATUS_INPUT_OUTPUT_ERROR;
	value.stream = fdopen(descriptor, "w");
	if (value.stream == NULL) {
		(void)close(descriptor);
		(void)unlink(value.temporary);
		return STATUS_INPUT_OUTPUT_ERROR;
	}
	*file = value;
	return STATUS_OK;
}

result_status prepare_file_commit(struct prepare_file *file)
{
	if (file == NULL || file->stream == NULL)
		return STATUS_INVALID_ARGUMENT;
	bool valid = !ferror(file->stream) && fflush(file->stream) == 0 &&
		     fsync(fileno(file->stream)) == 0;
	valid = fclose(file->stream) == 0 && valid;
	file->stream = NULL;
	if (valid)
		valid = rename(file->temporary, file->final) == 0;
	char parent[PATH_MAX];
	(void)snprintf(parent, sizeof(parent), "%s", file->final);
	char *const slash = strrchr(parent, '/');
	if (slash != NULL)
		*(slash == parent ? slash + 1 : slash) = '\0';
	const int directory =
		valid ? open(parent, O_RDONLY | O_DIRECTORY | O_CLOEXEC) : -1;
	valid = directory >= 0 && fsync(directory) == 0 && valid;
	if (directory >= 0)
		(void)close(directory);
	prepare_file_abort(file);
	return valid ? STATUS_OK : STATUS_INPUT_OUTPUT_ERROR;
}

result_status prepare_manifest_write(const char *path,
				     const struct prepare_manifest *manifest)
{
	if (manifest == NULL || manifest->known > manifest->chunks ||
	    manifest->chunks > PREPARE_CHUNKS_MOST)
		return STATUS_INVALID_ARGUMENT;
	struct prepare_file file;
	result_status status = prepare_file_open(path, &file);
	if (status != STATUS_OK)
		return status;
	(void)fprintf(file.stream, "{\"schema\":1,\"source\":");
	prepare_json_text(file.stream, manifest->source);
	(void)fputs(",\"pack_id\":", file.stream);
	prepare_json_text(file.stream, manifest->pack_identity);
	(void)fprintf(file.stream,
		      ",\"bytes\":%" PRIu64 ",\"chunk_bytes\":%" PRIu64
		      ",\"chunks\":[",
		      manifest->bytes, manifest->chunk_bytes);
	for (uint32_t chunk = 0; chunk < manifest->known; chunk++) {
		if (chunk != 0)
			(void)fputc(',', file.stream);
		prepare_json_text(file.stream, manifest->hashes[chunk]);
	}
	(void)fputs("]}\n", file.stream);
	return prepare_file_commit(&file);
}
