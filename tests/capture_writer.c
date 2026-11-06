/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * capture_writer.c: the writer of capture_writer.h. Test support: no part of the engine.
 *
 * Contract: capture_writer.h. How this file keeps it: the records are held until close; tensors.bin is opened once,
 * made at its size with ftruncate and written with pwrite at each slab's place; a record counts its slabs put and
 * takes only the next one, so that the count says whether every slab was put once. The types are named as
 * capture_reader.h names them. The manifest is written with stdio and checked once at its close.
 */
#include "capture_writer.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "anemone/hooks.h"

#define MANIFEST_FILE_NAME "manifest.json"
#define TENSORS_FILE_NAME "tensors.bin"
#define FOLDER_MODE 0755
#define FILE_MODE 0644
#define LARGEST_COUNT ((uint64_t)INT64_MAX)             /* of bytes: an offset into a file is an off_t */
#define NAME_CHARACTERS "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._"
#define FIRST_PRINTABLE ' '
#define LAST_PRINTABLE '~'

struct written_record {
    char name[TEST_CAPTURE_WRITER_NAME_CAPACITY];
    const char *description;
    uint64_t dimensions[TEST_CAPTURE_WRITER_MOST_DIMENSIONS];
    uint64_t byte_offset;
    uint64_t byte_count;
    uint64_t slab_count;
    uint64_t slabs_put;                                  /* the slabs before it, each put once */
    uint32_t dimension_count;
    uint32_t type;
    uint32_t layer;
    uint32_t position;
};

struct test_capture_writer {
    struct written_record *records;
    char *folder;
    uint32_t *conversion;                                /* TEST_CAPTURE_WRITER_MOST_SLAB_VALUES words */
    uint64_t byte_count;
    uint32_t record_count;
    uint32_t record_capacity;
    int descriptor;                                      /* tensors.bin; -1 before open */
    bool failed;                                         /* a put failed */
};

result_status
test_capture_writer_create(uint32_t record_capacity,
			   struct test_capture_writer **writer_out)
{
    struct test_capture_writer *const writer = writer_out == NULL ? NULL : calloc(1, sizeof(*writer));
    if (writer == NULL) {
	    LOG_ERROR("capture writer: %s",
		      writer_out == NULL ? "a missing argument" : "no memory");
	    return writer_out == NULL ? STATUS_INVALID_ARGUMENT :
					STATUS_OUT_OF_MEMORY;
    }
    writer->records = calloc(record_capacity == 0 ? 1u : record_capacity, sizeof(struct written_record));
    writer->conversion = calloc(TEST_CAPTURE_WRITER_MOST_SLAB_VALUES, sizeof(uint32_t));
    writer->record_capacity = record_capacity;
    writer->descriptor = -1;
    if (writer->records == NULL || writer->conversion == NULL) {
        free(writer->records);
        free(writer->conversion);
        free(writer);
	LOG_ERROR("capture writer: no memory for %" PRIu32 " records",
		  record_capacity);
	return STATUS_OUT_OF_MEMORY;
    }
    *writer_out = writer;
    return STATUS_OK;
}

/* The bytes a value takes in tensors.bin: a bfloat16 is stored as a float32; 0 for a type this writer does not take. */
static uint32_t stored_bytes_of(enum test_capture_type type)
{
    return type == TEST_CAPTURE_BFLOAT16 ? (uint32_t)sizeof(float)
         : type == TEST_CAPTURE_FLOAT32 || type == TEST_CAPTURE_INT64 ? test_capture_element_bytes(type) : 0;
}

/* A name: 1 to capacity - 1 bytes of NAME_CHARACTERS. */
static bool name_is_plain(const char *name)
{
    const size_t length = name == NULL ? 0 : strnlen(name, TEST_CAPTURE_WRITER_NAME_CAPACITY);
    return length != 0 && length < TEST_CAPTURE_WRITER_NAME_CAPACITY && strspn(name, NAME_CHARACTERS) == length;
}

/* A description: printable characters that a JSON string holds as they are (no quote, no backslash). */
static bool description_is_plain(const char *description)
{
    bool plain = description != NULL && *description != '\0';
    for (const char *next = description; plain && *next != '\0'; next++) {
        plain = *next >= FIRST_PRINTABLE && *next <= LAST_PRINTABLE && *next != '"' && *next != '\\';
    }
    return plain;
}

result_status
test_capture_writer_add(struct test_capture_writer *writer, const char *name,
			enum test_capture_type type, const uint64_t *dimensions,
			uint32_t dimension_count, uint64_t slab_count,
			uint32_t layer, uint32_t position,
			const char *description, uint32_t *index_out)
{
    uint64_t byte_count = stored_bytes_of(type);
    bool valid = writer != NULL && writer->descriptor < 0 && writer->record_count < writer->record_capacity &&
                 name_is_plain(name) && byte_count != 0 && dimensions != NULL &&
                 dimension_count - 1u < TEST_CAPTURE_WRITER_MOST_DIMENSIONS && description_is_plain(description) &&
                 index_out != NULL && slab_count != 0;
    for (uint32_t dimension = 0; valid && dimension < dimension_count; dimension++) {
        valid = dimensions[dimension] != 0 && byte_count <= LARGEST_COUNT / dimensions[dimension];
        byte_count *= valid ? dimensions[dimension] : 1u;
    }
    for (uint32_t index = 0; valid && index < writer->record_count; index++) {
        valid = strcmp(writer->records[index].name, name) != 0;
    }
    valid = valid && byte_count % slab_count == 0 && byte_count <= LARGEST_COUNT - writer->byte_count &&
            (type != TEST_CAPTURE_BFLOAT16 || byte_count / slab_count / sizeof(float) <=
                                              TEST_CAPTURE_WRITER_MOST_SLAB_VALUES);
    if (!valid) {
	    LOG_ERROR(
		    "capture writer: the record %s is refused (a name given twice or not plain, a type, shape or "
		    "slab count this writer does not take, no room, or the folder is open)",
		    name != NULL ? name : "(none)");
	    return STATUS_INVALID_ARGUMENT;
    }
    struct written_record *const record = &writer->records[writer->record_count];
    *record = (struct written_record){.description = description, .byte_offset = writer->byte_count,
                                      .byte_count = byte_count, .slab_count = slab_count,
                                      .dimension_count = dimension_count, .type = (uint32_t)type, .layer = layer,
                                      .position = position};
    memcpy(record->name, name, strlen(name) + 1u);
    memcpy(record->dimensions, dimensions, dimension_count * sizeof(dimensions[0]));
    writer->byte_count += byte_count;
    *index_out = writer->record_count++;
    return STATUS_OK;
}

result_status test_capture_writer_open(struct test_capture_writer *writer,
				       const char *folder)
{
    char path[TEST_CAPTURE_PATH_CAPACITY];
    const int length = writer == NULL || folder == NULL ? -1
                     : snprintf(path, sizeof(path), "%s/" MANIFEST_FILE_NAME, folder);
    if (length <= 0 || (size_t)length >= sizeof(path) || writer->descriptor >= 0) {
	    LOG_ERROR(
		    "capture writer: a missing argument, a folder path too long, or a folder already open");
	    return STATUS_INVALID_ARGUMENT;
    }
    writer->folder = strdup(folder);
    if (writer->folder == NULL || mkdir(folder, FOLDER_MODE) != 0) {
	    LOG_ERROR("capture writer: the folder %s could not be made: %s",
		      folder,
		      writer->folder == NULL ? "no memory" : strerror(errno));
	    return writer->folder == NULL ? STATUS_OUT_OF_MEMORY :
					    STATUS_INPUT_OUTPUT_ERROR;
    }
    (void)snprintf(path, sizeof(path), "%s/" TENSORS_FILE_NAME, folder);
    const int descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, FILE_MODE);
    if (descriptor < 0 || ftruncate(descriptor, (off_t)writer->byte_count) != 0) {
	    LOG_ERROR("capture writer: %s could not be made of %" PRIu64
		      " bytes: %s",
		      path, writer->byte_count, strerror(errno));
	    if (descriptor >= 0) {
		    (void)close(descriptor);
	    }
	    return STATUS_INPUT_OUTPUT_ERROR;
    }
    writer->descriptor = descriptor;
    return STATUS_OK;
}

result_status test_capture_writer_put(struct test_capture_writer *writer,
				      uint32_t index, uint64_t slab,
				      const void *values)
{
    struct written_record *const record = writer != NULL && index < writer->record_count ? &writer->records[index]
                                                                                          : NULL;
    if (record == NULL || values == NULL || writer->descriptor < 0 || slab >= record->slab_count ||
        slab != record->slabs_put) {
	    LOG_ERROR(
		    "capture writer: a put of no record, without values, before the folder is open, beyond the "
		    "record's slabs, or of another slab than its next");
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint64_t slab_bytes = record->byte_count / record->slab_count;
    const unsigned char *next = values;
    if (record->type == TEST_CAPTURE_BFLOAT16) {          /* the high half of a float32: exact */
        for (uint64_t value = 0; value < slab_bytes / sizeof(uint32_t); value++) {
            uint16_t half = 0;
            memcpy(&half, next + value * sizeof(half), sizeof(half));
            writer->conversion[value] = (uint32_t)half << (sizeof(half) * CHAR_BIT);
        }
        next = (const unsigned char *)writer->conversion;
    }
    uint64_t offset = record->byte_offset + slab * slab_bytes;
    for (uint64_t left = slab_bytes; left > 0;) {
        const ssize_t written = pwrite(writer->descriptor, next, (size_t)left, (off_t)offset);
        if (written <= 0 && !(written < 0 && errno == EINTR)) {
		LOG_ERROR("capture writer: %s/" TENSORS_FILE_NAME
			  " could not be written at byte %" PRIu64 ": %s",
			  writer->folder, offset,
			  written < 0 ? strerror(errno) : "nothing written");
		writer->failed = true;
		return STATUS_INPUT_OUTPUT_ERROR;
	}
        const uint64_t done = written > 0 ? (uint64_t)written : 0;
        next += done;
        left -= done;
        offset += done;
    }
    record->slabs_put++;
    return STATUS_OK;
}

/* The manifest: every record, complete only when asked and every record holds what was put whole. */
MUST_CHECK static result_status
manifest_write(const struct test_capture_writer *writer, bool complete,
	       const char *conditions, const char *results)
{
    char path[TEST_CAPTURE_PATH_CAPACITY];
    struct stat facts = {0};
    for (uint32_t index = 0; index < writer->record_count; index++) {
        complete = complete && writer->records[index].slabs_put == writer->records[index].slab_count;
    }
    complete = complete && !writer->failed && fsync(writer->descriptor) == 0 &&
               fstat(writer->descriptor, &facts) == 0 && (uint64_t)facts.st_size == writer->byte_count;
    (void)snprintf(path, sizeof(path), "%s/" MANIFEST_FILE_NAME, writer->folder);
    FILE *const file = fopen(path, "wx");
    if (file == NULL) {
	    LOG_ERROR("capture writer: %s could not be made: %s", path,
		      strerror(errno));
	    return STATUS_INPUT_OUTPUT_ERROR;
    }
    (void)fprintf(file, "{\"complete\": %s, \"conditions\": %s, \"results\": %s, \"tensors\": [",
                  complete ? "true" : "false", conditions, results);
    for (uint32_t index = 0; index < writer->record_count; index++) {
        const struct written_record *const record = &writer->records[index];
        const enum test_capture_type stored = record->type == TEST_CAPTURE_BFLOAT16 ? TEST_CAPTURE_FLOAT32
                                            : (enum test_capture_type)record->type;
        (void)fprintf(file, "%s\n {\"name\": \"%s\", \"dtype\": \"%s\", \"shape\": [", index == 0 ? "" : ",",
                      record->name, test_capture_type_name(stored));
        for (uint32_t dimension = 0; dimension < record->dimension_count; dimension++) {
            (void)fprintf(file, "%s%" PRIu64, dimension == 0 ? "" : ", ", record->dimensions[dimension]);
        }
        (void)fprintf(file, "], \"byte_offset\": %" PRIu64 ", \"byte_count\": %" PRIu64 ", \"layer\": ",
                      record->byte_offset, record->byte_count);
        if (record->layer == TEST_CAPTURE_NO_LAYER) {
            (void)fprintf(file, "null");
        } else {
            (void)fprintf(file, "%" PRIu32, record->layer);
        }
        (void)fprintf(file, ", \"description\": \"%s\"", record->description);
        if (record->type == TEST_CAPTURE_BFLOAT16) {
            (void)fprintf(file, ", \"reference_dtype\": \"%s\"", test_capture_type_name(TEST_CAPTURE_BFLOAT16));
        }
        if (record->position != TEST_CAPTURE_WRITER_NO_POSITION) {
            (void)fprintf(file, ", \"position\": %" PRIu32, record->position);
        }
        (void)fprintf(file, "}");
    }
    (void)fprintf(file, "\n]}\n");
    const bool failed = ferror(file) != 0;
    if (fclose(file) != 0 || failed) {
	    LOG_ERROR("capture writer: %s could not be written whole", path);
	    return STATUS_INPUT_OUTPUT_ERROR;
    }
    return STATUS_OK;
}

result_status test_capture_writer_close(struct test_capture_writer *writer,
					bool complete, const char *conditions,
					const char *results)
{
    if (writer == NULL) {
	    return STATUS_OK;
    }
    result_status status =
	    writer->descriptor < 0 ?
		    STATUS_OK :
	    conditions == NULL || results == NULL ?
		    STATUS_INVALID_ARGUMENT :
		    manifest_write(writer, complete, conditions, results);
    if (status == STATUS_INVALID_ARGUMENT) {
	    LOG_ERROR(
		    "capture writer: closed without conditions or results: no manifest");
    }
    if (writer->descriptor >= 0 && close(writer->descriptor) != 0 &&
	status == STATUS_OK) {
	    LOG_ERROR("capture writer: %s/" TENSORS_FILE_NAME
		      " could not be closed: %s",
		      writer->folder, strerror(errno));
	    status = STATUS_INPUT_OUTPUT_ERROR;
    }
    free(writer->records);
    free(writer->conversion);
    free(writer->folder);
    free(writer);
    return status;
}
