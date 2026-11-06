/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * capture_reader.c: the reader of capture_reader.h. Test support: no part of the engine.
 *
 * Contract: capture_reader.h. How this file keeps it:
 *   - The manifest is read whole, checked by the engine's JSON reader, walked once for its records and freed. A
 *     capture keeps the records, their names (unescaped, each with its NUL, in one block as long as the manifest: a
 *     name is never longer than its JSON string) and the records sorted by name for the search. tensors.bin stays
 *     open and is read with pread, so that reads share nothing but the descriptor.
 *   - A bfloat16 range is read CHUNK_VALUES float32 values at a time into the stack, checked and narrowed.
 *   - Fault points: test_capture_memory at each allocation, test_capture_file_read at each read of a file.
 */
#include "capture_reader.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "anemone/hooks.h"
#include "json.h"

#define MANIFEST_FILE_NAME "manifest.json"
#define TENSORS_FILE_NAME "tensors.bin"
#define JSON_TRUE "true"
#define JSON_NULL "null"
#define LARGEST_COUNT ((uint64_t)INT64_MAX)       /* of bytes or elements: an offset into a file is an off_t */
#define DECIMAL_BASE 10u
#define COMPARE_SCRATCH_BYTES 256                  /* a key or a dtype unescaped: a longer one is none asked for */
#define KEY_SCRATCH_BYTES (1u << 20u)              /* the keys of the objects open at one place, to find one twice */
#define MOST_BYTES_PER_READ (1ull << 30u)
#define CHUNK_VALUES 4096u
#define BFLOAT16_SHIFT 16u
#define LOW_HALF_MASK 0xFFFFu
#define SHAPE_TEXT_CAPACITY 256                    /* "[", 8 numbers of up to 19 digits, 7 ", ", "]" and a NUL */

_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "tensors.bin holds the host's bytes, little-endian");

/* How the writer stores each type, and what a read delivers of it. */
struct type_form {
    const char *numpy_name;
    const char *reference_name;                    /* NULL where numpy's type is the reference's */
    uint32_t stored_bytes;
    uint32_t delivered_bytes;
};
static const struct type_form type_forms[TEST_CAPTURE_TYPE_COUNT] = {
    [TEST_CAPTURE_FLOAT16] = {"float16", NULL, 2, 2},
    [TEST_CAPTURE_BFLOAT16] = {"float32", "bfloat16", 4, 2},
    [TEST_CAPTURE_FLOAT32] = {"float32", NULL, 4, 4},
    [TEST_CAPTURE_FLOAT64] = {"float64", NULL, 8, 8},
    [TEST_CAPTURE_FLOAT8_E4M3] = {"uint8", "float8_e4m3fn", 1, 1},
    [TEST_CAPTURE_FLOAT8_E8M0] = {"uint8", "float8_e8m0fnu", 1, 1},
    [TEST_CAPTURE_INT8] = {"int8", NULL, 1, 1},
    [TEST_CAPTURE_UINT8] = {"uint8", NULL, 1, 1},
    [TEST_CAPTURE_INT16] = {"int16", NULL, 2, 2},
    [TEST_CAPTURE_INT32] = {"int32", NULL, 4, 4},
    [TEST_CAPTURE_INT64] = {"int64", NULL, 8, 8},
};

enum manifest_member { MANIFEST_COMPLETE, MANIFEST_TENSORS, MANIFEST_MEMBER_COUNT };
static const char *const manifest_member_names[MANIFEST_MEMBER_COUNT] = {"complete", "tensors"};
enum record_member { RECORD_NAME, RECORD_DTYPE, RECORD_SHAPE, RECORD_BYTE_OFFSET, RECORD_BYTE_COUNT, RECORD_LAYER,
                     RECORD_REFERENCE_DTYPE, RECORD_MEMBER_COUNT };
static const char *const record_member_names[RECORD_MEMBER_COUNT] = {
    "name", "dtype", "shape", "byte_offset", "byte_count", "layer", "reference_dtype"};

struct name_entry {
    const char *name;
    uint32_t tensor_index;
    uint32_t reserved_zero;
};

struct test_capture {
    struct test_capture_tensor *tensors;           /* in the order of the file */
    struct name_entry *by_name;                    /* sorted by name */
    char *names;
    int tensors_descriptor;
    uint32_t tensor_count;
};

/* Reads byte_count bytes at byte_offset of an open file; one error line, naming the file, when it cannot. */
MUST_CHECK static result_status read_file(int descriptor, const char *file_name,
					  uint64_t byte_offset,
					  void *destination,
					  uint64_t byte_count)
{
    for (uint64_t delivered_byte_count = 0; delivered_byte_count < byte_count;) {
        const uint64_t left_byte_count = byte_count - delivered_byte_count;
        const size_t asked_byte_count =
            (size_t)(left_byte_count < MOST_BYTES_PER_READ ? left_byte_count : MOST_BYTES_PER_READ);
        errno = 0;      /* an injected failure sets none, and an older EINTR would have it asked again */
	const ssize_t read_byte_count =
		FAULT_POINT(test_capture_file_read) ?
			-1 :
			pread(descriptor,
			      (unsigned char *)destination +
				      delivered_byte_count,
			      asked_byte_count,
			      (off_t)(byte_offset + delivered_byte_count));
	if (read_byte_count <= 0 && (read_byte_count == 0 || errno != EINTR)) {
		LOG_ERROR("capture: the read of %zu bytes at offset %" PRIu64
			  " of %s failed: %s",
			  asked_byte_count, byte_offset + delivered_byte_count,
			  file_name,
			  read_byte_count == 0 ? "the file ends before them" :
						 strerror(errno));
		return STATUS_INPUT_OUTPUT_ERROR;
	}
        delivered_byte_count += read_byte_count > 0 ? (uint64_t)read_byte_count : 0;
    }
    return STATUS_OK;
}

/* Opens a regular file of the folder and sizes it; one error line when it cannot. */
MUST_CHECK static result_status file_open(const char *path,
					  const char *file_name,
					  int *descriptor_out,
					  uint64_t *byte_count_out)
{
    struct stat facts = {0};
    const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    const bool is_sized = descriptor >= 0 && fstat(descriptor, &facts) == 0;
    if (!is_sized || !S_ISREG(facts.st_mode) || facts.st_size < 0) {
	    LOG_ERROR("capture: %s is no regular file that opens: %s",
		      file_name,
		      is_sized ? "it is something else" : strerror(errno));
	    if (descriptor >= 0) {
		    (void)close(descriptor);
	    }
	    return STATUS_INPUT_OUTPUT_ERROR;
    }
    *descriptor_out = descriptor;
    *byte_count_out = (uint64_t)facts.st_size;
    return STATUS_OK;
}

/* The whole manifest, with a NUL behind it, in memory the caller frees. */
MUST_CHECK static result_status
manifest_text_of(const char *path, char **text_out, size_t *length_out)
{
    int descriptor = -1;
    uint64_t byte_count = 0;
    result_status status =
	    file_open(path, MANIFEST_FILE_NAME, &descriptor, &byte_count);
    if (status != STATUS_OK) {
	    return status;
    }
    char *const text = byte_count > TEST_CAPTURE_MOST_MANIFEST_BYTES ||
				       FAULT_POINT(test_capture_memory) ?
			       NULL :
			       malloc((size_t)byte_count + 1);
    status = byte_count > TEST_CAPTURE_MOST_MANIFEST_BYTES ?
		     STATUS_CORRUPT_PACK :
	     text == NULL ? STATUS_OUT_OF_MEMORY :
			    read_file(descriptor, MANIFEST_FILE_NAME, 0, text,
				      byte_count);
    (void)close(descriptor);
    if (byte_count > TEST_CAPTURE_MOST_MANIFEST_BYTES) {
	    LOG_ERROR("capture: the manifest holds %" PRIu64
		      " bytes, above the limit",
		      byte_count);
    }
    if (status != STATUS_OK) {
	    free(text);
	    return status;
    }
    text[byte_count] = '\0';
    *text_out = text;
    *length_out = (size_t)byte_count;
    return STATUS_OK;
}

static bool is_kind(struct json_value value, char first_byte)
{
    return value.text != NULL && value.text[value.begin] == first_byte;
}

static bool is_word(struct json_value value, const char *word)
{
    const size_t length = strlen(word);
    return value.text != NULL && value.end - value.begin == length &&
           memcmp(value.text + value.begin, word, length) == 0;
}

/* The members of an object that names names, each in values_out; text NULL for one the object lacks. */
static void members_of(struct json_scratch *scratch, struct json_value object, const char *const *names,
                       uint32_t name_count, struct json_value *values_out)
{
    struct json_value key = {0};
    struct json_value value = {0};
    for (uint32_t index = 0; index < name_count; index++) {
        values_out[index] = (struct json_value){NULL, 0, 0};
    }
    for (size_t cursor = object.begin;
	 json_next(object, &cursor, &key, &value);) {
	    for (uint32_t index = 0; index < name_count; index++) {
		    if (json_string_is(scratch, key, names[index])) {
			    values_out[index] = value;
		    }
	    }
    }
}

/* The value of a whole number (digits only) up to LARGEST_COUNT; false for any other value. */
static bool whole_number_of(struct json_value value, uint64_t *number_out)
{
    uint64_t number = 0;
    if (value.text == NULL || value.end == value.begin) {
        return false;
    }
    for (size_t position = value.begin; position < value.end; position++) {
        const char digit = value.text[position];
        const uint64_t digit_value = (uint64_t)(unsigned char)digit - (uint64_t)(unsigned char)'0';
        if (digit < '0' || digit > '9' || number > (LARGEST_COUNT - digit_value) / DECIMAL_BASE) {
            return false;
        }
        number = number * DECIMAL_BASE + digit_value;
    }
    *number_out = number;
    return true;
}

/* The type a dtype and a reference dtype (text NULL for none) name together, or TEST_CAPTURE_TYPE_COUNT. */
static uint32_t type_of(struct json_scratch *scratch, struct json_value dtype, struct json_value reference_dtype)
{
    for (uint32_t type = 0; type < TEST_CAPTURE_TYPE_COUNT; type++) {
        const struct type_form *const form = &type_forms[type];
	const bool reference_agrees =
		form->reference_name == NULL ?
			reference_dtype.text == NULL :
			json_string_is(scratch, reference_dtype,
				       form->reference_name);
	if (reference_agrees &&
	    json_string_is(scratch, dtype, form->numpy_name)) {
		return type;
	}
    }
    return TEST_CAPTURE_TYPE_COUNT;
}

/* What is wrong with the shape, the count and the place of a record, or NULL; its type is known. */
static const char *size_flaw(struct json_value shape, struct json_value byte_offset, struct json_value byte_count,
                             uint64_t expected_offset, struct test_capture_tensor *tensor)
{
    struct json_value dimension = {0};
    uint64_t element_count = 1;                     /* LARGEST_COUNT + 1 once the product is beyond it */
    for (size_t cursor = shape.begin;
	 json_next(shape, &cursor, NULL, &dimension);) {
	    uint64_t size = 0;
	    if (tensor->dimension_count == TEST_CAPTURE_MAXIMUM_DIMENSIONS) {
		    return "it has more dimensions than this reader takes";
	    }
	    if (!whole_number_of(dimension, &size)) {
		    return "a dimension is no whole number";
	    }
	    element_count = size != 0 && element_count > LARGEST_COUNT / size ?
				    LARGEST_COUNT + 1 :
				    element_count * size;
	    tensor->dimensions[tensor->dimension_count] = size;
	    tensor->dimension_count++;
    }
    const uint64_t stored_bytes = type_forms[tensor->type].stored_bytes;
    if (!whole_number_of(byte_offset, &tensor->byte_offset) || !whole_number_of(byte_count, &tensor->byte_count)) {
        return "its byte offset or byte count is no whole number";
    }
    if (element_count > LARGEST_COUNT / stored_bytes || element_count * stored_bytes != tensor->byte_count) {
        return "its byte count is not its elements times the size of one";
    }
    if (tensor->byte_offset != expected_offset) {
        return "it does not begin where the record before ends";
    }
    tensor->element_count = element_count;
    return NULL;
}

/* What is wrong with one record, by the first rule it breaks, or NULL with the record in *tensor. Its name is
 * unescaped into the names, behind those before it. */
static const char *record_flaw(struct json_scratch *scratch, struct json_scratch *names, struct json_value record,
                               uint64_t expected_offset, struct test_capture_tensor *tensor)
{
    struct json_value members[RECORD_MEMBER_COUNT];
    uint64_t layer = TEST_CAPTURE_NO_LAYER;
    if (!is_kind(record, '{')) {
        return "it is no object";
    }
    members_of(scratch, record, record_member_names, RECORD_MEMBER_COUNT, members);
    if (!is_kind(members[RECORD_NAME], '"') || !is_kind(members[RECORD_DTYPE], '"') ||
        !is_kind(members[RECORD_SHAPE], '[') || members[RECORD_LAYER].text == NULL) {
        return "a member is missing or of another kind";
    }
    const struct json_bytes name = json_unescape(names, members[RECORD_NAME]);
    if (names->is_short || names->used >= names->capacity) {
        return "the names do not fit the manifest's length";        /* cannot be: a name is no longer than its string */
    }
    names->bytes[names->used] = '\0';
    names->used++;
    tensor->name = name.data;
    tensor->type = type_of(scratch, members[RECORD_DTYPE], members[RECORD_REFERENCE_DTYPE]);
    if (name.count == 0 || memchr(name.data, '\0', name.count) != NULL) {
        return "its name is empty or holds a NUL";
    }
    if (tensor->type == TEST_CAPTURE_TYPE_COUNT) {
        return "its dtype and reference_dtype are no pair this reader knows";
    }
    if (!is_word(members[RECORD_LAYER], JSON_NULL) &&
        (!whole_number_of(members[RECORD_LAYER], &layer) || layer >= TEST_CAPTURE_NO_LAYER)) {
        return "its layer is neither null nor a whole number below UINT32_MAX";
    }
    tensor->layer = (uint32_t)layer;
    return size_flaw(members[RECORD_SHAPE], members[RECORD_BYTE_OFFSET], members[RECORD_BYTE_COUNT], expected_offset,
                     tensor);
}

static int name_order(const void *first, const void *second)
{
    const struct name_entry *const left = first;
    const struct name_entry *const right = second;
    return strcmp(left->name, right->name);
}

/* What is wrong with the manifest as a whole, or NULL; the records' array in *tensors_out. keys is the scratch
 * memory of the search for a key twice. */
static const char *manifest_flaw(struct json_scratch *scratch, struct json_scratch *keys, const char *text,
                                 size_t length, struct json_value *tensors_out)
{
    struct json_value manifest = {0};
    struct json_value members[MANIFEST_MEMBER_COUNT];
    if (json_read_document(text, length, &manifest) != JSON_DOCUMENT_VALID ||
	!is_kind(manifest, '{')) {
	    return "the manifest is not one JSON object";
    }
    const bool has_repeated_key = json_has_repeated_key(keys, manifest);
    if (keys->is_short) {
        return "the manifest has more keys open at one place than this reader checks";
    }
    if (has_repeated_key) {
        return "an object of the manifest holds a key twice";
    }
    members_of(scratch, manifest, manifest_member_names, MANIFEST_MEMBER_COUNT, members);
    if (!is_word(members[MANIFEST_COMPLETE], JSON_TRUE)) {
        return "the manifest does not say complete";
    }
    if (!is_kind(members[MANIFEST_TENSORS], '[')) {
        return "the manifest's tensors are no array";
    }
    *tensors_out = members[MANIFEST_TENSORS];
    return NULL;
}

/* Fills the capture from the manifest's text; tensors.bin is open and holds file_byte_count bytes. */
MUST_CHECK static result_status records_of(struct test_capture *capture,
					   const char *text, size_t length,
					   uint64_t file_byte_count)
{
    char compare_bytes[COMPARE_SCRATCH_BYTES];
    struct json_scratch scratch = {compare_bytes, sizeof(compare_bytes), 0, false};
    struct json_value tensors = {0};
    struct json_value record = {0};
    uint64_t record_count = 0;
    uint64_t expected_offset = 0;
    char *const key_bytes =
	    FAULT_POINT(test_capture_memory) ? NULL : malloc(KEY_SCRATCH_BYTES);
    struct json_scratch keys = {key_bytes, KEY_SCRATCH_BYTES, 0, false};
    if (key_bytes == NULL) {
	    return STATUS_OUT_OF_MEMORY;
    }
    const char *flaw = manifest_flaw(&scratch, &keys, text, length, &tensors);
    free(key_bytes);
    for (size_t cursor = tensors.begin;
	 flaw == NULL && json_next(tensors, &cursor, NULL, &record);) {
	    record_count++;
    }
    flaw = flaw == NULL && record_count >= UINT32_MAX ? "the manifest holds more records than this reader takes" : flaw;
    if (flaw != NULL) {
	    LOG_ERROR("capture: %s", flaw);
	    return STATUS_CORRUPT_PACK;
    }
    const size_t slots = record_count > 0 ? (size_t)record_count : 1;
    const bool memory_fails = FAULT_POINT(test_capture_memory);
    capture->tensors = memory_fails ? NULL : calloc(slots, sizeof(*capture->tensors));
    capture->by_name = memory_fails ? NULL : calloc(slots, sizeof(*capture->by_name));
    capture->names = memory_fails ? NULL : malloc(length + 1);
    if (capture->tensors == NULL || capture->by_name == NULL || capture->names == NULL) {
	    return STATUS_OUT_OF_MEMORY;
    }
    struct json_scratch names = {capture->names, length + 1, 0, false};
    for (size_t cursor = tensors.begin;
	 capture->tensor_count < record_count &&
	 json_next(tensors, &cursor, NULL, &record);) {
	    struct test_capture_tensor *const tensor =
		    &capture->tensors[capture->tensor_count];
	    flaw = record_flaw(&scratch, &names, record, expected_offset,
			       tensor);
	    if (flaw != NULL) {
		    LOG_ERROR("capture: record %" PRIu32 " breaks the rule: %s",
			      capture->tensor_count, flaw);
		    return STATUS_CORRUPT_PACK;
	    }
	    expected_offset +=
		    tensor->byte_count; /* each at most 2^63 - 1, and equal to where the next begins */
	    if (expected_offset > LARGEST_COUNT) {
		    LOG_ERROR("capture: the records end beyond %" PRIu64
			      " bytes",
			      LARGEST_COUNT);
		    return STATUS_CORRUPT_PACK;
	    }
	    capture->by_name[capture->tensor_count] =
		    (struct name_entry){ tensor->name, capture->tensor_count,
					 0 };
	    capture->tensor_count++;
    }
    qsort(capture->by_name, capture->tensor_count, sizeof(*capture->by_name), name_order);
    for (uint32_t index = 1; index < capture->tensor_count; index++) {
        if (name_order(&capture->by_name[index - 1], &capture->by_name[index]) == 0) {
		LOG_ERROR("capture: two records are named %s",
			  capture->by_name[index].name);
		return STATUS_CORRUPT_PACK;
	}
    }
    if (expected_offset != file_byte_count) {
	    LOG_ERROR("capture: the records end at %" PRIu64
		      " bytes, %s holds %" PRIu64,
		      expected_offset, TENSORS_FILE_NAME, file_byte_count);
	    return STATUS_CORRUPT_PACK;
    }
    return STATUS_OK;
}

/* The path of a file of the folder, or false when it does not fit. */
static bool path_in(const char *folder_path, const char *file_name, char *path_out)
{
    const int length = snprintf(path_out, TEST_CAPTURE_PATH_CAPACITY, "%s/%s", folder_path, file_name);
    return length > 0 && length < TEST_CAPTURE_PATH_CAPACITY;
}

result_status test_capture_open(const char *folder_path,
				struct test_capture **capture_out)
{
    char manifest_path[TEST_CAPTURE_PATH_CAPACITY];
    char tensors_path[TEST_CAPTURE_PATH_CAPACITY];
    char *text = NULL;
    size_t length = 0;
    uint64_t file_byte_count = 0;
    if (folder_path == NULL || capture_out == NULL || !path_in(folder_path, MANIFEST_FILE_NAME, manifest_path) ||
        !path_in(folder_path, TENSORS_FILE_NAME, tensors_path)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    struct test_capture *const capture = FAULT_POINT(test_capture_memory) ?
						 NULL :
						 calloc(1, sizeof(*capture));
    if (capture == NULL) {
	    return STATUS_OUT_OF_MEMORY;
    }
    capture->tensors_descriptor = -1;
    result_status status = manifest_text_of(manifest_path, &text, &length);
    if (status == STATUS_OK) {
	    status = file_open(tensors_path, TENSORS_FILE_NAME,
			       &capture->tensors_descriptor, &file_byte_count);
    }
    if (status == STATUS_OK) {
	    status = records_of(capture, text, length, file_byte_count);
    }
    free(text);
    if (status != STATUS_OK) {
	    test_capture_close(capture);
	    return status;
    }
    *capture_out = capture;
    return STATUS_OK;
}

void test_capture_close(struct test_capture *capture)
{
    if (capture != NULL) {
        if (capture->tensors_descriptor >= 0) {
            (void)close(capture->tensors_descriptor);
        }
        free(capture->tensors);
        free(capture->by_name);
        free(capture->names);
        free(capture);
    }
}

uint32_t test_capture_tensor_count(const struct test_capture *capture)
{
    return capture != NULL ? capture->tensor_count : 0;
}

const struct test_capture_tensor *test_capture_tensor_at(const struct test_capture *capture, uint32_t tensor_index)
{
    return capture != NULL && tensor_index < capture->tensor_count ? &capture->tensors[tensor_index] : NULL;
}

const struct test_capture_tensor *test_capture_find(const struct test_capture *capture, const char *name)
{
    const struct name_entry key = {name, 0, 0};
    const struct name_entry *const found =
        capture != NULL && name != NULL && capture->tensor_count > 0
            ? bsearch(&key, capture->by_name, capture->tensor_count, sizeof(key), name_order) : NULL;
    return found != NULL ? &capture->tensors[found->tensor_index] : NULL;
}

uint32_t test_capture_element_bytes(enum test_capture_type type)
{
    return (uint32_t)type < TEST_CAPTURE_TYPE_COUNT ? type_forms[type].delivered_bytes : 0;
}

const char *test_capture_type_name(enum test_capture_type type)
{
    if ((uint32_t)type >= TEST_CAPTURE_TYPE_COUNT) {
        return "";
    }
    return type_forms[type].reference_name != NULL ? type_forms[type].reference_name : type_forms[type].numpy_name;
}

/* Reads float32 values of a bfloat16 tensor and delivers the high half of each, whose low half must be zero. */
MUST_CHECK static result_status
read_bfloat16(const struct test_capture *capture,
	      const struct test_capture_tensor *tensor, uint64_t first_element,
	      uint64_t element_count, unsigned char *destination)
{
    uint32_t values[CHUNK_VALUES] = {0};
    for (uint64_t done = 0; done < element_count;) {
        const uint64_t count = element_count - done < CHUNK_VALUES ? element_count - done : CHUNK_VALUES;
        const uint64_t byte_offset = tensor->byte_offset + (first_element + done) * sizeof(values[0]);
	const result_status status =
		read_file(capture->tensors_descriptor, TENSORS_FILE_NAME,
			  byte_offset, values, count * sizeof(values[0]));
	if (status != STATUS_OK) {
		return status;
	}
	for (uint64_t index = 0; index < count; index++) {
            const uint16_t word = (uint16_t)(values[index] >> BFLOAT16_SHIFT);
            if ((values[index] & LOW_HALF_MASK) != 0) {
		    LOG_ERROR("capture: value %" PRIu64
			      " of %s is no bfloat16 value",
			      first_element + done + index, tensor->name);
		    return STATUS_CORRUPT_PACK;
	    }
            memcpy(destination + (done + index) * sizeof(word), &word, sizeof(word));
        }
        done += count;
    }
    return STATUS_OK;
}

/* The shape of a record as "[1, 435, 5120]", for an error line: SHAPE_TEXT_CAPACITY holds the most dimensions. */
static void shape_text_of(const struct test_capture_tensor *tensor, char text[SHAPE_TEXT_CAPACITY])
{
    size_t length = 0;
    for (uint32_t dimension = 0; dimension < tensor->dimension_count; dimension++) {
        const int written = snprintf(text + length, SHAPE_TEXT_CAPACITY - length, "%s%" PRIu64,
                                     dimension == 0 ? "[" : ", ", tensor->dimensions[dimension]);
        length += written > 0 ? (size_t)written : 0;
    }
    (void)snprintf(text + length, SHAPE_TEXT_CAPACITY - length, "%s]", tensor->dimension_count == 0 ? "[" : "");
}

result_status test_capture_read(const struct test_capture *capture,
				const char *name, enum test_capture_type type,
				const uint64_t *dimensions,
				uint32_t dimension_count,
				uint64_t first_element, uint64_t element_count,
				void *destination,
				uint64_t destination_byte_count)
{
    if (capture == NULL || name == NULL || destination == NULL || (dimensions == NULL && dimension_count != 0)) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct test_capture_tensor *const tensor = test_capture_find(capture, name);
    const bool shape_agrees = tensor != NULL && dimension_count == tensor->dimension_count &&
                              (dimension_count == 0 ||
                               memcmp(dimensions, tensor->dimensions, dimension_count * sizeof(dimensions[0])) == 0);
    const char *const refusal =
        tensor == NULL ? "no record has the name"
        : (uint32_t)type >= TEST_CAPTURE_TYPE_COUNT || tensor->type != (uint32_t)type ? "another type was asked"
        : !shape_agrees ? "another shape was asked"
        : element_count == 0 || first_element > tensor->element_count ||
              element_count > tensor->element_count - first_element ? "the range asked is not inside it"
        : destination_byte_count != element_count * type_forms[type].delivered_bytes
              ? "the destination is not the size of the elements" : NULL;
    if (refusal != NULL) {
        char shape_text[SHAPE_TEXT_CAPACITY] = "";
        if (tensor != NULL) {
            shape_text_of(tensor, shape_text);
        }
	LOG_ERROR("test_capture_read: %s (%s %s): %s", name,
		  tensor != NULL ?
			  test_capture_type_name(
				  (enum test_capture_type)tensor->type) :
			  "absent",
		  shape_text, refusal);
	return STATUS_INVALID_ARGUMENT;
    }
    if (type == TEST_CAPTURE_BFLOAT16) {
        return read_bfloat16(capture, tensor, first_element, element_count, destination);
    }
    return read_file(capture->tensors_descriptor, TENSORS_FILE_NAME,
                     tensor->byte_offset + first_element * type_forms[type].stored_bytes, destination,
                     destination_byte_count);
}
