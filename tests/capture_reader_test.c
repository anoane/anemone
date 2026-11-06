/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * capture_reader_test: tests/capture_reader.c against capture_reader.h, on capture folders it writes as the oracle's
 * writer does; and, given the paths of capture folders, on those.
 *
 *   - A folder with a record of every type (a 0-dimensional one, an empty one, one whose name has escapes, a
 *     bfloat16 one longer than a chunk) and members the reader leaves alone (NaN, nesting, facts) opens: the
 *     records in order, each found by its name, each read whole and in part as the reference held it.
 *   - Each rule of the header broken once in such a folder is refused with its status and one error line that names
 *     it; a folder that cannot be read is an input or output error; a bfloat16 value with bits in its low half is
 *     refused when it is read, and only then.
 *   - Each refusal of test_capture_read, with its line; the types' sizes and names.
 *   - No memory, and a read that fails, at every place (with the checking hooks); four threads reading at once.
 *   - With arguments, the paths of capture folders: each opens, and every tensor of it is read whole with its own
 *     type and shape (its bfloat16 values checked). The figures are printed; without them the test says that it
 *     read no real capture.
 * A failure is one line on the standard output: the case, and what was expected of it.
 */
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "anemone/hooks.h"
#include "log_expectation.h"
#include "anemone/status.h"
#include "capture_reader.h"

#define ROOT_TEMPLATE "/tmp/anemone_capture_reader_test_XXXXXX"
#define PATH_BYTES 256u                         /* of a folder the test writes */
#define FILE_PATH_BYTES 512u                    /* and of a file in it */
#define MANIFEST_CAPACITY 65536u
#define TENSORS_CAPACITY 65536u
#define CAPTURE_CAPACITY 4096u
#define NUMBER_TEXT_BYTES 32u
#define PATTERN_RECORD_STEP 41u
#define PATTERN_ELEMENT_STEP 37u
#define BFLOAT16_SHIFT 16u
#define FIRST_PART_ELEMENT 3u                   /* of the part of a record read alone */
#define PART_ELEMENTS 5u
#define THREAD_COUNT 4u
#define THREAD_ROUNDS 50u
#define MOST_FAULTED_CALLS 16u                  /* more than the allocations and reads of one open */
#define REAL_SLICE_BYTES (64u << 20u)           /* of a real tensor read at once */
#define MANY_KEYS 70000u                        /* in one object: more than the reader's scratch checks (16 B each) */
#define KEY_TEXT_BYTES 16u                      /* of one member "k00000": 0, of them */
#define NO_RECORD UINT32_MAX
#define NULL_LAYER TEST_CAPTURE_NO_LAYER          /* of a record whose layer the manifest writes as null */
#define FORMAT_ARGUMENT 1                       /* of append, for the compiler's check of its format */
#define FIRST_VALUE_ARGUMENT 2
#define COUNT_OF(table) (sizeof(table) / sizeof((table)[0]))

static int failures = 0;
static const char *context_text = "";
static char root_path[] = ROOT_TEMPLATE;
static char capture_path[] = "/tmp/anemone_capture_reader_test_log_XXXXXX";
static int capture_descriptor = -1;             /* the temporary file that stands in for the error stream */
static int error_stream_descriptor = -1;        /* a duplicate of descriptor 2, kept to put it back */
static uint32_t folder_count = 0;

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)printf("FAILED: %s: %s\n", context_text, what);
        failures++;
    }
}

static void capture_begin(void)
{
    expect(capture_descriptor >= 0 && fflush(stderr) == 0 && ftruncate(capture_descriptor, 0) == 0 &&
               lseek(capture_descriptor, 0, SEEK_SET) == 0 && dup2(capture_descriptor, STDERR_FILENO) == STDERR_FILENO,
           "the error stream is pointed at the temporary file");
}

/* Checks what was logged since capture_begin: nothing (words NULL) or one error line with the words. */
static void capture_end_expecting(const char *words, const char *what)
{
    char text[CAPTURE_CAPACITY];
    const bool is_open = capture_descriptor >= 0 && error_stream_descriptor >= 0 && fflush(stderr) == 0;
    const ssize_t byte_count = is_open ? pread(capture_descriptor, text, sizeof(text) - 1, 0) : -1;
    expect(is_open && dup2(error_stream_descriptor, STDERR_FILENO) == STDERR_FILENO && byte_count >= 0,
           "the temporary file is read back and the error stream restored");
    text[byte_count >= 0 ? (size_t)byte_count : 0] = '\0';
    const bool is_right = test_log_is_expected(text, words, true);
    if (!is_right) {
        (void)printf("FAILED: %s: %s: expected %s \"%s\", got \"%s\"\n", context_text, what,
                     words == NULL ? "no line" : "one error line with", words == NULL ? "" : words, text);
        failures++;
    }
}

/* ---- the folders the test writes ---- */

/* How the oracle's writer names each type: numpy's name, and the reference's where numpy has none. */
static const char *const numpy_names[TEST_CAPTURE_TYPE_COUNT] = {
    "float16", "float32", "float32", "float64", "uint8", "uint8", "int8", "uint8", "int16", "int32", "int64"};
static const char *const reference_names[TEST_CAPTURE_TYPE_COUNT] = {
    NULL, "bfloat16", NULL, NULL, "float8_e4m3fn", "float8_e8m0fnu", NULL, NULL, NULL, NULL, NULL};
static const uint32_t stored_bytes[TEST_CAPTURE_TYPE_COUNT] = {2, 4, 4, 8, 1, 1, 1, 1, 2, 4, 8};

struct test_record {
    const char *name_json;                      /* as the manifest writes it */
    const char *name;                           /* as a test asks for it */
    uint32_t type;
    uint32_t dimension_count;
    uint64_t dimensions[TEST_CAPTURE_MAXIMUM_DIMENSIONS];
    uint32_t layer;                             /* TEST_CAPTURE_NO_LAYER: null */
    const char *facts_json;                     /* members the reader leaves alone, behind the description */
};

enum good_record { RECORD_TOKEN_IDS, RECORD_STREAMS, RECORD_COLLAPSE, RECORD_CODES, RECORD_SCALES, RECORD_HALVES,
                   RECORD_DOUBLES, RECORD_SIGNED_BYTES, RECORD_BYTES, RECORD_SHORTS, RECORD_WORDS, RECORD_SCALAR,
                   RECORD_EMPTY, RECORD_ESCAPED, RECORD_LONG, GOOD_RECORD_COUNT };
static const struct test_record good_records[GOOD_RECORD_COUNT] = {
    [RECORD_TOKEN_IDS] = {"\"input.token_ids\"", "input.token_ids", TEST_CAPTURE_INT64, 2, {1, 5}, NULL_LAYER, ""},
    [RECORD_STREAMS] = {"\"layer_0.block.input\"", "layer_0.block.input", TEST_CAPTURE_BFLOAT16, 4, {1, 5, 4, 8}, 0,
                        ""},
    [RECORD_COLLAPSE] = {"\"layer_0.block.input_collapse_weights\"", "layer_0.block.input_collapse_weights",
                         TEST_CAPTURE_FLOAT32, 3, {1, 5, 4}, 0, ""},
    [RECORD_CODES] = {"\"layer_0.linear.attn.wq_a.quantized_input\"", "layer_0.linear.attn.wq_a.quantized_input",
                      TEST_CAPTURE_FLOAT8_E4M3, 3, {1, 5, 32}, 0,
                      ", \"weight\": \"layers.0.attn.wq_a.weight\", \"weight_dtype\": \"float8_e4m3fn\", "
                      "\"rows_of_the_call\": [0, 1], \"positions\": {\"first\": 3, \"ratio\": 1.5e-3}"},
    [RECORD_SCALES] = {"\"layer_0.linear.attn.wq_a.input_scales\"", "layer_0.linear.attn.wq_a.input_scales",
                       TEST_CAPTURE_FLOAT8_E8M0, 3, {1, 5, 1}, 0, ""},
    [RECORD_HALVES] = {"\"values.float16\"", "values.float16", TEST_CAPTURE_FLOAT16, 1, {3}, 39, ""},
    [RECORD_DOUBLES] = {"\"values.float64\"", "values.float64", TEST_CAPTURE_FLOAT64, 1, {2}, NULL_LAYER, ""},
    [RECORD_SIGNED_BYTES] = {"\"values.int8\"", "values.int8", TEST_CAPTURE_INT8, 1, {4}, NULL_LAYER, ""},
    [RECORD_BYTES] = {"\"values.uint8\"", "values.uint8", TEST_CAPTURE_UINT8, 1, {4}, NULL_LAYER, ""},
    [RECORD_SHORTS] = {"\"values.int16\"", "values.int16", TEST_CAPTURE_INT16, 2, {2, 1}, NULL_LAYER, ""},
    [RECORD_WORDS] = {"\"values.int32\"", "values.int32", TEST_CAPTURE_INT32, 1, {3}, 1, ""},
    [RECORD_SCALAR] = {"\"values.scalar\"", "values.scalar", TEST_CAPTURE_INT64, 0, {0}, NULL_LAYER, ""},
    [RECORD_EMPTY] = {"\"values.empty\"", "values.empty", TEST_CAPTURE_FLOAT32, 2, {0, 4}, NULL_LAYER, ""},
    [RECORD_ESCAPED] = {"\"a \\\"quoted\\\" name \\u00e9\\n\"", "a \"quoted\" name \xc3\xa9\n", TEST_CAPTURE_UINT8, 1,
                        {2}, 14, ""},
    [RECORD_LONG] = {"\"layer_20.attention.output\"", "layer_20.attention.output", TEST_CAPTURE_BFLOAT16, 2,
                     {3, 4000}, 20, ""},
};

/* One change of a good folder: a member of a record (or of the manifest) written otherwise, or left out (text
 * NULL); the whole manifest written otherwise; tensors.bin longer or shorter. */
enum change_place { CHANGE_NOTHING, CHANGE_NAME, CHANGE_DTYPE, CHANGE_SHAPE, CHANGE_OFFSET, CHANGE_COUNT, CHANGE_LAYER,
                    CHANGE_FACTS, CHANGE_COMPLETE, CHANGE_TENSORS, CHANGE_MANIFEST, CHANGE_FILE_LONGER,
                    CHANGE_FILE_SHORTER, CHANGE_LOW_HALF };
struct change {
    uint32_t place;                             /* enum change_place */
    uint32_t record_index;                      /* of a change of a record, else NO_RECORD */
    const char *text;
};

static char manifest_text[MANIFEST_CAPACITY];
static size_t manifest_length = 0;
static unsigned char tensors_bytes[TENSORS_CAPACITY];
static size_t tensors_length = 0;

static void append(const char *format, ...) __attribute__((format(printf, FORMAT_ARGUMENT, FIRST_VALUE_ARGUMENT)));
static void append(const char *format, ...)
{
    va_list values;
    va_start(values, format);
    const int written = vsnprintf(manifest_text + manifest_length, MANIFEST_CAPACITY - manifest_length, format, values);
    va_end(values);
    expect(written >= 0 && (size_t)written < MANIFEST_CAPACITY - manifest_length, "the manifest fits its buffer");
    manifest_length += written >= 0 && (size_t)written < MANIFEST_CAPACITY - manifest_length ? (size_t)written : 0;
}

/* A member, its key and its text, unless a change leaves it out or writes it otherwise; *is_first while the object
 * has no member yet. */
static void append_member(const struct change *change, uint32_t place, uint32_t record_index, bool *is_first,
                          const char *key, const char *text)
{
    const bool is_changed = change->place == place && change->record_index == record_index;
    if (!is_changed || change->text != NULL) {
        append("%s\"%s\": %s", *is_first ? "" : ", ", key, is_changed ? change->text : text);
        *is_first = false;
    }
}

static uint64_t element_count_of(const struct test_record *record)
{
    uint64_t element_count = 1;
    for (uint32_t dimension = 0; dimension < record->dimension_count; dimension++) {
        element_count *= record->dimensions[dimension];
    }
    return element_count;
}

/* The bfloat16 word, or the stored byte, of an element of a record: never the same twice in a record. */
static uint16_t pattern_word(uint32_t record_index, uint64_t element)
{
    return (uint16_t)((uint64_t)record_index * PATTERN_RECORD_STEP + element * PATTERN_ELEMENT_STEP + 1u);
}

/* Writes the good folder with one change into a new folder; its path in path_out. */
static void write_folder(const struct change *change, char path_out[PATH_BYTES])
{
    char path[FILE_PATH_BYTES];
    char number[NUMBER_TEXT_BYTES];
    uint64_t byte_offset = 0;
    manifest_length = 0;
    tensors_length = 0;
    (void)snprintf(path_out, PATH_BYTES, "%s/folder_%" PRIu32, root_path, folder_count++);
    expect(mkdir(path_out, S_IRWXU) == 0, "a folder is made");
    append("{\n \"complete\": %s", change->place == CHANGE_COMPLETE && change->text != NULL ? change->text : "true");
    if (change->place == CHANGE_COMPLETE && change->text == NULL) {
        manifest_length = 0;
        append("{\n \"other\": true");
    }
    append(",\n \"conditions\": {\"capture\": \"forward\", \"nested\": [[1, 2.5, null, false], {\"x\": \"\\u2019\"}]},"
           "\n \"results\": {\"seconds\": NaN, \"first\": -Infinity},\n \"tensors\": ");
    if (change->place == CHANGE_TENSORS) {
        append("%s", change->text);
    }
    const uint32_t record_count = change->place == CHANGE_TENSORS ? 0 : GOOD_RECORD_COUNT;
    for (uint32_t record_index = 0; record_index < record_count; record_index++) {
        const struct test_record *const record = &good_records[record_index];
        const uint32_t type = record->type;
        const uint64_t element_count = element_count_of(record);
        char shape[PATH_BYTES] = "[";
        for (uint32_t dimension = 0; dimension < record->dimension_count; dimension++) {
            const size_t length = strlen(shape);
            (void)snprintf(shape + length, sizeof(shape) - length, "%s%" PRIu64, dimension == 0 ? "" : ", ",
                           record->dimensions[dimension]);
        }
        (void)snprintf(shape + strlen(shape), sizeof(shape) - strlen(shape), "]");
        bool is_first = true;
        append("%s\n  {", record_index == 0 ? "[" : ",");
        append_member(change, CHANGE_NAME, record_index, &is_first, "name", record->name_json);
        (void)snprintf(number, sizeof(number), "\"%s\"", numpy_names[type]);
        append_member(change, CHANGE_DTYPE, record_index, &is_first, "dtype", number);
        append_member(change, CHANGE_SHAPE, record_index, &is_first, "shape", shape);
        (void)snprintf(number, sizeof(number), "%" PRIu64, byte_offset);
        append_member(change, CHANGE_OFFSET, record_index, &is_first, "byte_offset", number);
        (void)snprintf(number, sizeof(number), "%" PRIu64, element_count * stored_bytes[type]);
        append_member(change, CHANGE_COUNT, record_index, &is_first, "byte_count", number);
        (void)snprintf(number, sizeof(number), "%" PRIu32, record->layer);
        append_member(change, CHANGE_LAYER, record_index, &is_first, "layer",
                      record->layer == TEST_CAPTURE_NO_LAYER ? "null" : number);
        append(", \"description\": \"what the \\\"reference\\\" held\\n\\ttab\"");
        if (reference_names[type] != NULL) {
            append(", \"reference_dtype\": \"%s\"", reference_names[type]);
        }
        append("%s%s", record->facts_json,
               change->place == CHANGE_FACTS && change->record_index == record_index ? change->text : "");
        append("}");
        for (uint64_t element = 0; element < element_count; element++) {
            const uint16_t word = pattern_word(record_index, element);
            const bool spoils_low_half = change->place == CHANGE_LOW_HALF && change->record_index == record_index &&
                                         element == element_count - 1u;
            const uint32_t stored = ((uint32_t)word << BFLOAT16_SHIFT) | (spoils_low_half ? 1u : 0u);
            for (uint32_t byte = 0; byte < stored_bytes[type]; byte++) {
                const bool fits = tensors_length < TENSORS_CAPACITY;
                expect(fits, "the tensors fit their buffer");
                tensors_bytes[fits ? tensors_length : 0] =
                    (unsigned char)(type == TEST_CAPTURE_BFLOAT16 ? stored >> (byte * CHAR_BIT) : word + byte);
                tensors_length += fits ? 1u : 0u;
            }
        }
        byte_offset += element_count * stored_bytes[type];
    }
    append("%s\n}\n", change->place == CHANGE_TENSORS ? "" : "\n ]");
    if (change->place == CHANGE_MANIFEST) {
        manifest_length = 0;
        append("%s", change->text);
    }
    tensors_length += change->place == CHANGE_FILE_LONGER ? 1u : 0u;
    tensors_length -= change->place == CHANGE_FILE_SHORTER ? 1u : 0u;
    const unsigned char *const contents[] = {(const unsigned char *)manifest_text, tensors_bytes};
    const size_t lengths[] = {manifest_length, tensors_length};
    const char *const names[] = {"manifest.json", "tensors.bin"};
    for (uint32_t file = 0; file < COUNT_OF(names); file++) {
        (void)snprintf(path, sizeof(path), "%s/%s", path_out, names[file]);
        const int descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
        expect(descriptor >= 0 && write(descriptor, contents[file], lengths[file]) == (ssize_t)lengths[file] &&
                   close(descriptor) == 0, "a file of the folder is written");
    }
}

/* Removes a folder the test wrote and the two files in it. */
static void remove_folder(const char *folder)
{
    char path[FILE_PATH_BYTES];
    (void)snprintf(path, sizeof(path), "%s/manifest.json", folder);
    (void)unlink(path);
    (void)snprintf(path, sizeof(path), "%s/tensors.bin", folder);
    (void)unlink(path);
    expect(rmdir(folder) == 0, "a folder of the test is removed");
}

/* ---- cases ---- */

/* Reads a whole record of the good folder, or a part, and compares it with what was written. */
static bool reads_back(const struct test_capture *capture, uint32_t record_index, uint64_t first, uint64_t count)
{
    const struct test_record *const record = &good_records[record_index];
    const uint32_t element_bytes = test_capture_element_bytes((enum test_capture_type)record->type);
    unsigned char *const values = malloc(count * element_bytes + 1u);
    bool is_equal = values != NULL &&
		    test_capture_read(capture, record->name,
				      (enum test_capture_type)record->type,
				      record->dimensions,
				      record->dimension_count, first, count,
				      values,
				      count * element_bytes) == STATUS_OK;
    for (uint64_t element = 0; is_equal && element < count; element++) {
        const uint16_t word = pattern_word(record_index, first + element);
        for (uint32_t byte = 0; byte < element_bytes; byte++) {
            const unsigned char expected =
                (unsigned char)(record->type == TEST_CAPTURE_BFLOAT16 ? (uint32_t)word >> (byte * CHAR_BIT)
                                                                      : word + byte);
            is_equal = is_equal && values[element * element_bytes + byte] == expected;
        }
    }
    free(values);
    return is_equal;
}

static void test_good_folder(const char *folder)
{
    struct test_capture *capture = NULL;
    context_text = "a good folder";
    capture_begin();
    expect(test_capture_open(folder, &capture) == STATUS_OK && capture != NULL,
	   "it opens");
    capture_end_expecting(NULL, "no line");
    if (capture == NULL) {
        return;
    }
    expect(test_capture_tensor_count(capture) == GOOD_RECORD_COUNT &&
               test_capture_tensor_at(capture, GOOD_RECORD_COUNT) == NULL && test_capture_find(capture, NULL) == NULL &&
               test_capture_find(capture, "input.token_id") == NULL, "the count; nothing past it or of another name");
    for (uint32_t record_index = 0; record_index < GOOD_RECORD_COUNT; record_index++) {
        const struct test_record *const record = &good_records[record_index];
        const struct test_capture_tensor *const tensor = test_capture_tensor_at(capture, record_index);
        const uint64_t element_count = element_count_of(record);
        expect(tensor != NULL && test_capture_find(capture, record->name) == tensor &&
                   strcmp(tensor->name, record->name) == 0 && tensor->type == record->type &&
                   tensor->dimension_count == record->dimension_count &&
                   memcmp(tensor->dimensions, record->dimensions, sizeof(tensor->dimensions)) == 0 &&
                   tensor->element_count == element_count &&
                   tensor->byte_count == element_count * stored_bytes[record->type] && tensor->layer == record->layer,
               "a record in its place, found by its name, as it was written");
        expect(element_count == 0 || reads_back(capture, record_index, 0, element_count), "read whole");
        if (element_count >= FIRST_PART_ELEMENT + PART_ELEMENTS) {
            expect(reads_back(capture, record_index, FIRST_PART_ELEMENT, PART_ELEMENTS), "read in part");
        }
    }
    test_capture_close(capture);
}

/* A read of the good folder that must be refused with one line holding the words (none: no line). */
static void read_refused(const struct test_capture *capture, const char *name, enum test_capture_type type,
                         const uint64_t *dimensions, uint32_t dimension_count, uint64_t first, uint64_t count,
                         uint64_t byte_count, const char *words, const char *what)
{
    uint64_t values[PART_ELEMENTS + 1u] = {0};
    context_text = what;
    capture_begin();
    expect(test_capture_read(capture, name, type, dimensions, dimension_count,
			     first, count, byte_count == 0 ? NULL : values,
			     byte_count) == STATUS_INVALID_ARGUMENT,
	   "refused");
    capture_end_expecting(words, "its line");
}

static void test_read_refusals(const char *folder)
{
    struct test_capture *capture = NULL;
    const struct test_record *const words = &good_records[RECORD_WORDS];
    const uint64_t other_dimensions[] = {words->dimensions[0] + 1u};
    const uint64_t word_bytes = sizeof(int32_t);
    expect(test_capture_open(folder, &capture) == STATUS_OK,
	   "the good folder opens");
    read_refused(NULL, words->name, TEST_CAPTURE_INT32, words->dimensions, 1, 0, 1, word_bytes, NULL, "no capture");
    read_refused(capture, NULL, TEST_CAPTURE_INT32, words->dimensions, 1, 0, 1, word_bytes, NULL, "no name");
    read_refused(capture, words->name, TEST_CAPTURE_INT32, NULL, 1, 0, 1, word_bytes, NULL, "no dimensions");
    read_refused(capture, words->name, TEST_CAPTURE_INT32, words->dimensions, 1, 0, 1, 0, NULL, "no destination");
    read_refused(capture, "values.int33", TEST_CAPTURE_INT32, words->dimensions, 1, 0, 1, word_bytes,
                 "no record has the name", "a name no record has");
    read_refused(capture, words->name, TEST_CAPTURE_UINT8, words->dimensions, 1, 0, 1, 1, "another type",
                 "another type");
    read_refused(capture, words->name, TEST_CAPTURE_TYPE_COUNT, words->dimensions, 1, 0, 1, 1, "another type",
                 "a type outside the enumeration");
    read_refused(capture, good_records[RECORD_STREAMS].name, TEST_CAPTURE_FLOAT32,
                 good_records[RECORD_STREAMS].dimensions, good_records[RECORD_STREAMS].dimension_count, 0, 1,
                 sizeof(float), "(bfloat16 [1, 5, 4, 8]): another type", "bfloat16 asked as float32");
    read_refused(capture, words->name, TEST_CAPTURE_INT32, other_dimensions, 1, 0, 1, word_bytes, "another shape",
                 "another dimension");
    read_refused(capture, words->name, TEST_CAPTURE_INT32, words->dimensions, 0, 0, 1, word_bytes, "another shape",
                 "fewer dimensions");
    read_refused(capture, words->name, TEST_CAPTURE_INT32, words->dimensions, 1, 0, 0, word_bytes, "not inside it",
                 "no elements");
    read_refused(capture, words->name, TEST_CAPTURE_INT32, words->dimensions, 1, 1, words->dimensions[0], word_bytes,
                 "not inside it", "a range past the end");
    read_refused(capture, words->name, TEST_CAPTURE_INT32, words->dimensions, 1, UINT64_MAX, 1, word_bytes,
                 "not inside it", "a first element past the end");
    read_refused(capture, words->name, TEST_CAPTURE_INT32, words->dimensions, 1, 0, 1, word_bytes + 1u,
                 "not the size", "a destination of another size");
    context_text = "the types";
    expect(test_capture_element_bytes(TEST_CAPTURE_BFLOAT16) == sizeof(uint16_t) &&
               test_capture_element_bytes(TEST_CAPTURE_INT64) == sizeof(int64_t) &&
               test_capture_element_bytes(TEST_CAPTURE_TYPE_COUNT) == 0 &&
               strcmp(test_capture_type_name(TEST_CAPTURE_FLOAT8_E8M0), "float8_e8m0fnu") == 0 &&
               strcmp(test_capture_type_name(TEST_CAPTURE_INT16), "int16") == 0 &&
               strcmp(test_capture_type_name(TEST_CAPTURE_TYPE_COUNT), "") == 0, "sizes and names");
    test_capture_close(capture);
    test_capture_close(NULL);
}

/* The changes of a good folder that break a rule, each with its status and the words of its line. */
struct broken_case {
    struct change change;
    result_status status;
    const char *words;
};
static const struct broken_case broken_cases[] = {
	{ { CHANGE_COMPLETE, NO_RECORD, "false" },
	  STATUS_CORRUPT_PACK,
	  "does not say complete" },
	{ { CHANGE_COMPLETE, NO_RECORD, "1" },
	  STATUS_CORRUPT_PACK,
	  "does not say complete" },
	{ { CHANGE_COMPLETE, NO_RECORD, NULL },
	  STATUS_CORRUPT_PACK,
	  "does not say complete" },
	{ { CHANGE_TENSORS, NO_RECORD, "{}" },
	  STATUS_CORRUPT_PACK,
	  "are no array" },
	{ { CHANGE_TENSORS, NO_RECORD, "[1]" },
	  STATUS_CORRUPT_PACK,
	  "it is no object" },
	{ { CHANGE_MANIFEST, NO_RECORD,
	    "{\"complete\": true, \"tensors\": []" },
	  STATUS_CORRUPT_PACK,
	  "not one JSON object" },
	{ { CHANGE_MANIFEST, NO_RECORD, "[]" },
	  STATUS_CORRUPT_PACK,
	  "not one JSON object" },
	{ { CHANGE_MANIFEST, NO_RECORD, "" },
	  STATUS_CORRUPT_PACK,
	  "not one JSON object" },
	{ { CHANGE_FACTS, RECORD_BYTES, ", \"layer\": 3" },
	  STATUS_CORRUPT_PACK,
	  "holds a key twice" },
	{ { CHANGE_NAME, RECORD_BYTES, NULL },
	  STATUS_CORRUPT_PACK,
	  "record 8 breaks the rule: a member is missing" },
	{ { CHANGE_NAME, RECORD_BYTES, "8" },
	  STATUS_CORRUPT_PACK,
	  "a member is missing or of another kind" },
	{ { CHANGE_NAME, RECORD_BYTES, "\"\"" },
	  STATUS_CORRUPT_PACK,
	  "empty or holds a NUL" },
	{ { CHANGE_NAME, RECORD_BYTES, "\"a\\u0000b\"" },
	  STATUS_CORRUPT_PACK,
	  "empty or holds a NUL" },
	{ { CHANGE_NAME, RECORD_BYTES, "\"values.int8\"" },
	  STATUS_CORRUPT_PACK,
	  "two records are named values.int8" },
	{ { CHANGE_DTYPE, RECORD_BYTES, NULL },
	  STATUS_CORRUPT_PACK,
	  "a member is missing" },
	{ { CHANGE_DTYPE, RECORD_BYTES, "\"bool\"" },
	  STATUS_CORRUPT_PACK,
	  "no pair this reader knows" },
	{ { CHANGE_FACTS, RECORD_TOKEN_IDS,
	    ", \"reference_dtype\": \"bfloat16\"" },
	  STATUS_CORRUPT_PACK,
	  "no pair this reader knows" },
	{ { CHANGE_FACTS, RECORD_COLLAPSE, ", \"reference_dtype\": null" },
	  STATUS_CORRUPT_PACK,
	  "no pair this reader knows" },
	{ { CHANGE_FACTS, RECORD_COLLAPSE,
	    ", \"reference_dtype\": \"float8_e4m3fn\"" },
	  STATUS_CORRUPT_PACK,
	  "no pair this reader knows" },
	{ { CHANGE_SHAPE, RECORD_WORDS, NULL },
	  STATUS_CORRUPT_PACK,
	  "a member is missing" },
	{ { CHANGE_SHAPE, RECORD_WORDS, "3" },
	  STATUS_CORRUPT_PACK,
	  "a member is missing" },
	{ { CHANGE_SHAPE, RECORD_WORDS, "[1, 1, 1, 1, 1, 1, 1, 1, 3]" },
	  STATUS_CORRUPT_PACK,
	  "more dimensions" },
	{ { CHANGE_SHAPE, RECORD_WORDS, "[-3]" },
	  STATUS_CORRUPT_PACK,
	  "no whole number" },
	{ { CHANGE_SHAPE, RECORD_WORDS, "[3.0]" },
	  STATUS_CORRUPT_PACK,
	  "no whole number" },
	{ { CHANGE_SHAPE, RECORD_WORDS, "[4]" },
	  STATUS_CORRUPT_PACK,
	  "not its elements times" },
	{ { CHANGE_SHAPE, RECORD_WORDS, "[4294967296, 4294967296, 0, 3]" },
	  STATUS_CORRUPT_PACK,
	  "not its elements times" },
	{ { CHANGE_SHAPE, RECORD_WORDS, "[4294967296, 4294967296]" },
	  STATUS_CORRUPT_PACK,
	  "not its elements times" },
	/* (2^62 + 1) (2^62 + 3) is 3 modulo 2^64: a product that wrapped would hold the record's 12 bytes */
	{ { CHANGE_SHAPE, RECORD_WORDS,
	    "[4611686018427387905, 4611686018427387907]" },
	  STATUS_CORRUPT_PACK,
	  "not its elements times" },
	{ { CHANGE_COUNT, RECORD_WORDS, "13" },
	  STATUS_CORRUPT_PACK,
	  "not its elements times" },
	{ { CHANGE_COUNT, RECORD_WORDS, "12.0" },
	  STATUS_CORRUPT_PACK,
	  "no whole number" },
	{ { CHANGE_COUNT, RECORD_WORDS, "9223372036854775808" },
	  STATUS_CORRUPT_PACK,
	  "no whole number" },
	{ { CHANGE_COUNT, RECORD_WORDS, NULL },
	  STATUS_CORRUPT_PACK,
	  "no whole number" },
	{ { CHANGE_OFFSET, RECORD_WORDS, "0" },
	  STATUS_CORRUPT_PACK,
	  "does not begin where" },
	{ { CHANGE_OFFSET, RECORD_WORDS, "\"0\"" },
	  STATUS_CORRUPT_PACK,
	  "no whole number" },
	{ { CHANGE_LAYER, RECORD_WORDS, NULL },
	  STATUS_CORRUPT_PACK,
	  "a member is missing" },
	{ { CHANGE_LAYER, RECORD_WORDS, "-1" },
	  STATUS_CORRUPT_PACK,
	  "neither null nor" },
	{ { CHANGE_LAYER, RECORD_WORDS, "4294967295" },
	  STATUS_CORRUPT_PACK,
	  "neither null nor" },
	{ { CHANGE_LAYER, RECORD_WORDS, "\"1\"" },
	  STATUS_CORRUPT_PACK,
	  "neither null nor" },
	{ { CHANGE_FILE_LONGER, NO_RECORD, "" },
	  STATUS_CORRUPT_PACK,
	  "tensors.bin holds" },
	{ { CHANGE_FILE_SHORTER, NO_RECORD, "" },
	  STATUS_CORRUPT_PACK,
	  "tensors.bin holds" },
};

static void test_broken_folders(void)
{
    for (uint32_t case_index = 0; case_index < COUNT_OF(broken_cases); case_index++) {
        const struct broken_case *const broken = &broken_cases[case_index];
        struct test_capture *capture = NULL;
        char folder[PATH_BYTES];
        char what[PATH_BYTES];
        (void)snprintf(what, sizeof(what), "broken folder %" PRIu32 " (%s)", case_index, broken->words);
        context_text = what;
        write_folder(&broken->change, folder);
        capture_begin();
        expect(test_capture_open(folder, &capture) == broken->status && capture == NULL, "refused, nothing given");
        capture_end_expecting(broken->words, "the rule broken is named");
        remove_folder(folder);
    }
}

static void test_unreadable_folders(void)
{
    static const struct change no_change = {CHANGE_NOTHING, NO_RECORD, NULL};
    struct test_capture *capture = NULL;
    char folder[PATH_BYTES];
    char path[FILE_PATH_BYTES];
    char long_path[TEST_CAPTURE_PATH_CAPACITY];
    context_text = "folders that cannot be read";
    (void)snprintf(path, sizeof(path), "%s/folder_that_is_not", root_path);
    capture_begin();
    expect(test_capture_open(path, &capture) == STATUS_INPUT_OUTPUT_ERROR,
	   "a folder that does not exist");
    capture_end_expecting("manifest.json is no regular file that opens", "its line");
    write_folder(&no_change, folder);
    (void)snprintf(path, sizeof(path), "%s/tensors.bin", folder);
    expect(unlink(path) == 0, "tensors.bin is removed");
    capture_begin();
    expect(test_capture_open(folder, &capture) == STATUS_INPUT_OUTPUT_ERROR,
	   "a folder without tensors.bin");
    capture_end_expecting("tensors.bin is no regular file that opens", "its line");
    (void)snprintf(path, sizeof(path), "%s/manifest.json", folder);
    expect(unlink(path) == 0 && mkdir(path, S_IRWXU) == 0, "manifest.json is made a folder");
    capture_begin();
    expect(test_capture_open(folder, &capture) == STATUS_INPUT_OUTPUT_ERROR,
	   "a manifest that is a folder");
    capture_end_expecting("it is something else", "its line");
    expect(rmdir(path) == 0, "the folder of the manifest is removed");
    const int descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
    expect(descriptor >= 0 && ftruncate(descriptor, (off_t)TEST_CAPTURE_MOST_MANIFEST_BYTES + 1) == 0 &&
               close(descriptor) == 0, "a manifest one byte above the limit, with no bytes on the drive");
    capture_begin();
    expect(test_capture_open(folder, &capture) == STATUS_CORRUPT_PACK,
	   "a manifest above the limit");
    capture_end_expecting("above the limit", "its line");
    remove_folder(folder);
    memset(long_path, 'p', sizeof(long_path) - 1);
    long_path[sizeof(long_path) - 1] = '\0';
    capture_begin();
    expect(test_capture_open(long_path, &capture) == STATUS_INVALID_ARGUMENT &&
		   test_capture_open(NULL, &capture) ==
			   STATUS_INVALID_ARGUMENT &&
		   test_capture_open(root_path, NULL) ==
			   STATUS_INVALID_ARGUMENT &&
		   capture == NULL,
	   "a path too long, no path, nowhere to put the capture");
    capture_end_expecting(NULL, "no line");
}

/* A manifest whose one object holds more keys than the search for a key twice has room for. */
static void test_many_keys(void)
{
    static const struct change no_change = {CHANGE_NOTHING, NO_RECORD, NULL};
    struct test_capture *capture = NULL;
    char folder[PATH_BYTES];
    char path[FILE_PATH_BYTES];
    size_t length = 0;
    context_text = "a manifest with too many keys at one place";
    write_folder(&no_change, folder);
    char *const text = malloc((size_t)MANY_KEYS * KEY_TEXT_BYTES + manifest_length);
    expect(text != NULL, "memory for the manifest");
    if (text == NULL) {
        remove_folder(folder);
        return;
    }
    const char *const last_brace = strrchr(manifest_text, '}');     /* the good manifest up to its last brace */
    length = last_brace != NULL ? (size_t)(last_brace - manifest_text) : 0;
    memcpy(text, manifest_text, length);
    for (uint32_t key = 0; key < MANY_KEYS; key++) {
        const int written = snprintf(text + length, KEY_TEXT_BYTES, ", \"k%05" PRIu32 "\": 0", key);
        length += written > 0 && (size_t)written < KEY_TEXT_BYTES ? (size_t)written : 0;
    }
    text[length] = '}';
    length++;
    (void)snprintf(path, sizeof(path), "%s/manifest.json", folder);
    const int descriptor = open(path, O_WRONLY | O_TRUNC | O_CLOEXEC);
    expect(descriptor >= 0 && write(descriptor, text, length) == (ssize_t)length && close(descriptor) == 0,
           "the manifest is written again");
    free(text);
    capture_begin();
    expect(test_capture_open(folder, &capture) == STATUS_CORRUPT_PACK,
	   "refused");
    capture_end_expecting("more keys open at one place", "its line");
    remove_folder(folder);
}

static void test_low_half(void)
{
    static const struct change spoiled = {CHANGE_LOW_HALF, RECORD_LONG, NULL};
    const struct test_record *const record = &good_records[RECORD_LONG];
    const uint64_t element_count = element_count_of(record);
    struct test_capture *capture = NULL;
    char folder[PATH_BYTES];
    context_text = "a bfloat16 value with a bit in its low half";
    write_folder(&spoiled, folder);
    uint16_t *const words = calloc(element_count, sizeof(*words));
    expect(words != NULL && test_capture_open(folder, &capture) == STATUS_OK,
	   "the folder opens");
    capture_begin();
    expect(words != NULL &&
		   test_capture_read(capture, record->name,
				     TEST_CAPTURE_BFLOAT16, record->dimensions,
				     record->dimension_count, 0, element_count,
				     words, element_count * sizeof(*words)) ==
			   STATUS_CORRUPT_PACK,
	   "the read that covers it is refused");
    capture_end_expecting("value 11999 of layer_20.attention.output is no bfloat16 value", "its line");
    expect(reads_back(capture, RECORD_LONG, 0, element_count - 1u), "the read of the values before it is not");
    free(words);
    test_capture_close(capture);
    remove_folder(folder);
}

static void test_faults(const char *folder)
{
#if CHECKING_HOOKS_ENABLED
	struct test_capture *capture = NULL;
	uint32_t value = 0;
	context_text = "no memory";
	for (uint32_t call = 1; call <= MOST_FAULTED_CALLS; call++) {
		expect(fault_arm("test_capture_memory", call),
		       "the fault of the memory is armed");
		capture_begin();
		const result_status status =
			test_capture_open(folder, &capture);
		capture_end_expecting(NULL, "no line");
		if (fault_disarm() < call) {
			expect(status == STATUS_OK && call > 1,
			       "past the last allocation, it opens");
			break;
		}
		expect(status == STATUS_OUT_OF_MEMORY && capture == NULL,
		       "each allocation that fails");
	}
	test_capture_close(capture);
	capture = NULL;
	context_text = "a read that fails";
	expect(fault_arm("test_capture_file_read", 1),
	       "the fault of a read is armed");
	capture_begin();
	expect(test_capture_open(folder, &capture) ==
			       STATUS_INPUT_OUTPUT_ERROR &&
		       capture == NULL,
	       "the read of the manifest");
	capture_end_expecting("of manifest.json failed", "its line");
	expect(fault_disarm() == 1 &&
		       test_capture_open(folder, &capture) == STATUS_OK &&
		       fault_arm("test_capture_file_read", 1),
	       "the folder opens, the fault is armed again");
	capture_begin();
	expect(test_capture_read(capture, good_records[RECORD_WORDS].name,
				 TEST_CAPTURE_INT32,
				 good_records[RECORD_WORDS].dimensions, 1, 0, 1,
				 &value,
				 sizeof(value)) == STATUS_INPUT_OUTPUT_ERROR,
	       "the read of a tensor");
	capture_end_expecting("of tensors.bin failed", "its line");
	expect(fault_disarm() == 1, "the fault point of a read was reached");
	test_capture_close(capture);
#else
    (void)folder;
#endif
}

struct thread_case {
    const struct test_capture *capture;
    bool failed;
};

static void *read_everything(void *argument)
{
    struct thread_case *const thread_case = argument;
    for (uint32_t round = 0; round < THREAD_ROUNDS; round++) {
        for (uint32_t record_index = 0; record_index < GOOD_RECORD_COUNT; record_index++) {
            const uint64_t element_count = element_count_of(&good_records[record_index]);
            thread_case->failed = thread_case->failed ||
                                  (element_count > 0 &&
                                   !reads_back(thread_case->capture, record_index, 0, element_count));
        }
    }
    return NULL;
}

static void test_threads(const char *folder)
{
    struct test_capture *capture = NULL;
    struct thread_case cases[THREAD_COUNT];
    pthread_t threads[THREAD_COUNT];
    context_text = "four threads reading one capture";
    expect(test_capture_open(folder, &capture) == STATUS_OK, "it opens");
    for (uint32_t index = 0; index < THREAD_COUNT; index++) {
        cases[index] = (struct thread_case){capture, false};
        expect(pthread_create(&threads[index], NULL, read_everything, &cases[index]) == 0, "a thread starts");
    }
    for (uint32_t index = 0; index < THREAD_COUNT; index++) {
        expect(pthread_join(threads[index], NULL) == 0 && !cases[index].failed, "a thread read every record back");
    }
    test_capture_close(capture);
}

/* Every tensor of a real capture folder, read whole with its own type and shape, in slices of REAL_SLICE_BYTES. */
static void test_real_folder(const char *folder, unsigned char *slice)
{
    struct test_capture *capture = NULL;
    uint64_t stored_byte_count = 0;
    const char *const last_slash = strrchr(folder, '/');
    context_text = "a real capture folder";
    expect(test_capture_open(folder, &capture) == STATUS_OK, "it opens");
    for (uint32_t tensor_index = 0; tensor_index < test_capture_tensor_count(capture); tensor_index++) {
        const struct test_capture_tensor *const tensor = test_capture_tensor_at(capture, tensor_index);
        const enum test_capture_type type = (enum test_capture_type)tensor->type;
        const uint64_t slice_elements = REAL_SLICE_BYTES / test_capture_element_bytes(type);
        for (uint64_t first = 0; first < tensor->element_count; first += slice_elements) {
            const uint64_t count = tensor->element_count - first < slice_elements ? tensor->element_count - first
                                                                                 : slice_elements;
	    expect(test_capture_read(
			   capture, tensor->name, type, tensor->dimensions,
			   tensor->dimension_count, first, count, slice,
			   count * test_capture_element_bytes(type)) ==
			   STATUS_OK,
		   tensor->name);
	}
        stored_byte_count += tensor->byte_count;
    }
    (void)printf("capture %s: %" PRIu32 " tensors, %" PRIu64 " bytes, every tensor read with its type and shape\n",
                 last_slash != NULL ? last_slash + 1 : folder, test_capture_tensor_count(capture), stored_byte_count);
    test_capture_close(capture);
}

int main(int argument_count, char **arguments)
{
    static const struct change no_change = {CHANGE_NOTHING, NO_RECORD, NULL};
    char good_folder[PATH_BYTES];
    error_stream_descriptor = dup(STDERR_FILENO);
    capture_descriptor = mkstemp(capture_path);
    if (error_stream_descriptor < 0 || capture_descriptor < 0 || mkdtemp(root_path) == NULL) {
        (void)printf("FAILED: the temporary file or folder could not be made: %s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    write_folder(&no_change, good_folder);
    test_good_folder(good_folder);
    test_read_refusals(good_folder);
    test_broken_folders();
    test_unreadable_folders();
    test_many_keys();
    test_low_half();
    test_faults(good_folder);
    test_threads(good_folder);
    remove_folder(good_folder);
    context_text = "the end of the test";
    expect(rmdir(root_path) == 0 && close(capture_descriptor) == 0 && unlink(capture_path) == 0 &&
               close(error_stream_descriptor) == 0, "the temporary folder and file are removed");
    unsigned char *const slice = argument_count > 1 ? malloc(REAL_SLICE_BYTES) : NULL;
    expect(argument_count == 1 || slice != NULL, "memory for a slice of a real tensor");
    for (int argument = 1; slice != NULL && argument < argument_count; argument++) {
        test_real_folder(arguments[argument], slice);
    }
    free(slice);
    if (argument_count == 1) {
        (void)printf("capture_reader_test: no folder was given, no real capture was read\n");
    }
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("capture_reader_test passed (%" PRIu32 " folders written)\n", folder_count);
    return EXIT_SUCCESS;
}
