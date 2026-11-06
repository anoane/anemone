/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * capture_reader.h: reads a capture folder of the reference oracle (tools/oracle/reference_oracle.py, and every tool
 * that writes with its CaptureWriter): manifest.json and tensors.bin, so that a test feeds a kernel the reference's
 * own input and holds its output to the reference's. Test support: no part of the engine.
 *
 * Contract:
 *   - The format is the writer's. tensors.bin holds the tensors one after another, each its values in C order as
 *     the host stores them. manifest.json is one JSON object whose "complete" is true and whose "tensors" is an
 *     array of records in the order of the file, each an object with "name" (a string, unique), "dtype" (numpy's
 *     name: float16, float32, float64, int8, uint8, int16, int32, int64), "shape" (an array of whole numbers),
 *     "byte_offset" and "byte_count" (whole numbers), "layer" (a whole number or null) and, where numpy has no type
 *     for the reference's, "reference_dtype": "bfloat16" on float32 values (each exact), "float8_e4m3fn" or
 *     "float8_e8m0fnu" on their bytes as uint8. Every other member is left unread.
 *   - test_capture_open checks the whole manifest before it gives a capture. It refuses with one error line, as
 *     STATUS_CORRUPT_PACK (the status of a file that contradicts itself): a manifest that is not one object
 *     json.loads reads (source/json.h), or that holds one key twice in an object; "complete" that is not true;
 *     "tensors" that is no array, or a record that is no object, lacks a member or has one of another kind; a name
 *     that is empty, holds a NUL or is another record's; a dtype or reference_dtype this reader does not know, or
 *     the two not together as above; more than TEST_CAPTURE_MAXIMUM_DIMENSIONS dimensions; a byte count that is
 *     not the elements times the stored size of one, or above INT64_MAX; a byte offset that is not where the
 *     record before ends (0 for the first); a layer of UINT32_MAX or more; a tensors.bin whose size is not where
 *     the last record ends; a manifest above TEST_CAPTURE_MOST_MANIFEST_BYTES. A file that cannot be opened, sized
 *     or read, or is no regular file, is STATUS_INPUT_OUTPUT_ERROR; a missing argument, or a folder path
 *     that does not fit TEST_CAPTURE_PATH_CAPACITY with the file names, STATUS_INVALID_ARGUMENT.
 *   - A record's type is what the reference held: a bfloat16 tensor is TEST_CAPTURE_BFLOAT16, though the file
 *     stores float32. test_capture_read delivers the reference's own bytes, element_count elements from
 *     first_element on in C order: a bfloat16 value as its two bytes (the high half of its float32, whose low half
 *     must be zero: else STATUS_CORRUPT_PACK with one error line), any other type as stored. It refuses,
 *     with STATUS_INVALID_ARGUMENT and one error line, a name no record has, a type or a shape other than
 *     the record's, a range of no elements or not inside the tensor, and a destination_byte_count that is not
 *     element_count times test_capture_element_bytes of the type. A missing argument is INVALID_ARGUMENT.
 *   - A capture does not change once open; any number of threads may read it. Its records and names live until
 *     test_capture_close, which accepts NULL.
 */
#ifndef TEST_CAPTURE_READER_H
#define TEST_CAPTURE_READER_H

#include <stdint.h>

#include "anemone/status.h"

#define TEST_CAPTURE_MAXIMUM_DIMENSIONS 8
#define TEST_CAPTURE_NO_LAYER UINT32_MAX              /* the layer of a record whose layer is null */
#define TEST_CAPTURE_MOST_MANIFEST_BYTES 1073741824ull /* 1 GiB: a manifest is read whole into memory */
#define TEST_CAPTURE_PATH_CAPACITY 4096               /* bytes of the path of a file in the folder, its NUL included */

enum test_capture_type {
    TEST_CAPTURE_FLOAT16,
    TEST_CAPTURE_BFLOAT16,
    TEST_CAPTURE_FLOAT32,
    TEST_CAPTURE_FLOAT64,
    TEST_CAPTURE_FLOAT8_E4M3,
    TEST_CAPTURE_FLOAT8_E8M0,
    TEST_CAPTURE_INT8,
    TEST_CAPTURE_UINT8,
    TEST_CAPTURE_INT16,
    TEST_CAPTURE_INT32,
    TEST_CAPTURE_INT64,
    TEST_CAPTURE_TYPE_COUNT                           /* number of types; not a type */
};

struct test_capture_tensor {
    const char *name;                                 /* NUL-terminated */
    uint64_t byte_offset;                             /* in tensors.bin */
    uint64_t byte_count;                              /* as stored */
    uint64_t element_count;
    uint64_t dimensions[TEST_CAPTURE_MAXIMUM_DIMENSIONS];   /* the first dimension_count; the rest zero */
    uint32_t dimension_count;
    uint32_t type;                                    /* enum test_capture_type */
    uint32_t layer;                                   /* or TEST_CAPTURE_NO_LAYER */
    uint32_t reserved_zero;
};

struct test_capture;

/* On failure capture_out is untouched. */
MUST_CHECK result_status test_capture_open(const char *folder_path,
					   struct test_capture **capture_out);
void test_capture_close(struct test_capture *capture);

/* For an open capture only: the records in the order of the file; NULL past the last, or for a name no record has
 * (also for a NULL name). */
uint32_t test_capture_tensor_count(const struct test_capture *capture);
const struct test_capture_tensor *test_capture_tensor_at(const struct test_capture *capture, uint32_t tensor_index);
const struct test_capture_tensor *test_capture_find(const struct test_capture *capture, const char *name);

/* The bytes test_capture_read delivers for one element of that type, and its name as the manifest gives it (the
 * reference's dtype where the record names one); 0 and "" for a value outside the enumeration. */
uint32_t test_capture_element_bytes(enum test_capture_type type);
const char *test_capture_type_name(enum test_capture_type type);

/* dimensions holds dimension_count numbers: the whole shape the caller expects the record to have. */
MUST_CHECK result_status
test_capture_read(const struct test_capture *capture, const char *name,
		  enum test_capture_type type, const uint64_t *dimensions,
		  uint32_t dimension_count, uint64_t first_element,
		  uint64_t element_count, void *destination,
		  uint64_t destination_byte_count);

#endif
