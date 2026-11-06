/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * capture_writer.h: writes a capture folder in the format capture_reader.h reads (that of reference_oracle.py's
 * CaptureWriter: tensors.bin and manifest.json), for values the engine computed, so that
 * tools/oracle/compare_engine_capture.py holds them to a capture of the reference. Test support: no part of the
 * engine.
 *
 * Contract:
 *   - A writer lays its records out first (test_capture_writer_add, one behind another from byte 0 in the order they
 *     are added), then makes the folder (test_capture_writer_open: a folder that exists is refused; tensors.bin is
 *     made at its whole size at once), then takes the values (test_capture_writer_put) and ends with the manifest
 *     (test_capture_writer_close), which says complete only when it is asked to, every record was put whole and the
 *     file holds what the records say. A writer is used by one thread.
 *   - A record: a name (unique, 1 to TEST_CAPTURE_WRITER_NAME_CAPACITY - 1 bytes of letters, digits, '.' and '_'),
 *     a type of capture_reader.h among FLOAT32, BFLOAT16 (stored as float32 with reference_dtype bfloat16) and INT64,
 *     1 to TEST_CAPTURE_WRITER_MOST_DIMENSIONS dimensions, a layer (or TEST_CAPTURE_NO_LAYER), a position for the
 *     manifest (or TEST_CAPTURE_WRITER_NO_POSITION), a description (printable, no quote or backslash: a constant
 *     the writer points to until close), and slab_count, which cuts its values into that many slabs of equal size (a
 *     record of every position: one a position). put writes one slab from values as a card holds them: a bfloat16
 *     value as its two bytes, any other as stored. A record's slabs are put in order, each once: a put of any other
 *     slab than its next (one put again, one skipped, one beyond the record) is refused, as is a record whose
 *     bfloat16 slab would hold more than TEST_CAPTURE_WRITER_MOST_SLAB_VALUES values.
 *   - Refusals are STATUS_INVALID_ARGUMENT with one error line (a NULL argument, a record against the rules
 *     above, more records than the capacity, an add after open, a put before it); a file that cannot be made or
 *     written is STATUS_INPUT_OUTPUT_ERROR. conditions and results are JSON objects, written as given.
 *     close frees the writer, also on failure, and accepts NULL; a writer whose folder was never made writes nothing.
 */
#ifndef TEST_CAPTURE_WRITER_H
#define TEST_CAPTURE_WRITER_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/status.h"
#include "capture_reader.h"

#define TEST_CAPTURE_WRITER_NAME_CAPACITY 96
#define TEST_CAPTURE_WRITER_MOST_DIMENSIONS 4u
#define TEST_CAPTURE_WRITER_NO_POSITION UINT32_MAX
#define TEST_CAPTURE_WRITER_MOST_SLAB_VALUES 20480u     /* the four streams of a row */

struct test_capture_writer;

MUST_CHECK result_status test_capture_writer_create(
	uint32_t record_capacity, struct test_capture_writer **writer_out);
/* index_out receives the record's index, from 0 in the order of the adds. */
MUST_CHECK result_status test_capture_writer_add(
	struct test_capture_writer *writer, const char *name,
	enum test_capture_type type, const uint64_t *dimensions,
	uint32_t dimension_count, uint64_t slab_count, uint32_t layer,
	uint32_t position, const char *description, uint32_t *index_out);
MUST_CHECK result_status test_capture_writer_open(
	struct test_capture_writer *writer, const char *folder);
MUST_CHECK result_status
test_capture_writer_put(struct test_capture_writer *writer, uint32_t index,
			uint64_t slab, const void *values);
MUST_CHECK result_status
test_capture_writer_close(struct test_capture_writer *writer, bool complete,
			  const char *conditions, const char *results);

#endif
