/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * storage_seam.h: where the bytes of a pack come from.
 *
 * Contract:
 *   - A reader gives positioned reads of ONE pack that exists as identical copies on one or more drives. The
 *     caller picks the copy for each read (it knows what each drive delivers); every copy returns the same bytes.
 *     copy_index is the index into the copy_paths given to reader_open.
 *   - reader_open opens every copy or nothing: when it refuses, nothing is left open and reader_out is untouched.
 *     It refuses a NULL argument or path and a copy_count of 0 or above STORAGE_MAXIMUM_COPIES
 *     (STATUS_INVALID_ARGUMENT); a copy it cannot open or size (STATUS_INPUT_OUTPUT_ERROR);
 *     copies of different sizes, and a size that is zero or no multiple of the block alignment
 *     (STATUS_CORRUPT_PACK); a copy whose read bound is above STORAGE_READ_LIMIT_SECONDS
 *     (STATUS_UNSUPPORTED); a lack of memory (STATUS_OUT_OF_MEMORY). It keeps no pointer into
 *     copy_paths. Loading from fewer copies is another reader_open with fewer paths.
 *   - A read is synchronous and holds no cursor, so any number of threads may read through one reader at once.
 *   - The byte offset, the byte count and the destination address of a read are multiples of the reader's block
 *     alignment, which is the largest of its copies'; the pack's extent alignment must be a multiple of it (the
 *     loader checks). A read delivers every byte asked for, or fails; it never returns a short count. It refuses
 *     (STATUS_INVALID_ARGUMENT) a count of zero, a range outside the pack, a value off the alignment and
 *     a copy_index at or above copy_count. A failed read leaves the reader usable, so the range can be read from
 *     another copy, and the destination bytes unspecified.
 *   - A read is the plain positioned read and takes no limit of its own. Its bound is the kernel's: the drive's
 *     request timeout times its attempts, for each request a read is cut into. reader_open reads that figure
 *     for every copy and refuses a copy above STORAGE_READ_LIMIT_SECONDS. A copy whose drive states no
 *     such figure (it is no NVMe drive, or the file system hides the drive) is accepted with a warning that says
 *     so: the engine knows no bound for its reads.
 *   - Also refused: a NULL reader, destination or output (STATUS_INVALID_ARGUMENT, by every function
 *     that returns a status); a copy that is no regular file (STATUS_INPUT_OUTPUT_ERROR); a block
 *     alignment that is no power of two and a size no off_t can hold (STATUS_CORRUPT_PACK).
 *   - reader_close accepts NULL and is called with no read in flight.
 *   - The table is bound once at load. The loader is its first user; nothing calls through it per token once the
 *     n-gram tables are in host memory. On a machine whose memory cannot hold the tables their rows come from
 *     the drives while the model serves: the row reader then issues ONE batched read per token, with a wait
 *     limit, through reader_read_many, which joins this table with piece 2 (the n-gram rows).
 * Today's only implementation is direct reads on Linux; another one is another table.
 */
#ifndef STORAGE_SEAM_H
#define STORAGE_SEAM_H

#include <stdint.h>

#include "anemone/status.h"

#define STORAGE_MAXIMUM_COPIES 8
/* What a stock Linux NVMe queue allows one read: its request timeout times its attempts. */
#define STORAGE_READ_LIMIT_SECONDS 180

struct storage_description {
	uint64_t byte_count; /* the size of the pack: the same on every copy */
	uint32_t copy_count; /* as given to reader_open */
	uint32_t block_alignment_bytes; /* the largest of the copies'; a power of two */
};

struct storage_reader;

struct storage_operations_record {
	const char *name;
	MUST_CHECK result_status (*reader_open)(
		const char *const *copy_paths, uint32_t copy_count,
		struct storage_reader **reader_out);
	void (*reader_close)(struct storage_reader *reader);
	MUST_CHECK result_status (*reader_describe)(
		const struct storage_reader *reader,
		struct storage_description *description_out);
	MUST_CHECK result_status (*reader_read)(
		const struct storage_reader *reader, uint32_t copy_index,
		uint64_t byte_offset, void *destination, uint64_t byte_count);
};

extern const struct storage_operations_record storage_direct;

#endif
