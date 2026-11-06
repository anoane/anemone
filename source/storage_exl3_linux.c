/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Separate load-time storage for indexed EXL3 plus unchanged native extents. */
#include "anemone/storage_exl3.h"

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include "anemone/hooks.h"
#include "storage_file_bounds.h"

#define MAP_MAGIC_BYTES 8u
#define MAP_RESERVED_BYTES 40u
#define READ_PIECE_BYTES 1048576u
#define DIRECT_PIECE_BYTES \
	(64ull             \
	 << 20u) /* one direct read: the kernel cuts it into the drive's own commands */
#define BINARY_SEARCH_RADIX 2u
#define MILLISECONDS_PER_SECOND 1000ull

struct map_header {
	unsigned char magic[MAP_MAGIC_BYTES];
	uint64_t byte_count;
	uint32_t source_count;
	uint32_t segment_count;
	unsigned char reserved[MAP_RESERVED_BYTES];
};
struct map_segment {
	uint64_t byte_offset;
	uint64_t byte_count;
	uint64_t source_offset;
	uint32_t source_index;
	uint32_t reserved_zero;
};
struct storage_reader {
	struct map_header header;
	int sources[EXL3_MAP_MAXIMUM_SOURCES];
	struct map_segment *segments;
	struct storage_reader *flat; /* copies 1..: a direct reader, or NULL */
	struct storage_description flat_description;
	/* each source opened once more with O_DIRECT (-1: not), and the alignment its direct reads need */
	int direct_sources[EXL3_MAP_MAXIMUM_SOURCES];
	uint32_t direct_alignment[EXL3_MAP_MAXIMUM_SOURCES];
};

static enum exl3_read_mode read_mode = EXL3_READS_DIRECT;

void storage_exl3_read_mode(enum exl3_read_mode mode)
{
	read_mode = mode < EXL3_READ_MODES ? mode : EXL3_READS_BUFFERED;
}
_Static_assert(sizeof(struct map_header) == EXL3_MAP_HEADER_BYTES,
	       "map header ABI");
_Static_assert(sizeof(struct map_segment) == EXL3_MAP_SEGMENT_BYTES,
	       "map segment ABI");

static bool read_pieces(int descriptor, uint64_t offset, void *destination,
			uint64_t count, uint64_t piece_most)
{
	unsigned char *bytes = destination;
	while (count != 0) {
		const size_t piece = count > piece_most ? (size_t)piece_most :
							  (size_t)count;
		const ssize_t got =
			pread(descriptor, bytes, piece, (off_t)offset);
		if (got < 0 && errno == EINTR) {
			continue;
		}
		if (got <= 0) {
			return false;
		}
		const uint64_t delivered = (uint64_t)got;
		count -= delivered;
		offset += delivered;
		bytes += delivered;
	}
	return true;
}

static bool read_exact(int descriptor, uint64_t offset, void *destination,
		       uint64_t count)
{
	return read_pieces(descriptor, offset, destination, count,
			   READ_PIECE_BYTES);
}

/* The source once more for direct reads, with the larger of the alignments statx states for them (else a block). */
static void direct_open(struct storage_reader *reader, uint32_t source,
			const char *path)
{
	reader->direct_sources[source] =
		read_mode == EXL3_READS_DIRECT ||
				read_mode == EXL3_READS_DIRECT_DROPPED ?
			open(path,
			     O_RDONLY | O_DIRECT | O_CLOEXEC | O_NONBLOCK) :
			-1;
	struct statx status = { .stx_mask = 0 };
	const bool stated = reader->direct_sources[source] >= 0 &&
			    statx(reader->direct_sources[source], "",
				  AT_EMPTY_PATH, STATX_DIOALIGN,
				  &status) == 0 &&
			    (status.stx_mask & STATX_DIOALIGN) != 0 &&
			    status.stx_dio_mem_align != 0 &&
			    status.stx_dio_offset_align != 0;
	reader->direct_alignment[source] =
		!stated ? EXL3_MAP_BLOCK_BYTES :
		status.stx_dio_mem_align > status.stx_dio_offset_align ?
			  status.stx_dio_mem_align :
			  status.stx_dio_offset_align;
}

static void map_close(struct storage_reader *reader)
{
	if (reader != NULL) {
		for (uint32_t source = 0; source < EXL3_MAP_MAXIMUM_SOURCES;
		     source++) {
			if (reader->sources[source] >= 0) {
				(void)close(reader->sources[source]);
			}
			if (reader->direct_sources[source] >= 0) {
				(void)close(reader->direct_sources[source]);
			}
		}
		free(reader->segments);
		storage_direct.reader_close(reader->flat);
		free(reader);
	}
}

MUST_CHECK static result_status map_open(const char *const *paths,
					 uint32_t copy_count,
					 struct storage_reader **reader_out)
{
	static const unsigned char magic[MAP_MAGIC_BYTES] = { 'A', 'N', 'E',
							      'X', 'L', '3',
							      0,   0 };
	if (paths == NULL || copy_count == 0 ||
	    copy_count > STORAGE_MAXIMUM_COPIES || paths[0] == NULL ||
	    reader_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	struct storage_reader *const reader =
		FAULT_POINT(exl3_map_memory) ? NULL :
					       calloc(1, sizeof(*reader));
	if (reader == NULL) {
		return STATUS_OUT_OF_MEMORY;
	}
	for (uint32_t source = 0; source < EXL3_MAP_MAXIMUM_SOURCES; source++) {
		reader->sources[source] = -1;
		reader->direct_sources[source] = -1;
	}
	reader->sources[0] =
		FAULT_POINT(exl3_map_open) ?
			-1 :
			open(paths[0], O_RDONLY | O_CLOEXEC | O_NONBLOCK);
	result_status status = STATUS_CORRUPT_PACK;
	if (reader->sources[0] < 0 ||
	    !read_exact(reader->sources[0], 0, &reader->header,
			sizeof(reader->header))) {
		status = STATUS_INPUT_OUTPUT_ERROR;
		goto refused;
	}
	const struct map_header *const header = &reader->header;
	if (memcmp(header->magic, magic, sizeof(magic)) != 0 ||
	    header->byte_count == 0 || header->byte_count > INT64_MAX ||
	    header->byte_count % EXL3_MAP_BLOCK_BYTES != 0 ||
	    header->source_count == 0 ||
	    header->source_count > EXL3_MAP_MAXIMUM_SOURCES ||
	    header->segment_count == 0 ||
	    header->segment_count > EXL3_MAP_MAXIMUM_SEGMENTS) {
		goto refused;
	}
	for (uint32_t byte = 0; byte < MAP_RESERVED_BYTES; byte++) {
		if (header->reserved[byte] != 0) {
			goto refused;
		}
	}
	uint64_t sizes[EXL3_MAP_MAXIMUM_SOURCES] = { 0 };
	for (uint32_t source = 0; source < header->source_count; source++) {
		char path[EXL3_MAP_PATH_BYTES];
		if (!read_exact(reader->sources[0],
				sizeof(*header) +
					(uint64_t)source * sizeof(path),
				path, sizeof(path)) ||
		    memchr(path, 0, sizeof(path)) == NULL ||
		    (source == 0 ? path[0] != 0 : path[0] != '/')) {
			goto refused;
		}
		if (source != 0) {
			reader->sources[source] =
				open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
		}
		struct stat metadata;
		if (reader->sources[source] < 0 ||
		    fstat(reader->sources[source], &metadata) != 0 ||
		    !S_ISREG(metadata.st_mode) || metadata.st_size <= 0) {
			status = STATUS_INPUT_OUTPUT_ERROR;
			goto refused;
		}
		const uint64_t bound =
			FAULT_POINT(exl3_map_read_bound) ?
				UINT64_MAX :
				storage_drive_read_bound_milliseconds(
					(uint32_t)major(metadata.st_dev),
					(uint32_t)minor(metadata.st_dev));
		if (bound >
		    STORAGE_READ_LIMIT_SECONDS * MILLISECONDS_PER_SECOND) {
			status = STATUS_UNSUPPORTED;
			goto refused;
		}
		if (bound == 0) {
			LOG_WARNING_ENTRY(
				"EXL3 source %u: the filesystem states no read bound",
				source);
		}
		sizes[source] = (uint64_t)metadata.st_size;
		direct_open(reader, source, source == 0 ? paths[0] : path);
	}
	const uint64_t segment_bytes =
		(uint64_t)header->segment_count * sizeof(struct map_segment);
	reader->segments = FAULT_POINT(exl3_map_memory) ?
				   NULL :
				   malloc((size_t)segment_bytes);
	if (reader->segments == NULL) {
		status = STATUS_OUT_OF_MEMORY;
		goto refused;
	}
	const uint64_t segments_offset =
		sizeof(*header) +
		(uint64_t)header->source_count * EXL3_MAP_PATH_BYTES;
	if (!read_exact(reader->sources[0], segments_offset, reader->segments,
			segment_bytes)) {
		goto refused;
	}
	uint64_t previous_end = 0;
	for (uint32_t index = 0; index < header->segment_count; index++) {
		const struct map_segment *const segment =
			&reader->segments[index];
		if (segment->reserved_zero != 0 ||
		    segment->source_index >= header->source_count ||
		    segment->byte_count == 0 ||
		    segment->byte_offset < previous_end ||
		    segment->byte_offset > header->byte_count ||
		    segment->byte_count >
			    header->byte_count - segment->byte_offset ||
		    segment->source_offset > sizes[segment->source_index] ||
		    segment->byte_count > sizes[segment->source_index] -
						  segment->source_offset ||
		    (segment->source_index == 0 &&
		     segment->source_offset <
			     segments_offset + segment_bytes)) {
			goto refused;
		}
		previous_end = segment->byte_offset + segment->byte_count;
	}
	/* the flat copies hold the logical bytes: as many of them, in whole blocks of the map's */
	if (copy_count > 1) {
		status = storage_direct.reader_open(paths + 1, copy_count - 1,
						    &reader->flat);
		if (status != STATUS_OK ||
		    storage_direct.reader_describe(reader->flat,
						   &reader->flat_description) !=
			    STATUS_OK ||
		    reader->flat_description.byte_count != header->byte_count ||
		    EXL3_MAP_BLOCK_BYTES % reader->flat_description
						   .block_alignment_bytes !=
			    0) {
			status = status == STATUS_OK ? STATUS_CORRUPT_PACK :
						       status;
			goto refused;
		}
	}
	*reader_out = reader;
	return STATUS_OK;
refused:
	map_close(reader);
	return status;
}

MUST_CHECK static result_status
map_describe(const struct storage_reader *reader,
	     struct storage_description *description_out)
{
	if (reader == NULL || description_out == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
	*description_out = (struct storage_description){
		reader->header.byte_count,
		1u + (reader->flat != NULL ?
			      reader->flat_description.copy_count :
			      0u),
		EXL3_MAP_BLOCK_BYTES
	};
	return STATUS_OK;
}

MUST_CHECK static result_status map_read(const struct storage_reader *reader,
					 uint32_t copy_index, uint64_t offset,
					 void *destination, uint64_t count)
{
	if (reader != NULL && copy_index != 0 && reader->flat != NULL)
		return storage_direct.reader_read(reader->flat, copy_index - 1u,
						  offset, destination, count);
	if (reader == NULL || destination == NULL || copy_index != 0 ||
	    count == 0 || offset > reader->header.byte_count ||
	    count > reader->header.byte_count - offset ||
	    offset % EXL3_MAP_BLOCK_BYTES != 0 ||
	    count % EXL3_MAP_BLOCK_BYTES != 0 ||
	    (uintptr_t)destination % EXL3_MAP_BLOCK_BYTES != 0 ||
	    count > SIZE_MAX) {
		return STATUS_INVALID_ARGUMENT;
	}
	/* bytes no segment holds read as zero: the holes are cleared as the segments are walked, each byte written once */
	unsigned char *const bytes = destination;
	uint64_t cleared_to = offset;
	uint32_t low = 0, high = reader->header.segment_count;
	while (low < high) {
		const uint32_t middle =
			low + (high - low) / BINARY_SEARCH_RADIX;
		const struct map_segment *const segment =
			&reader->segments[middle];
		if (segment->byte_offset + segment->byte_count <= offset) {
			low = middle + 1u;
		} else {
			high = middle;
		}
	}
	for (uint32_t index = low; index < reader->header.segment_count;
	     index++) {
		const struct map_segment *const segment =
			&reader->segments[index];
		if (segment->byte_offset >= offset + count) {
			break;
		}
		const uint64_t begin = offset > segment->byte_offset ?
					       offset :
					       segment->byte_offset;
		const uint64_t end =
			offset + count < segment->byte_offset +
						 segment->byte_count ?
				offset + count :
				segment->byte_offset + segment->byte_count;
		memset(bytes + (cleared_to - offset), 0,
		       (size_t)(begin - cleared_to));
		cleared_to = end;
		/* a piece aligned for direct reads bypasses the page cache: no second copy of what the caller keeps */
		const uint32_t source = segment->source_index;
		const uint64_t source_offset =
			segment->source_offset + begin - segment->byte_offset;
		unsigned char *const piece = bytes + (begin - offset);
		const uint64_t alignment = reader->direct_alignment[source];
		/* the mode now, not at open: the engine may leave DIRECT_DROPPED for DROPPED after it (main.c) */
		const bool direct =
			(read_mode == EXL3_READS_DIRECT ||
			 read_mode == EXL3_READS_DIRECT_DROPPED) &&
			reader->direct_sources[source] >= 0 &&
			((source_offset | (end - begin) | (uintptr_t)piece) %
			 alignment) == 0;
		if (FAULT_POINT(exl3_map_read) ||
		    !(direct ? read_pieces(reader->direct_sources[source],
					   source_offset, piece, end - begin,
					   DIRECT_PIECE_BYTES) :
			       read_exact(reader->sources[source],
					  source_offset, piece, end - begin))) {
			return STATUS_INPUT_OUTPUT_ERROR;
		}
		if (!direct && (read_mode == EXL3_READS_DROPPED ||
				read_mode == EXL3_READS_DIRECT_DROPPED))
			(void)posix_fadvise(reader->sources[source],
					    (off_t)source_offset,
					    (off_t)(end - begin),
					    POSIX_FADV_DONTNEED);
	}
	memset(bytes + (cleared_to - offset), 0,
	       (size_t)(offset + count - cleared_to));
	return STATUS_OK;
}

const struct storage_operations_record storage_exl3 = {
	.name = "exl3_map",
	.reader_open = map_open,
	.reader_close = map_close,
	.reader_describe = map_describe,
	.reader_read = map_read
};

const struct storage_operations_record *storage_for_path(const char *path)
{
	static const char suffix[] = ".exl3map";
	const size_t length = path == NULL ? 0 : strlen(path);
	return length >= sizeof(suffix) - 1u &&
			       strcmp(path + length - (sizeof(suffix) - 1u),
				      suffix) == 0 ?
		       &storage_exl3 :
		       &storage_direct;
}
