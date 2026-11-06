/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * storage_direct_linux.c: storage_direct, the storage of storage_seam.h as direct positioned reads on Linux.
 *
 * Contract, beside the seam's:
 *   - A copy is a regular file opened with O_DIRECT. Its block alignment is the larger of the two that statx states
 *     for direct reads (memory and offset), or FALLBACK_BLOCK_ALIGNMENT_BYTES with a warning where none is stated.
 *   - The read bound is known for an NVMe drive only, found through the device number of the file: the io_timeout
 *     of its queue times one more than the max_retries of nvme_core. Any other drive, or a figure that cannot be
 *     read, is a copy with no known bound: one warning, and the copy is accepted.
 *   - An error line shows no more of a path than its last component.
 *   - Readers share no state and a reader never changes after open, so reads take no lock.
 */
#include "anemone/storage_seam.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "anemone/hooks.h"
#include "storage_file_bounds.h"

#define FALLBACK_BLOCK_ALIGNMENT_BYTES 4096u    /* whole blocks of every drive in use, and a page of the host */
#define COPY_FORMAT "copy %" PRIu32 " (%s)"     /* its index and the last component of its path */
#define MOST_COPY_BYTES ((uint64_t)INT64_MAX)   /* what the offset of a positioned read can hold */
#define MOST_BYTES_PER_PREAD ((uint64_t)1 << 30u) /* under the kernel's cap, which cuts a read at a page, not a block */
#define READ_LIMIT_MILLISECONDS ((uint64_t)STORAGE_READ_LIMIT_SECONDS * 1000)
#define DEVICE_LINK_CAPACITY 64                 /* "/sys/dev/block/<major>:<minor>" with two 32-bit numbers */
#define NVME_NAME_PREFIX "nvme"
#define SYSFS_FIGURE_CAPACITY 24                /* more than the digits of a 64-bit figure and its line end */
#define SYSFS_NO_FIGURE UINT64_MAX              /* a figure is at most 32 bits, so two multiply within 64 */
#define DECIMAL_BASE 10

struct storage_reader {
	struct storage_description
		description; /* copy_count: during open, the copies that are open so far */
	int descriptors[STORAGE_MAXIMUM_COPIES];
};

/* The figure in a file of sysfs (digits and a line end, at most 32 bits), or SYSFS_NO_FIGURE where there is none. */
static uint64_t sysfs_figure(const char *directory, const char *file_name)
{
    char figure_path[PATH_MAX];
    char text[SYSFS_FIGURE_CAPACITY] = "";      /* stays a string: a read fills all but its last byte at most */
    char *figure_end = text;
    const int path_length = snprintf(figure_path, sizeof(figure_path), "%s/%s", directory, file_name);
    const int descriptor =
        path_length < 0 || (size_t)path_length >= sizeof(figure_path) ? -1 : open(figure_path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        return SYSFS_NO_FIGURE;
    }
    const bool is_read = read(descriptor, text, sizeof(text) - 1) > 0;
    (void)close(descriptor);
    const uint64_t figure = strtoull(text, &figure_end, DECIMAL_BASE);  /* too many digits: the largest value */
    return (is_read && isdigit((unsigned char)text[0]) != 0 && figure_end[0] == '\n' && figure_end[1] == '\0' &&
            figure <= UINT32_MAX) ? figure : SYSFS_NO_FIGURE;
}

/* The milliseconds within which one read from the drive behind a device number returns, or 0 where the engine
 * knows no bound: the file system hides the drive, it is no NVMe drive, or a figure cannot be read. */
static uint64_t drive_read_bound_milliseconds(uint32_t device_major, uint32_t device_minor)
{
    char device_link[DEVICE_LINK_CAPACITY];
    char device_path[PATH_MAX] = "";
    uint64_t timeout_milliseconds = SYSFS_NO_FIGURE;
    uint64_t retry_count = SYSFS_NO_FIGURE;
    const int link_length = snprintf(device_link, sizeof(device_link), "/sys/dev/block/%" PRIu32 ":%" PRIu32,
                                     device_major, device_minor);
    const bool is_resolved = link_length > 0 && (size_t)link_length < sizeof(device_link) &&
                             realpath(device_link, device_path) != NULL;
    const char *const device_name = is_resolved ? strrchr(device_path, '/') : NULL;
    if (device_name != NULL && strncmp(device_name + 1, NVME_NAME_PREFIX, sizeof(NVME_NAME_PREFIX) - 1) == 0) {
        /* a partition is named after its drive, holds its own number, and lies in the directory of its drive */
        const bool is_partition = sysfs_figure(device_path, "partition") != SYSFS_NO_FIGURE;
        timeout_milliseconds = sysfs_figure(device_path, is_partition ? "../queue/io_timeout" : "queue/io_timeout");
        retry_count = sysfs_figure("/sys/module/nvme_core/parameters", "max_retries");
    }
    return (timeout_milliseconds == SYSFS_NO_FIGURE || retry_count == SYSFS_NO_FIGURE)
               ? 0 : timeout_milliseconds * (retry_count + 1);
}

/* The EXL3 map shares this admission policy; the native reader keeps its call and arithmetic. */
uint64_t storage_drive_read_bound_milliseconds(uint32_t device_major,
					       uint32_t device_minor)
{
    return drive_read_bound_milliseconds(device_major, device_minor);
}

static void direct_reader_close(struct storage_reader *reader)
{
    for (uint32_t copy_index = 0; reader != NULL && copy_index < reader->description.copy_count; copy_index++) {
        (void)close(reader->descriptors[copy_index]);
    }
    free(reader);
}

/* Opens the next copy of a reader and checks it against the seam and the copies before it. The descriptor is the
 * reader's as soon as it is open, so that direct_reader_close releases it whatever is refused afterwards. */
MUST_CHECK static result_status copy_open(struct storage_reader *reader,
					  const char *copy_path)
{
	struct storage_description *const description = &reader->description;
	const uint32_t copy_index = description->copy_count;
	const char *copy_name = NULL;
#if LOGGING_HOOKS_ENABLED
	const char *const last_separator = strrchr(copy_path, '/');
	copy_name = last_separator != NULL ? last_separator + 1 : copy_path;
#endif
    struct statx file_status = {.stx_mask = 0};
    uint32_t alignment_bytes = FALLBACK_BLOCK_ALIGNMENT_BYTES;
    /* O_NONBLOCK: a named pipe given as a copy returns at once and is refused below; a regular file ignores it. */
    const int descriptor =
	    FAULT_POINT(storage_open) ?
		    -1 :
		    open(copy_path,
			 O_RDONLY | O_DIRECT | O_CLOEXEC | O_NONBLOCK);
    if (descriptor < 0) {
	    LOG_ERROR(COPY_FORMAT " does not open for direct reads: %s",
		      copy_index, copy_name, strerror(errno));
	    return STATUS_INPUT_OUTPUT_ERROR;
    }
    reader->descriptors[copy_index] = descriptor;
    description->copy_count = copy_index + 1;
    const bool is_sized =
	    (FAULT_POINT(storage_size) ?
		     -1 :
		     statx(descriptor, "", AT_EMPTY_PATH,
			   STATX_TYPE | STATX_SIZE | STATX_DIOALIGN,
			   &file_status)) == 0;
    if (!is_sized || (~file_status.stx_mask & (STATX_TYPE | STATX_SIZE)) != 0 || !S_ISREG(file_status.stx_mode)) {
	    LOG_ERROR(COPY_FORMAT " is no regular file of a known size: %s",
		      copy_index, copy_name,
		      is_sized ? "it is something else" : strerror(errno));
	    return STATUS_INPUT_OUTPUT_ERROR;
    }
    if ((file_status.stx_mask & STATX_DIOALIGN) != 0 && file_status.stx_dio_mem_align != 0 &&
        file_status.stx_dio_offset_align != 0) {
        alignment_bytes = file_status.stx_dio_mem_align > file_status.stx_dio_offset_align
                              ? file_status.stx_dio_mem_align : file_status.stx_dio_offset_align;
    } else {
	    LOG_WARNING_ENTRY(
		    COPY_FORMAT
		    ": no alignment for direct reads is stated, %" PRIu32
		    " bytes are assumed",
		    copy_index, copy_name, alignment_bytes);
    }
    /* every alignment a power of two and the one size whole blocks of each: whole blocks of the largest as well */
    if ((copy_index > 0 && file_status.stx_size != description->byte_count) || file_status.stx_size == 0 ||
        file_status.stx_size > MOST_COPY_BYTES || (alignment_bytes & (alignment_bytes - 1)) != 0 ||
        file_status.stx_size % alignment_bytes != 0) {
	    LOG_ERROR(COPY_FORMAT
		      " holds %" PRIu64 " bytes in blocks of %" PRIu32
		      ": a copy holds as many as "
		      "copy 0, not none, in whole blocks of a power of two",
		      copy_index, copy_name, (uint64_t)file_status.stx_size,
		      alignment_bytes);
	    return STATUS_CORRUPT_PACK;
    }
    description->byte_count = file_status.stx_size;
    description->block_alignment_bytes = alignment_bytes > description->block_alignment_bytes
                                             ? alignment_bytes : description->block_alignment_bytes;
    const uint64_t bound_milliseconds =
	    FAULT_POINT(storage_read_bound) ?
		    UINT64_MAX :
		    drive_read_bound_milliseconds(file_status.stx_dev_major,
						  file_status.stx_dev_minor);
    if (bound_milliseconds > READ_LIMIT_MILLISECONDS) {
	    LOG_ERROR(COPY_FORMAT ": its drive lets one read take %" PRIu64
				  " ms, the limit is %" PRIu64,
		      copy_index, copy_name, bound_milliseconds,
		      READ_LIMIT_MILLISECONDS);
	    return STATUS_UNSUPPORTED;
    }
    if (bound_milliseconds == 0) {
	    LOG_WARNING_ENTRY(
		    COPY_FORMAT
		    ": the engine knows no bound for reads of this copy: it lies on no NVMe "
		    "drive, or its drive states no request timeout",
		    copy_index, copy_name);
    } else {
	    LOG_INFORMATION(COPY_FORMAT " read bound: %" PRIu64 " ms",
			    copy_index, copy_name, bound_milliseconds);
    }
    return STATUS_OK;
}

MUST_CHECK static result_status
direct_reader_open(const char *const *copy_paths, uint32_t copy_count,
		   struct storage_reader **reader_out)
{
	struct storage_reader *reader = NULL;
	result_status status = STATUS_OK;
	if (copy_paths == NULL || reader_out == NULL || copy_count == 0 ||
	    copy_count > STORAGE_MAXIMUM_COPIES) {
		return STATUS_INVALID_ARGUMENT;
	}
    for (uint32_t copy_index = 0; copy_index < copy_count; copy_index++) {
        if (copy_paths[copy_index] == NULL) {
		return STATUS_INVALID_ARGUMENT;
	}
    }
    reader = FAULT_POINT(storage_allocation) ? NULL :
					       calloc(1, sizeof(*reader));
    if (reader == NULL) {
	    return STATUS_OUT_OF_MEMORY;
    }
    for (uint32_t copy_index = 0;
	 status == STATUS_OK && copy_index < copy_count; copy_index++) {
	    status = copy_open(reader, copy_paths[copy_index]);
    }
    if (status == STATUS_OK) {
	    *reader_out = reader;
    } else {
	    direct_reader_close(reader);
    }
    return status;
}

MUST_CHECK static result_status
direct_reader_describe(const struct storage_reader *reader,
		       struct storage_description *description_out)
{
    if (reader == NULL || description_out == NULL) {
	    return STATUS_INVALID_ARGUMENT;
    }
    *description_out = reader->description;
    return STATUS_OK;
}

MUST_CHECK static result_status
direct_reader_read(const struct storage_reader *reader, uint32_t copy_index,
		   uint64_t byte_offset, void *destination, uint64_t byte_count)
{
    uint64_t delivered_byte_count = 0;
    if (reader == NULL || destination == NULL || copy_index >= reader->description.copy_count || byte_count == 0 ||
        byte_offset > reader->description.byte_count || byte_count > reader->description.byte_count - byte_offset ||
        ((byte_offset | byte_count | (uintptr_t)destination) & (reader->description.block_alignment_bytes - 1)) != 0) {
	    return STATUS_INVALID_ARGUMENT;
    }
    TIMER_BEGIN(storage_read);
    while (delivered_byte_count < byte_count) {
        const uint64_t left_byte_count = byte_count - delivered_byte_count;
        const size_t asked_byte_count =
            (size_t)(left_byte_count < MOST_BYTES_PER_PREAD ? left_byte_count : MOST_BYTES_PER_PREAD);
        errno = 0;      /* an injected failure sets none, and an older EINTR would have it asked again */
	const ssize_t read_byte_count =
		FAULT_POINT(storage_read) ?
			-1 :
			pread(reader->descriptors[copy_index],
			      (unsigned char *)destination +
				      delivered_byte_count,
			      asked_byte_count,
			      (off_t)(byte_offset + delivered_byte_count));
	if (read_byte_count <= 0 && (read_byte_count == 0 || errno != EINTR)) {  /* after EINTR it is asked again */
		LOG_ERROR("copy %" PRIu32
			  ": the read of %zu bytes at offset %" PRIu64
			  " failed: %s",
			  copy_index, asked_byte_count,
			  byte_offset + delivered_byte_count,
			  read_byte_count == 0 ? "the copy ends before them" :
						 strerror(errno));
		return STATUS_INPUT_OUTPUT_ERROR;
	}
        delivered_byte_count += read_byte_count > 0 ? (uint64_t)read_byte_count : 0;
    }
    TIMER_END(storage_read);
    COUNTER_ADD(storage_read_bytes, byte_count);
    return STATUS_OK;
}

const struct storage_operations_record storage_direct = {
	.name = "direct",
	.reader_open = direct_reader_open,
	.reader_close = direct_reader_close,
	.reader_describe = direct_reader_describe,
	.reader_read = direct_reader_read,
};
