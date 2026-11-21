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
#include "engine/storage/storage_seam.h"
#include "engine/engram/engram_disk.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include <time.h>

#include "common/hooks.h"
#include "storage_file_bounds.h"

#define FALLBACK_BLOCK_ALIGNMENT_BYTES \
	4096u /* whole blocks of every drive in use, and a page of the host */
#define COPY_FORMAT     \
	"copy %" PRIu32 \
	" (%s)" /* its index and the last component of its path */
#define MOST_COPY_BYTES \
	((uint64_t)     \
		 INT64_MAX) /* what the offset of a positioned read can hold */
#define MOST_BYTES_PER_PREAD \
	((uint64_t)1         \
	 << 30u) /* under the kernel's cap, which cuts a read at a page, not a block */
#define READ_LIMIT_MILLISECONDS ((uint64_t)STORAGE_READ_LIMIT_SECONDS * 1000)
#define DEVICE_LINK_CAPACITY \
	64 /* "/sys/dev/block/<major>:<minor>" with two 32-bit numbers */
#define NVME_NAME_PREFIX "nvme"
#define SYSFS_FIGURE_CAPACITY \
	24 /* more than the digits of a 64-bit figure and its line end */
#define SYSFS_NO_FIGURE \
	UINT64_MAX /* a figure is at most 32 bits, so two multiply within 64 */
#define DECIMAL_BASE 10

struct storage_reader {
	struct storage_description
		description; /* copy_count: during open, the copies that are open so far */
	int descriptors[STORAGE_MAXIMUM_COPIES];
	uint32_t table_count; /* zero: a normal full-pack reader; otherwise independent Engram-only sources */
	uint64_t logical_starts[ENGRAM_MAXIMUM_TABLES];
	uint64_t table_bytes[ENGRAM_MAXIMUM_TABLES];
	uint64_t physical_starts[STORAGE_MAXIMUM_COPIES][ENGRAM_MAXIMUM_TABLES];
};

/* The figure in a file of sysfs (digits and a line end, at most 32 bits), or SYSFS_NO_FIGURE where there is none. */
static uint64_t sysfs_figure(const char *directory, const char *file_name)
{
	char figure_path[PATH_MAX];
	char text[SYSFS_FIGURE_CAPACITY] =
		""; /* stays a string: a read fills all but its last byte at most */
	char *figure_end = text;
	const int path_length = snprintf(figure_path, sizeof(figure_path),
					 "%s/%s", directory, file_name);
	const int descriptor = path_length < 0 || (size_t)path_length >=
							  sizeof(figure_path) ?
				       -1 :
				       open(figure_path, O_RDONLY | O_CLOEXEC);
	if (descriptor < 0) {
		return SYSFS_NO_FIGURE;
	}
	const bool is_read = read(descriptor, text, sizeof(text) - 1) > 0;
	(void)close(descriptor);
	const uint64_t figure =
		strtoull(text, &figure_end,
			 DECIMAL_BASE); /* too many digits: the largest value */
	return (is_read && isdigit((unsigned char)text[0]) != 0 &&
		figure_end[0] == '\n' && figure_end[1] == '\0' &&
		figure <= UINT32_MAX) ?
		       figure :
		       SYSFS_NO_FIGURE;
}

/* The milliseconds within which one read from the drive behind a device number returns, or 0 where the engine
 * knows no bound: the file system hides the drive, it is no NVMe drive, or a figure cannot be read. */
static uint64_t drive_read_bound_milliseconds(uint32_t device_major,
					      uint32_t device_minor)
{
	char device_link[DEVICE_LINK_CAPACITY];
	char device_path[PATH_MAX] = "";
	uint64_t timeout_milliseconds = SYSFS_NO_FIGURE;
	uint64_t retry_count = SYSFS_NO_FIGURE;
	const int link_length = snprintf(device_link, sizeof(device_link),
					 "/sys/dev/block/%" PRIu32 ":%" PRIu32,
					 device_major, device_minor);
	const bool is_resolved = link_length > 0 &&
				 (size_t)link_length < sizeof(device_link) &&
				 realpath(device_link, device_path) != NULL;
	const char *const device_name =
		is_resolved ? strrchr(device_path, '/') : NULL;
	if (device_name != NULL && strncmp(device_name + 1, NVME_NAME_PREFIX,
					   sizeof(NVME_NAME_PREFIX) - 1) == 0) {
		/* a partition is named after its drive, holds its own number, and lies in the directory of its drive */
		const bool is_partition =
			sysfs_figure(device_path, "partition") !=
			SYSFS_NO_FIGURE;
		timeout_milliseconds = sysfs_figure(
			device_path, is_partition ? "../queue/io_timeout" :
						    "queue/io_timeout");
		retry_count = sysfs_figure("/sys/module/nvme_core/parameters",
					   "max_retries");
	}
	return (timeout_milliseconds == SYSFS_NO_FIGURE ||
		retry_count == SYSFS_NO_FIGURE) ?
		       0 :
		       timeout_milliseconds * (retry_count + 1);
}

/* The EXL3 map shares this admission policy; the native reader keeps its call and arithmetic. */
uint64_t storage_drive_read_bound_milliseconds(uint32_t device_major,
					       uint32_t device_minor)
{
	return drive_read_bound_milliseconds(device_major, device_minor);
}

static void direct_reader_close(struct storage_reader *reader)
{
	for (uint32_t copy_index = 0;
	     reader != NULL && copy_index < reader->description.copy_count;
	     copy_index++) {
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
	struct statx file_status = { .stx_mask = 0 };
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
	const bool is_sized = (FAULT_POINT(storage_size) ?
				       -1 :
				       statx(descriptor, "", AT_EMPTY_PATH,
					     STATX_TYPE | STATX_SIZE |
						     STATX_INO | STATX_DIOALIGN,
					     &file_status)) == 0;
	if (!is_sized ||
	    (~file_status.stx_mask & (STATX_TYPE | STATX_SIZE)) != 0 ||
	    !S_ISREG(file_status.stx_mode)) {
		LOG_ERROR(COPY_FORMAT " is no regular file of a known size: %s",
			  copy_index, copy_name,
			  is_sized ? "it is something else" : strerror(errno));
		return STATUS_INPUT_OUTPUT_ERROR;
	}
	if ((file_status.stx_mask & STATX_DIOALIGN) != 0 &&
	    file_status.stx_dio_mem_align != 0 &&
	    file_status.stx_dio_offset_align != 0) {
		alignment_bytes =
			file_status.stx_dio_mem_align >
					file_status.stx_dio_offset_align ?
				file_status.stx_dio_mem_align :
				file_status.stx_dio_offset_align;
	} else {
		LOG_WARNING_ENTRY(
			COPY_FORMAT
			": no alignment for direct reads is stated, %" PRIu32
			" bytes are assumed",
			copy_index, copy_name, alignment_bytes);
	}
	/* every alignment a power of two and the one size whole blocks of each: whole blocks of the largest as well */
	if ((copy_index > 0 &&
	     file_status.stx_size != description->byte_count) ||
	    file_status.stx_size == 0 ||
	    file_status.stx_size > MOST_COPY_BYTES ||
	    (alignment_bytes & (alignment_bytes - 1)) != 0 ||
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
	if (copy_index == 0 && (file_status.stx_mask & STATX_INO) != 0) {
		description->first_device = makedev(file_status.stx_dev_major,
						    file_status.stx_dev_minor);
		description->first_inode = file_status.stx_ino;
	}
	description->block_alignment_bytes =
		alignment_bytes > description->block_alignment_bytes ?
			alignment_bytes :
			description->block_alignment_bytes;
	const uint64_t bound_milliseconds =
		FAULT_POINT(storage_read_bound) ?
			UINT64_MAX :
			drive_read_bound_milliseconds(
				file_status.stx_dev_major,
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

/* A mapped reader only accepts reads wholly inside one immutable table extent. */
static bool table_offset(const struct storage_reader *reader, uint32_t copy,
			 uint64_t logical, uint64_t bytes,
			 uint64_t *physical_out)
{
	if (reader->table_count == 0) {
		*physical_out = logical;
		return true;
	}
	for (uint32_t table = 0; table < reader->table_count; table++) {
		const uint64_t start = reader->logical_starts[table],
			       length = reader->table_bytes[table];
		if (logical >= start && logical - start <= length &&
		    bytes <= length - (logical - start)) {
			*physical_out = reader->physical_starts[copy][table] +
					logical - start;
			return true;
		}
	}
	return false;
}

MUST_CHECK static result_status
direct_reader_read(const struct storage_reader *reader, uint32_t copy_index,
		   uint64_t byte_offset, void *destination, uint64_t byte_count)
{
	uint64_t delivered_byte_count = 0;
	if (reader == NULL || destination == NULL ||
	    copy_index >= reader->description.copy_count || byte_count == 0 ||
	    byte_offset > reader->description.byte_count ||
	    byte_count > reader->description.byte_count - byte_offset ||
	    ((byte_offset | byte_count | (uintptr_t)destination) &
	     (reader->description.block_alignment_bytes - 1)) != 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	uint64_t physical_offset = 0;
	if (!table_offset(reader, copy_index, byte_offset, byte_count,
			  &physical_offset))
		return STATUS_INVALID_ARGUMENT;
	struct timespec started = { 0 };
	if (reader->table_count != 0 &&
	    clock_gettime(CLOCK_MONOTONIC, &started) != 0)
		return STATUS_TIMED_OUT;
	TIMER_BEGIN(storage_read);
	while (delivered_byte_count < byte_count) {
		if (reader->table_count != 0) {
			struct timespec now;
			if (clock_gettime(CLOCK_MONOTONIC, &now) != 0 ||
			    now.tv_sec - started.tv_sec >=
				    STORAGE_READ_LIMIT_SECONDS)
				return STATUS_TIMED_OUT;
		}
		const uint64_t left_byte_count =
			byte_count - delivered_byte_count;
		const size_t asked_byte_count =
			(size_t)(left_byte_count < MOST_BYTES_PER_PREAD ?
					 left_byte_count :
					 MOST_BYTES_PER_PREAD);
		errno = 0; /* an injected failure sets none, and an older EINTR would have it asked again */
		const ssize_t read_byte_count =
			FAULT_POINT(storage_read) ?
				-1 :
				pread(reader->descriptors[copy_index],
				      (unsigned char *)destination +
					      delivered_byte_count,
				      asked_byte_count,
				      (off_t)(physical_offset +
					      delivered_byte_count));
		if (read_byte_count <= 0 &&
		    (read_byte_count == 0 ||
		     errno != EINTR)) { /* after EINTR it is asked again */
			LOG_ERROR("copy %" PRIu32
				  ": the read of %zu bytes at offset %" PRIu64
				  " failed: %s",
				  copy_index, asked_byte_count,
				  byte_offset + delivered_byte_count,
				  read_byte_count == 0 ?
					  "the copy ends before them" :
					  strerror(errno));
			return STATUS_INPUT_OUTPUT_ERROR;
		}
		delivered_byte_count +=
			read_byte_count > 0 ? (uint64_t)read_byte_count : 0;
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

result_status engram_disk_sources_open(const char *const *paths, uint32_t count,
				       const struct pack_record *logical_pack,
				       const struct engram_record *engram,
				       struct storage_reader **reader_out)
{
	const struct engram_description *description =
		engram_description_of(engram);
	if (paths == NULL || count == 0 || count > STORAGE_MAXIMUM_COPIES ||
	    logical_pack == NULL || description == NULL || reader_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t copy = 0; copy < count; copy++)
		if (paths[copy] == NULL)
			return STATUS_INVALID_ARGUMENT;
	struct storage_reader *mapped = calloc(1, sizeof(*mapped));
	if (mapped == NULL)
		return STATUS_OUT_OF_MEMORY;
	mapped->table_count = description->table_count;
	mapped->description.byte_count =
		pack_header_of(logical_pack)->file_byte_count;
	result_status status = STATUS_OK;
	for (uint32_t table = 0; table < description->table_count; table++) {
		const uint32_t unit = description->table_unit_index[table];
		if (unit >= pack_header_of(logical_pack)->unit_count ||
		    pack_units_of(logical_pack)[unit].unit_kind !=
			    PLACEMENT_UNIT_HOST_ONLY ||
		    pack_units_of(logical_pack)[unit].extent_count != 1) {
			direct_reader_close(mapped);
			return STATUS_INVALID_ARGUMENT;
		}
		const struct pack_extent *extent =
			&pack_extents_of(logical_pack)
				[pack_units_of(logical_pack)
					 [description->table_unit_index[table]]
						 .first_extent_index];
		mapped->logical_starts[table] = extent->byte_offset;
		mapped->table_bytes[table] = extent->byte_count;
	}
	for (uint32_t copy = 0; status == STATUS_OK && copy < count; copy++) {
		struct storage_reader *single = NULL;
		struct pack_record *source_pack = NULL;
		struct engram_record *source_engram = NULL;
		status = direct_reader_open(&paths[copy], 1, &single);
		if (status == STATUS_OK)
			status = pack_open_entry(&storage_direct, single,
						 &source_pack);
		if (status == STATUS_OK)
			status = engram_open(&storage_direct, single,
					     source_pack, &source_engram);
		if (status == STATUS_OK &&
		    !engram_hashing_equal(engram, source_engram))
			status = STATUS_CORRUPT_PACK;
		const struct engram_description *source_description =
			engram_description_of(source_engram);
		for (uint32_t table = 0;
		     status == STATUS_OK && table < description->table_count;
		     table++) {
			if (source_description == NULL) {
				status = STATUS_CORRUPT_PACK;
				break;
			}
			const struct pack_extent *source = &pack_extents_of(
				source_pack)[pack_units_of(source_pack)
						     [source_description
							      ->table_unit_index
								      [table]]
							     .first_extent_index];
			const struct pack_extent *logical =
				&pack_extents_of(logical_pack)
					[pack_units_of(logical_pack)
						 [description->table_unit_index
							  [table]]
							 .first_extent_index];
			if (source->byte_count != logical->byte_count ||
			    source->digest != logical->digest)
				status = STATUS_CORRUPT_PACK;
			mapped->physical_starts[copy][table] =
				source->byte_offset;
		}
		if (status == STATUS_OK) {
			mapped->descriptors[copy] = single->descriptors[0];
			mapped->description.copy_count++;
			if (single->description.block_alignment_bytes >
			    mapped->description.block_alignment_bytes)
				mapped->description.block_alignment_bytes =
					single->description
						.block_alignment_bytes;
			single->description.copy_count =
				0; /* descriptor ownership transferred */
		}
		engram_close(source_engram);
		pack_close(source_pack);
		direct_reader_close(single);
	}
	if (status != STATUS_OK) {
		direct_reader_close(mapped);
		return status;
	}
	*reader_out = mapped;
	return STATUS_OK;
}

/* Runtime Engram gathers use independent prepared rings; startup's sequential reader remains unchanged. */
#include "storage_batch.h"
#include <linux/io_uring.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <sys/syscall.h>

#define BATCH_LANES 8u
#define BATCH_POLL_MILLISECONDS 1000
#define NANOSECONDS_PER_SECOND 1000000000ull

struct batch_lane {
	int descriptor;
	struct io_uring_params parameters;
	void *submission;
	void *completion;
	struct io_uring_sqe *entries;
	size_t submission_bytes;
	size_t completion_bytes;
	unsigned char *buffers;
	pthread_mutex_t mutex;
	bool mutex_ready;
	bool quarantined;
};
struct storage_batch {
	const struct storage_reader *reader;
	struct batch_lane lanes[BATCH_LANES];
	uint32_t copy_count;
	atomic_uint next_lane;
};

static void batch_lane_close(struct batch_lane *lane)
{
	if (lane->descriptor >= 0)
		(void)close(lane->descriptor);
	if (lane->quarantined)
		return; /* pending I/O refers only to these retained ring-owned buffers */
	if (lane->entries != NULL && lane->entries != MAP_FAILED)
		(void)munmap(lane->entries, lane->parameters.sq_entries *
						    sizeof(*lane->entries));
	if (lane->completion != NULL && lane->completion != MAP_FAILED &&
	    lane->completion != lane->submission)
		(void)munmap(lane->completion, lane->completion_bytes);
	if (lane->submission != NULL && lane->submission != MAP_FAILED)
		(void)munmap(lane->submission, lane->submission_bytes);
	free(lane->buffers);
	if (lane->mutex_ready)
		(void)pthread_mutex_destroy(&lane->mutex);
}

void storage_batch_close(struct storage_batch *batch)
{
	if (batch == NULL)
		return;
	for (uint32_t index = 0; index < BATCH_LANES; index++)
		batch_lane_close(&batch->lanes[index]);
	free(batch);
}

MUST_CHECK static result_status
batch_lane_open(struct batch_lane *lane, const struct storage_reader *reader)
{
	lane->descriptor = (int)syscall(__NR_io_uring_setup,
					STORAGE_BATCH_DEPTH, &lane->parameters);
	if (lane->descriptor < 0)
		return STATUS_UNSUPPORTED;
	struct io_uring_params *const parameters = &lane->parameters;
	lane->submission_bytes = parameters->sq_off.array +
				 parameters->sq_entries * sizeof(uint32_t);
	lane->completion_bytes =
		parameters->cq_off.cqes +
		parameters->cq_entries * sizeof(struct io_uring_cqe);
	if ((parameters->features & IORING_FEAT_SINGLE_MMAP) != 0 &&
	    lane->completion_bytes > lane->submission_bytes)
		lane->submission_bytes = lane->completion_bytes;
	lane->submission = mmap(NULL, lane->submission_bytes,
				PROT_READ | PROT_WRITE, MAP_SHARED,
				lane->descriptor, IORING_OFF_SQ_RING);
	lane->completion = (parameters->features & IORING_FEAT_SINGLE_MMAP) !=
					   0 ?
				   lane->submission :
				   mmap(NULL, lane->completion_bytes,
					PROT_READ | PROT_WRITE, MAP_SHARED,
					lane->descriptor, IORING_OFF_CQ_RING);
	lane->entries = mmap(NULL,
			     parameters->sq_entries * sizeof(*lane->entries),
			     PROT_READ | PROT_WRITE, MAP_SHARED,
			     lane->descriptor, IORING_OFF_SQES);
	if (lane->submission == MAP_FAILED || lane->completion == MAP_FAILED ||
	    lane->entries == MAP_FAILED)
		return STATUS_OUT_OF_MEMORY;
	lane->buffers =
		aligned_alloc(PACK_BLOCK_BYTES,
			      (size_t)(STORAGE_BATCH_DEPTH * PACK_BLOCK_BYTES));
	if (lane->buffers == NULL ||
	    pthread_mutex_init(&lane->mutex, NULL) != 0)
		return STATUS_OUT_OF_MEMORY;
	lane->mutex_ready = true;
	if (syscall(__NR_io_uring_register, lane->descriptor,
		    IORING_REGISTER_FILES, reader->descriptors,
		    reader->description.copy_count) != 0)
		return STATUS_UNSUPPORTED;
	return STATUS_OK;
}

result_status storage_batch_open(const struct storage_reader *reader,
				 struct storage_batch **batch_out)
{
	if (reader == NULL || batch_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	for (uint32_t copy = 0; copy < reader->description.copy_count; copy++) {
		struct statx status = { 0 };
		if (statx(reader->descriptors[copy], "", AT_EMPTY_PATH,
			  STATX_DIOALIGN, &status) != 0 ||
		    (status.stx_mask & STATX_DIOALIGN) == 0 ||
		    status.stx_dio_mem_align == 0 ||
		    status.stx_dio_offset_align == 0 ||
		    PACK_BLOCK_BYTES % status.stx_dio_mem_align != 0 ||
		    PACK_BLOCK_BYTES % status.stx_dio_offset_align != 0)
			return STATUS_UNSUPPORTED;
	}
	struct storage_batch *const batch = calloc(1, sizeof(*batch));
	if (batch == NULL)
		return STATUS_OUT_OF_MEMORY;
	batch->reader = reader;
	batch->copy_count = reader->description.copy_count;
	atomic_init(&batch->next_lane, 0);
	for (uint32_t index = 0; index < BATCH_LANES; index++)
		batch->lanes[index].descriptor = -1;
	result_status status = STATUS_OK;
	for (uint32_t index = 0; status == STATUS_OK && index < BATCH_LANES;
	     index++)
		status = batch_lane_open(&batch->lanes[index], reader);
	if (status != STATUS_OK) {
		storage_batch_close(batch);
		return status;
	}
	*batch_out = batch;
	return STATUS_OK;
}

MUST_CHECK static result_status
batch_submit(struct batch_lane *lane, const uint32_t *copies,
	     const uint64_t *offsets, uint32_t count, result_status *results)
{
	uint32_t *const tail = (void *)((unsigned char *)lane->submission +
					lane->parameters.sq_off.tail);
	const uint32_t *const mask =
		(void *)((unsigned char *)lane->submission +
			 lane->parameters.sq_off.ring_mask);
	uint32_t *const array = (void *)((unsigned char *)lane->submission +
					 lane->parameters.sq_off.array);
	uint32_t local_tail = __atomic_load_n(tail, __ATOMIC_ACQUIRE);
	for (uint32_t index = 0; index < count; index++) {
		const uint32_t slot = local_tail & *mask;
		struct io_uring_sqe *const entry = &lane->entries[slot];
		memset(entry, 0, sizeof(*entry));
		entry->opcode = IORING_OP_READ;
		entry->flags = IOSQE_FIXED_FILE;
		entry->fd = (int)copies[index];
		entry->off = offsets[index];
		entry->addr =
			(uint64_t)(uintptr_t)(lane->buffers +
					      (size_t)index * PACK_BLOCK_BYTES);
		entry->len = PACK_BLOCK_BYTES;
		entry->user_data = index;
		array[slot] = slot;
		local_tail++;
	}
	struct timespec start;
	if (clock_gettime(CLOCK_MONOTONIC, &start) != 0)
		return STATUS_TIMED_OUT;
	__atomic_store_n(tail, local_tail, __ATOMIC_RELEASE);
	for (uint32_t submitted = 0; submitted < count;) {
		const long result = syscall(__NR_io_uring_enter,
					    lane->descriptor, count - submitted,
					    0, 0, NULL, 0);
		if (result < 0 && errno == EINTR) {
			struct timespec now;
			if (clock_gettime(CLOCK_MONOTONIC, &now) != 0 ||
			    now.tv_sec - start.tv_sec >=
				    STORAGE_READ_LIMIT_SECONDS) {
				lane->quarantined = true;
				return STATUS_TIMED_OUT;
			}
			continue;
		}
		if (result <= 0) {
			lane->quarantined = true;
			return STATUS_INPUT_OUTPUT_ERROR;
		}
		submitted += (uint32_t)result;
	}
	uint32_t *const head = (void *)((unsigned char *)lane->completion +
					lane->parameters.cq_off.head);
	const uint32_t *const completion_tail =
		(void *)((unsigned char *)lane->completion +
			 lane->parameters.cq_off.tail);
	const uint32_t *const completion_mask =
		(void *)((unsigned char *)lane->completion +
			 lane->parameters.cq_off.ring_mask);
	const struct io_uring_cqe *const completions =
		(void *)((unsigned char *)lane->completion +
			 lane->parameters.cq_off.cqes);
	uint32_t local_head = __atomic_load_n(head, __ATOMIC_ACQUIRE);
	for (uint32_t done = 0; done < count;) {
		while (local_head !=
		       __atomic_load_n(completion_tail, __ATOMIC_ACQUIRE)) {
			const struct io_uring_cqe *const completion =
				&completions[local_head & *completion_mask];
			if (completion->user_data >= count) {
				lane->quarantined = true;
				return STATUS_INPUT_OUTPUT_ERROR;
			}
			results[completion->user_data] =
				completion->res == (int)PACK_BLOCK_BYTES ?
					STATUS_OK :
					STATUS_INPUT_OUTPUT_ERROR;
			local_head++;
			done++;
		}
		__atomic_store_n(head, local_head, __ATOMIC_RELEASE);
		if (done == count)
			break;
		struct timespec now;
		if (clock_gettime(CLOCK_MONOTONIC, &now) != 0 ||
		    now.tv_sec - start.tv_sec >= STORAGE_READ_LIMIT_SECONDS) {
			lane->quarantined = true;
			return STATUS_TIMED_OUT;
		}
		struct pollfd wait = { .fd = lane->descriptor,
				       .events = POLLIN };
		if ((poll(&wait, 1, BATCH_POLL_MILLISECONDS) < 0 &&
		     errno != EINTR) ||
		    (wait.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
			lane->quarantined = true;
			return STATUS_INPUT_OUTPUT_ERROR;
		}
	}
	return STATUS_OK;
}

result_status storage_batch_read(struct storage_batch *batch,
				 const uint32_t *copies,
				 const uint64_t *offsets, uint32_t count,
				 void *blocks_out, result_status *results_out)
{
	if (batch == NULL || copies == NULL || offsets == NULL ||
	    blocks_out == NULL || results_out == NULL || count == 0 ||
	    count > STORAGE_BATCH_DEPTH)
		return STATUS_INVALID_ARGUMENT;
	uint64_t physical_offsets[STORAGE_BATCH_DEPTH];
	for (uint32_t index = 0; index < count; index++)
		if (copies[index] >= batch->copy_count ||
		    offsets[index] % PACK_BLOCK_BYTES != 0 ||
		    !table_offset(batch->reader, copies[index], offsets[index],
				  PACK_BLOCK_BYTES, &physical_offsets[index]))
			return STATUS_INVALID_ARGUMENT;
	struct batch_lane *const lane =
		&batch->lanes[atomic_fetch_add_explicit(&batch->next_lane, 1,
							memory_order_relaxed) %
			      BATCH_LANES];
	struct timespec deadline;
	if (clock_gettime(CLOCK_REALTIME, &deadline) != 0)
		return STATUS_TIMED_OUT;
	deadline.tv_sec += STORAGE_READ_LIMIT_SECONDS;
	if (pthread_mutex_timedlock(&lane->mutex, &deadline) != 0)
		return STATUS_TIMED_OUT;
	result_status status =
		lane->quarantined ? STATUS_INPUT_OUTPUT_ERROR :
				    batch_submit(lane, copies, physical_offsets,
						 count, results_out);
	if (status == STATUS_OK)
		memcpy(blocks_out, lane->buffers,
		       (size_t)count * PACK_BLOCK_BYTES);
	if (pthread_mutex_unlock(&lane->mutex) != 0)
		return STATUS_DEVICE_ERROR;
	return status;
}

int storage_direct_descriptor(const struct storage_reader *reader,
			      uint32_t copy)
{
	return reader != NULL && copy < reader->description.copy_count ?
		       reader->descriptors[copy] :
		       -1;
}

bool storage_direct_physical_offset(const struct storage_reader *reader,
				    uint32_t copy, uint64_t logical,
				    uint64_t bytes, uint64_t *physical_out)
{
	return reader != NULL && physical_out != NULL &&
	       copy < reader->description.copy_count &&
	       logical <= reader->description.byte_count &&
	       bytes <= reader->description.byte_count - logical &&
	       table_offset(reader, copy, logical, bytes, physical_out);
}
