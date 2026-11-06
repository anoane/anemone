/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Original-weight tower qualification reuses the product pack mapper, launch planner and CUDA backend. */
#include "vision_original.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "anemone/backend_records.h"
#include "model_vision.h"

#define ORIGINAL_WAIT_NS 240000000000ull
#define ORIGINAL_DEADLINE_SECONDS 900u
#define ORIGINAL_STAGING_BYTES 16777216ull
#define ORIGINAL_GRID_DIMENSIONS 2u
#define ORIGINAL_MAX_GRID_HEIGHT 3u
#define ORIGINAL_MAX_GRID_WIDTH 3063u
#define ORIGINAL_NS_PER_SECOND 1000000000ull
#define ORIGINAL_REPEATS 3u
#define ORIGINAL_TRACE_PATH_BYTES 4096u
#define ORIGINAL_TRACE_RANGE_CAPACITY 9u
#define ORIGINAL_DECIMAL_RADIX 10
#define ORIGINAL_PATCH_BYTES                                  \
	((uint64_t)VISION_MAXIMUM_PATCHES * VISION_CHANNELS * \
	 VISION_PATCH_SIZE * VISION_PATCH_SIZE * VISION_BFLOAT16_BYTES)

static const struct backend_operations *const backend = &backend_cuda_value;

struct original_partitions {
	uint32_t mlp, mlp_up, aligner;
	bool has_aligner_bias_last, aligner_bias_last;
};

static bool aligner_bias_last_read(struct original_partitions *partitions)
{
	const char *const value = getenv("VISION_TEST_ALIGNER_BIAS_LAST");
	partitions->has_aligner_bias_last = value != NULL;
	partitions->aligner_bias_last = false;
	if (value == NULL || strcmp(value, "0") == 0) {
		return true;
	}
	if (strcmp(value, "1") == 0) {
		partitions->aligner_bias_last = true;
		return true;
	}
	return false;
}

static bool partition_read(const char *name, uint32_t input_width,
			   uint32_t *width_out)
{
	const char *const value = getenv(name);
	*width_out = 0;
	if (value == NULL) {
		return true;
	}
	if (value[0] < '0' || value[0] > '9') {
		return false;
	}
	char *end = NULL;
	errno = 0;
	const unsigned long width =
		strtoul(value, &end, ORIGINAL_DECIMAL_RADIX);
	if (errno != 0 || end == NULL || *end != '\0' || width == 0 ||
	    width > input_width || width % VISION_LINEAR_TILE != 0 ||
	    width > (VISION_LINEAR_K_PARTITION_MASK >>
		     VISION_LINEAR_K_PARTITION_SHIFT)) {
		return false;
	}
	*width_out = (uint32_t)width;
	return true;
}

/* Diagnostic overrides affect only this helper's copied launch records, in both full and isolated execution. */
static bool tower_prepare(const struct vision_description *description,
			  uint64_t arena_bytes,
			  const struct vision_kernels *kernels,
			  const struct original_partitions *partitions,
			  struct kernel_launch *launches, uint32_t *count_out)
{
	if (vision_prepare(description, arena_bytes, kernels, launches,
			   VISION_MAXIMUM_LAUNCHES, count_out) != STATUS_OK) {
		return false;
	}
	for (uint32_t index = 0; index < *count_out; index++) {
		if (launches[index].kernel !=
		    kernels->operations[VISION_LINEAR]) {
			continue;
		}
		struct vision_arguments record;
		memcpy(&record, launches[index].argument, sizeof(record));
		uint32_t width = 0;
		if (record.input_width == VISION_MAXIMUM_INTERMEDIATE &&
		    record.output_width == VISION_MAXIMUM_WIDTH) {
			width = partitions->mlp;
		} else if (record.input_width == VISION_MAXIMUM_WIDTH &&
			   record.output_width ==
				   VISION_GATE_COMPONENTS *
					   VISION_MAXIMUM_INTERMEDIATE) {
			width = partitions->mlp_up;
		} else if (record.input_width ==
				   VISION_MAXIMUM_WIDTH *
					   VISION_DOWNSAMPLE_RATIO *
					   VISION_DOWNSAMPLE_RATIO &&
			   record.output_width ==
				   VISION_MAXIMUM_LANGUAGE_WIDTH) {
			width = partitions->aligner;
			if (partitions->has_aligner_bias_last) {
				record.flags =
					(record.flags &
					 ~VISION_LINEAR_BIAS_LAST) |
					(partitions->aligner_bias_last ?
						 VISION_LINEAR_BIAS_LAST :
						 0u);
			}
		}
		if (width != 0) {
			record.flags =
				(record.flags &
				 ~VISION_LINEAR_K_PARTITION_MASK) |
				(width << VISION_LINEAR_K_PARTITION_SHIFT);
		}
		memcpy(launches[index].argument, &record, sizeof(record));
	}
	return true;
}

static uint64_t now_ns(void)
{
	struct timespec now;
	return clock_gettime(CLOCK_MONOTONIC, &now) == 0 && now.tv_sec >= 0 ?
		       (uint64_t)now.tv_sec * ORIGINAL_NS_PER_SECOND +
			       (uint64_t)now.tv_nsec :
		       0;
}

static bool patch_read(const char *path,
		       uint32_t grid[ORIGINAL_GRID_DIMENSIONS],
		       unsigned char **bytes_out, uint64_t *count_out)
{
	FILE *const file = fopen(path, "rb");
	if (file == NULL) {
		return false;
	}
	bool passed = fread(grid, sizeof(*grid), ORIGINAL_GRID_DIMENSIONS,
			    file) == ORIGINAL_GRID_DIMENSIONS &&
		      grid[0] != 0 && grid[1] != 0 &&
		      grid[0] <= VISION_MAXIMUM_PATCHES / grid[1];
	if (passed) {
		const uint64_t cell_height =
			(grid[0] + VISION_DOWNSAMPLE_RATIO - 1u) /
			VISION_DOWNSAMPLE_RATIO;
		const uint64_t cell_width =
			(grid[1] + VISION_DOWNSAMPLE_RATIO - 1u) /
			VISION_DOWNSAMPLE_RATIO;
		passed =
			cell_height * (cell_width + 1u) + VISION_HALF_DIVISOR <=
			VISION_MAXIMUM_TOKENS;
	}
	const uint64_t bytes =
		passed ? (uint64_t)grid[0] * grid[1] * VISION_CHANNELS *
				 VISION_PATCH_SIZE * VISION_PATCH_SIZE *
				 VISION_BFLOAT16_BYTES :
			 0;
	passed = passed && bytes <= ORIGINAL_PATCH_BYTES;
	unsigned char *const patches = passed ? malloc((size_t)bytes) : NULL;
	passed = passed && patches != NULL &&
		 fread(patches, 1, (size_t)bytes, file) == bytes &&
		 fgetc(file) == EOF && ferror(file) == 0;
	passed = fclose(file) == 0 && passed;
	if (!passed) {
		free(patches);
		return false;
	}
	*bytes_out = patches;
	*count_out = bytes;
	return true;
}

/* The real pack keeps the entire tower in one stage extent. The metadata layout alone includes other stages;
 * only this one checked extent is read/uploaded, and only its ranges are consumed by the vision planner. */
static bool tower_map(const struct pack_record *pack,
		      struct load_layout **layout_out,
		      struct model_vision *vision_out,
		      const struct pack_extent **extent_out)
{
	const struct pack_header *const header = pack_header_of(pack);
	uint32_t *const cards = calloc(header->unit_count, sizeof(*cards));
	if (cards == NULL) {
		return false;
	}
	const struct pack_unit_record_2 *const units = pack_units_of(pack);
	for (uint32_t index = 0; index < header->unit_count; index++) {
		cards[index] = units[index].unit_kind ==
					       PLACEMENT_UNIT_HOST_ONLY ?
				       PLACEMENT_IN_HOST_MEMORY :
				       0u;
	}
	const uint32_t layers[MODEL_LAYERS] = { 0 };
	bool passed = load_layout_create(pack, cards, header->unit_count, 1,
					 layout_out) == STATUS_OK;
	free(cards);
	passed = passed &&
		 model_vision_weights(pack, *layout_out, layers, vision_out) ==
			 STATUS_OK &&
		 vision_out->enabled;
	if (!passed) {
		return false;
	}
	const char patch_name[] = "vision.patch_embed.proj.weight";
	const uint32_t patch = pack_find_tensor(
		pack, patch_name, (uint32_t)sizeof(patch_name) - 1u);
	if (patch == PACK_NOT_FOUND) {
		return false;
	}
	const struct pack_tensor *const tensors = pack_tensors_of(pack);
	const uint32_t extent = tensors[patch].extent_index;
	const char *const names = pack_name_bytes_of(pack);
	for (uint32_t index = 0; index < header->tensor_count; index++) {
		const char *const name =
			names + tensors[index].name_byte_offset;
		const uint32_t size = tensors[index].name_byte_count;
		const bool tower =
			(size >= sizeof("vision.") - 1u &&
			 memcmp(name, "vision.", sizeof("vision.") - 1u) ==
				 0) ||
			(size >= sizeof("aligner.") - 1u &&
			 memcmp(name, "aligner.", sizeof("aligner.") - 1u) ==
				 0) ||
			(size >= sizeof("image_") - 1u &&
			 memcmp(name, "image_", sizeof("image_") - 1u) == 0);
		if (tower && tensors[index].extent_index != extent) {
			return false;
		}
	}
	*extent_out = &pack_extents_of(pack)[extent];
	return (*layout_out)->byte_offset_of_extent[extent] == 0;
}

static bool tower_upload(const struct storage_reader *reader,
			 const struct pack_extent *extent,
			 struct device_card *card, void *staging)
{
	for (uint64_t offset = 0; offset < extent->byte_count;) {
		const uint64_t remaining = extent->byte_count - offset;
		const uint64_t count = remaining < ORIGINAL_STAGING_BYTES ?
					       remaining :
					       ORIGINAL_STAGING_BYTES;
		uint64_t ticket = 0;
		if (storage_direct.reader_read(reader, 0,
					       extent->byte_offset + offset,
					       staging, count) != STATUS_OK ||
		    backend->upload_begin(card, offset, staging, count,
					  &ticket) != STATUS_OK ||
		    backend->upload_wait(card, ticket, ORIGINAL_WAIT_NS) !=
			    STATUS_OK) {
			return false;
		}
		offset += count;
	}
	uint64_t digest = 0;
	return backend->digest(card, 0, extent->byte_count, ORIGINAL_WAIT_NS,
			       &digest) == STATUS_OK &&
	       digest == extent->digest;
}

static bool trace_write(struct device_card *card, void *staging,
			uint64_t offset, uint64_t bytes, const char *path)
{
	FILE *const file = fopen(path, "wb");
	if (file == NULL) {
		return false;
	}
	bool passed = true;
	while (passed && bytes != 0) {
		const uint64_t count = bytes < ORIGINAL_STAGING_BYTES ?
					       bytes :
					       ORIGINAL_STAGING_BYTES;
		uint64_t ticket = 0;
		passed = backend->download_begin(card, offset, staging, count,
						 &ticket) == STATUS_OK &&
			 backend->upload_wait(card, ticket, ORIGINAL_WAIT_NS) ==
				 STATUS_OK &&
			 fwrite(staging, 1, (size_t)count, file) == count;
		offset += count;
		bytes -= count;
	}
	return fclose(file) == 0 && passed;
}

struct original_trace_range {
	uint64_t offset, bytes;
	uint32_t writer_pass;
};

static bool trace_restore(struct device_card *card, void *staging,
			  const struct original_trace_range *range,
			  const char *directory)
{
	char path[ORIGINAL_TRACE_PATH_BYTES];
	const int length = snprintf(path, sizeof(path), "%s/pass_%03u.bf16",
				    directory, range->writer_pass);
	if (length <= 0 || (uint32_t)length >= sizeof(path)) {
		return false;
	}
	FILE *const file = fopen(path, "rb");
	if (file == NULL) {
		return false;
	}
	bool passed = true;
	for (uint64_t offset = 0; passed && offset < range->bytes;) {
		const uint64_t remaining = range->bytes - offset;
		const uint64_t count = remaining < ORIGINAL_STAGING_BYTES ?
					       remaining :
					       ORIGINAL_STAGING_BYTES;
		uint64_t ticket = 0;
		passed = fread(staging, 1, (size_t)count, file) == count &&
			 backend->upload_begin(card, range->offset + offset,
					       staging, count,
					       &ticket) == STATUS_OK &&
			 backend->upload_wait(card, ticket, ORIGINAL_WAIT_NS) ==
				 STATUS_OK;
		offset += count;
	}
	passed = passed && fgetc(file) == EOF && ferror(file) == 0;
	return fclose(file) == 0 && passed;
}

/* Diagnostic replay is outside every timing scope. Every pass rewrites its complete output before capture. */
static bool tower_trace(struct device_card *card, void *staging,
			const struct kernel_launch *launches, uint32_t count,
			uint64_t arena_bytes,
			const struct vision_kernels *kernels)
{
	const char *const directory = getenv("VISION_TRACE_DIRECTORY");
	if (directory == NULL) {
		return true;
	}
	if (directory[0] == '\0') {
		return false;
	}
	const char *const reference_directory =
		getenv("VISION_TRACE_REFERENCE_DIRECTORY");
	if (reference_directory != NULL &&
	    (reference_directory[0] == '\0' ||
	     strcmp(reference_directory, directory) == 0)) {
		return false;
	}
	struct original_trace_range ranges[ORIGINAL_TRACE_RANGE_CAPACITY] = {
		{ 0 }
	};
	uint32_t range_count = 0;
	for (uint32_t pass = 0; pass < count; pass++) {
		for (uint32_t index = 0;
		     reference_directory != NULL && index < range_count;
		     index++) {
			if (!trace_restore(card, staging, &ranges[index],
					   reference_directory)) {
				return false;
			}
		}
		struct vision_arguments record;
		memcpy(&record, launches[pass].argument, sizeof(record));
		const uint64_t components =
			launches[pass].kernel ==
					kernels->operations[VISION_ROPE] ?
				VISION_QKV_COMPONENTS :
				1u;
		const uint64_t bytes = (uint64_t)record.rows *
				       record.output_width * components *
				       VISION_BFLOAT16_BYTES;
		char path[ORIGINAL_TRACE_PATH_BYTES];
		const int length = snprintf(path, sizeof(path),
					    "%s/pass_%03u.bf16", directory,
					    pass);
		if (bytes == 0 || record.destination_offset > arena_bytes ||
		    bytes > arena_bytes - record.destination_offset ||
		    length <= 0 || (uint32_t)length >= sizeof(path) ||
		    backend->launch_run(card, launches + pass, 1) !=
			    STATUS_OK ||
		    backend->launch_wait(card, ORIGINAL_WAIT_NS) != STATUS_OK ||
		    !trace_write(card, staging, record.destination_offset,
				 bytes, path)) {
			return false;
		}
		uint32_t destination = 0;
		while (destination < range_count &&
		       ranges[destination].offset !=
			       record.destination_offset) {
			destination++;
		}
		if (destination == ORIGINAL_TRACE_RANGE_CAPACITY) {
			return false;
		}
		range_count += destination == range_count ? 1u : 0u;
		ranges[destination] = (struct original_trace_range){
			record.destination_offset, bytes, pass
		};
	}
	(void)printf(
		"original vision traced_passes=%u isolated=%u directory=%s\n",
		count, reference_directory != NULL ? 1u : 0u, directory);
	return true;
}

static bool tower_run(struct device_card *card,
		      const struct vision_description *description,
		      uint64_t arena_bytes,
		      const struct vision_kernels *kernels,
		      const struct original_partitions *partitions)
{
	struct kernel_launch launches[VISION_MAXIMUM_LAUNCHES];
	uint32_t count = 0, fill_kernel = 0;
	if (!tower_prepare(description, arena_bytes, kernels, partitions,
			   launches, &count) ||
	    backend->kernel_prepare(card, KERNEL_FILE_FILL, KERNEL_FILL_WORDS,
				    FILL_RECORD_BYTES,
				    &fill_kernel) != STATUS_OK) {
		return false;
	}
	const struct fill_record record = { description->scratch.byte_offset,
					    description->scratch.byte_count /
						    FILL_WORD_BYTES,
					    0, 0 };
	struct kernel_launch flush = {
		.kernel = fill_kernel,
		.grid = { FILL_MOST_BLOCKS, 1, 1 },
		.block = { FILL_THREADS_PER_BLOCK_VALUE, 1, 1 },
		.argument_byte_count = FILL_RECORD_BYTES
	};
	memcpy(flush.argument, &record, sizeof(record));
	for (uint32_t repeat = 0; repeat <= ORIGINAL_REPEATS; repeat++) {
		if (backend->launch_run(card, &flush, 1) != STATUS_OK ||
		    backend->launch_wait(card, ORIGINAL_WAIT_NS) != STATUS_OK) {
			return false;
		}
		const uint64_t began = now_ns();
		if (began == 0 ||
		    backend->launch_run(card, launches, count) != STATUS_OK ||
		    backend->launch_wait(card, ORIGINAL_WAIT_NS) != STATUS_OK) {
			return false;
		}
		const uint64_t ended = now_ns();
		if (ended <= began) {
			return false;
		}
		(void)printf(
			"original vision grid=%ux%u launches=%u scratch_flush_bytes=%llu warmup=%u encoder_ns=%llu\n",
			description->grid_height, description->grid_width,
			count,
			(unsigned long long)description->scratch.byte_count,
			repeat == 0 ? 1u : 0u,
			(unsigned long long)(ended - began));
	}
	return true;
}

bool test_vision_original_card(const char *pack_path, const char *patch_path,
			       const char *output_path)
{
	const char *const guarded = getenv("UNDER_GPU_RUN");
	if (guarded == NULL || strcmp(guarded, "1") != 0) {
		(void)puts("original vision requires gpu_run.sh");
		return false;
	}
	if (getenv("VISION_TRACE_REFERENCE_DIRECTORY") != NULL &&
	    getenv("VISION_TRACE_DIRECTORY") == NULL) {
		(void)puts("isolated vision requires VISION_TRACE_DIRECTORY");
		return false;
	}
	struct original_partitions partitions;
	if (!aligner_bias_last_read(&partitions) ||
	    !partition_read("VISION_TEST_MLP_K_PARTITION",
			    VISION_MAXIMUM_INTERMEDIATE, &partitions.mlp) ||
	    !partition_read("VISION_TEST_MLP_UP_K_PARTITION",
			    VISION_MAXIMUM_WIDTH, &partitions.mlp_up) ||
	    !partition_read("VISION_TEST_ALIGNER_K_PARTITION",
			    VISION_MAXIMUM_WIDTH * VISION_DOWNSAMPLE_RATIO *
				    VISION_DOWNSAMPLE_RATIO,
			    &partitions.aligner)) {
		(void)puts(
			"original vision refuses invalid experimental linear settings");
		return false;
	}
	(void)printf(
		"original vision experiment mlp_k_partition=%u mlp_up_k_partition=%u aligner_k_partition=%u "
		"aligner_bias_last=%u aligner_bias_last_override=%u\n",
		partitions.mlp, partitions.mlp_up, partitions.aligner,
		partitions.aligner_bias_last ? 1u : 0u,
		partitions.has_aligner_bias_last ? 1u : 0u);
	(void)alarm(ORIGINAL_DEADLINE_SECONDS);
	uint32_t count = 0, grid[ORIGINAL_GRID_DIMENSIONS] = { 0 };
	struct card_description_record card_description;
	if (backend->card_count(&count) != STATUS_OK || count != 1 ||
	    backend->card_describe(0, &card_description) != STATUS_OK ||
	    (strstr(card_description.name, "170HX") == NULL &&
	     strstr(card_description.name, "RTX PRO 6000") == NULL)) {
		return false;
	}
	unsigned char *patches = NULL;
	uint64_t patch_bytes = 0;
	struct storage_reader *reader = NULL;
	struct pack_record *pack = NULL;
	struct load_layout *layout = NULL;
	struct model_vision vision;
	struct device_card *card = NULL;
	void *staging = NULL;
	const struct pack_extent *extent = NULL;
	struct vision_scratch scratch;
	const struct vision_config config = vision_official_config();
	bool passed = patch_read(patch_path, grid, &patches, &patch_bytes) &&
		      storage_direct.reader_open(&pack_path, 1, &reader) ==
			      STATUS_OK &&
		      pack_open_entry(&storage_direct, reader, &pack) ==
			      STATUS_OK &&
		      tower_map(pack, &layout, &vision, &extent);
	const uint64_t patch_offset = passed ? extent->byte_count : 0;
	const uint64_t output_offset =
		patch_offset + ((patch_bytes + VISION_ALIGNMENT - 1u) /
				VISION_ALIGNMENT * VISION_ALIGNMENT);
	const uint64_t cells =
		(uint64_t)((grid[0] + VISION_DOWNSAMPLE_RATIO - 1u) /
			   VISION_DOWNSAMPLE_RATIO) *
		((grid[1] + VISION_DOWNSAMPLE_RATIO - 1u) /
		 VISION_DOWNSAMPLE_RATIO);
	const uint64_t output_bytes =
		cells * VISION_MAXIMUM_LANGUAGE_WIDTH * VISION_BFLOAT16_BYTES;
	passed = passed &&
		 vision_scratch_plan(&config, ORIGINAL_MAX_GRID_HEIGHT,
				     ORIGINAL_MAX_GRID_WIDTH,
				     output_offset + output_bytes,
				     &scratch) == STATUS_OK &&
		 output_bytes <= ORIGINAL_STAGING_BYTES;
	const uint64_t arena_bytes =
		passed ? output_offset + output_bytes + scratch.byte_count : 0;
	struct vision_kernels kernels;
	const uint32_t arithmetic_class =
		card_description.arithmetic_class == ARITHMETIC_CLASS_SM_120 ?
			ARITHMETIC_CLASS_SM_120_NATIVE :
			card_description.arithmetic_class;
	passed = passed &&
		 backend->card_open(0, arena_bytes, &card) == STATUS_OK &&
		 backend->staging_allocate(ORIGINAL_STAGING_BYTES, &staging) ==
			 STATUS_OK &&
		 vision_kernels_prepare(backend, card, arithmetic_class,
					&kernels) == STATUS_OK &&
		 tower_upload(reader, extent, card, staging);
	uint64_t ticket = 0;
	if (passed) {
		memcpy(staging, patches, (size_t)patch_bytes);
		passed = backend->upload_begin(card, patch_offset, staging,
					       patch_bytes,
					       &ticket) == STATUS_OK &&
			 backend->upload_wait(card, ticket, ORIGINAL_WAIT_NS) ==
				 STATUS_OK;
	}
	const struct vision_description description = {
		.config = config,
		.weights = &vision.weights,
		.patches = { patch_offset, patch_bytes },
		.output = { output_offset, output_bytes },
		.scratch = { .byte_offset = output_offset + output_bytes,
			     .byte_count = passed ? scratch.byte_count : 0 },
		.grid_height = grid[0],
		.grid_width = grid[1]
	};
	passed = passed &&
		 tower_run(card, &description, arena_bytes, &kernels,
			   &partitions) &&
		 backend->download_begin(card, output_offset, staging,
					 output_bytes, &ticket) == STATUS_OK &&
		 backend->upload_wait(card, ticket, ORIGINAL_WAIT_NS) ==
			 STATUS_OK;
	if (passed) {
		FILE *const file = fopen(output_path, "wb");
		passed = file != NULL;
		if (file != NULL) {
			const bool written = fwrite(staging, 1,
						    (size_t)output_bytes,
						    file) == output_bytes;
			passed = fclose(file) == 0 && written;
		}
	}
	/* Persist the complete forward result before an isolated replay can replace the final arena output. */
	if (passed && getenv("VISION_TRACE_DIRECTORY") != NULL) {
		struct kernel_launch launches[VISION_MAXIMUM_LAUNCHES];
		uint32_t launch_count = 0;
		passed = tower_prepare(&description, arena_bytes, &kernels,
				       &partitions, launches, &launch_count) &&
			 tower_trace(card, staging, launches, launch_count,
				     arena_bytes, &kernels);
	}
	if (card != NULL) {
		backend->card_close(card, ORIGINAL_WAIT_NS);
	}
	if (staging != NULL) {
		backend->staging_free(staging);
	}
	load_layout_destroy(layout);
	pack_close(pack);
	storage_direct.reader_close(reader);
	free(patches);
	(void)printf("original vision %s output_bytes=%llu %s\n",
		     card_description.name, (unsigned long long)output_bytes,
		     passed ? "passed" : "failed");
	return passed;
}
