/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * backend_cuda_launch.c: what a card runs, on the CUDA driver interface: kernel_prepare, launch_run and launch_wait
 * of backend_seam.h, and the modules the kernels come from. Where the seam leaves a choice:
 *   - A module is the build of one kernel file for the card's architecture, from the table of
 *     backend_cuda_kernel_images.h. A card loads it for the first kernel it prepares from the file and holds it
 *     until it is closed; the digest's is loaded when the card is opened. A module loaded for a kernel_prepare that
 *     is then refused is unloaded again: a refusal leaves nothing behind.
 *   - A kernel handle is the number of kernels the card had prepared before it: cards that prepare the same kernels
 *     in the same order give the same handles, so one prepared launch serves each of them. A handle of another card
 *     that is below this card's count names this card's kernel of that number; only its record size can tell.
 *   - kernel_prepare answers INVALID_ARGUMENT for a missing argument, a function name of no bytes or of
 *     KERNEL_FUNCTION_NAME_CAPACITY and more, a record of no bytes or of more than LAUNCH_ARGUMENT_BYTES,
 *     and a kernel file no architecture has a build of: a name this program does not know. UNSUPPORTED is a kernel
 *     file that only another architecture has a build of (the build makes every file for every architecture, so no
 *     card it opens shows that today) and one kernel more than BACKEND_MAXIMUM_KERNELS. Every refusal is one
 *     error line. It loads the function, so that the first launch pays nothing, and asks the driver for the most
 *     threads and the most dynamic shared memory a block of that function may have on this card: no launch gets
 *     more.
 *   - launch_run checks every launch before it queues the first. Beyond the list of the seam it refuses a missing
 *     table, a count of 0, a grid or block dimension above the card's, and a byte behind the record that is not
 *     zero: what the driver would reject never reaches it. One error line names the first launch refused. When the
 *     driver does reject a launch, those before it in the call stay queued and the card has failed.
 *   - launch_wait records the card's launch event behind everything queued when it is called, and polls it as
 *     upload_wait polls the event of a ticket. Waits at the same time share the event: the earlier one then waits
 *     for what the later one saw as well. It reports no ticket finished: that is upload_wait's alone.
 *   - A driver error in any of the three fails the card, as backend_cuda.c says of an upload. On a failed card
 *     they answer DEVICE_ERROR behind the refusals of their arguments.
 *   - Beside the driver calls there is one fault point, backend_cuda_kernel_table_full: no build has as many
 *     kernels as a card may prepare.
 */
#include "backend_cuda_card.h"

#include <inttypes.h>
#include <string.h>

#include "anemone/exchange_records.h"

#define KERNEL_PARAMETER_COUNT 2u               /* of every kernel: its record, then the address of the arena */

enum kernel_parameter { RECORD_PARAMETER, ADDRESS_PARAMETER };
enum launch_dimension { DIMENSION_X, DIMENSION_Y, DIMENSION_Z, DIMENSION_COUNT };

_Static_assert(DIMENSION_COUNT == LAUNCH_DIMENSIONS,
	       "a launch has the three dimensions of the driver");

static const CUdevice_attribute most_grid_attributes[DIMENSION_COUNT] = {
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z};
static const CUdevice_attribute most_block_attributes[DIMENSION_COUNT] = {
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z};
static const unsigned char zero_bytes[LAUNCH_ARGUMENT_BYTES] = {
	0
}; /* what lies behind a record */

bool cuda_launch_limits_read(struct device_card *card, CUdevice device)
{
    for (uint32_t dimension = 0; dimension < DIMENSION_COUNT; dimension++) {
        int most_grid = 0;
        int most_block = 0;
        if (!DRIVER_CALL(cuDeviceGetAttribute, &most_grid, most_grid_attributes[dimension], device) ||
            !DRIVER_CALL(cuDeviceGetAttribute, &most_block, most_block_attributes[dimension], device) ||
            most_grid < 1 || most_block < 1) {
		LOG_ERROR(
			"a card states no limits of a launch (dimension %" PRIu32
			")",
			dimension);
		return false;
        }
        card->launch_limits.most_grid[dimension] = (uint32_t)most_grid;
        card->launch_limits.most_block[dimension] = (uint32_t)most_block;
    }
    return true;
}

/* The build of a kernel file for an architecture, or for the first that has one if architecture is NULL; NULL if
 * there is none. */
static const struct cuda_kernel_image *
kernel_image_of(const char *architecture, const char *kernel_file_name)
{
	for (uint32_t image_index = 0; image_index < cuda_kernel_image_count;
	     image_index++) {
		const struct cuda_kernel_image *const image =
			&cuda_kernel_images[image_index];
		if (image->bytes != NULL && image->byte_count != 0 &&
		    strcmp(image->kernel_name, kernel_file_name) == 0 &&
		    (architecture == NULL ||
		     strcmp(image->architecture, architecture) == 0)) {
			return image;
		}
	}
    return NULL;
}

/* The index of the card's module of an image: the one it holds, or loaded now as its last. module_count, and no
 * module more, when the driver does not load it. There is room: a card holds at most one module more than it has
 * prepared kernels, and none is loaded for a kernel the table of kernels has no room for. */
static uint32_t module_index_of(struct open_card_state *state,
				const struct cuda_kernel_image *image)
{
    uint32_t module_index = 0;
    while (module_index < state->module_count && state->modules[module_index].image != image) {
        module_index++;
    }
    ASSERT(state->module_count < MODULE_CAPACITY ||
	   module_index < state->module_count);
    if (module_index == state->module_count && module_index < MODULE_CAPACITY &&
        DRIVER_CALL(cuModuleLoadData, &state->modules[module_index].module, image->bytes)) {
        state->modules[module_index].image = image;
        state->module_count++;
    }
    return module_index;
}

MUST_CHECK result_status cuda_module_of(const struct device_card *card,
					struct open_card_state *state,
					const char *kernel_file_name,
					CUmodule *module_out)
{
	const struct cuda_kernel_image *const image =
		kernel_image_of(card->architecture, kernel_file_name);
	const uint32_t module_index = image == NULL ?
					      state->module_count :
					      module_index_of(state, image);
	if (module_index == state->module_count) {
		return image == NULL ? STATUS_UNSUPPORTED : STATUS_DEVICE_ERROR;
	}
    *module_out = state->modules[module_index].module;
    return STATUS_OK;
}

void cuda_modules_unload(struct open_card_state *state)
{
    for (uint32_t module_index = 0; module_index < state->module_count; module_index++) {
        (void)DRIVER_CALL(cuModuleUnload, state->modules[module_index].module);
    }
    state->module_count = 0;
    state->kernel_count = 0;
}

/* Finds a function in a module, loads it and reads what the card's build says of it: everything of *kernel_out but
 * its kernel file. INVALID_ARGUMENT with one error line when the module has no such function or it is not a kernel
 * that takes a record of argument_byte_count bytes and then an address; DEVICE_ERROR when the driver fails. */
MUST_CHECK static result_status kernel_read(CUmodule module,
					    const char *function_name,
					    uint32_t argument_byte_count,
					    struct prepared_kernel *kernel_out)
{
    CUfunction function = NULL;
    size_t parameter_count = 0;
    size_t record_offset = 0;
    size_t record_byte_count = 0;
    size_t address_offset = 0;
    size_t address_byte_count = 0;
    int most_threads = 0;
    int most_shared_bytes = 0;
    CUdevice device = 0;
    int card_most_shared_bytes = 0;             /* of a block that asks for it */
    int static_shared_bytes = 0;
    const CUresult result =
	    FAULT_POINT(backend_cuda_cuModuleGetFunction) ?
		    CUDA_ERROR_UNKNOWN :
		    cuModuleGetFunction(&function, module, function_name);
    if (result == CUDA_ERROR_NOT_FOUND) {
	    LOG_ERROR("kernel_prepare: the kernel file has no function %s",
		      function_name);
	    return STATUS_INVALID_ARGUMENT;
    }
    if (!cuda_driver_succeeded("cuModuleGetFunction", result) ||
	!DRIVER_CALL(cuFuncLoad, function) ||
	!DRIVER_CALL(cuFuncGetParamCount, function, &parameter_count)) {
	    return STATUS_DEVICE_ERROR;
    }
    if (parameter_count != KERNEL_PARAMETER_COUNT) {
	    LOG_ERROR(
		    "kernel_prepare: %s takes %zu arguments: a kernel takes a record and an address",
		    function_name, parameter_count);
	    return STATUS_INVALID_ARGUMENT;
    }
    if (!DRIVER_CALL(cuFuncGetParamInfo, function, RECORD_PARAMETER, &record_offset, &record_byte_count) ||
        !DRIVER_CALL(cuFuncGetParamInfo, function, ADDRESS_PARAMETER, &address_offset, &address_byte_count) ||
        !DRIVER_CALL(cuFuncGetAttribute, &most_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function) ||
        !DRIVER_CALL(cuCtxGetDevice, &device) ||
        !DRIVER_CALL(cuDeviceGetAttribute, &card_most_shared_bytes,
                     CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device) ||
        !DRIVER_CALL(cuFuncGetAttribute, &static_shared_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function) ||
        !DRIVER_CALL(cuFuncSetAttribute, function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                     card_most_shared_bytes - static_shared_bytes) ||
        !DRIVER_CALL(cuFuncGetAttribute, &most_shared_bytes, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                     function)) {
	    return STATUS_DEVICE_ERROR;
    }
    if (most_threads < 1 || most_shared_bytes < 0) {
	    LOG_ERROR(
		    "kernel_prepare: the driver gives a block of %s %d threads and %d bytes of shared memory",
		    function_name, most_threads, most_shared_bytes);
	    return STATUS_DEVICE_ERROR;
    }
    if (record_offset != 0 || record_byte_count != argument_byte_count || address_offset < record_byte_count ||
        address_byte_count != sizeof(CUdeviceptr)) {
	    LOG_ERROR(
		    "kernel_prepare: %s takes %zu bytes and then %zu bytes, not a record of %" PRIu32
		    " bytes and then an address",
		    function_name, record_byte_count, address_byte_count,
		    argument_byte_count);
	    return STATUS_INVALID_ARGUMENT;
    }
    *kernel_out = (struct prepared_kernel){.function = function,
                                           .argument_byte_count = argument_byte_count,
                                           .most_threads_per_block = (uint32_t)most_threads,
                                           .most_dynamic_shared_byte_count = (uint32_t)most_shared_bytes};
    return STATUS_OK;
}

/* kernel_prepare on an entered card. */
MUST_CHECK static result_status
kernel_prepare_entered(const struct device_card *card,
		       struct open_card_state *state,
		       const char *kernel_file_name, const char *function_name,
		       uint32_t argument_byte_count, uint32_t *kernel_out)
{
    const size_t name_length = function_name == NULL ? 0 : strnlen(function_name, KERNEL_FUNCTION_NAME_CAPACITY);
    const uint32_t module_count_before = state->module_count;
    struct prepared_kernel kernel = {0};
    if (kernel_file_name == NULL || function_name == NULL ||
	kernel_out == NULL || name_length == 0 ||
	name_length == KERNEL_FUNCTION_NAME_CAPACITY ||
	argument_byte_count == 0 ||
	argument_byte_count > LAUNCH_ARGUMENT_BYTES) {
	    LOG_ERROR(
		    "kernel_prepare: a missing argument, or a function name or a record of no bytes or too many");
	    return STATUS_INVALID_ARGUMENT;
    }
    const struct cuda_kernel_image *const image =
	    kernel_image_of(card->architecture, kernel_file_name);
    if (image == NULL && kernel_image_of(NULL, kernel_file_name) == NULL) {
	    LOG_ERROR("kernel_prepare: no architecture has a build of %s",
		      kernel_file_name);
	    return STATUS_INVALID_ARGUMENT;
    }
    if (state->first_failed_ticket != 0) {
	    return STATUS_DEVICE_ERROR;
    }
    if (image == NULL) {
	    LOG_ERROR("kernel_prepare: no build of %s for %s", kernel_file_name,
		      card->architecture);
	    return STATUS_UNSUPPORTED;
    }
    for (uint32_t kernel_index = 0; kernel_index < state->kernel_count; kernel_index++) {
        const struct prepared_kernel *const prepared = &state->kernels[kernel_index];
        if (prepared->image == image && strcmp(state->kernel_function_names[kernel_index], function_name) == 0) {
            if (prepared->argument_byte_count != argument_byte_count) {
		    LOG_ERROR("kernel_prepare: %s takes a record of %" PRIu32
			      " bytes, not of %" PRIu32,
			      function_name, prepared->argument_byte_count,
			      argument_byte_count);
		    return STATUS_INVALID_ARGUMENT;
	    }
            *kernel_out = kernel_index;         /* prepared before: the same handle */
	    return STATUS_OK;
	}
    }
    if (state->kernel_count >= BACKEND_MAXIMUM_KERNELS ||
	FAULT_POINT(backend_cuda_kernel_table_full)) {
	    LOG_ERROR("kernel_prepare: the card has prepared %" PRIu32
		      " kernels and takes no more",
		      state->kernel_count);
	    return STATUS_UNSUPPORTED;
    }
    const uint32_t module_index = module_index_of(state, image);
    const result_status status =
	    module_index == state->module_count ?
		    STATUS_DEVICE_ERROR :
		    kernel_read(state->modules[module_index].module,
				function_name, argument_byte_count, &kernel);
    if (status != STATUS_OK) {
	    if (state->module_count >
		module_count_before) { /* loaded for this call: it is not left behind */
		    state->module_count = module_count_before;
		    (void)DRIVER_CALL(
			    cuModuleUnload,
			    state->modules[module_count_before].module);
	    }
	    if (status == STATUS_DEVICE_ERROR) {
		    state->first_failed_ticket =
			    state->last_ticket_finished + 1;
	    }
	    return status;
    }
    kernel.image = image;
    state->kernels[state->kernel_count] = kernel;
    memcpy(state->kernel_function_names[state->kernel_count], function_name, name_length);
    state->kernel_function_names[state->kernel_count][name_length] = '\0';
    *kernel_out = state->kernel_count;
    state->kernel_count++;
    return STATUS_OK;
}

MUST_CHECK result_status cuda_kernel_prepare(struct device_card *card,
					     const char *kernel_file_name,
					     const char *function_name,
					     uint32_t argument_byte_count,
					     uint32_t *kernel_out)
{
	result_status status = STATUS_OK;
	struct open_card_state *const state =
		cuda_card_enter(card, true, &status);
	if (state == NULL) {
		return status;
	}
    status = kernel_prepare_entered(card, state, kernel_file_name, function_name, argument_byte_count, kernel_out);
    cuda_card_leave(card);
    return status;
}

/* The side runs behind a graph node of branch (before: the branch of the launch before): a SIDE run's first node opens
 * a run and a later one ends it anew, SIDE_MORE opens a run of its own, JOIN closes them all. */
static uint32_t side_mark(uint32_t branch, uint32_t before, CUgraphNode node,
			  CUgraphNode side_lasts[LAUNCH_MOST_SIDE_RUNS],
			  uint32_t side_count)
{
	if (branch == LAUNCH_JOIN)
		return 0;
	if (branch == LAUNCH_SIDE_MORE ||
	    (branch == LAUNCH_SIDE && before != LAUNCH_SIDE)) {
		side_lasts[side_count] = node;
		return side_count + 1u;
	}
	if (branch == LAUNCH_SIDE)
		side_lasts[side_count - 1u] = node;
	return side_count;
}

/* A copy's place: the card's own arena, or the owner's arena behind an address word peer_connect wrote; false when
 * it does not hold byte_count bytes from the place's offset. */
static bool copy_place(const struct open_card_state *state,
		       struct exchange_place place, uint64_t byte_count,
		       CUdeviceptr *address_out)
{
    CUdeviceptr base = state->arena_address;
    uint64_t limit = state->arena_byte_count;
    if (place.address_word_offset != EXCHANGE_OWN_ARENA) {
	    uint32_t entry = 0;
	    while (entry < state->peer_word_count &&
		   state->peer_words[entry].word != place.address_word_offset)
		    entry++;
	    if (entry == state->peer_word_count)
		    return false;
	    base = state->peer_words[entry].address;
	    limit = state->peer_words[entry].byte_count;
    }
    if (place.offset > limit || byte_count > limit - place.offset)
        return false;
    *address_out = base + place.offset;
    return true;
}

/* Whether launch_run may hand a launch to the driver: its kernel is prepared, its record has the kernel's size,
 * every dimension is at least 1 and at most the card's, its block has no more threads and no more dynamic shared
 * memory than the driver gives the kernel, its branch one of LAUNCH_* and the bytes behind its record zero. */
static bool launch_is_valid(const struct card_launch_limits *limits,
			    const struct open_card_state *state,
			    const struct kernel_launch *launch)
{
    uint64_t thread_count = 1;
    if (launch->kernel == BACKEND_COPY) {
	    if (launch->branch > LAUNCH_SIDE_MORE ||
		launch->dynamic_shared_byte_count != 0 ||
		launch->argument_byte_count != EXCHANGE_COPY_ARGUMENT_BYTES ||
		memcmp(launch->argument + EXCHANGE_COPY_ARGUMENT_BYTES,
		       zero_bytes,
		       LAUNCH_ARGUMENT_BYTES - EXCHANGE_COPY_ARGUMENT_BYTES) !=
			0) {
		    return false;
	    }
	    struct exchange_copy_arguments copy;
	    memcpy(&copy, launch->argument, sizeof(copy));
	    CUdeviceptr source = 0, destination = 0;
	    const bool own =
		    copy.source.address_word_offset == EXCHANGE_OWN_ARENA &&
		    copy.destination.address_word_offset == EXCHANGE_OWN_ARENA;
	    return copy.byte_count != 0 &&
		   copy_place(state, copy.source, copy.byte_count, &source) &&
		   copy_place(state, copy.destination, copy.byte_count,
			      &destination) &&
		   (!own || source + copy.byte_count <= destination ||
		    destination + copy.byte_count <= source);
    }
    if (launch->kernel >= state->kernel_count) {
        return false;
    }
    const struct prepared_kernel *const kernel = &state->kernels[launch->kernel];
    /* the record's size is looked at before the bytes behind it are: no byte outside the argument is read */
    bool is_valid =
	    launch->branch <= LAUNCH_SIDE_MORE &&
	    launch->argument_byte_count == kernel->argument_byte_count &&
	    launch->argument_byte_count <= LAUNCH_ARGUMENT_BYTES &&
	    launch->dynamic_shared_byte_count <=
		    kernel->most_dynamic_shared_byte_count &&
	    memcmp(launch->argument + launch->argument_byte_count, zero_bytes,
		   LAUNCH_ARGUMENT_BYTES - launch->argument_byte_count) == 0;
    for (uint32_t dimension = 0; is_valid && dimension < DIMENSION_COUNT; dimension++) {
        thread_count *= launch->block[dimension];       /* below 2^31 before it, as the kernel's most is: no wrap */
        is_valid = launch->grid[dimension] != 0 && launch->grid[dimension] <= limits->most_grid[dimension] &&
                   launch->block[dimension] != 0 && launch->block[dimension] <= limits->most_block[dimension] &&
                   thread_count <= kernel->most_threads_per_block;
    }
    return is_valid;
}

/* Shared by eager launches and graph preparation: a late bad entry cannot queue any earlier entry. */
MUST_CHECK static result_status
launches_validate(const struct device_card *card,
		  const struct open_card_state *state,
		  const struct kernel_launch *launches, uint32_t launch_count)
{
    uint32_t valid_count = 0;
    while (launches != NULL && valid_count < launch_count &&
           launch_is_valid(&card->launch_limits, state, &launches[valid_count])) {
        valid_count++;
    }
    if (launches == NULL || launch_count == 0 || valid_count < launch_count) {
	    LOG_ERROR("launch table: refused at launch %" PRIu32 " of %" PRIu32
		      ": nothing is queued",
		      valid_count, launch_count);
	    return STATUS_INVALID_ARGUMENT;
    }
    return state->first_failed_ticket == 0 ? STATUS_OK : STATUS_DEVICE_ERROR;
}

MUST_CHECK result_status cuda_launch_run(struct device_card *card,
					 const struct kernel_launch *launches,
					 uint32_t launch_count)
{
	result_status status = STATUS_OK;
	struct open_card_state *const state =
		cuda_card_enter(card, true, &status);
	if (state == NULL) {
		return status;
	}
    status = launches_validate(card, state, launches, launch_count);
    for (uint32_t launch_index = 0;
	 status == STATUS_OK && launch_index < launch_count; launch_index++) {
	    const struct kernel_launch *const launch = &launches[launch_index];
	    if (launch->kernel == BACKEND_COPY) {
		    struct exchange_copy_arguments copy;
		    memcpy(&copy, launch->argument, sizeof(copy));
		    CUdeviceptr source = 0, destination = 0; /* checked above */
		    if (!copy_place(state, copy.source, copy.byte_count,
				    &source) ||
			!copy_place(state, copy.destination, copy.byte_count,
				    &destination) ||
			!DRIVER_CALL(cuMemcpyDtoDAsync, destination, source,
				     copy.byte_count, state->stream)) {
			    state->first_failed_ticket =
				    state->last_ticket_finished + 1;
			    status = STATUS_DEVICE_ERROR;
		    }
		    continue;
	    }
	    /* the driver copies each parameter from where its pointer points and writes nothing there: its interface
         * has no const for it */
	    void *kernel_parameters[] = { (void *)(uintptr_t)launch->argument,
					  &state->arena_address };
	    if (!DRIVER_CALL(
			cuLaunchKernel, state->kernels[launch->kernel].function,
			launch->grid[DIMENSION_X], launch->grid[DIMENSION_Y],
			launch->grid[DIMENSION_Z], launch->block[DIMENSION_X],
			launch->block[DIMENSION_Y], launch->block[DIMENSION_Z],
			launch->dynamic_shared_byte_count, state->stream,
			kernel_parameters, NULL)) {
		    state->first_failed_ticket =
			    state->last_ticket_finished + 1;
		    status = STATUS_DEVICE_ERROR;
	    }
    }
    if (status == STATUS_OK) {
	    COUNTER_ADD(backend_cuda_launches, launch_count);
    }
    cuda_card_leave(card);
    return status;
}

/* Build a dependency chain without executing or capturing the card's existing stream work. The driver copies
 * arguments at node creation. Definitions and uploaded executables live until card_close, including a partial
 * graph after a driver failure; close destroys them before their modules and arena. */
MUST_CHECK result_status cuda_graph_capture(
	struct device_card *card, const struct kernel_launch *launches,
	uint32_t launch_count, uint32_t *graph_out)
{
	result_status status = STATUS_OK;
	struct open_card_state *const state =
		cuda_card_enter(card, true, &status);
	if (state == NULL) {
		return status;
	}
	status = graph_out == NULL ?
			 STATUS_INVALID_ARGUMENT :
			 launches_validate(card, state, launches, launch_count);
	if (status == STATUS_OK &&
	    state->graph_count == BACKEND_MAXIMUM_GRAPHS) {
		status = STATUS_BUSY_VALUE;
	}
	if (status == STATUS_OK) {
		const uint32_t graph_index = state->graph_count;
		CUgraphNode previous = NULL,
			    fork = NULL; /* the branch's fork */
		CUgraphNode side_lasts[LAUNCH_MOST_SIDE_RUNS] = {
			NULL
		}; /* each side run's end */
		uint32_t side_count = 0;
		bool is_built = DRIVER_CALL(cuGraphCreate,
					    &state->graphs[graph_index], 0);
		for (uint32_t launch_index = 0;
		     is_built && launch_index < launch_count; launch_index++) {
			const struct kernel_launch *const launch =
				&launches[launch_index];
			const uint32_t
				branch = launch->branch,
				before = launch_index == 0 ?
						 LAUNCH_CHAIN :
						 launches[launch_index - 1u]
							 .branch;
			/* the side run begins behind the fork, SIDE_MORE runs too; RESUME takes the fork; JOIN the previous and
             * every side run (a run's first launch forks at the graph's root: behind none) */
			const bool after_side = before == LAUNCH_SIDE ||
						before == LAUNCH_SIDE_MORE;
			if (branch == LAUNCH_SIDE && !after_side) {
				fork = previous;
				side_count = 0;
			}
			if ((branch == LAUNCH_SIDE &&
			     before == LAUNCH_SIDE_MORE) ||
			    (branch == LAUNCH_SIDE_MORE &&
			     (!after_side ||
			      side_count == LAUNCH_MOST_SIDE_RUNS)) ||
			    (branch == LAUNCH_RESUME && !after_side) ||
			    (branch == LAUNCH_JOIN && side_count == 0)) {
				is_built = false;
				break;
			}
			CUgraphNode depends[1u + LAUNCH_MOST_SIDE_RUNS] = {
				branch == LAUNCH_RESUME ||
						branch == LAUNCH_SIDE_MORE ?
					fork :
					previous
			};
			size_t depend_count = depends[0] == NULL ? 0u : 1u;
			for (uint32_t run = 0;
			     branch == LAUNCH_JOIN && run < side_count; run++)
				depends[depend_count++] = side_lasts[run];
			if (launch->kernel == BACKEND_COPY) {
				struct exchange_copy_arguments copy;
				memcpy(&copy, launch->argument, sizeof(copy));
				CUdeviceptr
					source = 0,
					destination =
						0; /* checked by launches_validate */
				if (!copy_place(state, copy.source,
						copy.byte_count, &source) ||
				    !copy_place(state, copy.destination,
						copy.byte_count,
						&destination)) {
					is_built = false;
					break;
				}
				const CUDA_MEMCPY3D parameters = {
					.srcMemoryType = CU_MEMORYTYPE_DEVICE,
					.srcDevice = source,
					.srcPitch = copy.byte_count,
					.dstMemoryType = CU_MEMORYTYPE_DEVICE,
					.dstDevice = destination,
					.dstPitch = copy.byte_count,
					.WidthInBytes = copy.byte_count,
					.Height = 1,
					.Depth = 1
				};
				CUgraphNode next = NULL;
				is_built = DRIVER_CALL(
					cuGraphAddMemcpyNode, &next,
					state->graphs[graph_index],
					depend_count == 0 ? NULL : depends,
					depend_count, &parameters,
					card->context);
				side_count = side_mark(branch, before, next,
						       side_lasts, side_count);
				previous = next;
				continue;
			}
			void *kernel_parameters[] = {
				(void *)(uintptr_t)launch->argument,
				&state->arena_address
			};
			CUDA_KERNEL_NODE_PARAMS parameters = {
				.func = state->kernels[launch->kernel].function,
				.gridDimX = launch->grid[DIMENSION_X],
				.gridDimY = launch->grid[DIMENSION_Y],
				.gridDimZ = launch->grid[DIMENSION_Z],
				.blockDimX = launch->block[DIMENSION_X],
				.blockDimY = launch->block[DIMENSION_Y],
				.blockDimZ = launch->block[DIMENSION_Z],
				.sharedMemBytes =
					launch->dynamic_shared_byte_count,
				.kernelParams = kernel_parameters
			};
			CUgraphNode next = NULL;
			is_built =
				DRIVER_CALL(cuGraphAddKernelNode, &next,
					    state->graphs[graph_index],
					    depend_count == 0 ? NULL : depends,
					    depend_count, &parameters);
			side_count = side_mark(branch, before, next, side_lasts,
					       side_count);
			previous = next;
		}
		if (!is_built ||
		    !DRIVER_CALL(cuGraphInstantiateWithFlags,
				 &state->graph_executables[graph_index],
				 state->graphs[graph_index], 0) ||
		    !DRIVER_CALL(cuGraphUpload,
				 state->graph_executables[graph_index],
				 state->stream)) {
			state->first_failed_ticket =
				state->last_ticket_finished + 1;
			status = STATUS_DEVICE_ERROR;
		} else {
			*graph_out = graph_index;
			state->graph_count++;
		}
	}
	cuda_card_leave(card);
	return status;
}

MUST_CHECK result_status cuda_graph_run(struct device_card *card,
					uint32_t graph)
{
	result_status status = STATUS_OK;
	struct open_card_state *const state =
		cuda_card_enter(card, true, &status);
	if (state == NULL) {
		return status;
	}
    if (graph >= state->graph_count) {
	    status = STATUS_INVALID_ARGUMENT;
    } else if (state->first_failed_ticket != 0) {
	    status = STATUS_DEVICE_ERROR;
    } else if (!DRIVER_CALL(cuGraphLaunch, state->graph_executables[graph], state->stream)) {
        state->first_failed_ticket = state->last_ticket_finished + 1;
	status = STATUS_DEVICE_ERROR;
    }
    cuda_card_leave(card);
    return status;
}

MUST_CHECK result_status cuda_launch_wait(struct device_card *card,
					  uint64_t wait_limit_nanoseconds)
{
	const uint64_t started_nanoseconds = cuda_monotonic_nanoseconds();
	bool event_is_recorded = false;
	result_status status = STATUS_OK;
	do {
		struct open_card_state *const state =
			cuda_card_enter(card, true, &status);
		if (state == NULL) {
			return status;
		}
		if (state->first_failed_ticket != 0) {
			status = STATUS_DEVICE_ERROR;
		} else if (!event_is_recorded &&
			   !DRIVER_CALL(cuEventRecord,
					state->events[LAUNCH_EVENT],
					state->stream)) {
			state->first_failed_ticket =
				state->last_ticket_finished + 1;
			status = STATUS_DEVICE_ERROR;
		} else {
			event_is_recorded = true;
			status = cuda_event_poll(state,
						 state->events[LAUNCH_EVENT]);
		}
		cuda_card_leave(card);
	} while (status == STATUS_TIMED_OUT &&
		 cuda_sleep_before_next_poll(started_nanoseconds,
					     wait_limit_nanoseconds));
    return status;
}
