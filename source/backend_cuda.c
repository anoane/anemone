/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * backend_cuda.c: the table backend_cuda_value of backend_seam.h, on the CUDA driver interface: the cards, staging
 * memory, the copies and the digest. What a card runs is in backend_cuda_launch.c; backend_cuda_card.h is what the
 * two share, and no other file of the engine includes a vendor header. Where the seam leaves a choice:
 *   - The first call initialises the driver, once, and retains every card's primary context for the life of the
 *     process (also when a later card then fails it). If it fails, an error line says why and every function answers
 *     DEVICE_ERROR (UNSUPPORTED above BACKEND_MAXIMUM_CARDS, as for staging on a machine with no card).
 *   - A card handle is the address of a record of this file: a closed card and an address that is no card are
 *     STATUS_INVALID_ARGUMENT. A card's mutex guards its open state: every call of a card takes it and makes
 *     the card's context current; a call that waits asks the driver under the mutex and sleeps without it.
 *   - The staging mutex guards only the table of blocks and is never held with a card's. No mutex is held while the
 *     driver page-locks or frees a block, in the first card's context, which several threads may have current.
 *   - A download is an upload in the other direction: one ring, one sequence of tickets. A copy is unfinished, also
 *     for BUSY, until upload_wait has reported it finished. The first driver error a call of an open card reports
 *     fails the card and every ticket not yet reported finished, issued or not. A copy of a failed card, one whose
 *     begin failed too, may read or write its staging memory until card_close.
 *   - A card runs one digest kernel at a time: a digest waits inside its limit for its turn and for a kernel whose
 *     caller gave up, and with a limit of 0 begins none. card_close waits its limit for the uploads, then with no
 *     limit for the driver to end what is in flight (one error line): nothing else stops a copy that reads staging.
 *   - arithmetic_class is ARITHMETIC_CLASS_SM_120 or _SM_80, or 0 for an older card, which card_open
 *     refuses (UNSUPPORTED). An arena or staging block the driver does not give is OUT_OF_MEMORY, whatever its reason.
 *   - Every driver call is a fault point named backend_cuda_<function>; reading a link file is backend_cuda_link_read.
 */
#include "backend_cuda_card.h"

#include <ctype.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "anemone/pack_digest.h"
#if TIMING_HOOKS_ENABLED
#include <math.h>
#endif

#define NANOSECONDS_PER_SECOND 1000000000ull
#define POLL_INTERVAL_NANOSECONDS \
	20000ull /* between two looks at an event: a card's hand-off waits for it (step_pipeline.c) */
#define COMPUTE_CAPABILITY_MAJOR_SM_120 12
#define COMPUTE_CAPABILITY_MAJOR_SM_80 8
#define MAXIMUM_MULTIPROCESSOR_COUNT \
	4096 /* far above any card: it bounds the grid of the digest kernel */
#define DIGEST_THREADS_PER_BLOCK 256u
#define DIGEST_BLOCKS_PER_MULTIPROCESSOR \
	8u /* about what a multiprocessor runs at once */
#define NO_CARD_INDEX \
	BACKEND_MAXIMUM_CARDS /* a record behind the cards, for an index past them */
#define LINK_TEXT_CAPACITY \
	128 /* the path of a file of the bus device, then what is read of it */
#define DECIMAL 10
#define ADDRESS_HALF_BITS 32u
#if TIMING_HOOKS_ENABLED
#define MICROSECONDS_PER_MILLISECOND 1000.0
#endif

/* The whole GT/s current_link_speed begins with ("2.5 GT/s PCIe" gives 2), at the index of its link generation. */
static const unsigned long link_speeds[] = { 0, 2, 5, 8, 16, 32, 64 };

static pthread_once_t backend_once = PTHREAD_ONCE_INIT;
static result_status backend_status =
	STATUS_DEVICE_ERROR; /* written once, inside the initialisation */
static uint32_t backend_card_count = 0;
static struct device_card cards[NO_CARD_INDEX + 1];
struct staging_block {
	void *address;
	uint64_t byte_count;
	uint64_t reader_ticket[BACKEND_MAXIMUM_CARDS];
	bool released, reclaiming, registered;
};
static pthread_mutex_t staging_mutex =
	PTHREAD_MUTEX_INITIALIZER; /* guards the three below */
static struct staging_block
	staging_blocks[BACKEND_MAXIMUM_STAGING_BLOCKS]; /* the first are alive: */
static uint32_t staging_block_count = 0; /* this many */
static uint32_t staging_blocks_awaited =
	0; /* being page-locked by the driver now: the limit counts them too */

uint64_t cuda_monotonic_nanoseconds(void)
{
	struct timespec now;
	if (clock_gettime(CLOCK_MONOTONIC, &now) != 0 || now.tv_sec < 0 ||
	    now.tv_nsec < 0) {
		return 0;
	}
	return (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND +
	       (uint64_t)now.tv_nsec;
}

bool cuda_sleep_before_next_poll(uint64_t started_nanoseconds,
				 uint64_t wait_limit_nanoseconds)
{
	const uint64_t now = cuda_monotonic_nanoseconds();
	const uint64_t left =
		wait_limit_nanoseconds -
		(now - started_nanoseconds); /* if the limit has not passed */
	struct timespec interval = { 0 };
	if (now == 0 || now < started_nanoseconds ||
	    now - started_nanoseconds >= wait_limit_nanoseconds) {
		return false;
	}
	interval.tv_nsec = (long)(left < POLL_INTERVAL_NANOSECONDS ?
					  left :
					  POLL_INTERVAL_NANOSECONDS);
	(void)clock_nanosleep(
		CLOCK_MONOTONIC, 0, &interval,
		NULL); /* cut short by a signal, it is a sooner look */
	return true;
}

bool cuda_driver_succeeded(const char *function_name, CUresult result)
{
	const char *error_name = NULL;
	if (result == CUDA_SUCCESS) {
		return true;
	}
	if (cuGetErrorName(result, &error_name) != CUDA_SUCCESS ||
	    error_name == NULL) {
		error_name = "an unnamed error";
	}
	LOG_ERROR("%s failed: %s", function_name, error_name);
	return false;
}

/* Run once: describes every card, whose record is all zero, and retains its context; backend_status says how. */
static void backend_initialise(void)
{
	const CUresult result = FAULT_POINT(backend_cuda_cuInit) ?
					CUDA_ERROR_UNKNOWN :
					cuInit(0);
	int device_count =
		0; /* stays 0 where the driver finds no card: its CUDA_ERROR_NO_DEVICE is no failure */
	if (result != CUDA_ERROR_NO_DEVICE &&
	    (!cuda_driver_succeeded("cuInit", result) ||
	     !DRIVER_CALL(cuDeviceGetCount, &device_count))) {
		return;
	}
	if (device_count < 0 || device_count > BACKEND_MAXIMUM_CARDS) {
		LOG_ERROR(
			"the driver counts %d cards: more than this engine drives",
			device_count);
		backend_status = STATUS_UNSUPPORTED;
		return;
	}
	for (int device_index = 0; device_index < device_count;
	     device_index++) {
		struct device_card *const card = &cards[device_index];
		struct card_description_record *const description =
			&card->description;
		char *const bus_address =
			description
				->bus_address; /* made lower case: as the file system spells it */
		CUdevice device = 0;
		size_t memory_total_bytes = 0;
		int capability = 0;
		int multiprocessors = 0;
		/* The driver is not offered the last byte of a text;
		 * bytes after its terminator are zeroed below. */
		if (!DRIVER_CALL(cuDeviceGet, &device, device_index) ||
		    !DRIVER_CALL(cuDeviceGetName, description->name,
				 CARD_NAME_CAPACITY - 1, device) ||
		    !DRIVER_CALL(cuDeviceGetPCIBusId, bus_address,
				 CARD_BUS_ADDRESS_CAPACITY - 1, device) ||
		    !DRIVER_CALL(cuDeviceTotalMem, &memory_total_bytes,
				 device) ||
		    !DRIVER_CALL(cuDeviceGetAttribute, &capability,
				 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
				 device) ||
		    !DRIVER_CALL(cuDeviceGetAttribute, &multiprocessors,
				 CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
				 device) ||
		    multiprocessors < 1 ||
		    multiprocessors > MAXIMUM_MULTIPROCESSOR_COUNT ||
		    !cuda_launch_limits_read(card, device) ||
		    pthread_mutex_init(&card->mutex, NULL) != 0 ||
		    !DRIVER_CALL(cuDevicePrimaryCtxRetain, &card->context,
				 device)) {
			LOG_ERROR("card %d cannot be used (%d multiprocessors)",
				  device_index, multiprocessors);
			return;
		}
		for (char *byte = bus_address; *byte != '\0'; byte++) {
			*byte = (char)tolower((unsigned char)*byte);
		}
		memset(description->name + strlen(description->name), 0,
		       sizeof(description->name) - strlen(description->name));
		memset(bus_address + strlen(bus_address), 0,
		       sizeof(description->bus_address) - strlen(bus_address));
		const bool is_sm_120 = capability >=
				       COMPUTE_CAPABILITY_MAJOR_SM_120;
		const bool is_sm_80 = !is_sm_120 &&
				      capability >=
					      COMPUTE_CAPABILITY_MAJOR_SM_80;
		description->arithmetic_class =
			is_sm_120 ? ARITHMETIC_CLASS_SM_120 :
			is_sm_80  ? ARITHMETIC_CLASS_SM_80 :
				    0;
		card->architecture = is_sm_120 ? "sm_120a" :
				     is_sm_80  ? "sm_80" :
						 "";
		description->memory_total_bytes = memory_total_bytes;
		description->memory_reserved_bytes =
			BACKEND_CUDA_MEMORY_RESERVED_BYTES;
		card->digest_most_blocks = (uint32_t)multiprocessors *
					   DIGEST_BLOCKS_PER_MULTIPROCESSOR;
	}
	LOG_INFORMATION("the CUDA driver is initialised: %d cards",
			device_count);
	backend_card_count = (uint32_t)device_count;
	backend_status = STATUS_OK;
}

MUST_CHECK static result_status backend_start(void)
{
	return pthread_once(&backend_once, backend_initialise) == 0 ?
		       backend_status :
		       STATUS_DEVICE_ERROR;
}

struct open_card_state *cuda_card_enter(struct device_card *card,
					bool must_be_open,
					result_status *status_out)
{
	const uintptr_t record_index = ((uintptr_t)card - (uintptr_t)cards) /
				       sizeof(cards[0]); /* wraps below them */
	*status_out = backend_start();
	if (*status_out != STATUS_OK) {
		return NULL;
	}
	*status_out = STATUS_INVALID_ARGUMENT;
	if (card == NULL || record_index >= backend_card_count ||
	    card != &cards[record_index]) {
		return NULL;
	}
	*status_out = STATUS_DEVICE_ERROR;
	if (pthread_mutex_lock(&card->mutex) != 0) {
		return NULL;
	}
	if (must_be_open && !card->state.is_open) {
		*status_out = STATUS_INVALID_ARGUMENT;
	} else if (DRIVER_CALL(cuCtxPushCurrent, card->context)) {
		*status_out = STATUS_OK;
		return &card->state;
	} else if (must_be_open) { /* an open card reports a device error: it begins no new work */
		card->state.first_failed_ticket =
			card->state.last_ticket_finished + 1;
	}
	(void)pthread_mutex_unlock(&card->mutex);
	return NULL;
}

void cuda_card_leave(struct device_card *card)
{
	CUcontext popped_context = NULL;
	(void)DRIVER_CALL(cuCtxPopCurrent, &popped_context);
	(void)pthread_mutex_unlock(&card->mutex);
}

MUST_CHECK result_status cuda_event_poll(struct open_card_state *state,
					 CUevent event)
{
	const CUresult result = FAULT_POINT(backend_cuda_cuEventQuery) ?
					CUDA_ERROR_UNKNOWN :
					cuEventQuery(event);
	if (result == CUDA_ERROR_NOT_READY) {
		return STATUS_TIMED_OUT;
	}
	if (cuda_driver_succeeded("cuEventQuery", result)) {
		return STATUS_OK;
	}
	state->first_failed_ticket = state->last_ticket_finished + 1;
	return STATUS_DEVICE_ERROR;
}

/* Gives back each thing an open card obtained and leaves it closed. Its context is current, nothing is in flight. */
static void card_release(struct open_card_state *state)
{
	for (uint32_t event_index = 0; event_index < CARD_EVENT_COUNT;
	     event_index++) {
		(void)(state->events[event_index] == NULL ||
		       DRIVER_CALL(cuEventDestroy, state->events[event_index]));
	}
#if TIMING_HOOKS_ENABLED
	for (uint32_t event_index = 0; event_index < MEASURE_EVENT_COUNT;
	     event_index++) {
		(void)(state->measurement_events[event_index] == NULL ||
		       DRIVER_CALL(cuEventDestroy,
				   state->measurement_events[event_index]));
	}
#endif
	(void)(state->stream == NULL ||
	       DRIVER_CALL(cuStreamDestroy, state->stream));
	for (uint32_t graph_index = 0; graph_index < BACKEND_MAXIMUM_GRAPHS;
	     graph_index++) {
		(void)(state->graph_executables[graph_index] == NULL ||
		       DRIVER_CALL(cuGraphExecDestroy,
				   state->graph_executables[graph_index]));
		(void)(state->graphs[graph_index] == NULL ||
		       DRIVER_CALL(cuGraphDestroy, state->graphs[graph_index]));
	}
	cuda_modules_unload(state);
	(void)(state->sum_address == 0 ||
	       DRIVER_CALL(cuMemFree, state->sum_address));
	(void)(state->arena_address == 0 ||
	       DRIVER_CALL(cuMemFree, state->arena_address));
	*state = (struct open_card_state){ 0 };
}

MUST_CHECK static result_status card_count(uint32_t *card_count_out)
{
	const result_status status = backend_start();
	if (status == STATUS_OK && card_count_out != NULL) {
		*card_count_out = backend_card_count;
	}
	return status == STATUS_OK && card_count_out == NULL ?
		       STATUS_INVALID_ARGUMENT :
		       status;
}

/* What a file of the card's bus device begins with: the link generation of that speed, or the number of a width; 0
 * when unread or unknown. Only the characters of a bus address become part of the path. No mutex is held. */
static uint32_t link_file_value(const struct device_card *card,
				const char *file_name, bool is_speed)
{
	const char *const bus_address = card->description.bus_address;
	char text[LINK_TEXT_CAPACITY];
	const int path_length = snprintf(text, sizeof(text),
					 "/sys/bus/pci/devices/%s/%s",
					 bus_address, file_name);
	const bool path_is_sound =
		path_length > 0 && (size_t)path_length < sizeof(text) &&
		bus_address[0] != '\0' &&
		bus_address[strspn(bus_address, "0123456789abcdef:.")] == '\0';
	const int descriptor =
		!path_is_sound || FAULT_POINT(backend_cuda_link_read) ?
			-1 :
			open(text, O_RDONLY | O_CLOEXEC);
	const ssize_t byte_count =
		descriptor < 0 ? -1 : read(descriptor, text, sizeof(text) - 1);
	if (descriptor >= 0) {
		(void)close(descriptor);
	}
	text[byte_count > 0 ? byte_count : 0] = '\0';
	const unsigned long number = strtoul(
		text, NULL,
		DECIMAL); /* 0 without digits; ULONG_MAX when too long */
	uint32_t generation = sizeof(link_speeds) / sizeof(link_speeds[0]) - 1;
	while (generation > 0 && link_speeds[generation] != number) {
		generation--;
	}
	return is_speed		    ? generation :
	       number <= UINT32_MAX ? (uint32_t)number :
				      0;
}

MUST_CHECK static result_status
card_describe(uint32_t card_index,
	      struct card_description_record *description_out)
{
	struct device_card *const card =
		&cards[card_index < NO_CARD_INDEX ? card_index : NO_CARD_INDEX];
	size_t free_bytes = 0;
	size_t total_bytes = 0;
	result_status status = STATUS_OK;
	if (cuda_card_enter(card, false, &status) == NULL) {
		return status;
	}
	const bool is_read =
		description_out != NULL &&
		DRIVER_CALL(cuMemGetInfo, &free_bytes, &total_bytes);
	cuda_card_leave(card);
	if (!is_read) {
		return description_out == NULL ? STATUS_INVALID_ARGUMENT :
						 STATUS_DEVICE_ERROR;
	}
	const uint32_t speed_now =
		link_file_value(card, "current_link_speed",
				true); /* the lowest while it idles */
	const uint32_t card_most =
		link_file_value(card, "max_link_speed", true);
	const uint32_t port_most = link_file_value(
		card, "../max_link_speed", true); /* the port: its parent */
	const uint32_t stated = card_most < port_most ? card_most : port_most;
	*description_out =
		card->description; /* every byte: the two texts are zero-filled, nothing is padding */
	description_out->memory_free_bytes = free_bytes;
	description_out->link_lane_count =
		link_file_value(card, "current_link_width", false);
	description_out->link_generation = speed_now > stated ? speed_now :
								stated;
	return STATUS_OK;
}

MUST_CHECK static result_status card_open(uint32_t card_index,
					  uint64_t arena_byte_count,
					  struct device_card **card_out)
{
	struct device_card *const card =
		&cards[card_index < NO_CARD_INDEX ? card_index : NO_CARD_INDEX];
	CUmodule digest_module = NULL;
	result_status status = STATUS_OK;
	struct open_card_state *const state =
		cuda_card_enter(card, false, &status);
	if (state == NULL) {
		return status;
	}
	/* the digest's module is the card's first: UNSUPPORTED when this build has no kernels for the card */
	status = card_out == NULL || arena_byte_count == 0 || state->is_open ?
			 STATUS_INVALID_ARGUMENT :
			 cuda_module_of(card, state, "kernel_pack_digest",
					&digest_module);
	if (status == STATUS_OK) {
		bool is_obtained =
			DRIVER_CALL(cuMemAlloc, &state->arena_address,
				    (size_t)arena_byte_count) &&
			DRIVER_CALL(cuModuleGetFunction,
				    &state->digest_function, digest_module,
				    "kernel_pack_digest") &&
			DRIVER_CALL(cuMemAlloc, &state->sum_address,
				    sizeof(uint64_t)) &&
			DRIVER_CALL(cuStreamCreate, &state->stream,
				    CU_STREAM_NON_BLOCKING);
		for (uint32_t event_index = 0;
		     is_obtained && event_index < CARD_EVENT_COUNT;
		     event_index++) {
			is_obtained = DRIVER_CALL(cuEventCreate,
						  &state->events[event_index],
						  CU_EVENT_DISABLE_TIMING);
		}
#if TIMING_HOOKS_ENABLED
		for (uint32_t event_index = 0;
		     is_obtained && event_index < MEASURE_EVENT_COUNT;
		     event_index++) {
			is_obtained = DRIVER_CALL(
				cuEventCreate,
				&state->measurement_events[event_index],
				CU_EVENT_DEFAULT);
		}
#endif
		state->arena_byte_count = arena_byte_count;
		state->is_open = is_obtained;
		status = is_obtained		   ? STATUS_OK :
			 state->arena_address == 0 ? STATUS_OUT_OF_MEMORY :
						     STATUS_DEVICE_ERROR;
		if (!is_obtained) {
			card_release(state);
		}
	}
	if (status == STATUS_OK) {
		*card_out = card;
	}
	cuda_card_leave(card);
	return status;
}

/* The index of the block a range begins in and lies wholly inside, or staging_block_count. Staging mutex held. */
static uint32_t staging_block_holding(const void *range, uint64_t byte_count)
{
	for (uint32_t block_index = 0; block_index < staging_block_count;
	     block_index++) {
		const uint64_t block_byte_count =
			staging_blocks[block_index].byte_count;
		const uint64_t offset = (uintptr_t)range -
					(uintptr_t)staging_blocks[block_index]
						.address; /* wraps below it */
		if (offset < block_byte_count &&
		    byte_count <= block_byte_count - offset) {
			return block_index;
		}
	}
	return staging_block_count;
}

MUST_CHECK static result_status staging_allocate(uint64_t byte_count,
						 void **page_locked_memory_out)
{
	const uint64_t alignment = BACKEND_STAGING_ALIGNMENT_BYTES;
	const bool is_valid = page_locked_memory_out != NULL &&
			      byte_count != 0 &&
			      byte_count <= UINT64_MAX - alignment;
	CUcontext popped_context = NULL;
	void *address = NULL;
	const result_status status = backend_start();
	if (status != STATUS_OK) {
		return status;
	}
	if (!is_valid) {
		return STATUS_INVALID_ARGUMENT;
	}
	if (backend_card_count == 0) {
		return STATUS_UNSUPPORTED;
	}
	if (pthread_mutex_lock(&staging_mutex) != 0) {
		return STATUS_DEVICE_ERROR;
	}
	const bool has_room = staging_block_count + staging_blocks_awaited <
			      BACKEND_MAXIMUM_STAGING_BLOCKS;
	staging_blocks_awaited +=
		has_room ?
			1 :
			0; /* its place in the table is kept while the driver works */
	(void)pthread_mutex_unlock(&staging_mutex);
	if (!has_room) {
		return STATUS_BUSY_VALUE;
	}
	byte_count = (byte_count + alignment - 1) / alignment * alignment;
	/* Mapped here on 2 MiB pages and registered, not allocated by the driver: measured 9.1 against 1.9 GiB/s. Still
     * seconds for a large block, so no mutex is held. */
	address = mmap(NULL, (size_t)byte_count, PROT_READ | PROT_WRITE,
		       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	address = address == MAP_FAILED ? NULL : address;
	if (address != NULL &&
	    madvise(address, (size_t)byte_count, MADV_HUGEPAGE) != 0) {
		LOG_WARNING_ENTRY(
			"no 2 MiB pages for staging memory: page-locking will be slow");
	}
	const bool has_context =
		address != NULL &&
		DRIVER_CALL(cuCtxPushCurrent, cards[0].context);
	if (address != NULL &&
	    !(has_context &&
	      DRIVER_CALL(cuMemHostRegister, address, (size_t)byte_count,
			  CU_MEMHOSTREGISTER_PORTABLE))) {
		(void)munmap(address, (size_t)byte_count);
		address = NULL;
	}
	if (has_context) {
		(void)DRIVER_CALL(cuCtxPopCurrent, &popped_context);
	}
	if (pthread_mutex_lock(&staging_mutex) != 0) {
		return STATUS_DEVICE_ERROR;
	} /* the block stays unlisted */
	staging_blocks_awaited--;
	if (address != NULL) {
		staging_blocks[staging_block_count++] =
			(struct staging_block){ .address = address,
						.byte_count = byte_count,
						.registered = true };
	}
	(void)pthread_mutex_unlock(&staging_mutex);
	if (address != NULL) {
		*page_locked_memory_out = address;
	}
	return address != NULL ? STATUS_OK : STATUS_OUT_OF_MEMORY;
}

/* Staging mutex guards ownership and copy readers. Card operations take their
 * card mutex FIRST; no holder of staging_mutex enters a card. A reserved next
 * ticket is recorded before submission: event failure can queue a copy while
 * issuing no public ticket, and that reservation survives until proven drain. */
static bool staging_has_readers(const struct staging_block *block)
{
	for (uint32_t card = 0; card < BACKEND_MAXIMUM_CARDS; card++) {
		if (block->reader_ticket[card] != 0) {
			return true;
		}
	}
	return false;
}

/* Card mutex held. A successful public wait retires only covered reservations;
 * a successful actual stream drain retires every reader of this card. */
static bool staging_readers_retire(uint32_t card, uint64_t ticket, bool drained)
{
	bool reclaim = false;
	if (pthread_mutex_lock(&staging_mutex) != 0) {
		return false;
	}
	for (uint32_t index = 0; index < staging_block_count; index++) {
		uint64_t *const reader =
			&staging_blocks[index].reader_ticket[card];
		if (drained || *reader <= ticket) {
			*reader = 0;
		}
		reclaim = reclaim ||
			  (staging_blocks[index].released &&
			   !staging_has_readers(&staging_blocks[index]));
	}
	(void)pthread_mutex_unlock(&staging_mutex);
	return reclaim;
}

/* Teardown/retirement only. Fixed stack snapshot bounds each attempt to one
 * pass, even if unregister/unmap fail. No callback/user owner is retained.
 * Reclaiming excludes concurrent frees; released blocks accept no new copy. */
static void staging_reclaim(void)
{
	void *candidates[BACKEND_MAXIMUM_STAGING_BLOCKS];
	uint32_t count = 0;
	if (pthread_mutex_lock(&staging_mutex) != 0) {
		return;
	}
	for (uint32_t index = 0; index < staging_block_count; index++) {
		const struct staging_block *const block =
			&staging_blocks[index];
		if (block->released && !block->reclaiming &&
		    !staging_has_readers(block)) {
			candidates[count++] = block->address;
		}
	}
	(void)pthread_mutex_unlock(&staging_mutex);
	for (uint32_t next = 0; next < count; next++) {
		void *const address = candidates[next];
		if (pthread_mutex_lock(&staging_mutex) != 0) {
			return;
		}
		uint32_t index = staging_block_holding(address, 0);
		const bool found = index < staging_block_count &&
				   staging_blocks[index].address == address &&
				   staging_blocks[index].released &&
				   !staging_blocks[index].reclaiming &&
				   !staging_has_readers(&staging_blocks[index]);
		const uint64_t bytes =
			found ? staging_blocks[index].byte_count : 0;
		bool registered = found && staging_blocks[index].registered;
		if (found) {
			staging_blocks[index].reclaiming = true;
		}
		(void)pthread_mutex_unlock(&staging_mutex);
		if (!found) {
			continue;
		}
		CUcontext popped = NULL;
		if (registered &&
		    DRIVER_CALL(cuCtxPushCurrent, cards[0].context)) {
			registered = !DRIVER_CALL(cuMemHostUnregister, address);
			(void)DRIVER_CALL(cuCtxPopCurrent, &popped);
		}
		const bool unmapped = !registered &&
				      munmap(address, (size_t)bytes) == 0;
		if (pthread_mutex_lock(&staging_mutex) != 0) {
			return;
		}
		index = staging_block_holding(address, 0);
		if (index < staging_block_count &&
		    staging_blocks[index].address == address) {
			if (unmapped) {
				staging_blocks[index] =
					staging_blocks[--staging_block_count];
			} else {
				staging_blocks[index].registered = registered;
				staging_blocks[index].reclaiming = false;
			}
		}
		(void)pthread_mutex_unlock(&staging_mutex);
	}
}

/* Caller relinquishes ownership. A pending/failed copy is kept by this backend
 * table, not the destroyed step driver. Failed unregister/unmap stays listed. */
static void staging_free(void *page_locked_memory)
{
	if (page_locked_memory == NULL ||
	    pthread_mutex_lock(&staging_mutex) != 0) {
		return;
	}
	const uint32_t index = staging_block_holding(page_locked_memory, 0);
	const bool found = index < staging_block_count &&
			   staging_blocks[index].address == page_locked_memory;
	if (found) {
		staging_blocks[index].released = true;
	}
	(void)pthread_mutex_unlock(&staging_mutex);
	if (!found) {
		LOG_ERROR(
			"staging_free: %p is not the start of a staging block",
			page_locked_memory);
		return;
	}
	staging_reclaim();
}

/* Queues one copy between a range of the arena and a range of one staging block, and gives it the next ticket: a
 * download when it has a destination, else an upload from source. */
MUST_CHECK static result_status
copy_begin(struct device_card *card, uint64_t arena_byte_offset,
	   const void *source, void *destination, uint64_t byte_count,
	   uint64_t *ticket_out)
{
	const void *const staged_range = destination != NULL ? destination :
							       source;
	result_status status = STATUS_OK;
	struct open_card_state *const state =
		cuda_card_enter(card, true, &status);
	if (state == NULL) {
		return status;
	}
	const CUdeviceptr arena_range =
		state->arena_address +
		arena_byte_offset; /* used once the range is checked */
	const uint64_t ticket = state->last_ticket_issued + 1;
	const uint32_t reader = (uint32_t)(card - cards);
	const bool staging_is_locked = pthread_mutex_lock(&staging_mutex) == 0;
	const uint32_t block =
		staging_is_locked ?
			staging_block_holding(staged_range, byte_count) :
			BACKEND_MAXIMUM_STAGING_BLOCKS;
	const bool range_is_staged = staging_is_locked &&
				     block < staging_block_count &&
				     !staging_blocks[block].released;
	status = ticket_out == NULL || byte_count == 0 || !range_is_staged ||
				 arena_byte_offset > state->arena_byte_count ||
				 byte_count > state->arena_byte_count -
						      arena_byte_offset ?
			 STATUS_INVALID_ARGUMENT :
		 state->first_failed_ticket != 0 ? STATUS_DEVICE_ERROR :
		 ticket - state->last_ticket_finished > UPLOAD_RING ?
						   STATUS_BUSY_VALUE :
						   STATUS_OK;
	status = status == STATUS_OK && ticket == 0 ? STATUS_UNSUPPORTED :
						      status;
	if (status == STATUS_OK) {
		staging_blocks[block].reader_ticket[reader] = ticket;
	}
	if (staging_is_locked) {
		(void)pthread_mutex_unlock(&staging_mutex);
	}
	if (status == STATUS_OK &&
	    (destination != NULL ?
		     DRIVER_CALL(cuMemcpyDtoHAsync, destination, arena_range,
				 (size_t)byte_count, state->stream) :
		     DRIVER_CALL(cuMemcpyHtoDAsync, arena_range, source,
				 (size_t)byte_count, state->stream)) &&
	    DRIVER_CALL(cuEventRecord, state->events[ticket % UPLOAD_RING],
			state->stream)) {
		state->last_ticket_issued = ticket;
		*ticket_out = ticket;
		if (destination == NULL) {
			COUNTER_ADD(backend_cuda_upload_bytes, byte_count);
		}
		if (destination != NULL) {
			COUNTER_ADD(backend_cuda_download_bytes, byte_count);
		}
	} else if (status == STATUS_OK) {
		state->first_failed_ticket = state->last_ticket_finished + 1;
		status = STATUS_DEVICE_ERROR;
	}
	cuda_card_leave(card);
	return status;
}

MUST_CHECK static result_status upload_begin(struct device_card *card,
					     uint64_t arena_byte_offset,
					     const void *page_locked_source,
					     uint64_t byte_count,
					     uint64_t *upload_ticket_out)
{
	return copy_begin(card, arena_byte_offset, page_locked_source, NULL,
			  byte_count, upload_ticket_out);
}

MUST_CHECK static result_status download_begin(struct device_card *card,
					       uint64_t arena_byte_offset,
					       void *page_locked_destination,
					       uint64_t byte_count,
					       uint64_t *download_ticket_out)
{
	return copy_begin(card, arena_byte_offset, NULL,
			  page_locked_destination, byte_count,
			  download_ticket_out);
}

MUST_CHECK static result_status upload_wait(struct device_card *card,
					    uint64_t ticket,
					    uint64_t wait_limit_nanoseconds)
{
	const uint64_t started_nanoseconds = cuda_monotonic_nanoseconds();
	result_status status = STATUS_OK;
	bool reclaim = false;
	do {
		struct open_card_state *const state =
			cuda_card_enter(card, true, &status);
		if (state == NULL) {
			return status;
		}
		status = state->first_failed_ticket != 0 &&
					 ticket >= state->first_failed_ticket ?
				 STATUS_DEVICE_ERROR :
			 ticket == 0 || ticket > state->last_ticket_issued ?
				 STATUS_INVALID_ARGUMENT :
			 ticket <= state->last_ticket_finished ?
				 STATUS_OK :
				 cuda_event_poll(
					 state,
					 state->events[ticket % UPLOAD_RING]);
		if (status == STATUS_OK &&
		    ticket > state->last_ticket_finished) {
			state->last_ticket_finished =
				ticket; /* the stream is a queue: every earlier one has finished too */
		}
		if (status == STATUS_OK) {
			reclaim = staging_readers_retire(
				(uint32_t)(card - cards), ticket, false);
		}
		cuda_card_leave(card);
	} while (status == STATUS_TIMED_OUT &&
		 cuda_sleep_before_next_poll(started_nanoseconds,
					     wait_limit_nanoseconds));
	if (status == STATUS_OK && reclaim) {
		staging_reclaim();
	}
	return status;
}

static void card_close(struct device_card *card,
		       uint64_t wait_limit_nanoseconds)
{
	result_status status = STATUS_OK;
	struct open_card_state *state =
		card == NULL ? NULL : cuda_card_enter(card, true, &status);
	if (card == NULL) {
		return;
	}
	if (state != NULL) {
		const uint64_t last_ticket = state->last_ticket_issued;
		cuda_card_leave(card);
		if (upload_wait(card, last_ticket, wait_limit_nanoseconds) ==
		    STATUS_TIMED_OUT) {
			LOG_ERROR(
				"card %s: its close waits past its limit for its uploads",
				card->description.bus_address);
		}
		state = cuda_card_enter(card, true, &status);
	}
	if (state == NULL) {
		LOG_ERROR(
			"card_close: not an open card of this backend, or its context cannot be made current");
		return;
	}
	if (!DRIVER_CALL(cuStreamSynchronize, state->stream)) {
		state->first_failed_ticket = state->last_ticket_finished + 1;
		cuda_card_leave(card);
		return; /* Failed card and staging remain owned; no completion proof. */
	}
	(void)staging_readers_retire((uint32_t)(card - cards), 0, true);
	card_release(state);
	cuda_card_leave(card);
	staging_reclaim();
}

/* The call that launched the kernel owns it while it waits for it: until it has read the sum, failed, or given up at
 * its limit. The next call then lets that kernel finish before it launches its own. */
MUST_CHECK static result_status digest(struct device_card *card,
				       uint64_t arena_byte_offset,
				       uint64_t byte_count,
				       uint64_t wait_limit_nanoseconds,
				       uint64_t *digest_out)
{
	const uint64_t started_nanoseconds = cuda_monotonic_nanoseconds();
	unsigned long long word_count = byte_count / PACK_DIGEST_WORD_BYTES;
	const uint64_t blocks_needed =
		(word_count + (DIGEST_THREADS_PER_BLOCK - 1)) /
		DIGEST_THREADS_PER_BLOCK;
	bool owns_the_kernel = false;
	result_status status = STATUS_OK;
	result_status give_up_status = STATUS_OK;
	uint64_t sum = 0;
	do {
		struct open_card_state *const state =
			cuda_card_enter(card, true, &status);
		if (state == NULL) {
			return status;
		}
		CUdeviceptr words_address =
			state->arena_address + arena_byte_offset;
		const CUevent event = state->events[DIGEST_EVENT];
		status =
			digest_out == NULL ||
					arena_byte_offset %
							PACK_DIGEST_WORD_BYTES !=
						0 ||
					byte_count % PACK_DIGEST_WORD_BYTES !=
						0 ||
					arena_byte_offset >
						state->arena_byte_count ||
					byte_count > state->arena_byte_count -
							     arena_byte_offset ?
				STATUS_INVALID_ARGUMENT :
			state->first_failed_ticket != 0 ?
				STATUS_DEVICE_ERROR :
			byte_count == 0 || owns_the_kernel ?
				STATUS_OK :
			wait_limit_nanoseconds == 0 ?
				STATUS_TIMED_OUT /* who only asks begins no kernel */
			:
			state->digest_has_a_waiter ?
				STATUS_TIMED_OUT /* the turn of another call */
			:
			state->digest_kernel_is_in_flight ?
				cuda_event_poll(state,
						event) /* its caller gave up */
				:
				STATUS_OK;
		if (status == STATUS_OK && byte_count != 0) {
			if (!owns_the_kernel) {
				/* Zeroing and the kernel queue behind earlier
				 * uploads in this stream. */
				void *kernel_parameters[] = {
					&words_address, &word_count,
					&state->sum_address
				};
				const unsigned int block_count =
					blocks_needed < card->digest_most_blocks ?
						(unsigned int)blocks_needed :
						card->digest_most_blocks;
				owns_the_kernel =
					DRIVER_CALL(cuMemsetD32Async,
						    state->sum_address, 0,
						    sizeof(sum) /
							    sizeof(uint32_t),
						    state->stream) &&
					DRIVER_CALL(cuLaunchKernel,
						    state->digest_function,
						    block_count, 1, 1,
						    DIGEST_THREADS_PER_BLOCK, 1,
						    1, 0, state->stream,
						    kernel_parameters, NULL) &&
					DRIVER_CALL(cuEventRecord, event,
						    state->stream);
				status = owns_the_kernel ? STATUS_OK :
							   STATUS_DEVICE_ERROR;
				COUNTER_ADD(backend_cuda_digest_bytes,
					    byte_count);
			}
			if (status == STATUS_OK) {
				status = cuda_event_poll(state, event);
			}
			if (status == STATUS_OK &&
			    !DRIVER_CALL(cuMemcpyDtoH, &sum, state->sum_address,
					 sizeof(sum))) {
				status = STATUS_DEVICE_ERROR;
			}
			if (status == STATUS_DEVICE_ERROR) {
				state->first_failed_ticket =
					state->last_ticket_finished + 1;
			}
			state->digest_kernel_is_in_flight =
				owns_the_kernel && status == STATUS_TIMED_OUT;
			state->digest_has_a_waiter =
				state->digest_kernel_is_in_flight;
		}
		cuda_card_leave(card);
	} while (status == STATUS_TIMED_OUT &&
		 cuda_sleep_before_next_poll(started_nanoseconds,
					     wait_limit_nanoseconds));
	if (status == STATUS_TIMED_OUT && owns_the_kernel &&
	    cuda_card_enter(card, true, &give_up_status) != NULL) {
		card->state.digest_has_a_waiter =
			false; /* the kernel runs on: the next digest lets it finish first */
		cuda_card_leave(card);
	}
	if (status == STATUS_OK) {
		*digest_out = sum;
	}
	return status;
}

/* Fails the open card behind a driver error of a mark (the top: the first driver error fails the card). */
MUST_CHECK static result_status mark_status(struct open_card_state *state,
					    bool succeeded)
{
	if (!succeeded && state->first_failed_ticket == 0) {
		state->first_failed_ticket = state->last_ticket_finished + 1;
	}
	return succeeded ? STATUS_OK : STATUS_DEVICE_ERROR;
}

/* backend_seam.h's marks: mark m is the card's event MARK_EVENT_FIRST + m, recorded on its stream. */
MUST_CHECK static result_status mark_record(struct device_card *card,
					    uint32_t mark)
{
	result_status status = STATUS_OK;
	struct open_card_state *const state =
		cuda_card_enter(card, true, &status);
	if (state == NULL) {
		return status;
	}
	status = mark >= BACKEND_MAXIMUM_MARKS ?
			 STATUS_INVALID_ARGUMENT :
		 state->first_failed_ticket != 0 ?
			 STATUS_DEVICE_ERROR :
			 mark_status(
				 state,
				 DRIVER_CALL(
					 cuEventRecord,
					 state->events[MARK_EVENT_FIRST + mark],
					 state->stream));
	cuda_card_leave(card);
	return status;
}

/* The owner's event is read under its mutex, then the card's stream waits for it under the card's: never both held
 * (the driver lets a stream wait for another context's event). */
MUST_CHECK static result_status
mark_follow(struct device_card *card, struct device_card *owner, uint32_t mark)
{
	result_status status = STATUS_OK;
	const struct open_card_state *const owned =
		cuda_card_enter(owner, true, &status);
	if (owned == NULL) {
		return status;
	}
	const CUevent event = mark < BACKEND_MAXIMUM_MARKS ?
				      owned->events[MARK_EVENT_FIRST + mark] :
				      NULL;
	cuda_card_leave(owner);
	struct open_card_state *const state =
		cuda_card_enter(card, true, &status);
	if (state == NULL) {
		return status;
	}
	status = event == NULL ?
			 STATUS_INVALID_ARGUMENT :
		 state->first_failed_ticket != 0 ?
			 STATUS_DEVICE_ERROR :
			 mark_status(state,
				     DRIVER_CALL(cuStreamWaitEvent,
						 state->stream, event, 0));
	cuda_card_leave(card);
	return status;
}

/* Load-time opt-in. Group drain before close/rewrite is the caller's contract; never hold two card locks. */
MUST_CHECK static result_status peer_connect(struct device_card *accessor,
					     struct device_card *owner,
					     uint64_t owner_offset,
					     uint64_t word)
{
	result_status status = STATUS_OK;
	struct open_card_state *state = cuda_card_enter(owner, true, &status);
	if (state == NULL)
		return status;
	const CUdeviceptr owner_address = state->arena_address;
	const uint64_t owner_bytes = state->arena_byte_count;
	const bool owner_failed = state->first_failed_ticket != 0;
	cuda_card_leave(owner);
	if (owner_offset >= owner_bytes ||
	    owner_address > UINT64_MAX - owner_offset)
		return STATUS_INVALID_ARGUMENT;
	if (owner_failed)
		return STATUS_DEVICE_ERROR;
	state = cuda_card_enter(accessor, true, &status);
	if (state == NULL)
		return status;
	if (word % BACKEND_ADDRESS_WORD_BYTES != 0 ||
	    word > state->arena_byte_count ||
	    BACKEND_ADDRESS_WORD_BYTES > state->arena_byte_count - word) {
		status = STATUS_INVALID_ARGUMENT;
	} else if (state->first_failed_ticket != 0) {
		status = STATUS_DEVICE_ERROR;
	} else if (accessor != owner) {
		CUdevice accessor_device = 0, owner_device = 0;
		int available = 0;
		const bool described =
			DRIVER_CALL(cuCtxGetDevice, &accessor_device) &&
			DRIVER_CALL(cuDeviceGetByPCIBusId, &owner_device,
				    owner->description.bus_address) &&
			DRIVER_CALL(cuDeviceCanAccessPeer, &available,
				    accessor_device, owner_device);
		if (!described) {
			status = mark_status(state, false);
		} else if (available == 0) {
			status = STATUS_UNSUPPORTED;
		} else {
			const CUresult connected =
				FAULT_POINT(
					backend_cuda_cuCtxEnablePeerAccess) ?
					CUDA_ERROR_UNKNOWN :
					cuCtxEnablePeerAccess(owner->context,
							      0);
			if (connected == CUDA_SUCCESS ||
			    connected ==
				    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
				status = STATUS_OK;
			} else if (connected ==
					   CUDA_ERROR_PEER_ACCESS_UNSUPPORTED ||
				   connected == CUDA_ERROR_TOO_MANY_PEERS ||
				   connected == CUDA_ERROR_NOT_SUPPORTED) {
				status = STATUS_UNSUPPORTED;
			} else {
				status = mark_status(
					state, cuda_driver_succeeded(
						       "cuCtxEnablePeerAccess",
						       connected));
			}
		}
	}
	if (status == STATUS_OK) {
		const CUdeviceptr address = owner_address + owner_offset;
		/* the word's host record, for copies that name it (backend_seam.h); past the record's capacity a word stays
		 * unrecorded, and a copy naming it is refused */
		uint32_t entry = 0;
		while (entry < state->peer_word_count && state->peer_words[entry].word != word)
			entry++;
		if (entry < PEER_WORD_CAPACITY)
			state->peer_words[entry] = (struct peer_word){ word, address, owner_bytes - owner_offset };
		state->peer_word_count += entry == state->peer_word_count && entry < PEER_WORD_CAPACITY ? 1u : 0u;
	}
	if (status == STATUS_OK) {
		const CUdeviceptr address = owner_address + owner_offset;
		status = mark_status(
			state,
			DRIVER_CALL(cuMemsetD32Async,
				    state->arena_address + word,
				    (unsigned int)address, 1, state->stream) &&
				DRIVER_CALL(cuMemsetD32Async,
					    state->arena_address + word +
						    sizeof(uint32_t),
					    (unsigned int)(address >>
							   ADDRESS_HALF_BITS),
					    1, state->stream));
	}
	cuda_card_leave(accessor);
	return status;
}

#if TIMING_HOOKS_ENABLED
MUST_CHECK static result_status
timer_cuda_measure_record(struct device_card *card, bool is_begin)
{
	result_status status = STATUS_OK;
	struct open_card_state *const state =
		cuda_card_enter(card, true, &status);
	if (state == NULL)
		return status;
	if (state->first_failed_ticket != 0) {
		status = STATUS_DEVICE_ERROR;
	} else if ((is_begin &&
		    state->measurement_phase != CUDA_MEASURE_IDLE) ||
		   (!is_begin &&
		    state->measurement_phase != CUDA_MEASURE_BEGUN)) {
		status = is_begin ? STATUS_BUSY_VALUE : STATUS_INVALID_ARGUMENT;
	} else {
		status = mark_status(
			state,
			DRIVER_CALL(cuEventRecord,
				    state->measurement_events
					    [is_begin ? MEASURE_BEGIN_EVENT :
							MEASURE_END_EVENT],
				    state->stream));
		if (status == STATUS_OK)
			state->measurement_phase = is_begin ?
							   CUDA_MEASURE_BEGUN :
							   CUDA_MEASURE_ENDED;
	}
	cuda_card_leave(card);
	return status;
}

MUST_CHECK static result_status
timer_cuda_measure_begin(struct device_card *card)
{
	return timer_cuda_measure_record(card, true);
}

MUST_CHECK static result_status timer_cuda_measure_end(struct device_card *card)
{
	return timer_cuda_measure_record(card, false);
}

MUST_CHECK static result_status
timer_cuda_measure_elapsed_us(struct device_card *card,
			      uint64_t wait_limit_nanoseconds,
			      double *microseconds_out)
{
	if (microseconds_out == NULL)
		return STATUS_INVALID_ARGUMENT;
	const uint64_t started_nanoseconds = cuda_monotonic_nanoseconds();
	result_status status = STATUS_OK;
	do {
		struct open_card_state *const state =
			cuda_card_enter(card, true, &status);
		if (state == NULL)
			return status;
		status = state->first_failed_ticket != 0 ?
				 STATUS_DEVICE_ERROR :
			 state->measurement_phase != CUDA_MEASURE_ENDED ?
				 STATUS_INVALID_ARGUMENT :
				 cuda_event_poll(state,
						 state->measurement_events
							 [MEASURE_END_EVENT]);
		if (status == STATUS_OK) {
			float milliseconds = 0;
			status = mark_status(
				state,
				DRIVER_CALL(cuEventElapsedTime, &milliseconds,
					    state->measurement_events
						    [MEASURE_BEGIN_EVENT],
					    state->measurement_events
						    [MEASURE_END_EVENT]));
			if (status == STATUS_OK &&
			    (!isfinite(milliseconds) || milliseconds < 0))
				status = mark_status(state, false);
			if (status == STATUS_OK) {
				*microseconds_out =
					(double)milliseconds *
					MICROSECONDS_PER_MILLISECOND;
				state->measurement_phase = CUDA_MEASURE_IDLE;
			}
		}
		cuda_card_leave(card);
	} while (status == STATUS_TIMED_OUT &&
		 cuda_sleep_before_next_poll(started_nanoseconds,
					     wait_limit_nanoseconds));
	return status;
}
#endif

/* The two formats this build's kernels decode, on a card of either arithmetic class. Needs no open card. */
static bool format_supported(uint32_t card_index,
			     enum weight_format_code format)
{
	const bool is_decoded = format == WEIGHT_FORMAT_E2M1_SCALED ||
				format == WEIGHT_FORMAT_E4M3_SCALED;
	return is_decoded && backend_start() == STATUS_OK &&
	       card_index < backend_card_count &&
	       cards[card_index].description.arithmetic_class != 0;
}

const struct backend_operations backend_cuda_value = {
	.name = "cuda",
	.card_count = card_count,
	.card_describe = card_describe,
	.card_open = card_open,
	.card_close = card_close,
	.staging_allocate = staging_allocate,
	.staging_free = staging_free,
	.staging_retire = staging_free,
	.upload_begin = upload_begin,
	.upload_wait = upload_wait,
	.digest = digest,
	.download_begin = download_begin,
	.kernel_prepare = cuda_kernel_prepare,
	.launch_run = cuda_launch_run,
	.launch_wait = cuda_launch_wait,
	.format_supported = format_supported,
	.graph_capture = cuda_graph_capture,
	.graph_run = cuda_graph_run,
	.mark_record = mark_record,
	.mark_follow = mark_follow,
	.peer_connect = peer_connect,
#if TIMING_HOOKS_ENABLED
	.measure_begin = timer_cuda_measure_begin,
	.measure_end = timer_cuda_measure_end,
	.measure_elapsed_us = timer_cuda_measure_elapsed_us,
#endif
};
