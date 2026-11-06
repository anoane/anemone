/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * host_memory_backend.c: the backend of host_memory_backend.h, whose cards are arenas in host memory. Test support:
 * no part of the engine.
 *
 * Contract: host_memory_backend.h and backend_seam.h. How this file keeps them:
 *   - One mutex guards everything, the copy of an upload, a digest and a stand-in included: the calls of every
 *     card are serialised, which the seam allows, and the thread sanitizer follows every access. A wait sleeps
 *     outside it, and error lines are written outside it.
 *   - An upload is in flight from upload_begin until upload_wait was asked about it, or about a later ticket of
 *     its card, often enough. A digest waits for the uploads begun before it: it finishes them, or finds the one
 *     that a test made fail. A download is an upload in the other direction: one function begins both.
 *   - Everything a card does happens inside the call that begins it, in the order of the calls: so a stand-in sees
 *     every copy and launch begun before it, and a download every launch.
 *   - A card fails as one ticket does: from the first ticket not yet reported finished, unless a test named an
 *     earlier one. That failure goes with the card when it is closed; the ticket a test named stays until reset.
 *   - Beyond the seam: a block is not freed while an upload that was reported neither finished nor failed reads
 *     it. The error line, and the block that stays alive, show a caller that freed staging memory too early.
 *   - reset leaves a card that is open as it is: it closes nothing.
 */
#include "host_memory_backend.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "anemone/hooks.h"
#include "anemone/exchange_records.h"
#include "anemone/pack_digest.h"
#include "anemone/weight_format_seam.h"
#include "backend_cuda_kernel_images.h"

#define CARD_NAME "test card in host memory"
#define NANOSECONDS_PER_SECOND 1000000000ull
#define NO_BLOCK \
	BACKEND_MAXIMUM_STAGING_BLOCKS /* in place of the index of a staging block */
#define UPLOAD_RING_SIZE \
	BACKEND_MAXIMUM_UPLOADS_IN_FLIGHT /* the tickets in flight have distinct places */
#define NAME_CAPACITY TEST_HOST_MEMORY_BACKEND_FUNCTION_NAME_CAPACITY

_Static_assert(sizeof(CARD_NAME) <= CARD_NAME_CAPACITY,
	       "the name of a card fits its description");

/* What a card of each class has: the name the build gives the architecture whose kernel modules it loads, and the
 * most dynamic shared memory a block may ask. */
struct card_class {
    const char *architecture;
    uint32_t arithmetic_class;
    uint32_t most_dynamic_shared_bytes;
};
static const struct card_class card_classes[] = {
	{ "sm_80", ARITHMETIC_CLASS_SM_80,
	  TEST_HOST_MEMORY_BACKEND_MOST_DYNAMIC_SHARED_BYTES_SM_80 },
	{ "sm_120a", ARITHMETIC_CLASS_SM_120,
	  TEST_HOST_MEMORY_BACKEND_MOST_DYNAMIC_SHARED_BYTES_SM_120 }
};
static const uint32_t most_grid[LAUNCH_DIMENSIONS] = {
	TEST_HOST_MEMORY_BACKEND_MOST_GRID_X,
	TEST_HOST_MEMORY_BACKEND_MOST_GRID_Y_Z,
	TEST_HOST_MEMORY_BACKEND_MOST_GRID_Y_Z
};
static const uint32_t most_block[LAUNCH_DIMENSIONS] = {
	TEST_HOST_MEMORY_BACKEND_MOST_BLOCK_X_Y,
	TEST_HOST_MEMORY_BACKEND_MOST_BLOCK_X_Y,
	TEST_HOST_MEMORY_BACKEND_MOST_BLOCK_Z
};
static const unsigned char zero_bytes[LAUNCH_ARGUMENT_BYTES] = {
	0
}; /* what lies behind a record */

struct recorded_kernel {
    const char *kernel_file_name;                /* the kernel table's own name */
    uint32_t argument_byte_count;
    char function_name[NAME_CAPACITY];
};

/* The members stand widest first: the analyser refuses padding in a type that fills a table. */
struct device_card {
	unsigned char *arena; /* NULL while the card is closed */
	uint64_t arena_byte_count;
	uint64_t last_ticket; /* of the uploads begun since the card was opened */
	uint64_t finished_ticket; /* this upload and every earlier one were reported finished */
	uint64_t first_failed_ticket; /* 0, or the first one not reported finished when a launch faulted */
	/* per upload in flight, by its ticket modulo the size: the block it reads, and how often it was asked about */
	const unsigned char *block_of_upload[UPLOAD_RING_SIZE];
	struct recorded_kernel kernels
		[BACKEND_MAXIMUM_KERNELS]; /* by handle: the first kernel_count */
	uint32_t poll_count_of_upload[UPLOAD_RING_SIZE];
	/* what a test set, which stays until reset, and the counters since reset */
	uint32_t link_generation;
	uint32_t link_lane_count;
	uint32_t polls_before_an_upload_finishes;
	uint32_t most_uploads_in_flight;
	uint64_t upload_count;
	uint64_t memory_byte_count;
	uint64_t flip_arena_byte_offset;
	uint64_t failing_ticket; /* the one a test named; 0: none */
	uint64_t digests_until_the_failing_one; /* 0: no digest fails */
	uint64_t launches_until_the_failing_one; /* 0: no launch faults */
	uint64_t mark_records; /* since reset: marks set, marks followed, followed before set */
	uint64_t mark_follows;
	uint64_t unset_mark_follows;
	uint32_t arithmetic_class;
	uint32_t kernel_count; /* prepared since the card was opened */
	bool flip_is_armed;
	bool has_failed; /* it reported a device error: it begins no new work */
	bool mark_is_set[BACKEND_MAXIMUM_MARKS]; /* since the card was opened */
};

struct staging_block {
    unsigned char *memory;
    uint64_t byte_count;
};

struct stand_in_entry {
    test_host_memory_backend_stand_in stand_in;
    void *context;
    char kernel_file_name[NAME_CAPACITY];
    char function_name[NAME_CAPACITY];
};

static pthread_mutex_t backend_mutex = PTHREAD_MUTEX_INITIALIZER;   /* guards everything below */
static struct device_card cards[BACKEND_MAXIMUM_CARDS];
static uint32_t configured_card_count = 0;
static struct staging_block staging_blocks[BACKEND_MAXIMUM_STAGING_BLOCKS];
static uint32_t staging_block_count = 0;
static uint64_t staging_allocations_until_the_failing_one = 0;      /* 0: no allocation fails */
static struct stand_in_entry stand_ins[TEST_HOST_MEMORY_BACKEND_MAXIMUM_STAND_INS];
static uint32_t stand_in_count = 0;
static struct test_host_memory_backend_work work_log[TEST_HOST_MEMORY_BACKEND_LOG_CAPACITY];
static uint64_t work_count = 0;                                     /* begun since reset: the first are logged */

/* Under the mutex: the cards a call may name. */
static uint32_t usable_card_count(void)
{
	return configured_card_count < BACKEND_MAXIMUM_CARDS ?
		       configured_card_count :
		       BACKEND_MAXIMUM_CARDS;
}

/* Under the mutex: whether the address is a card of this backend that is open. */
static bool card_is_open(const struct device_card *card)
{
	for (uint32_t card_index = 0; card_index < BACKEND_MAXIMUM_CARDS;
	     card_index++) {
		if (card == &cards[card_index]) {
			return card->arena != NULL;
		}
	}
    return false;
}

/* What a card of that arithmetic class has, or NULL for a class no build runs on. */
static const struct card_class *class_of(uint32_t arithmetic_class)
{
    for (uint32_t index = 0; index < sizeof(card_classes) / sizeof(card_classes[0]); index++) {
        if (card_classes[index].arithmetic_class == arithmetic_class) {
            return &card_classes[index];
        }
    }
    return NULL;
}

/* The kernel table's own name of a kernel file the program holds a build of for that architecture, or for any if
 * architecture is NULL; NULL if it holds none. */
static const char *kernel_file_built(const char *architecture, const char *kernel_file_name)
{
	for (uint32_t image_index = 0; image_index < cuda_kernel_image_count;
	     image_index++) {
		const struct cuda_kernel_image *const image =
			&cuda_kernel_images[image_index];
		if (image->bytes != NULL && image->byte_count != 0 &&
		    strcmp(image->kernel_name, kernel_file_name) == 0 &&
		    (architecture == NULL ||
		     strcmp(image->architecture, architecture) == 0)) {
			return image->kernel_name;
		}
	}
    return NULL;
}

/* Under the mutex: adds one entry to the work log, or only counts it once the log is full. */
static void log_work(uint32_t kind, const struct device_card *card,
		     uint64_t arena_byte_offset, uint64_t byte_count,
		     const struct kernel_launch *launch)
{
    if (work_count < TEST_HOST_MEMORY_BACKEND_LOG_CAPACITY) {
        struct test_host_memory_backend_work *const work = &work_log[work_count];
        memset(work, 0, sizeof(*work));
        work->kind = kind;
        work->card_index = (uint32_t)(card - cards);
        work->arena_byte_offset = arena_byte_offset;
        work->byte_count = byte_count;
        if (launch != NULL) {
            work->launch = *launch;
        }
    }
    work_count++;
}

/* Under the mutex: the card has failed, as when a ticket fails. Every ticket not yet reported finished fails, until
 * the card is closed; the ticket a test named is left as it is. */
static void fail_card(struct device_card *card)
{
    card->has_failed = true;
    if (card->first_failed_ticket == 0) {
        card->first_failed_ticket = card->finished_ticket + 1;
    }
}

/* Under the mutex: whether an upload of that ticket fails: the ticket a test named and every later one, and every
 * ticket from the first the card had not reported finished when it failed. */
static bool ticket_fails(const struct device_card *card, uint64_t upload_ticket)
{
    return (card->failing_ticket != 0 && upload_ticket >= card->failing_ticket) ||
           (card->first_failed_ticket != 0 && upload_ticket >= card->first_failed_ticket);
}

/* Under the mutex: counts one call down to the failing one. True for the call a test made fail, once. */
static bool is_the_failing_call(uint64_t *calls_until_the_failing_one)
{
    if (*calls_until_the_failing_one == 0) {
        return false;
    }
    *calls_until_the_failing_one -= 1;
    return *calls_until_the_failing_one == 0;
}

/* Under the mutex: the index of the block that holds the whole range, or NO_BLOCK. */
static uint32_t block_that_holds(const void *first_byte, uint64_t byte_count)
{
    const uintptr_t first_address = (uintptr_t)first_byte;
    for (uint32_t block_index = 0; block_index < staging_block_count; block_index++) {
        const uintptr_t block_address = (uintptr_t)staging_blocks[block_index].memory;
        if (first_address >= block_address && first_address - block_address <= staging_blocks[block_index].byte_count &&
            byte_count <= staging_blocks[block_index].byte_count - (first_address - block_address)) {
            return block_index;
        }
    }
    return NO_BLOCK;
}

/* Under the mutex: whether an upload of that ticket is past reading its staging memory. */
static bool upload_is_over(const struct device_card *card,
			   uint64_t upload_ticket)
{
    return upload_ticket <= card->finished_ticket || (card->has_failed && ticket_fails(card, upload_ticket));
}

/* Under the mutex: reports finished, in the order of the tickets, every upload that was asked about often
 * enough. An upload that fails never finishes, nor does one behind it. */
static void advance_finished_ticket(struct device_card *card)
{
    while (card->finished_ticket < card->last_ticket) {
        const uint64_t next_ticket = card->finished_ticket + 1;
        const uint32_t poll_count = card->poll_count_of_upload[next_ticket % UPLOAD_RING_SIZE];
        if (ticket_fails(card, next_ticket) || poll_count < card->polls_before_an_upload_finishes) {
            return;
        }
        card->finished_ticket = next_ticket;
    }
}

MUST_CHECK static result_status card_count(uint32_t *card_count_out)
{
	result_status status = STATUS_INVALID_ARGUMENT;
	(void)pthread_mutex_lock(&backend_mutex);
	if (card_count_out != NULL) {
		status = configured_card_count > BACKEND_MAXIMUM_CARDS ?
				 STATUS_UNSUPPORTED :
				 STATUS_OK;
	}
	if (status == STATUS_OK) {
		*card_count_out = configured_card_count;
	}
    (void)pthread_mutex_unlock(&backend_mutex);
    return status;
}

MUST_CHECK static result_status
card_describe(uint32_t card_index,
	      struct card_description_record *description_out)
{
	result_status status = STATUS_INVALID_ARGUMENT;
	(void)pthread_mutex_lock(&backend_mutex);
	if (description_out != NULL && card_index < usable_card_count()) {
		const struct device_card *const card = &cards[card_index];
		memset(description_out, 0, sizeof(*description_out));
		memcpy(description_out->name, CARD_NAME, sizeof(CARD_NAME));
		description_out->arithmetic_class = card->arithmetic_class;
		description_out->link_generation = card->link_generation;
		description_out->link_lane_count = card->link_lane_count;
		description_out->memory_total_bytes = card->memory_byte_count;
		description_out->memory_free_bytes =
			card->memory_byte_count - card->arena_byte_count;
		status = STATUS_OK;
	}
    (void)pthread_mutex_unlock(&backend_mutex);
    return status;
}

MUST_CHECK static result_status card_open(uint32_t card_index,
					  uint64_t arena_byte_count,
					  struct device_card **card_out)
{
	result_status status = STATUS_INVALID_ARGUMENT;
	(void)pthread_mutex_lock(&backend_mutex);
	if (card_out != NULL && card_index < usable_card_count() &&
	    arena_byte_count != 0 && cards[card_index].arena == NULL) {
		struct device_card *const card = &cards[card_index];
		/* zero-filled, so that a byte the loader never delivered is the same in every run */
		unsigned char *const arena =
			arena_byte_count <= card->memory_byte_count ?
				calloc(1, (size_t)arena_byte_count) :
				NULL;
		status = STATUS_OUT_OF_MEMORY;
		if (arena != NULL) {
			card->arena = arena;
			card->arena_byte_count = arena_byte_count;
			card->last_ticket = 0;
			card->finished_ticket = 0;
			card->first_failed_ticket = 0;
			card->kernel_count = 0;
			card->has_failed = false;
			memset(card->mark_is_set, 0, sizeof(card->mark_is_set));
			*card_out = card;
			status = STATUS_OK;
		}
    }
    (void)pthread_mutex_unlock(&backend_mutex);
    return status;
}

/* There is nothing to wait for: an upload copies when it begins. With the card go its uploads in flight and its
 * kernels. */
static void card_close(struct device_card *card,
		       uint64_t wait_limit_nanoseconds)
{
    bool was_open = false;
    (void)wait_limit_nanoseconds;
    (void)pthread_mutex_lock(&backend_mutex);
    was_open = card_is_open(card);
    if (was_open) {
        free(card->arena);
        card->arena = NULL;
        card->arena_byte_count = 0;
        card->last_ticket = 0;
        card->finished_ticket = 0;
        card->first_failed_ticket = 0;
        card->kernel_count = 0;
    }
    (void)pthread_mutex_unlock(&backend_mutex);
    if (card != NULL && !was_open) {
	    LOG_ERROR(
		    "card_close: the address is no open card of the test backend");
    }
}

MUST_CHECK static result_status staging_allocate(uint64_t byte_count,
						 void **page_locked_memory_out)
{
	result_status status = STATUS_OUT_OF_MEMORY;
	if (page_locked_memory_out == NULL || byte_count == 0) {
		return STATUS_INVALID_ARGUMENT;
	}
    (void)pthread_mutex_lock(&backend_mutex);
    if (is_the_failing_call(&staging_allocations_until_the_failing_one)) {
	    status = STATUS_OUT_OF_MEMORY;
    } else if (staging_block_count == BACKEND_MAXIMUM_STAGING_BLOCKS) {
	    status = STATUS_BUSY_VALUE;
    } else if (byte_count <= SIZE_MAX - BACKEND_STAGING_ALIGNMENT_BYTES) {
	    /* aligned_alloc wants a size that is a multiple of the alignment */
	    const size_t whole_pages =
		    ((size_t)byte_count + BACKEND_STAGING_ALIGNMENT_BYTES - 1) /
		    BACKEND_STAGING_ALIGNMENT_BYTES *
		    BACKEND_STAGING_ALIGNMENT_BYTES;
	    unsigned char *const memory =
		    aligned_alloc(BACKEND_STAGING_ALIGNMENT_BYTES, whole_pages);
	    if (memory != NULL) {
		    staging_blocks[staging_block_count].memory = memory;
		    staging_blocks[staging_block_count].byte_count = byte_count;
		    staging_block_count++;
		    *page_locked_memory_out = memory;
		    status = STATUS_OK;
	    }
    }
    (void)pthread_mutex_unlock(&backend_mutex);
    return status;
}

static void staging_free(void *page_locked_memory)
{
    const char *problem = NULL;
    uint32_t block_index = 0;
    if (page_locked_memory == NULL) {
        return;
    }
    (void)pthread_mutex_lock(&backend_mutex);
    while (block_index < staging_block_count && staging_blocks[block_index].memory != page_locked_memory) {
        block_index++;
    }
    if (block_index == staging_block_count) {
        problem = "the address is not the start of a staging block";
    }
    for (uint32_t card_index = 0;
	 card_index < BACKEND_MAXIMUM_CARDS && problem == NULL; card_index++) {
	    const struct device_card *const card = &cards[card_index];
	    for (uint64_t ticket = card->finished_ticket + 1;
		 ticket <= card->last_ticket; ticket++) {
		    if (!upload_is_over(card, ticket) &&
			card->block_of_upload[ticket % UPLOAD_RING_SIZE] ==
				page_locked_memory) {
			    problem =
				    "an upload that was reported neither finished nor failed still reads the block";
		    }
	    }
    }
    if (problem == NULL) {
        free(staging_blocks[block_index].memory);
        staging_blocks[block_index] = staging_blocks[staging_block_count - 1];
        staging_block_count--;
    }
    (void)pthread_mutex_unlock(&backend_mutex);
    if (problem != NULL) {
	    LOG_ERROR("staging_free: %s: it is left alone", problem);
    }
}

/* Makes one copy between a range of the arena and a range of one staging block, and gives it the next ticket: a
 * download when it has a destination, else an upload from source. */
MUST_CHECK static result_status
copy_begin(struct device_card *card, uint64_t arena_byte_offset,
	   const void *source, void *destination, uint64_t byte_count,
	   uint64_t *ticket_out)
{
    const void *const staged_range = destination != NULL ? destination : source;
    result_status status = STATUS_INVALID_ARGUMENT;
    (void)pthread_mutex_lock(&backend_mutex);
    const uint32_t block_index = staged_range != NULL ? block_that_holds(staged_range, byte_count) : NO_BLOCK;
    if (card_is_open(card) && ticket_out != NULL && byte_count != 0 && block_index != NO_BLOCK &&
        arena_byte_offset <= card->arena_byte_count && byte_count <= card->arena_byte_count - arena_byte_offset) {
        const uint64_t uploads_in_flight = card->last_ticket - card->finished_ticket;
        if (card->has_failed) {
		status = STATUS_DEVICE_ERROR;
	} else if (uploads_in_flight >= BACKEND_MAXIMUM_UPLOADS_IN_FLIGHT) {
		status = STATUS_BUSY_VALUE;
	} else {
		const uint64_t ticket = card->last_ticket + 1;
		if (destination != NULL) {
			memcpy(destination, card->arena + arena_byte_offset,
			       (size_t)byte_count);
		} else {
			memcpy(card->arena + arena_byte_offset, source,
			       (size_t)byte_count);
			card->upload_count++;
		}
		if (destination == NULL && card->flip_is_armed &&
		    card->flip_arena_byte_offset >= arena_byte_offset &&
		    card->flip_arena_byte_offset - arena_byte_offset <
			    byte_count) {
			unsigned char *const flipped_byte =
				&card->arena[card->flip_arena_byte_offset];
			*flipped_byte =
				(unsigned char)(*flipped_byte ^
						1u); /* the link corrupted this one transfer */
			card->flip_is_armed = false;
		}
		card->last_ticket = ticket;
		card->poll_count_of_upload[ticket % UPLOAD_RING_SIZE] = 0;
		card->block_of_upload[ticket % UPLOAD_RING_SIZE] =
			staging_blocks[block_index].memory;
		if (uploads_in_flight + 1 > card->most_uploads_in_flight) {
			card->most_uploads_in_flight =
				(uint32_t)(uploads_in_flight + 1);
		}
		advance_finished_ticket(
			card); /* with no polls asked for, it has finished already */
		log_work(destination != NULL ?
				 TEST_HOST_MEMORY_BACKEND_DOWNLOAD :
				 TEST_HOST_MEMORY_BACKEND_UPLOAD,
			 card, arena_byte_offset, byte_count, NULL);
		*ticket_out = ticket;
		status = STATUS_OK;
	}
    }
    (void)pthread_mutex_unlock(&backend_mutex);
    return status;
}

MUST_CHECK static result_status upload_begin(struct device_card *card,
					     uint64_t arena_byte_offset,
					     const void *page_locked_source,
					     uint64_t byte_count,
					     uint64_t *upload_ticket_out)
{
    return copy_begin(card, arena_byte_offset, page_locked_source, NULL, byte_count, upload_ticket_out);
}

MUST_CHECK static result_status download_begin(struct device_card *card,
					       uint64_t arena_byte_offset,
					       void *page_locked_destination,
					       uint64_t byte_count,
					       uint64_t *download_ticket_out)
{
    return copy_begin(card, arena_byte_offset, NULL, page_locked_destination, byte_count, download_ticket_out);
}

MUST_CHECK static result_status upload_wait(struct device_card *card,
					    uint64_t upload_ticket,
					    uint64_t wait_limit_nanoseconds)
{
	result_status status = STATUS_INVALID_ARGUMENT;
	(void)pthread_mutex_lock(&backend_mutex);
	if (card_is_open(card) && upload_ticket != 0 &&
	    upload_ticket <= card->last_ticket) {
		for (uint64_t ticket = card->finished_ticket + 1;
		     ticket <= upload_ticket; ticket++) {
			uint32_t *const poll_count =
				&card->poll_count_of_upload[ticket %
							    UPLOAD_RING_SIZE];
			*poll_count += *poll_count < UINT32_MAX ? 1u : 0u;
		}
		advance_finished_ticket(card);
		if (upload_ticket <= card->finished_ticket) {
			status = STATUS_OK;
		} else if (ticket_fails(card, upload_ticket)) {
			card->has_failed = true;
			status = STATUS_DEVICE_ERROR;
		} else {
			status = STATUS_TIMED_OUT;
		}
    }
    (void)pthread_mutex_unlock(&backend_mutex);
    if (status == STATUS_TIMED_OUT && wait_limit_nanoseconds > 0) {
	    const uint64_t sleep_nanoseconds =
		    wait_limit_nanoseconds <
				    TEST_HOST_MEMORY_BACKEND_WAIT_SLICE_NANOSECONDS ?
			    wait_limit_nanoseconds :
			    TEST_HOST_MEMORY_BACKEND_WAIT_SLICE_NANOSECONDS;
	    const struct timespec slice = {
		    .tv_sec = (time_t)(sleep_nanoseconds /
				       NANOSECONDS_PER_SECOND),
		    .tv_nsec =
			    (long)(sleep_nanoseconds % NANOSECONDS_PER_SECOND)
	    };
	    (void)nanosleep(&slice, NULL);
    }
    return status;
}

/* The limit is not used: every upload begun before the digest has copied its bytes already. */
MUST_CHECK static result_status digest(struct device_card *card,
				       uint64_t arena_byte_offset,
				       uint64_t byte_count,
				       uint64_t wait_limit_nanoseconds,
				       uint64_t *digest_out)
{
	result_status status = STATUS_INVALID_ARGUMENT;
	(void)wait_limit_nanoseconds;
	(void)pthread_mutex_lock(&backend_mutex);
	if (card_is_open(card) && digest_out != NULL &&
	    arena_byte_offset % PACK_DIGEST_WORD_BYTES == 0 &&
	    byte_count % PACK_DIGEST_WORD_BYTES == 0 &&
	    arena_byte_offset <= card->arena_byte_count &&
	    byte_count <= card->arena_byte_count - arena_byte_offset) {
		if (card->has_failed ||
		    is_the_failing_call(&card->digests_until_the_failing_one) ||
		    ticket_fails(card, card->last_ticket)) {
			card->has_failed = true;
			status = STATUS_DEVICE_ERROR;
		} else {
			card->finished_ticket = card->last_ticket;
			status = pack_digest_of_host_bytes(
				card->arena + arena_byte_offset, byte_count,
				digest_out);
			if (status == STATUS_OK) {
				log_work(TEST_HOST_MEMORY_BACKEND_DIGEST, card,
					 arena_byte_offset, byte_count, NULL);
			}
		}
	}
    (void)pthread_mutex_unlock(&backend_mutex);
    return status;
}

/* Why kernel_prepare refused a kernel on a card with a class, for its error line. */
enum prepare_refusal { PREPARE_ACCEPTED, PREPARE_ARGUMENTS, PREPARE_NO_BUILD, PREPARE_NO_BUILD_FOR_THE_CARD,
                       PREPARE_OTHER_RECORD, PREPARE_TABLE_FULL };

/* Under the mutex, on a sound open card with an architecture, for a kernel file of the kernel table: the handle of
 * the kernel, prepared now or before, or the refusal (with the size of the record it was prepared with). */
static enum prepare_refusal
kernel_record(struct device_card *card, const char *architecture,
	      const char *kernel_file_name, const char *function_name,
	      size_t name_length, uint32_t argument_byte_count,
	      uint32_t *kernel_out, uint32_t *prepared_byte_count_out)
{
    const char *const built_name = kernel_file_built(architecture, kernel_file_name);
    if (built_name == NULL) {
        return PREPARE_NO_BUILD_FOR_THE_CARD;
    }
    for (uint32_t kernel = 0; kernel < card->kernel_count; kernel++) {
        const struct recorded_kernel *const recorded = &card->kernels[kernel];
        if (recorded->kernel_file_name == built_name && strcmp(recorded->function_name, function_name) == 0) {
            *prepared_byte_count_out = recorded->argument_byte_count;
            if (recorded->argument_byte_count != argument_byte_count) {
                return PREPARE_OTHER_RECORD;
            }
            *kernel_out = kernel;               /* prepared before: the same handle */
            return PREPARE_ACCEPTED;
        }
    }
    if (card->kernel_count >= BACKEND_MAXIMUM_KERNELS) {
	    return PREPARE_TABLE_FULL;
    }
    struct recorded_kernel *const recorded = &card->kernels[card->kernel_count];
    recorded->kernel_file_name = built_name;
    recorded->argument_byte_count = argument_byte_count;
    memcpy(recorded->function_name, function_name, name_length);
    recorded->function_name[name_length] = '\0';
    *kernel_out = card->kernel_count;
    card->kernel_count++;
    return PREPARE_ACCEPTED;
}

/* A card of class 0 has no build of any kernel: what the seam refuses whatever the kernel, then that. A card with a
 * class records the kernel, as the header says. */
MUST_CHECK static result_status kernel_prepare(struct device_card *card,
					       const char *kernel_file_name,
					       const char *function_name,
					       uint32_t argument_byte_count,
					       uint32_t *kernel_out)
{
    const size_t name_length = function_name == NULL ? 0 : strnlen(function_name, NAME_CAPACITY);
    const bool arguments_are_given =
	    kernel_file_name != NULL && name_length != 0 &&
	    kernel_out != NULL && argument_byte_count != 0 &&
	    argument_byte_count <= LAUNCH_ARGUMENT_BYTES;
    enum prepare_refusal refusal = PREPARE_ACCEPTED;
    uint32_t prepared_byte_count = 0;
    uint32_t kernel_count = 0;
    result_status status = STATUS_INVALID_ARGUMENT;
    (void)pthread_mutex_lock(&backend_mutex);
    const bool is_open = card_is_open(card);
    const struct card_class *const card_class = is_open ? class_of(card->arithmetic_class) : NULL;
    const char *const architecture = card_class != NULL ? card_class->architecture : NULL;
    if (is_open && architecture == NULL && arguments_are_given) {
	    status = card->has_failed ? STATUS_DEVICE_ERROR :
					STATUS_UNSUPPORTED;
    } else if (architecture != NULL) {
        refusal = !arguments_are_given || name_length == NAME_CAPACITY ? PREPARE_ARGUMENTS
                : kernel_file_built(NULL, kernel_file_name) == NULL ? PREPARE_NO_BUILD : PREPARE_ACCEPTED;
        if (refusal == PREPARE_ACCEPTED && card->has_failed) {
		status = STATUS_DEVICE_ERROR;
	} else if (refusal == PREPARE_ACCEPTED) {
            refusal = kernel_record(card, architecture, kernel_file_name, function_name, name_length,
                                    argument_byte_count, kernel_out, &prepared_byte_count);
	    status = refusal == PREPARE_ACCEPTED     ? STATUS_OK :
		     refusal == PREPARE_OTHER_RECORD ? STATUS_INVALID_ARGUMENT :
						       STATUS_UNSUPPORTED;
	}
#if LOGGING_HOOKS_ENABLED
	kernel_count = card->kernel_count;
#endif
    }
    (void)pthread_mutex_unlock(&backend_mutex);
    if (refusal == PREPARE_ARGUMENTS) {
	    LOG_ERROR(
		    "kernel_prepare: a missing argument, or a function name or a record of no bytes or too many");
    } else if (refusal == PREPARE_NO_BUILD) {
	    LOG_ERROR("kernel_prepare: no architecture has a build of %s",
		      kernel_file_name);
    } else if (refusal == PREPARE_NO_BUILD_FOR_THE_CARD) {
	    LOG_ERROR("kernel_prepare: no build of %s for %s", kernel_file_name,
		      architecture);
    } else if (refusal == PREPARE_OTHER_RECORD) {
	    LOG_ERROR("kernel_prepare: %s takes a record of %" PRIu32
		      " bytes, not of %" PRIu32,
		      function_name, prepared_byte_count, argument_byte_count);
    } else if (refusal == PREPARE_TABLE_FULL) {
	    LOG_ERROR("kernel_prepare: the card has prepared %" PRIu32
		      " kernels and takes no more",
		      kernel_count);
    }
    return status;
}

/* Under the mutex: whether a launch of a card of that class may be queued: its kernel is prepared, its record has
 * the kernel's size, every dimension is at least 1 and at most the limit, its block has no more threads and no
 * more dynamic shared memory than the limits, and its reserved member and the bytes behind its record are zero. */
static bool launch_is_valid(const struct device_card *card,
			    const struct card_class *card_class,
			    const struct kernel_launch *launch)
{
    uint64_t thread_count = 1;
    if (launch->kernel == BACKEND_COPY) {
	    if (launch->branch != LAUNCH_CHAIN ||
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
	    return copy.source.address_word_offset == EXCHANGE_OWN_ARENA &&
		   copy.destination.address_word_offset == EXCHANGE_OWN_ARENA &&
		   copy.byte_count != 0 &&
		   copy.source.offset <= card->arena_byte_count &&
		   copy.byte_count <=
			   card->arena_byte_count - copy.source.offset &&
		   copy.destination.offset <= card->arena_byte_count &&
		   copy.byte_count <=
			   card->arena_byte_count - copy.destination.offset &&
		   (copy.source.offset + copy.byte_count <=
			    copy.destination.offset ||
		    copy.destination.offset + copy.byte_count <=
			    copy.source.offset);
    }
    if (launch->kernel >= card->kernel_count) {
        return false;
    }
    /* the record's size is looked at before the bytes behind it are: no byte outside the argument is read */
    bool is_valid =
	    launch->branch <= LAUNCH_SIDE_MORE &&
	    launch->argument_byte_count ==
		    card->kernels[launch->kernel].argument_byte_count &&
	    launch->argument_byte_count <= LAUNCH_ARGUMENT_BYTES &&
	    launch->dynamic_shared_byte_count <=
		    card_class->most_dynamic_shared_bytes &&
	    memcmp(launch->argument + launch->argument_byte_count, zero_bytes,
		   LAUNCH_ARGUMENT_BYTES - launch->argument_byte_count) == 0;
    for (uint32_t dimension = 0; is_valid && dimension < LAUNCH_DIMENSIONS;
	 dimension++) {
	    thread_count *=
		    launch->block[dimension]; /* at most the limit before it: no wrap */
	    is_valid = launch->grid[dimension] != 0 &&
		       launch->grid[dimension] <= most_grid[dimension] &&
		       launch->block[dimension] != 0 &&
		       launch->block[dimension] <= most_block[dimension] &&
		       thread_count <=
			       TEST_HOST_MEMORY_BACKEND_MOST_THREADS_PER_BLOCK;
    }
    return is_valid;
}

/* Under the mutex: the stand-in a test set for a kernel, or NULL. */
static const struct stand_in_entry *stand_in_of(const struct recorded_kernel *kernel)
{
    for (uint32_t index = 0; index < stand_in_count; index++) {
        if (strcmp(stand_ins[index].kernel_file_name, kernel->kernel_file_name) == 0 &&
            strcmp(stand_ins[index].function_name, kernel->function_name) == 0) {
            return &stand_ins[index];
        }
    }
    return NULL;
}

/* Under the mutex, on a sound card with a class: queues launches that were checked. Each is logged and, until the
 * card fails, runs its stand-in. */
static void run_launches(struct device_card *card,
			 const struct kernel_launch *launches,
			 uint32_t launch_count)
{
    for (uint32_t launch_index = 0; launch_index < launch_count; launch_index++) {
	    const struct kernel_launch *const launch = &launches[launch_index];
	    log_work(TEST_HOST_MEMORY_BACKEND_LAUNCH, card, 0, 0, launch);
	    if (!card->has_failed) {
		    if (is_the_failing_call(
				&card->launches_until_the_failing_one)) {
			    fail_card(card);
		    } else if (launch->kernel == BACKEND_COPY) {
			    struct exchange_copy_arguments copy;
			    memcpy(&copy, launch->argument, sizeof(copy));
			    memcpy(card->arena + copy.destination.offset,
				   card->arena + copy.source.offset,
				   copy.byte_count);
		    } else {
			    const struct stand_in_entry *const entry =
				    stand_in_of(&card->kernels[launch->kernel]);
			    if (entry != NULL &&
				!entry->stand_in(
					launch, (uint32_t)(card - cards),
					card->arena, card->arena_byte_count,
					entry->context)) {
				    fail_card(card);
			    }
		    }
	}
    }
}

/* A card of class 0: no handle was ever prepared, so no launch is looked at and nothing is queued. */
MUST_CHECK static result_status launch_run(struct device_card *card,
					   const struct kernel_launch *launches,
					   uint32_t launch_count)
{
    uint32_t valid_count = 0;
    bool is_refused = false;
    result_status status = STATUS_INVALID_ARGUMENT;
    (void)pthread_mutex_lock(&backend_mutex);
    const bool is_open = card_is_open(card);
    const struct card_class *const card_class = is_open ? class_of(card->arithmetic_class) : NULL;
    if (is_open && card_class == NULL) {
        if (launches != NULL && launch_count != 0) {
		status = card->has_failed ? STATUS_DEVICE_ERROR :
					    STATUS_UNSUPPORTED;
	}
    } else if (card_class != NULL) {
        while (launches != NULL && valid_count < launch_count &&
               launch_is_valid(card, card_class, &launches[valid_count])) {
            valid_count++;
        }
        is_refused = launches == NULL || launch_count == 0 || valid_count < launch_count;
	status = is_refused	  ? STATUS_INVALID_ARGUMENT :
		 card->has_failed ? STATUS_DEVICE_ERROR :
				    STATUS_OK;
	if (status == STATUS_OK) {
		run_launches(card, launches, launch_count);
	}
    }
    (void)pthread_mutex_unlock(&backend_mutex);
    if (is_refused) {
	    LOG_ERROR("launch_run: refused at launch %" PRIu32 " of %" PRIu32
		      ": nothing is queued",
		      valid_count, launch_count);
    }
    return status;
}

/* Nothing is ever left queued: copies happen when they begin, and launches inside launch_run. */
MUST_CHECK static result_status launch_wait(struct device_card *card,
					    uint64_t wait_limit_nanoseconds)
{
	result_status status = STATUS_INVALID_ARGUMENT;
	(void)wait_limit_nanoseconds;
	(void)pthread_mutex_lock(&backend_mutex);
	if (card_is_open(card)) {
		status = card->has_failed ? STATUS_DEVICE_ERROR : STATUS_OK;
	}
    (void)pthread_mutex_unlock(&backend_mutex);
    return status;
}

/* A copy finishes when it begins, so a mark has nothing to order: both count, and a follow checks the owner's mark. */
MUST_CHECK static result_status mark_record(struct device_card *card,
					    uint32_t mark)
{
	result_status status = STATUS_INVALID_ARGUMENT;
	(void)pthread_mutex_lock(&backend_mutex);
	if (card_is_open(card) && mark < BACKEND_MAXIMUM_MARKS) {
		status = card->has_failed ? STATUS_DEVICE_ERROR : STATUS_OK;
		card->mark_is_set[mark] = card->mark_is_set[mark] ||
					  status == STATUS_OK;
		card->mark_records += status == STATUS_OK ? 1u : 0u;
	}
    (void)pthread_mutex_unlock(&backend_mutex);
    return status;
}

MUST_CHECK static result_status
mark_follow(struct device_card *card, struct device_card *owner, uint32_t mark)
{
	result_status status = STATUS_INVALID_ARGUMENT;
	(void)pthread_mutex_lock(&backend_mutex);
	if (card_is_open(card) && card_is_open(owner) &&
	    mark < BACKEND_MAXIMUM_MARKS) {
		status = card->has_failed ? STATUS_DEVICE_ERROR : STATUS_OK;
		card->mark_follows += status == STATUS_OK ? 1u : 0u;
		card->unset_mark_follows +=
			status == STATUS_OK && !owner->mark_is_set[mark] ? 1u :
									   0u;
	}
    (void)pthread_mutex_unlock(&backend_mutex);
    return status;
}

/* The formats this build carries, on a card a test gave a class. Needs no open card. */
static bool format_supported(uint32_t card_index,
			     enum weight_format_code format)
{
    bool is_supported = false;
    (void)pthread_mutex_lock(&backend_mutex);
    is_supported = card_index < usable_card_count() &&
		   class_of(cards[card_index].arithmetic_class) != NULL &&
		   weight_format_operations_of(format) != NULL;
    (void)pthread_mutex_unlock(&backend_mutex);
    return is_supported;
}

const struct backend_operations test_host_memory_backend = {
	.name = "test host memory",
	.card_count = card_count,
	.card_describe = card_describe,
	.card_open = card_open,
	.card_close = card_close,
	.staging_allocate = staging_allocate,
	.staging_free = staging_free,
	.upload_begin = upload_begin,
	.upload_wait = upload_wait,
	.digest = digest,
	.download_begin = download_begin,
	.kernel_prepare = kernel_prepare,
	.launch_run = launch_run,
	.launch_wait = launch_wait,
	.format_supported = format_supported,
	.mark_record = mark_record,
	.mark_follow = mark_follow,
};

void test_host_memory_backend_reset(uint32_t card_count_to_report, uint64_t memory_bytes_per_card)
{
    (void)pthread_mutex_lock(&backend_mutex);
    configured_card_count = card_count_to_report;
    staging_allocations_until_the_failing_one = 0;
    stand_in_count = 0;
    work_count = 0;
    for (uint32_t card_index = 0; card_index < BACKEND_MAXIMUM_CARDS;
	 card_index++) {
	    if (cards[card_index].arena == NULL) {
		    memset(&cards[card_index], 0, sizeof(cards[card_index]));
	    }
	    cards[card_index].memory_byte_count = memory_bytes_per_card;
    }
    (void)pthread_mutex_unlock(&backend_mutex);
}

/* The card a setter or a counter names, with the mutex taken; NULL, and the mutex not taken, for no card. */
static struct device_card *locked_card(uint32_t card_index)
{
	if (card_index >= BACKEND_MAXIMUM_CARDS) {
		return NULL;
	}
    (void)pthread_mutex_lock(&backend_mutex);
    return &cards[card_index];
}

void test_host_memory_backend_set_link(uint32_t card_index, uint32_t link_generation, uint32_t link_lane_count)
{
	struct device_card *const card = locked_card(card_index);
	if (card != NULL) {
		card->link_generation = link_generation;
		card->link_lane_count = link_lane_count;
		(void)pthread_mutex_unlock(&backend_mutex);
	}
}

void test_host_memory_backend_set_arithmetic_class(uint32_t card_index, uint32_t arithmetic_class)
{
	struct device_card *const card = locked_card(card_index);
	if (card != NULL) {
		card->arithmetic_class = arithmetic_class;
		(void)pthread_mutex_unlock(&backend_mutex);
	}
}

void test_host_memory_backend_set_polls_before_an_upload_finishes(uint32_t card_index, uint32_t poll_count)
{
	struct device_card *const card = locked_card(card_index);
	if (card != NULL) {
		card->polls_before_an_upload_finishes = poll_count;
		(void)pthread_mutex_unlock(&backend_mutex);
	}
}

void test_host_memory_backend_fail_upload(uint32_t card_index, uint64_t upload_ticket)
{
	struct device_card *const card = locked_card(card_index);
	if (card != NULL) {
		card->failing_ticket = upload_ticket;
		(void)pthread_mutex_unlock(&backend_mutex);
	}
}

void test_host_memory_backend_flip_bit_in_next_upload(uint32_t card_index, uint64_t arena_byte_offset)
{
	struct device_card *const card = locked_card(card_index);
	if (card != NULL) {
		card->flip_is_armed = true;
		card->flip_arena_byte_offset = arena_byte_offset;
		(void)pthread_mutex_unlock(&backend_mutex);
	}
}

void test_host_memory_backend_fail_staging_allocation(uint64_t allocation_number)
{
    (void)pthread_mutex_lock(&backend_mutex);
    staging_allocations_until_the_failing_one = allocation_number;
    (void)pthread_mutex_unlock(&backend_mutex);
}

void test_host_memory_backend_fail_digest(uint32_t card_index, uint64_t digest_number)
{
	struct device_card *const card = locked_card(card_index);
	if (card != NULL) {
		card->digests_until_the_failing_one = digest_number;
		(void)pthread_mutex_unlock(&backend_mutex);
	}
}

void test_host_memory_backend_fail_launch(uint32_t card_index, uint64_t launch_number)
{
	struct device_card *const card = locked_card(card_index);
	if (card != NULL) {
		card->launches_until_the_failing_one = launch_number;
		(void)pthread_mutex_unlock(&backend_mutex);
	}
}

result_status test_host_memory_backend_set_stand_in(
	const char *kernel_file_name, const char *function_name,
	test_host_memory_backend_stand_in stand_in, void *context)
{
    const size_t file_length = kernel_file_name == NULL ? 0 : strnlen(kernel_file_name, NAME_CAPACITY);
    const size_t function_length = function_name == NULL ? 0 : strnlen(function_name, NAME_CAPACITY);
    result_status status = STATUS_OK;
    uint32_t index = 0;
    if (file_length == 0 || file_length == NAME_CAPACITY || function_length == 0 || function_length == NAME_CAPACITY) {
	    return STATUS_INVALID_ARGUMENT;
    }
    (void)pthread_mutex_lock(&backend_mutex);
    while (index < stand_in_count && (strcmp(stand_ins[index].kernel_file_name, kernel_file_name) != 0 ||
                                      strcmp(stand_ins[index].function_name, function_name) != 0)) {
        index++;
    }
    if (stand_in == NULL && index < stand_in_count) {
        stand_ins[index] = stand_ins[stand_in_count - 1];   /* the order of the table means nothing */
        stand_in_count--;
    } else if (stand_in != NULL && index == TEST_HOST_MEMORY_BACKEND_MAXIMUM_STAND_INS) {
	    status = STATUS_BUSY_VALUE;
    } else if (stand_in != NULL) {
        struct stand_in_entry *const entry = &stand_ins[index];
        memset(entry, 0, sizeof(*entry));
        memcpy(entry->kernel_file_name, kernel_file_name, file_length);
        memcpy(entry->function_name, function_name, function_length);
        entry->stand_in = stand_in;
        entry->context = context;
        stand_in_count += index == stand_in_count ? 1u : 0u;
    }
    (void)pthread_mutex_unlock(&backend_mutex);
    return status;
}

const unsigned char *
test_host_memory_backend_arena_of(const struct device_card *card)
{
    const unsigned char *arena = NULL;
    (void)pthread_mutex_lock(&backend_mutex);
    if (card_is_open(card)) {
        arena = card->arena;
    }
    (void)pthread_mutex_unlock(&backend_mutex);
    return arena;
}

uint32_t test_host_memory_backend_open_card_count(void)
{
    uint32_t open_card_count = 0;
    (void)pthread_mutex_lock(&backend_mutex);
    for (uint32_t card_index = 0; card_index < BACKEND_MAXIMUM_CARDS;
	 card_index++) {
	    open_card_count += cards[card_index].arena != NULL ? 1u : 0u;
    }
    (void)pthread_mutex_unlock(&backend_mutex);
    return open_card_count;
}

uint32_t test_host_memory_backend_staging_block_count(void)
{
    uint32_t block_count = 0;
    (void)pthread_mutex_lock(&backend_mutex);
    block_count = staging_block_count;
    (void)pthread_mutex_unlock(&backend_mutex);
    return block_count;
}

void test_host_memory_backend_marks(uint32_t card_index, uint64_t *records_out, uint64_t *follows_out,
                                    uint64_t *unset_follows_out)
{
	const struct device_card *const card = locked_card(card_index);
	*records_out = card != NULL ? card->mark_records : 0;
	*follows_out = card != NULL ? card->mark_follows : 0;
	*unset_follows_out = card != NULL ? card->unset_mark_follows : 0;
	if (card != NULL) {
		(void)pthread_mutex_unlock(&backend_mutex);
	}
}

uint64_t test_host_memory_backend_upload_count(uint32_t card_index)
{
    uint64_t upload_count = 0;
    const struct device_card *const card = locked_card(card_index);
    if (card != NULL) {
        upload_count = card->upload_count;
        (void)pthread_mutex_unlock(&backend_mutex);
    }
    return upload_count;
}

uint32_t test_host_memory_backend_most_uploads_in_flight(uint32_t card_index)
{
    uint32_t most_uploads_in_flight = 0;
    const struct device_card *const card = locked_card(card_index);
    if (card != NULL) {
        most_uploads_in_flight = card->most_uploads_in_flight;
        (void)pthread_mutex_unlock(&backend_mutex);
    }
    return most_uploads_in_flight;
}

bool test_host_memory_backend_kernel_of(uint32_t card_index, uint32_t kernel,
                                        struct test_host_memory_backend_kernel *kernel_out)
{
    bool is_found = false;
    const struct device_card *const card = locked_card(card_index);
    if (card != NULL) {
        is_found = kernel_out != NULL && card->arena != NULL && kernel < card->kernel_count;
        if (is_found) {
            const struct recorded_kernel *const recorded = &card->kernels[kernel];
            memset(kernel_out, 0, sizeof(*kernel_out));
            kernel_out->kernel_file_name = recorded->kernel_file_name;
            memcpy(kernel_out->function_name, recorded->function_name, sizeof(kernel_out->function_name));
            kernel_out->argument_byte_count = recorded->argument_byte_count;
        }
        (void)pthread_mutex_unlock(&backend_mutex);
    }
    return is_found;
}

uint64_t test_host_memory_backend_work_count(void)
{
    uint64_t count = 0;
    (void)pthread_mutex_lock(&backend_mutex);
    count = work_count;
    (void)pthread_mutex_unlock(&backend_mutex);
    return count;
}

bool test_host_memory_backend_work_at(uint64_t work_index, struct test_host_memory_backend_work *work_out)
{
    bool is_logged = false;
    (void)pthread_mutex_lock(&backend_mutex);
    is_logged = work_out != NULL && work_index < work_count && work_index < TEST_HOST_MEMORY_BACKEND_LOG_CAPACITY;
    if (is_logged) {
        *work_out = work_log[work_index];
    }
    (void)pthread_mutex_unlock(&backend_mutex);
    return is_logged;
}
