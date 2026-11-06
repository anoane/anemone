/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * loader.c: where a placement puts the extents of a pack, and the load that brings them there. Contract: loader.h.
 *
 * A destination is a lane: one for every card of the layout and one for host memory. A lane hands out its bytes as
 * pieces: the next run that is contiguous in the file and in the lane, cut at LOADER_CHUNK_BYTES. A piece
 * for a card travels in one of the lane's staging slots, which is free again once upload_wait has reported its
 * upload finished; a piece for host memory is read into its place in the host arena, which is mapped before any thread
 * starts. One mutex guards what the threads share: no thread holds it across a read, an upload_begin, a digest or a
 * sleep, and the first failure, recorded under it, ends every thread at its next step. A thread with nothing to take
 * sleeps a slice and looks again: none waits for an upload. A reader thread transfers until every piece has arrived,
 * then digests; the calling thread joins them all and sends what differed the same way again. Fault points:
 * loader_layout_allocation, loader_state_allocation, loader_host_arena, loader_thread_create, loader_clock.
 */
#include "anemone/loader.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>

#include "anemone/hooks.h"
#include "anemone/pack_digest.h"

#define NANOSECONDS_PER_SECOND 1000000000ull
#define WAIT_SLICE_NANOSECONDS 1000000           /* a thread with nothing to take looks again after this long */
#define HUGE_PAGE_BYTES 2097152ull               /* 2 MiB: the host arena is mapped as a whole number of them */
#define HOST_ARENA_MAPPED_BYTES(bytes) ((size_t)(((bytes) + HUGE_PAGE_BYTES - 1) / HUGE_PAGE_BYTES * HUGE_PAGE_BYTES))
#define LINK_GENERATION_COUNT 6
#define MAXIMUM_LANES \
	(BACKEND_MAXIMUM_CARDS + 1) /* the cards, then host memory */
#define MAXIMUM_READER_THREADS \
	(STORAGE_MAXIMUM_COPIES * LOADER_MAXIMUM_READER_THREADS_PER_COPY)

_Static_assert(LOADER_CHUNK_BYTES % PACK_BLOCK_BYTES == 0,
	       "a piece is cut between two blocks");
/* The bytes one lane of a link carries in a second, by generation from 1: its line rate less its encoding. */
static const double link_lane_bytes_per_second[LINK_GENERATION_COUNT] = {250e6, 500e6, 985e6, 1969e6, 3938e6, 7563e6};

/* A piece on its way: in a slot of its card's lane or, for host memory, in one of its thread, without memory. */
struct slot {
	void *memory; /* LOADER_CHUNK_BYTES of staging memory */
	struct device_card *card; /* NULL for host memory */
	uint64_t file_byte_offset;
	uint64_t destination_byte_offset;
	uint64_t byte_count;
	uint64_t upload_ticket; /* while it uploads */
	enum slot_state {
		SLOT_FREE,
		SLOT_READING,
		SLOT_UPLOADING,
		SLOT_ABANDONED
	} state; /* abandoned: its card failed */
};

struct lane {
	uint32_t destination; /* a card of the layout, or PLACEMENT_IN_HOST_MEMORY */
	uint32_t next_extent_index; /* the cursor: where the next piece begins */
	uint64_t next_byte_offset_in_extent;
	uint64_t bytes_not_taken;
	uint64_t arrival_nanoseconds; /* when its last piece arrived; zero while none has */
	double seconds_per_byte; /* at the nominal rate of the link; zero for host memory */
	uint32_t check_extent_index; /* the next extent to digest: the digesting ends at the cursor */
	bool is_being_checked; /* a card is digested by one thread at a time */
	struct slot slots[LOADER_STAGING_SLOTS_PER_CARD];
};

struct loader {
	const struct load_request *request;
	const struct pack_extent *extents;
	uint32_t extent_count;
	uint32_t copy_count;
	uint32_t lane_count; /* the cards of the layout, then host memory */
	pthread_mutex_t
		mutex; /* guards everything below while reader threads run */
	result_status status; /* the first failure */
	uint32_t next_thread_number; /* a thread takes it as it begins: it names the copy it reads */
	uint64_t bytes_not_delivered;
	uint64_t last_arrival_nanoseconds; /* the stall rule counts from here */
	bool copy_is_dropped[STORAGE_MAXIMUM_COPIES];
	struct load_result
		result; /* the host arena, set before the threads; the counts, as they grow */
	struct lane lanes[MAXIMUM_LANES];
	pthread_t threads[MAXIMUM_READER_THREADS];
	bool extent_differs
		[]; /* per extent: it arrived with another digest than the pack's */
};

/* The monotonic clock: the time since an earlier reading of it, or with zero the reading itself. The longest time
 * there is when the clock cannot be read, so that every wait ends. */
static uint64_t nanoseconds_since(uint64_t earlier_nanoseconds)
{
    struct timespec now = {0};
    const bool is_read = !FAULT_POINT(loader_clock) &&
			 clock_gettime(CLOCK_MONOTONIC, &now) == 0;
    const uint64_t now_nanoseconds = (uint64_t)now.tv_sec * NANOSECONDS_PER_SECOND + (uint64_t)now.tv_nsec;
    return is_read ? now_nanoseconds - earlier_nanoseconds : UINT64_MAX;
}

static uint64_t smaller_of(uint64_t left, uint64_t right)
{
    return left < right ? left : right;
}

result_status load_layout_create(const struct pack_record *pack,
				 const uint32_t *card_of_unit,
				 uint32_t unit_count, uint32_t card_count,
				 struct load_layout **layout_out)
{
	if (pack == NULL || card_of_unit == NULL || layout_out == NULL ||
	    card_count > BACKEND_MAXIMUM_CARDS ||
	    !pack_placement_is_valid(pack, card_of_unit, unit_count,
				     card_count)) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct pack_extent *const extents = pack_extents_of(pack);
	const uint32_t extent_count = pack_header_of(pack)->extent_count;
	/* one allocation: the two tables lie behind the layout, the wider one first */
	struct load_layout *const layout =
		FAULT_POINT(loader_layout_allocation) ?
			NULL :
			calloc(1, sizeof(*layout) +
					  extent_count * (sizeof(uint64_t) +
							  sizeof(uint32_t)));
	if (layout == NULL) {
		return STATUS_OUT_OF_MEMORY;
	}
    layout->byte_offset_of_extent = (void *)(layout + 1);
    layout->card_of_extent = (void *)(layout->byte_offset_of_extent + extent_count);
    layout->extent_count = extent_count;
    layout->card_count = card_count;
    for (uint32_t extent_index = 0; extent_index < extent_count; extent_index++) {
        const uint32_t card_index = card_of_unit[extents[extent_index].unit_index];
        uint64_t *const byte_count =
            card_index < card_count ? &layout->card_byte_count[card_index] : &layout->host_byte_count;
        layout->card_of_extent[extent_index] = card_index;
        layout->byte_offset_of_extent[extent_index] = *byte_count;
        *byte_count += extents[extent_index].byte_count;
    }
    *layout_out = layout;
    return STATUS_OK;
}

void load_layout_destroy(struct load_layout *layout)
{
    free(layout);                                /* the one allocation; NULL is accepted, as by free */
}

/* The first extent at or after extent_index that goes to the lane; the extent count when none does. */
static uint32_t lane_extent_from(const struct loader *loader, const struct lane *lane, uint32_t extent_index)
{
    const uint32_t *const card_of_extent = loader->request->layout->card_of_extent;
    while (extent_index < loader->extent_count && card_of_extent[extent_index] != lane->destination) {
        extent_index++;
    }
    return extent_index;
}

/* The slot of the lane that is in that state: of several, the one with the lowest ticket. NULL when none is. */
static struct slot *oldest_slot(struct lane *lane, enum slot_state state)
{
    struct slot *oldest = NULL;
    for (uint32_t slot_index = 0; slot_index < LOADER_STAGING_SLOTS_PER_CARD;
	 slot_index++) {
	    struct slot *const slot = &lane->slots[slot_index];
	    /* a slot that is being read into is not looked at beyond its state: its thread writes the ticket */
	    if (slot->state == state &&
		(oldest == NULL ||
		 slot->upload_ticket < oldest->upload_ticket)) {
		    oldest = slot;
	    }
    }
    return oldest;
}

/* Under the mutex: asks every card about its oldest uploads, only asks, and frees the slots of those that have
 * finished: their pieces have arrived. */
static void reclaim_finished_uploads(struct loader *loader)
{
	for (uint32_t lane_index = 0;
	     lane_index < loader->lane_count && loader->status == STATUS_OK;
	     lane_index++) {
		struct lane *const lane = &loader->lanes[lane_index];
		result_status status = STATUS_OK;
		for (struct slot *slot = oldest_slot(lane, SLOT_UPLOADING);
		     slot != NULL && status == STATUS_OK;
		     slot = oldest_slot(lane, SLOT_UPLOADING)) {
			status = loader->request->backend->upload_wait(
				slot->card, slot->upload_ticket, 0);
			if (status == STATUS_OK) {
				slot->state = SLOT_FREE;
				loader->bytes_not_delivered -= slot->byte_count;
				lane->arrival_nanoseconds =
					loader->last_arrival_nanoseconds =
						nanoseconds_since(0);
			}
		}
		loader->status = status == STATUS_TIMED_OUT ? STATUS_OK :
							      status;
	}
}

/* Under the mutex: the next piece of the lane that would finish last, at the nominal rate of its link, of those that
 * can take one now: a card with a free slot, or host memory, which comes last and takes host_piece. NULL: none can. */
static struct slot *take_piece(struct loader *loader, struct slot *host_piece)
{
	const struct load_layout *const layout = loader->request->layout;
	struct lane *lane = NULL;
	struct slot *piece = NULL;
	double most_seconds_left = -1.0;
	for (uint32_t lane_index = 0; lane_index < loader->lane_count;
	     lane_index++) {
		struct lane *const candidate = &loader->lanes[lane_index];
		struct slot *const slot =
			lane_index + 1 < loader->lane_count ?
				oldest_slot(candidate, SLOT_FREE) :
				host_piece;
		const double seconds_left = (double)candidate->bytes_not_taken *
					    candidate->seconds_per_byte;
		if (candidate->bytes_not_taken > 0 && slot != NULL &&
		    seconds_left > most_seconds_left) {
			most_seconds_left = seconds_left;
			lane = candidate;
			piece = slot;
		}
    }
    if (lane == NULL || piece == NULL) {
        return NULL;
    }
    uint32_t extent_index = lane->next_extent_index;
    const uint64_t byte_limit =
	    smaller_of(lane->bytes_not_taken, LOADER_CHUNK_BYTES);
    ASSERT(extent_index <
	   loader->extent_count); /* a lane with bytes left has an extent left */
    piece->file_byte_offset = loader->extents[extent_index].byte_offset + lane->next_byte_offset_in_extent;
    piece->destination_byte_offset = layout->byte_offset_of_extent[extent_index] + lane->next_byte_offset_in_extent;
    piece->byte_count = 0;
    piece->state = SLOT_READING;
    /* the next extent of the pack, when it goes to the same lane, follows this one in the file and in the lane */
    while (piece->byte_count < byte_limit && extent_index < loader->extent_count &&
           layout->card_of_extent[extent_index] == lane->destination) {
        const uint64_t extent_bytes_left = loader->extents[extent_index].byte_count - lane->next_byte_offset_in_extent;
        const uint64_t byte_count = smaller_of(extent_bytes_left, byte_limit - piece->byte_count);
        piece->byte_count += byte_count;
        lane->next_byte_offset_in_extent += byte_count;
        if (byte_count == extent_bytes_left) {
            lane->next_byte_offset_in_extent = 0;
            extent_index++;
        }
    }
    lane->next_extent_index = lane_extent_from(loader, lane, extent_index);
    lane->bytes_not_taken -= piece->byte_count;
    return piece;
}

/* What a reader thread does, and the calling thread for an extent that is sent again: transfers until every piece
 * has arrived, then digests. It reads the copy its number names until that one is dropped, and ends at a failure. */
static void *transfer_and_check(void *argument)
{
    struct loader *const loader = argument;
    const struct load_request *const request = loader->request;
    unsigned char *const host_arena = loader->result.host_arena;         /* a mapping: it begins on a page boundary */
    struct slot host_piece = {0};
    struct slot *piece = NULL;                   /* kept for another turn only to be read from another copy */
    result_status status =
	    STATUS_OK; /* of what this thread did last outside the mutex */
    (void)pthread_mutex_lock(&loader->mutex);
    uint32_t copy_index = loader->next_thread_number / request->reader_threads_per_copy;
    loader->next_thread_number++;
    for (;;) {
        struct lane *checked_lane = NULL;
        uint64_t digest = 0;
	loader->status = loader->status == STATUS_OK ? status : loader->status;
	status = STATUS_OK;
	reclaim_finished_uploads(loader);
	if (loader->status != STATUS_OK) {
		break;
	}
	piece = piece != NULL ? piece : take_piece(loader, &host_piece);
        if (piece != NULL) {
            for (uint32_t live_index = 0; live_index < loader->copy_count && loader->copy_is_dropped[copy_index];
                 live_index++) {                 /* the live copy with the lowest index takes over */
                copy_index = loader->copy_is_dropped[live_index] ? copy_index : live_index;
            }
            void *const memory = piece->card != NULL ? piece->memory : host_arena + piece->destination_byte_offset;
            (void)pthread_mutex_unlock(&loader->mutex);
            status = request->storage->reader_read(request->reader, copy_index, piece->file_byte_offset, memory,
                                                   piece->byte_count);
	    const bool copy_failed = status == STATUS_INPUT_OUTPUT_ERROR;
	    if (status == STATUS_OK && piece->card != NULL) {
		    status = request->backend->upload_begin(
			    piece->card, piece->destination_byte_offset, memory,
			    piece->byte_count, &piece->upload_ticket);
	    }
	    (void)pthread_mutex_lock(&loader->mutex);
            if (copy_failed && !loader->copy_is_dropped[copy_index]) {
                loader->copy_is_dropped[copy_index] = true;
                loader->result.copies_dropped++;
		LOG_WARNING_ENTRY("copy %" PRIu32
				  " failed a read: the load goes on without it",
				  copy_index);
	    }
            /* a piece of a dropped copy is read again while a copy is left; a card's piece arrives at its reclaim */
            const bool is_read_again = copy_failed && loader->result.copies_dropped < loader->copy_count;
	    const bool has_arrived = status == STATUS_OK && piece->card == NULL;
	    piece->state = is_read_again ? SLOT_READING :
			   status == STATUS_OK && !has_arrived ?
					   SLOT_UPLOADING :
			   status == STATUS_DEVICE_ERROR ? SLOT_ABANDONED :
							   SLOT_FREE;
	    loader->result.byte_count_read_of_copy[copy_index] +=
		    status == STATUS_OK ? piece->byte_count : 0;
	    if (has_arrived) {
                loader->bytes_not_delivered -= piece->byte_count;
                loader->lanes[loader->lane_count - 1].arrival_nanoseconds = loader->last_arrival_nanoseconds =
                    nanoseconds_since(0);
            }
	    status = is_read_again ? STATUS_OK : status;
	    piece = is_read_again ? piece : NULL;
        } else if (loader->bytes_not_delivered == 0) {
            for (uint32_t lane_index = 0; lane_index < loader->lane_count && checked_lane == NULL; lane_index++) {
                struct lane *const lane = &loader->lanes[lane_index];
                const bool has_work = lane->check_extent_index < lane->next_extent_index;
                checked_lane = has_work && !lane->is_being_checked ? lane : NULL;
            }
            if (checked_lane == NULL) {          /* nothing is left that this thread may digest */
                break;
            }
            const uint32_t extent_index = checked_lane->check_extent_index;
            const uint64_t byte_offset = request->layout->byte_offset_of_extent[extent_index];
            const uint64_t byte_count = loader->extents[extent_index].byte_count;
	    const bool is_on_a_card = checked_lane->destination !=
				      PLACEMENT_IN_HOST_MEMORY;
	    checked_lane->check_extent_index = lane_extent_from(loader, checked_lane, extent_index + 1);
            checked_lane->is_being_checked = is_on_a_card;
            (void)pthread_mutex_unlock(&loader->mutex);
	    status = is_on_a_card ?
			     request->backend->digest(
				     request->cards[checked_lane->destination],
				     byte_offset, byte_count,
				     request->stall_limit_nanoseconds,
				     &digest) :
			     pack_digest_of_host_bytes(host_arena + byte_offset,
						       byte_count, &digest);
	    (void)pthread_mutex_lock(&loader->mutex);
            checked_lane->is_being_checked = false;
	    loader->extent_differs[extent_index] =
		    status == STATUS_OK &&
		    digest != loader->extents[extent_index].digest;
	} else if (nanoseconds_since(loader->last_arrival_nanoseconds) > request->stall_limit_nanoseconds) {
		status = STATUS_TIMED_OUT;
	} else {                                 /* nothing to take now: the other threads hold it, or the links */
            (void)pthread_mutex_unlock(&loader->mutex);
            (void)nanosleep(&(struct timespec){.tv_nsec = WAIT_SLICE_NANOSECONDS}, NULL);
            (void)pthread_mutex_lock(&loader->mutex);
        }
    }
    (void)pthread_mutex_unlock(&loader->mutex);
    return NULL;
}

/* On the calling thread, the reader threads joined: every extent that differed takes the way of its lane again,
 * alone and from each live copy in turn, until its digest is the pack's. */
static void send_differing_extents_again(struct loader *loader)
{
	const struct load_layout *const layout = loader->request->layout;
	for (uint32_t extent_index = 0;
	     extent_index < loader->extent_count && loader->status == STATUS_OK;
	     extent_index++) {
		struct lane *const lane = &loader->lanes[smaller_of(
			layout->card_of_extent[extent_index],
			layout->card_count)];
		loader->result.extents_read_again +=
			loader->extent_differs[extent_index] ? 1u : 0u;
		for (uint32_t copy_index = 0;
		     copy_index < loader->copy_count &&
		     loader->extent_differs[extent_index] &&
		     loader->status == STATUS_OK;
		     copy_index++) {
			if (!loader->copy_is_dropped[copy_index]) {
				lane->next_extent_index = extent_index;
				lane->check_extent_index = extent_index;
				lane->bytes_not_taken =
					loader->extents[extent_index].byte_count;
				loader->bytes_not_delivered =
					lane->bytes_not_taken;
				loader->last_arrival_nanoseconds =
					nanoseconds_since(0);
				loader->next_thread_number =
					copy_index *
					loader->request->reader_threads_per_copy;
				(void)transfer_and_check(loader);
			}
		}
		if (loader->status == STATUS_OK &&
		    loader->extent_differs[extent_index]) {
			LOG_ERROR("extent %" PRIu32 " of unit %" PRIu32
				  " has the digest of the pack on no copy",
				  extent_index,
				  loader->extents[extent_index].unit_index);
			loader->status = STATUS_CORRUPT_PACK;
		}
	}
}

/* Returns the staging slots, the reader threads joined. A slot whose upload is not seen finished or failed within
 * the stall limit stays allocated: only closing its card makes it safe to free. */
static void release_slots(const struct loader *loader)
{
	const struct load_request *const request = loader->request;
	const uint64_t limit_nanoseconds = request->stall_limit_nanoseconds;
	const uint64_t started_nanoseconds = nanoseconds_since(0);
	uint32_t slots_left_allocated = 0;
	for (uint32_t slot_index = 0;
	     slot_index < MAXIMUM_LANES * LOADER_STAGING_SLOTS_PER_CARD;
	     slot_index++) {
		const struct slot *const slot =
			&loader->lanes[slot_index /
				       LOADER_STAGING_SLOTS_PER_CARD]
				 .slots[slot_index %
					LOADER_STAGING_SLOTS_PER_CARD];
		result_status status =
			slot->state == SLOT_UPLOADING ? STATUS_TIMED_OUT :
			slot->state == SLOT_ABANDONED ? STATUS_DEVICE_ERROR :
							STATUS_OK;
		for (uint64_t left = limit_nanoseconds;
		     status == STATUS_TIMED_OUT && left > 0;) {
			left = limit_nanoseconds -
			       smaller_of(
				       limit_nanoseconds,
				       nanoseconds_since(started_nanoseconds));
			status = request->backend->upload_wait(
				slot->card, slot->upload_ticket,
				left); /* may answer early */
		}
		const bool is_over =
			status ==
			STATUS_OK; /* a failed card may still read its slots: they stay */
		request->backend->staging_free(is_over ? slot->memory : NULL);
		slots_left_allocated += is_over ? 0u : 1u;
	}
    if (slots_left_allocated > 0) {
	    LOG_ERROR(
		    "%" PRIu32
		    " staging slots stay allocated: their uploads were not seen to finish",
		    slots_left_allocated);
    }
}

result_status load_run_entry(const struct load_request *request,
			     struct load_result *result_out)
{
	if (request == NULL || result_out == NULL || request->storage == NULL ||
	    request->reader == NULL || request->backend == NULL ||
	    request->pack == NULL || request->layout == NULL ||
	    request->layout->byte_offset_of_extent == NULL ||
	    request->layout->card_of_extent == NULL ||
	    request->layout->extent_count !=
		    pack_header_of(request->pack)->extent_count ||
	    request->layout->card_count > BACKEND_MAXIMUM_CARDS ||
	    request->reader_threads_per_copy == 0 ||
	    request->reader_threads_per_copy >
		    LOADER_MAXIMUM_READER_THREADS_PER_COPY ||
	    request->reserved_zero != 0 ||
	    request->stall_limit_nanoseconds == 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct load_layout *const layout = request->layout;
	struct storage_description storage = { 0 };
	struct loader *const loader =
		FAULT_POINT(loader_state_allocation) ?
			NULL :
			calloc(1,
			       sizeof(*loader) +
				       layout->extent_count *
					       sizeof(loader->extent_differs[0]));
	if (loader == NULL || pthread_mutex_init(&loader->mutex, NULL) != 0) {
		free(loader);
		return STATUS_OUT_OF_MEMORY;
	}
    loader->request = request;
    loader->extents = pack_extents_of(request->pack);
    loader->extent_count = layout->extent_count;
    loader->lane_count = layout->card_count + 1;
    /* What of the request takes a walk to check, before anything is obtained. The reader's alignment divides what a
     * direct read is given: whole pack blocks, its address from a page boundary on (a slot's start, the arena's). The
     * layout, followed extent by extent, so that one of another pack or card count, or changed since, is refused: an
     * extent lies where the ones before it of its destination end, and a byte count is what the extents occupy, no more
     * and no less. Every lane is then set at its first extent, and a card that receives bytes gets its staging slots
     * and the rate of its link: one that is not known counts as one lane of generation 1, the slowest, served first. */
    result_status status =
	    request->storage->reader_describe(request->reader, &storage);
    const bool reader_fits =
	    storage.copy_count >= 1 &&
	    storage.copy_count <= STORAGE_MAXIMUM_COPIES &&
	    storage.block_alignment_bytes != 0 &&
	    PACK_BLOCK_BYTES % storage.block_alignment_bytes == 0 &&
	    BACKEND_STAGING_ALIGNMENT_BYTES % storage.block_alignment_bytes ==
		    0;
    status = status == STATUS_OK && !reader_fits ? STATUS_INVALID_ARGUMENT :
						   status;
    loader->copy_count = storage.copy_count;
    for (uint32_t extent_index = 0;
	 extent_index < loader->extent_count && status == STATUS_OK;
	 extent_index++) {
	    const uint32_t card_index = layout->card_of_extent[extent_index];
	    const bool is_card = card_index < layout->card_count;
	    struct lane *const lane =
		    &loader->lanes[is_card ? card_index : layout->card_count];
	    const bool has_destination =
		    is_card ? request->cards != NULL &&
				      request->cards[card_index] != NULL &&
				      request->card_descriptions != NULL :
			      card_index == PLACEMENT_IN_HOST_MEMORY;
	    status =
		    has_destination &&
				    layout->byte_offset_of_extent[extent_index] ==
					    lane->bytes_not_taken ?
			    STATUS_OK :
			    STATUS_INVALID_ARGUMENT;
	    lane->bytes_not_taken += loader->extents[extent_index].byte_count;
    }
    for (uint32_t lane_index = 0;
	 lane_index < loader->lane_count && status == STATUS_OK; lane_index++) {
	    struct lane *const lane = &loader->lanes[lane_index];
	    const bool is_card = lane_index < layout->card_count;
	    lane->destination = is_card ? lane_index : PLACEMENT_IN_HOST_MEMORY;
	    lane->next_extent_index = lane_extent_from(loader, lane, 0);
	    lane->check_extent_index = lane->next_extent_index;
	    loader->bytes_not_delivered += lane->bytes_not_taken;
	    if (lane->bytes_not_taken !=
		(is_card ? layout->card_byte_count[lane_index] :
			   layout->host_byte_count)) {
		    status = STATUS_INVALID_ARGUMENT;
	    } else if (is_card && lane->bytes_not_taken > 0) {
		    const struct card_description_record *const card =
			    &request->card_descriptions[lane_index];
		    const bool link_is_known = card->link_generation >= 1 &&
					       card->link_lane_count >= 1 &&
					       card->link_generation <=
						       LINK_GENERATION_COUNT;
		    lane->seconds_per_byte =
			    1.0 /
			    (link_is_known ?
				     link_lane_bytes_per_second
						     [card->link_generation - 1] *
					     (double)card->link_lane_count :
				     link_lane_bytes_per_second[0]);
	    }
    }
    for (uint32_t slot_index = 0;
	 slot_index < layout->card_count * LOADER_STAGING_SLOTS_PER_CARD &&
	 status == STATUS_OK;
	 slot_index++) { /* the request is whole */
	    struct lane *const lane =
		    &loader->lanes[slot_index / LOADER_STAGING_SLOTS_PER_CARD];
	    struct slot *const slot =
		    &lane->slots[slot_index % LOADER_STAGING_SLOTS_PER_CARD];
	    if (lane->bytes_not_taken > 0) {
		    slot->card = request->cards[lane->destination];
		    status = request->backend->staging_allocate(
			    LOADER_CHUNK_BYTES, &slot->memory);
	    }
    }
    const bool needs_arena = status == STATUS_OK && layout->host_byte_count > 0;
    const size_t mapped_bytes = HOST_ARENA_MAPPED_BYTES(layout->host_byte_count);   /* 0 for a count that wraps */
    void *const host_arena = !needs_arena || mapped_bytes == 0 ||
					     FAULT_POINT(loader_host_arena) ?
				     MAP_FAILED :
				     mmap(NULL, mapped_bytes,
					  PROT_READ | PROT_WRITE,
					  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    loader->result.host_arena = host_arena != MAP_FAILED ? host_arena : NULL;     /* before any thread; not touched */
    status = needs_arena && host_arena == MAP_FAILED ? STATUS_OUT_OF_MEMORY :
						       status;
    if (host_arena != MAP_FAILED && madvise(host_arena, mapped_bytes, MADV_HUGEPAGE) != 0) {
	    LOG_WARNING_ENTRY(
		    "no 2 MiB pages for the host arena: the load goes on with small pages");
    }
    const uint32_t thread_limit = loader->copy_count * request->reader_threads_per_copy;
    const uint64_t started_nanoseconds = loader->last_arrival_nanoseconds = nanoseconds_since(0);
    uint64_t transfer_nanoseconds = 0;
    uint32_t thread_count = 0;
    while (status == STATUS_OK && thread_count < thread_limit &&
	   !FAULT_POINT(loader_thread_create) &&
	   pthread_create(&loader->threads[thread_count], NULL,
			  transfer_and_check, loader) == 0) {
	    thread_count++;
    }
    status = status == STATUS_OK && thread_count < thread_limit ?
		     STATUS_OUT_OF_MEMORY :
		     status;
    (void)pthread_mutex_lock(&loader->mutex);                    /* a failure ends the threads at their next step */
    loader->status = loader->status == STATUS_OK ? status : loader->status;
    (void)pthread_mutex_unlock(&loader->mutex);
    for (uint32_t thread_index = 0; thread_index < thread_count; thread_index++) {
        (void)pthread_join(loader->threads[thread_index], NULL);
    }
    for (uint32_t lane_index = 0; lane_index < loader->lane_count; lane_index++) {      /* before any is sent again */
        const uint64_t arrival_nanoseconds = loader->lanes[lane_index].arrival_nanoseconds;
        uint64_t *const arrival_out = lane_index + 1 < loader->lane_count
            ? &loader->result.card_arrival_nanoseconds[lane_index] : &loader->result.host_arrival_nanoseconds;
        *arrival_out = arrival_nanoseconds != 0 ? arrival_nanoseconds - started_nanoseconds : 0;
        transfer_nanoseconds = *arrival_out > transfer_nanoseconds ? *arrival_out : transfer_nanoseconds;
    }
    send_differing_extents_again(loader);
    status = loader->status;
    loader->result.transfer_nanoseconds = transfer_nanoseconds;
    loader->result.check_nanoseconds = nanoseconds_since(started_nanoseconds) - transfer_nanoseconds;
    release_slots(loader);
    if (status == STATUS_OK) {
	    *result_out = loader->result;
    } else {
	    load_host_arena_release(loader->result.host_arena,
				    layout->host_byte_count);
    }
    (void)pthread_mutex_destroy(&loader->mutex);
    free(loader);
    return status;
}

void load_host_arena_release(void *host_arena, uint64_t host_byte_count)
{
    if (host_arena != NULL && munmap(host_arena, HOST_ARENA_MAPPED_BYTES(host_byte_count)) != 0) {
	    LOG_ERROR("the host arena could not be unmapped: it stays mapped");
    }
}
