/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * loader.h: brings the extents of a pack to where a placement puts them, and proves that they arrived.
 *
 * Contract:
 *   - A layout says where each extent goes: the extents of one destination (a card's arena, or the host arena)
 *     follow one another in the order of the pack, from offset 0, with nothing between them. So the weights are
 *     the first card_byte_count bytes of a card's arena; what a model keeps beside them lies behind.
 *   - load_run_entry reads every extent from the copies of the pack and delivers it: into the arena of its
 *     card through page-locked staging memory, or straight into a host arena that the loader maps itself on
 *     2 MiB pages. The host arena is plain memory: nothing registers it with the cards during the load, because
 *     registering a large block holds every upload for as long as it takes (measured: 8 s of two slow links lost
 *     for 94 GiB). The mapping is layout->host_byte_count rounded up to whole 2 MiB pages, the bytes behind the
 *     last extent zero; where the kernel refuses such pages the load goes on with small ones (one warning). Its
 *     pages come at first touch inside the reads: STATUS_OUT_OF_MEMORY answers for the mapping alone. A
 *     card that is to read an extent of it later gets it through a staging slot.
 *   - Work is cut into pieces of at most LOADER_CHUNK_BYTES. reader_threads_per_copy threads read each
 *     copy. A thread takes its next piece for the destination that would finish last (bytes left over the nominal
 *     rate of its link; host memory counts as the fastest), skipping a card whose
 *     LOADER_STAGING_SLOTS_PER_CARD staging slots are all in use: a slow link is kept busy from the first
 *     byte and never holds a reader. No thread waits for an upload while another destination has work.
 *   - Nothing is trusted until checked: when all bytes are delivered, the digest of every extent is computed
 *     where it lies (on its card by the backend, in host memory by pack_digest_of_host_bytes) and
 *     compared with the pack's. An extent that differs is read again, from each copy in turn, until one gives
 *     the digest; if none does the load fails with STATUS_CORRUPT_PACK.
 *   - A copy whose read fails with STATUS_INPUT_OUTPUT_ERROR is dropped for the rest of the load (one
 *     warning) and its threads read the live copy of the lowest index; when no copy is left the load fails with
 *     that status. Any other status of a read fails the load with it.
 *   - Every wait has a bound: the load fails with STATUS_TIMED_OUT when stall_limit_nanoseconds pass and
 *     no piece arrives anywhere. The limit must be above the storage's own bound for one read
 *     (STORAGE_READ_LIMIT_SECONDS), or a slow drive is taken for a stall.
 *   - On success the result holds the host arena (NULL when the layout puts nothing in host memory); the caller
 *     owns it and returns it with load_host_arena_release. On failure the result is untouched, the bytes
 *     of the arenas are unspecified and the loader has released everything it obtained, the host arena included,
 *     except staging memory of an upload that it could not see finish or that its card reported failed: that is
 *     left allocated (one error line), because only closing its card makes it safe to free.
 *   - The request is checked first (STATUS_INVALID_ARGUMENT): a NULL member, a layout of another pack or
 *     card count or whose byte counts are not what its extents occupy, a missing card or description where the
 *     layout places bytes, a reader whose copies or block alignment do not fit the pack and the staging memory, a
 *     thread count of 0 or above LOADER_MAXIMUM_READER_THREADS_PER_COPY, a reserved field that is not
 *     zero, a stall limit of zero. A reader that cannot be described fails the load with the reader's status.
 */
#ifndef LOADER_H
#define LOADER_H

#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/pack.h"
#include "anemone/status.h"
#include "anemone/storage_seam.h"

#define LOADER_CHUNK_BYTES 16777216ull /* 16 MiB: one read and one upload */
#define LOADER_STAGING_SLOTS_PER_CARD \
	8 /* half of a card's uploads in flight (backend_seam.h) */
#define LOADER_DEFAULT_READER_THREADS_PER_COPY \
	2 /* measured: two plain readers reach a drive's rate */
#define LOADER_MAXIMUM_READER_THREADS_PER_COPY 16
#define LOADER_DEFAULT_STALL_LIMIT_NANOSECONDS \
	240000000000ull /* a third above the storage's read bound */

struct load_layout {
	uint64_t *byte_offset_of_extent; /* per extent: where it begins in its destination */
	uint32_t *card_of_extent; /* per extent: a card, or PLACEMENT_IN_HOST_MEMORY */
	uint64_t card_byte_count
		[BACKEND_MAXIMUM_CARDS]; /* what the extents of each card occupy */
	uint64_t host_byte_count;
	uint32_t extent_count;
	uint32_t card_count;
};

/* First user: the program's load, which sizes the arenas from it before it opens a card. Fails with
 * STATUS_INVALID_ARGUMENT for a NULL argument, a card count above BACKEND_MAXIMUM_CARDS and a
 * placement that pack_placement_is_valid refuses, with STATUS_OUT_OF_MEMORY for a lack of memory. */
MUST_CHECK result_status load_layout_create(const struct pack_record *pack,
					    const uint32_t *card_of_unit,
					    uint32_t unit_count,
					    uint32_t card_count,
					    struct load_layout **layout_out);
void load_layout_destroy(struct load_layout *layout); /* accepts NULL */

struct load_request {
	const struct storage_operations_record *storage;
	const struct storage_reader *reader;
	const struct backend_operations *backend;
	const struct pack_record *pack;
	const struct load_layout *layout;
	/* layout->card_count entries each. A card that receives bytes is open with an arena of at least its
     * card_byte_count and has its description (the loader reads the link); the others may be NULL. */
	struct device_card *const *cards;
	const struct card_description_record *card_descriptions;
	uint32_t reader_threads_per_copy;
	uint32_t reserved_zero;
	uint64_t stall_limit_nanoseconds;
};

struct load_result {
	void *host_arena; /* layout->host_byte_count bytes mapped by the loader, or NULL */
	uint64_t byte_count_read_of_copy[STORAGE_MAXIMUM_COPIES];
	uint64_t transfer_nanoseconds; /* from the first read to the last piece delivered */
	/* When the last piece of each destination arrived, counted like transfer_nanoseconds; 0 where nothing went. */
	uint64_t card_arrival_nanoseconds[BACKEND_MAXIMUM_CARDS];
	uint64_t host_arrival_nanoseconds;
	uint64_t check_nanoseconds; /* the digests, and whatever had to be read again */
	uint32_t extents_read_again;
	uint32_t copies_dropped;
};

MUST_CHECK result_status load_run_entry(const struct load_request *request,
					struct load_result *result_out);

/* Returns the host arena of a result to the system. host_byte_count must be the layout's (it cannot be checked);
 * a failure to unmap is one error line and the arena stays. Accepts NULL. */
void load_host_arena_release(void *host_arena, uint64_t host_byte_count);

#endif
