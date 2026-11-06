/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * host_memory_backend.h: a backend whose cards are arenas in host memory, so that the loader and the programs of
 * launches run under the sanitizers without a card, with the faults of a card and a link switched on by a test.
 * Test support: no part of the engine.
 *
 * Contract:
 *   - test_host_memory_backend obeys backend_seam.h: every refusal, every status, tickets, the limit of uploads
 *     in flight and of staging blocks, calls from any number of threads. Its digest is
 *     pack_digest_of_host_bytes over the arena.
 *   - test_host_memory_backend_reset closes nothing: it is called when no card is open and no staging block is
 *     alive (a test checks both counters first), sets the number of cards and the memory each reports as total
 *     and free, and clears every fault, counter, class, stand-in and the work log. Cards report
 *     memory_reserved_bytes 0, link 0 by 0 and arithmetic class 0 until a test sets a link or a class.
 *   - An upload copies its bytes into the arena when it begins, but is only reported finished once upload_wait
 *     was asked about it (or a later ticket of the card) polls_before_an_upload_finishes times; a wait with a
 *     limit above zero sleeps for the smaller of its limit and TEST_HOST_MEMORY_BACKEND_WAIT_SLICE_NANOSECONDS
 *     before it answers STATUS_TIMED_OUT. The default is 0 polls: finished at once. So a test decides
 *     without a clock whether the loader sees uploads in flight, a full ring, and waits that time out.
 *   - A download copies its bytes out of the arena when it begins, under every rule of an upload: the same refusals,
 *     the same sequence of tickets, the same ring and its limit, the same polls, and fail_upload names a ticket of
 *     either direction. It is no upload for flip_bit_in_next_upload and for the count of uploads.
 *   - Kernels on a card of class 0, the default: it has no build of any kernel. kernel_prepare and launch_run
 *     answer STATUS_UNSUPPORTED and look at no file, function or launch: only what backend_seam.h refuses
 *     whatever the kernel comes first, as STATUS_INVALID_ARGUMENT (a card that is not open, a missing
 *     argument, a function name or a record of no bytes, a record of more than LAUNCH_ARGUMENT_BYTES, a
 *     count of no launches), and then STATUS_DEVICE_ERROR on a card that has failed. launch_wait answers
 *     STATUS_OK on an open card whatever its limit (DEVICE_ERROR once it has failed); format_supported is
 *     false.
 *   - Kernels on a card a test gave ARITHMETIC_CLASS_SM_80 or _SM_120: the card keeps a registry, as the
 *     CUDA backend keeps its prepared kernels, and computes nothing itself.
 *       kernel_prepare refuses what that backend refuses and a card in host memory can see, each with one error
 *       line: a function name of no bytes or of TEST_HOST_MEMORY_BACKEND_FUNCTION_NAME_CAPACITY bytes or more,
 *       and a kernel file of which the program's table of kernel modules (backend_cuda_kernel_images.h) holds no
 *       build at all (INVALID_ARGUMENT); a file with no build for the class's architecture (UNSUPPORTED, which the
 *       build never makes); a function prepared before with another record size (INVALID_ARGUMENT); one kernel
 *       more than BACKEND_MAXIMUM_KERNELS (UNSUPPORTED). No module is read, so a function's name is taken
 *       as given. A handle is the number of kernels the card had prepared, so cards that prepare the same kernels
 *       in the same order give the same handles, and the same kernel again gives its handle. A closed card forgets
 *       its kernels.
 *       launch_run checks every launch before it queues the first, as backend_seam.h says, against the limits
 *       TEST_HOST_MEMORY_BACKEND_MOST_* (of the card's class for shared memory); a refusal is one error line and
 *       queues nothing. Then each launch, in order, runs the stand-in a test set for its kernel, if any.
 *       launch_wait answers STATUS_OK on a sound card: each launch ran inside launch_run.
 *       format_supported is true for every format this build carries (weight_format_operations_of).
 *   - Marks (backend_seam.h): a copy is done when it begins, so they order nothing; mark_record and mark_follow
 *     refuse what that header refuses and count, and a follow of an owner's mark not set since that card's open is
 *     counted apart (test_host_memory_backend_marks), so that a test sees that every mark followed was set first.
 *   - The work log holds, in the order it began on any card since reset, every upload, download and launch that
 *     was queued and every digest that was delivered, up to TEST_HOST_MEMORY_BACKEND_LOG_CAPACITY entries: work
 *     beyond is counted but not logged. A call that is refused or answers an error logs nothing.
 *   - A stand-in is what a kernel does in host memory: launch_run calls it for each launch of that kernel file and
 *     function on any card, behind every copy and launch of the card begun before it, with the launch, the card's
 *     index and its arena. It runs under the backend's mutex, so it calls no function of this backend, and returns
 *     false for a kernel that faulted. set_stand_in with NULL takes a stand-in away; setting one again replaces it.
 *   - Faults stay until reset:
 *       fail_upload                the upload of that card with that ticket fails: it and every later ticket
 *                                  report STATUS_DEVICE_ERROR, as backend_seam.h says
 *       flip_bit_in_next_upload    the next upload of that card that covers arena_byte_offset arrives with the
 *                                  lowest bit of that byte flipped, once: a link that corrupted one transfer
 *       fail_staging_allocation    the allocation_number-th staging_allocate from now on, counted from 1, fails
 *                                  with STATUS_OUT_OF_MEMORY, once
 *       fail_digest                the digest_number-th digest of that card from now on, counted from 1, fails
 *                                  with STATUS_DEVICE_ERROR and the card begins no new work
 *       fail_launch                the launch_number-th launch of that card from now on, counted from 1, faults,
 *                                  once, as when its stand-in returns false
 *     A launch that faults leaves launch_run's answer STATUS_OK, as a kernel's fault is seen only later on
 *     a card: the card has failed. No stand-in runs from that launch on, and launch_wait, every call that would
 *     begin work and every ticket not yet reported finished answer STATUS_DEVICE_ERROR. That goes with the
 *     card when it is closed: opened again it is sound, and the ticket fail_upload named is still the one to fail.
 *   - The setters are called with no call of the table in flight; the counters and the log may be read at any time.
 */
#ifndef TEST_HOST_MEMORY_BACKEND_H
#define TEST_HOST_MEMORY_BACKEND_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/status.h"

#define TEST_HOST_MEMORY_BACKEND_WAIT_SLICE_NANOSECONDS 50000
#define TEST_HOST_MEMORY_BACKEND_FUNCTION_NAME_CAPACITY 64    /* bytes, its NUL included: the CUDA backend's bound */
#define TEST_HOST_MEMORY_BACKEND_LOG_CAPACITY 16384           /* entries */
#define TEST_HOST_MEMORY_BACKEND_MAXIMUM_STAND_INS 32
/* What a launch may ask of a card of either class: the limits the driver states for the cards of both classes, and
 * per class the most dynamic shared memory a block may have once its function's limit is raised, on compute
 * capability 8.0 and 12.0. A card gives a function less (its static shared memory, and 48 KiB while nobody raised
 * its limit), which the host cannot see: no module is read. */
#define TEST_HOST_MEMORY_BACKEND_MOST_GRID_X 2147483647u
#define TEST_HOST_MEMORY_BACKEND_MOST_GRID_Y_Z 65535u
#define TEST_HOST_MEMORY_BACKEND_MOST_BLOCK_X_Y 1024u
#define TEST_HOST_MEMORY_BACKEND_MOST_BLOCK_Z 64u
#define TEST_HOST_MEMORY_BACKEND_MOST_THREADS_PER_BLOCK 1024u
#define TEST_HOST_MEMORY_BACKEND_MOST_DYNAMIC_SHARED_BYTES_SM_80 166912u
#define TEST_HOST_MEMORY_BACKEND_MOST_DYNAMIC_SHARED_BYTES_SM_120 101376u

enum test_host_memory_backend_work_kind {
    TEST_HOST_MEMORY_BACKEND_UPLOAD,
    TEST_HOST_MEMORY_BACKEND_DOWNLOAD,
    TEST_HOST_MEMORY_BACKEND_DIGEST,
    TEST_HOST_MEMORY_BACKEND_LAUNCH
};

/* One entry of the work log. */
struct test_host_memory_backend_work {
	struct kernel_launch
		launch; /* a launch as launch_run was given it; all zero for other work */
	uint64_t arena_byte_offset; /* the range of a copy or a digest; zero for a launch */
	uint64_t byte_count;
	uint32_t kind; /* enum test_host_memory_backend_work_kind */
	uint32_t card_index;
};

/* A prepared kernel, as a card of a class keeps it. */
struct test_host_memory_backend_kernel {
    const char *kernel_file_name;     /* the kernel table's own name: valid for the life of the program */
    char function_name[TEST_HOST_MEMORY_BACKEND_FUNCTION_NAME_CAPACITY];
    uint32_t argument_byte_count;
};

typedef bool (*test_host_memory_backend_stand_in)(
	const struct kernel_launch *launch, uint32_t card_index,
	unsigned char *arena, uint64_t arena_byte_count, void *context);

extern const struct backend_operations test_host_memory_backend;

void test_host_memory_backend_reset(uint32_t card_count, uint64_t memory_bytes_per_card);
void test_host_memory_backend_set_link(uint32_t card_index, uint32_t link_generation, uint32_t link_lane_count);
void test_host_memory_backend_set_arithmetic_class(uint32_t card_index, uint32_t arithmetic_class);
void test_host_memory_backend_set_polls_before_an_upload_finishes(uint32_t card_index, uint32_t poll_count);
void test_host_memory_backend_fail_upload(uint32_t card_index, uint64_t upload_ticket);
void test_host_memory_backend_flip_bit_in_next_upload(uint32_t card_index, uint64_t arena_byte_offset);
void test_host_memory_backend_fail_staging_allocation(uint64_t allocation_number);
void test_host_memory_backend_fail_digest(uint32_t card_index, uint64_t digest_number);
void test_host_memory_backend_fail_launch(uint32_t card_index, uint64_t launch_number);
/* INVALID_ARGUMENT for a missing or empty name, or one of TEST_HOST_MEMORY_BACKEND_FUNCTION_NAME_CAPACITY bytes or
 * more; BUSY for one stand-in more than TEST_HOST_MEMORY_BACKEND_MAXIMUM_STAND_INS. */
MUST_CHECK result_status test_host_memory_backend_set_stand_in(
	const char *kernel_file_name, const char *function_name,
	test_host_memory_backend_stand_in stand_in, void *context);

/* The arena of an open card, to compare with what a pack holds. */
const unsigned char *
test_host_memory_backend_arena_of(const struct device_card *card);

uint32_t test_host_memory_backend_open_card_count(void);
uint32_t test_host_memory_backend_staging_block_count(void);         /* alive now */
uint64_t test_host_memory_backend_upload_count(uint32_t card_index); /* begun since reset; a download is not counted */
/* since reset, of that card: marks it set, marks it followed, follows of an owner's mark not set since its open */
void test_host_memory_backend_marks(uint32_t card_index, uint64_t *records_out, uint64_t *follows_out,
                                    uint64_t *unset_follows_out);
uint32_t test_host_memory_backend_most_uploads_in_flight(uint32_t card_index);       /* copies of either direction */
/* A kernel of an open card by its handle; false, and kernel_out untouched, for none. */
bool test_host_memory_backend_kernel_of(uint32_t card_index, uint32_t kernel,
                                        struct test_host_memory_backend_kernel *kernel_out);
uint64_t test_host_memory_backend_work_count(void);                  /* begun since reset, logged or not */
/* The entry of the log at that index; false, and work_out untouched, beyond the entries logged. */
bool test_host_memory_backend_work_at(uint64_t work_index, struct test_host_memory_backend_work *work_out);

#endif
