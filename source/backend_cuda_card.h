/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * backend_cuda_card.h: what the sources of the CUDA backend share. Private to source/backend_cuda.c and
 * source/backend_cuda_launch.c: with this header they are the only files of the engine that include a vendor header.
 *
 * Contract:
 *   - backend_cuda.c holds the cards: their table, the driver's initialisation, opening and closing, staging memory,
 *     the copies and the digest. backend_cuda_launch.c holds what a card runs: its modules, its prepared kernels and
 *     their launches. Each defines the functions declared under its name below.
 *   - A call for a card stands between cuda_card_enter and cuda_card_leave: the first takes the
 *     card's mutex and makes its context current, and struct open_card_state is read and written only between the
 *     two. What struct device_card holds beside it is written once, while the driver is initialised.
 *   - A card has failed once first_failed_ticket is not 0. Whoever gets a driver error on an open card sets it to
 *     last_ticket_finished + 1 and answers STATUS_DEVICE_ERROR; whoever is asked to begin work on a failed
 *     card answers that status and asks the driver for nothing.
 *   - Every driver call is a DRIVER_CALL or stands behind a fault point of the same name, backend_cuda_<function>.
 *   - The first modules of a card are those it holds; the first kernels are those it has prepared, and a kernel
 *     handle of backend_seam.h is an index among them. A card holds at most one module more than it has prepared
 *     kernels: the digest's, loaded when it is opened.
 */
#ifndef BACKEND_CUDA_CARD_H
#define BACKEND_CUDA_CARD_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#include <cuda.h>

#include "anemone/backend_seam.h"
#include "anemone/hooks.h"
#include "backend_cuda_kernel_images.h"

#define UPLOAD_RING \
	BACKEND_MAXIMUM_UPLOADS_IN_FLIGHT /* the event of ticket t is at t modulo the ring */
#define DIGEST_EVENT UPLOAD_RING                /* behind the ring: the event of the digest kernel */
#define LAUNCH_EVENT (UPLOAD_RING + 1)          /* and the event launch_wait records behind everything queued */
#define MARK_EVENT_FIRST (UPLOAD_RING + 2)      /* behind it: the marks of backend_seam.h, mark m at this + m */
#define CARD_EVENT_COUNT (MARK_EVENT_FIRST + BACKEND_MAXIMUM_MARKS)
#if TIMING_HOOKS_ENABLED
#define MEASURE_BEGIN_EVENT 0u
#define MEASURE_END_EVENT 1u
#define MEASURE_EVENT_COUNT 2u
enum cuda_measurement_phase {
	CUDA_MEASURE_IDLE,
	CUDA_MEASURE_BEGUN,
	CUDA_MEASURE_ENDED
};
#endif
#define MODULE_CAPACITY            \
	(BACKEND_MAXIMUM_KERNELS + \
	 1) /* the digest's, and one for each prepared kernel */
#define PEER_WORD_CAPACITY 256u                 /* address words peer_connect wrote, which copies may name */

/* An address word peer_connect wrote into the card's arena: the owner's address it holds and the bytes behind it. */
struct peer_word {
    uint64_t word;
    CUdeviceptr address;
    uint64_t byte_count;
};
#define KERNEL_FUNCTION_NAME_CAPACITY 64        /* bytes of a kernel's function name, its terminating zero included */

/* The build of one kernel file for the card's architecture, loaded on the card. */
struct loaded_module {
	const struct cuda_kernel_image *image;
	CUmodule module;
};

/* What launch_run reads of a prepared kernel: half a cache line. */
struct prepared_kernel {
    CUfunction function;
    const struct cuda_kernel_image *
	    image; /* its kernel file: with the function's name, which kernel */
    uint32_t argument_byte_count;               /* of its record, as the card's build states it */
    uint32_t most_threads_per_block;            /* this and the next: what the driver gives a block of this function */
    uint32_t most_dynamic_shared_byte_count;
    uint32_t reserved_zero;
};

/* What an open card holds: all zero while it is closed. Guarded by the card's mutex. */
struct open_card_state {
    CUdeviceptr arena_address;
    uint64_t arena_byte_count;
    CUdeviceptr sum_address;                    /* where the digest kernel leaves its sum */
    CUstream stream;                            /* the copies and the kernels, in the order they began */
    CUevent events[CARD_EVENT_COUNT];           /* the ring of the copies, the digest's, the launch's, the marks */
#if TIMING_HOOKS_ENABLED
    CUevent measurement_events[MEASURE_EVENT_COUNT];
    enum cuda_measurement_phase measurement_phase;
#endif
    CUfunction digest_function;
    struct loaded_module modules[MODULE_CAPACITY];
    struct prepared_kernel kernels[BACKEND_MAXIMUM_KERNELS];
    CUgraph graphs[BACKEND_MAXIMUM_GRAPHS];
    CUgraphExec graph_executables[BACKEND_MAXIMUM_GRAPHS];
    struct peer_word peer_words[PEER_WORD_CAPACITY];
    uint64_t last_ticket_issued;
    uint64_t last_ticket_finished;              /* this ticket and every earlier one were reported finished */
    uint64_t first_failed_ticket;               /* 0 while the card is sound; then last_ticket_finished + 1 */
    uint32_t module_count;
    uint32_t kernel_count;
    uint32_t graph_count;
    uint32_t peer_word_count;
    char kernel_function_names[BACKEND_MAXIMUM_KERNELS]
			      [KERNEL_FUNCTION_NAME_CAPACITY]; /* by handle */
    bool is_open;
    bool digest_kernel_is_in_flight;
    bool digest_has_a_waiter;                   /* a call owns the kernel in flight and will read its sum */
};

/* The most the card takes in each dimension of a grid and of a block, in the order of struct kernel_launch. */
struct card_launch_limits {
	uint32_t most_grid[LAUNCH_DIMENSIONS];
	uint32_t most_block[LAUNCH_DIMENSIONS];
};

struct device_card {
	pthread_mutex_t mutex;
	CUcontext context;
	const char *
		architecture; /* of its kernel build; empty for a card too old for any */
	struct open_card_state state;
	struct card_description_record
		description; /* what never changes of it; the rest is zero */
	struct card_launch_limits launch_limits;
	uint32_t digest_most_blocks; /* the widest grid of the digest kernel */
};

/* A driver call that only succeeds or fails: true when it succeeded, and a fault point (see the top). */
#define DRIVER_CALL(function_name, ...)                                   \
	cuda_driver_succeeded(#function_name,                             \
			      FAULT_POINT(backend_cuda_##function_name) ? \
				      CUDA_ERROR_UNKNOWN :                \
				      function_name(__VA_ARGS__))

/* ---- backend_cuda.c ---- */

/* Zero when the clock cannot be read. */
uint64_t cuda_monotonic_nanoseconds(void);

/* False at once when the limit since started_nanoseconds has passed (or no clock); else sleeps until the next look. */
bool cuda_sleep_before_next_poll(uint64_t started_nanoseconds,
				 uint64_t wait_limit_nanoseconds);

/* True for a driver result that is success; otherwise one error line with the call and the driver's name of it. */
bool cuda_driver_succeeded(const char *function_name, CUresult result);

/* Takes a card's mutex and makes its context current: the card's state, or NULL with nothing held and *status_out
 * saying why (INVALID_ARGUMENT for an address that is no card, and for a closed card where an open one is needed). */
struct open_card_state *cuda_card_enter(struct device_card *card,
					bool must_be_open,
					result_status *status_out);
void cuda_card_leave(struct device_card *card);

/* Whether everything an event recorded is done: OK, TIMED_OUT for not yet, or DEVICE_ERROR and the card has failed. */
MUST_CHECK result_status cuda_event_poll(struct open_card_state *state,
					 CUevent event);

/* ---- backend_cuda_launch.c ---- */

/* Run once for a card, while the driver is initialised: false, with an error line, if the card does not tell the
 * limits of a launch. */
bool cuda_launch_limits_read(struct device_card *card, CUdevice device);

/* The module of a kernel file on an entered card: the one it holds, or loaded now and held until
 * cuda_modules_unload. UNSUPPORTED when the card's architecture has no build of the file, DEVICE_ERROR when
 * the driver does not load it. */
MUST_CHECK result_status cuda_module_of(const struct device_card *card,
					struct open_card_state *state,
					const char *kernel_file_name,
					CUmodule *module_out);

/* Unloads every module of an entered card whose queue is empty: none of its kernels is prepared afterwards. */
void cuda_modules_unload(struct open_card_state *state);

MUST_CHECK result_status cuda_kernel_prepare(struct device_card *card,
					     const char *kernel_file_name,
					     const char *function_name,
					     uint32_t argument_byte_count,
					     uint32_t *kernel_out);
MUST_CHECK result_status cuda_launch_run(struct device_card *card,
					 const struct kernel_launch *launches,
					 uint32_t launch_count);
MUST_CHECK result_status cuda_launch_wait(struct device_card *card,
					  uint64_t wait_limit_nanoseconds);
MUST_CHECK result_status cuda_graph_capture(
	struct device_card *card, const struct kernel_launch *launches,
	uint32_t launch_count, uint32_t *graph_out);
MUST_CHECK result_status cuda_graph_run(struct device_card *card,
					uint32_t graph);

#endif
