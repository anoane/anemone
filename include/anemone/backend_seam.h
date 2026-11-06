/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * backend_seam.h: the cards.
 *
 * Contract:
 *   - A backend owns everything that touches a card: its memory, the page-locked host memory that feeds it, the
 *     copies and the kernels. No other part of the engine includes a vendor header.
 *   - The host half of a backend is C11 (for CUDA: the driver interface, which is a C header). Kernel files hold
 *     device code only, are built once per architecture into modules the backend loads at card_open, and include
 *     no Anemone header except record layouts written to compile as both C11 and CUDA.
 *   - Any thread may call any function of this table: the backend makes the card current itself and serialises
 *     the calls of one card. Calls for different cards run concurrently. The three calls that wait for a card's
 *     work (upload_wait, digest, card_close) take a limit; no other call waits for an upload or a kernel.
 *   - card_count fails with STATUS_UNSUPPORTED above BACKEND_MAXIMUM_CARDS and answers 0 on a
 *     machine without cards. card_index is the order of enumeration in this process and may change between
 *     boots: a saved plan names a card by its bus_address. An index at or above the count is
 *     STATUS_INVALID_ARGUMENT. card_describe needs no open card and writes every byte of the
 *     description: the two strings are NUL-terminated and zero-filled to their capacity.
 *   - A card is opened once (opening an open card is STATUS_INVALID_ARGUMENT) with one arena of a size
 *     the caller has checked: at most memory_free_bytes minus memory_reserved_bytes. Everything the engine keeps
 *     on a card lives at a fixed offset inside that arena: there is no other allocation on a card after load.
 *   - Staging memory belongs to the backend, not to one card. It may be obtained before any card is open, is
 *     aligned to BACKEND_STAGING_ALIGNMENT_BYTES, and is page-locked for every card, so each of them
 *     uploads from it without blocking. At most BACKEND_MAXIMUM_STAGING_BLOCKS blocks exist at a time:
 *     one more is refused with STATUS_BUSY_VALUE.
 *   - An upload copies page-locked host memory into the arena and returns before the copy is done. upload_begin
 *     refuses (STATUS_INVALID_ARGUMENT) a count of zero, a range outside the arena, and a source range
 *     not wholly inside one staging block. The source stays valid and unchanged until
 *     upload_wait reports success or an actual drain retires this reader; failure is not completion.
 *   - A ticket is the card's upload sequence number: it starts at one, so zero is never a ticket, and is never
 *     reused. Unfinished means not yet reported finished by upload_wait: the limit below clears only through
 *     waits. A wait that returns STATUS_OK means that upload and every earlier one of the card have
 *     finished; once a ticket was reported finished or failed, every later wait on it returns the same status at
 *     once. At most BACKEND_MAXIMUM_UPLOADS_IN_FLIGHT uploads of a card may be unfinished: one more is
 *     refused with STATUS_BUSY_VALUE.
 *   - upload_wait and digest wait at most wait_limit_nanoseconds; 0 only asks (a digest of a range that is not
 *     empty then answers STATUS_TIMED_OUT and begins nothing). STATUS_TIMED_OUT from
 *     upload_wait means not finished: the ticket stays valid and its source memory must not be reused. From
 *     digest it means no value was delivered: the card stays usable and the digest may be asked again.
 *   - When an upload fails, its ticket and every later one of that card report STATUS_DEVICE_ERROR and
 *     the bytes of their arena ranges are unspecified. A card that has reported that status once, from an upload
 *     or from a digest, begins no new work (upload_begin and digest return it): only card_close is left. A copy
 *     queued before the failure may still read its source: staging memory of a failed card is freed only after
 *     that card has successfully drained/closed; calling close alone does not prove that on driver failure.
 *   - A digest is pack_digest (pack_digest.h) computed ON the card over arena bytes, so it proves what
 *     arrived, not what was sent. Offset and count are multiples of PACK_DIGEST_WORD_BYTES. It sees
 *     every upload of the card begun before it, and its limit covers the wait for them.
 *   - card_close waits at most its limit for the card's unfinished uploads (one error line if they are not done
 *     by then), then lets the vendor's interface drain what is queued, which no limit of ours covers, and
 *     releases the card only after a successful drain. A failed drain retains the failed card and its staging.
 *     Staging memory is freed only
 *     after every upload that reads it has finished or its card was closed; that is the caller's duty.
 *     card_close and staging_free accept NULL; an address that is not the start of a block is logged as an error
 *     and left alone.
 *   - A download is an upload in the other direction (arena to staging memory) with the same tickets, limits and
 *     failure rules; its destination range lies wholly inside one staging block and holds the bytes once
 *     upload_wait reports its ticket finished.
 *   - Kernels. A kernel is a function of one kernel file (source/kernel_<name>.cu), named by the file without
 *     directory and suffix and by the function. Every kernel takes exactly two arguments: ONE record by value
 *     (the records headers, include/anemone/<name>_records.h, compile as C11 and as CUDA), and the address of
 *     the card's arena; a record names memory by byte offsets into the arena, never by address, so a prepared
 *     launch is the same bytes on every card and in every run. kernel_prepare finds a kernel once at load and
 *     checks what the card's build says of it: the function exists, it takes two arguments, the first of exactly
 *     argument_byte_count bytes and the second an address; anything else is STATUS_INVALID_ARGUMENT with
 *     one error line (a kernel file this card's architecture has no build of: STATUS_UNSUPPORTED). At
 *     most BACKEND_MAXIMUM_KERNELS are prepared per card; preparing the same kernel again gives the same
 *     handle.
 *   - A launch (struct kernel_launch) is prepared at load and never changed: the kernel, its grid and block
 *     sizes, its dynamic shared memory and its record. launch_run queues launch_count launches on the card in
 *     their order, behind every upload, download and launch queued before, and returns without waiting; it
 *     refuses (STATUS_INVALID_ARGUMENT, nothing queued) a handle that was not prepared on this card (a copy
 *     aside, below), a record size that is not the kernel's, a grid or block dimension of zero, a block of more
 *     threads or more shared memory than the card allows, and reserved bytes that are not zero. launch_wait waits
 *     at most its limit for everything queued on the card (STATUS_TIMED_OUT: not done yet, ask again). A
 *     launch the card rejects, and any fault of a kernel the driver reports, is STATUS_DEVICE_ERROR with the
 *     rule of a failed upload: the card begins no new work. This is the eager form; graph_capture makes the graph
 *     form.
 *   - format_supported says whether the kernels of a card's arithmetic class decode a weight format, so that a
 *     plan fails at load and not at the first token.
 *   - Exchange memory (wide_prefill.md 5) is page-locked host memory mapped into every card, which kernels and the
 *     host read and write in place while cards run (exchange_records.h). exchange_allocate gives one block at a time
 *     (one more is STATUS_BUSY_VALUE), zero and aligned to BACKEND_EXCHANGE_ALIGNMENT_BYTES; a count of 0
 *     or one that does not round up to it in 64 bits, or no output, is INVALID_ARGUMENT. exchange_free frees it once
 *     every card it was attached to is closed, the caller's duty as with staging. exchange_attach writes into the
 *     card's arena the address under which the card's kernels reach the block, in place for every launch queued after
 *     it returns: a record names exchange memory by that word's offset. A card or block not given, a block that
 *     exchange_allocate did not give, or a word off a multiple of BACKEND_ADDRESS_WORD_BYTES or not wholly
 *     inside the arena: INVALID_ARGUMENT.
 *   - Peer slots: peer_attach does the same for the arena of another open card (the owner), which the accessor's
 *     kernels and copies then reach in place, until either card is closed. It allows only an ordered pair that the
 *     peer gate's record at gate_record_path passed under this boot's key (peer_to_peer.md 6: the driver, its module
 *     and parameters, the cards, the boot): a record absent, stale, malformed or of another verdict, or a pair it does
 *     not list, is STATUS_UNSUPPORTED with one line, and that hop stays in exchange memory. No other call
 *     opens peer access. A card or path not given, a card as its own owner, or a word exchange_attach would refuse:
 *     INVALID_ARGUMENT.
 *   - A launch whose kernel is BACKEND_COPY copies by the card's copy engine, in the table's order: its
 *     record is exchange_records.h's copy. This build supports nonoverlapping places in the card's own arena
 *     (EXCHANGE_OWN_ARENA) and places behind an address word peer_connect wrote into the card's arena (the
 *     owner's arena from the word's offset on: a copy engine transfer to or from a peer); it refuses other mapped
 *     words. Its bounds and record are checked as launches'; a copy may stand on a graph branch as a kernel does.
 *   - graph_capture checks a launch table as launch_run does and makes of it, in its order, one graph of the card
 *     that never changes (at most BACKEND_MAXIMUM_GRAPHS a card: one more is STATUS_BUSY_VALUE);
 *     graph_run queues it as launch_run queues the table, with the same effect on every byte; a handle the card did
 *     not capture is INVALID_ARGUMENT. A card's graphs go when it is closed.
 *   - Marks (optional, both or neither): a card has BACKEND_MAXIMUM_MARKS of them, by index. mark_record sets
 *     one behind everything queued on the card so far; mark_follow makes everything queued on card after the call
 *     wait, on the card, until owner's mark as last set before the call is passed (a mark never set: no wait). The
 *     host waits for neither. An index past the marks, or a card or owner not open: INVALID_ARGUMENT; a failed card,
 *     or one whose driver fails the call (which then fails it): DEVICE_ERROR. First user: step_pipeline.c's
 *     hand-offs, which pass from card to card without the host waiting in between.
 *   - The table is bound once at load. An entry a backend does not offer yet is NULL, and a caller that needs it
 *     answers STATUS_UNSUPPORTED. CUDA offers prepared kernel graphs; exchange memory, peer slots and
 *     copies remain absent. Still to come in this header: kernel_build_digest in the description.
 * Today's only implementation is CUDA; another backend is another table and its kernels.
 */
#ifndef BACKEND_SEAM_H
#define BACKEND_SEAM_H

#include <stdint.h>

#include <stdbool.h>

#include "anemone/status.h"
#include "anemone/weight_format_seam.h"

#define BACKEND_MAXIMUM_CARDS 16
#define BACKEND_MAXIMUM_KERNELS \
	256 /* target, native draft, wide, fused and BF16 tensor-core families a card */
#define BACKEND_NO_KERNEL UINT32_MAX /* in place of a kernel handle */
#define BACKEND_COPY  \
	(UINT32_MAX - \
	 1u) /* in place of a kernel handle: a copy (exchange_records.h) */
#define BACKEND_EXCHANGE_ALIGNMENT_BYTES \
	4096 /* a host page; flag lines and slots lie on their own lines */
#define BACKEND_ADDRESS_WORD_BYTES 8 /* a uint64 address in an arena */
#define BACKEND_MAXIMUM_GRAPHS \
	512 /* per card: TP4 thin variants plus wide/full runs; PP retains its existing capture boundaries */
#define BACKEND_MAXIMUM_MARKS \
	256 /* per card: a pipeline slot's hand-off runs each take one (step_pipeline.c) */
#define LAUNCH_DIMENSIONS 3
#define LAUNCH_ARGUMENT_BYTES 192 /* the largest record a kernel may take */
#define BACKEND_MAXIMUM_UPLOADS_IN_FLIGHT \
	16 /* per card: twice the eight staging slots of a loader lane */
#define BACKEND_STAGING_ALIGNMENT_BYTES \
	4096 /* a host page; a reader's block alignment must divide it */
#define BACKEND_MAXIMUM_STAGING_BLOCKS \
	256 /* a loader's slots on every card, and room */
#define CARD_NAME_CAPACITY 96
#define CARD_BUS_ADDRESS_CAPACITY 32
#define ARITHMETIC_CLASS_SM_80 \
	80u /* compute capability 8.0 up to below 12.0: kernels built for sm_80 */
#define ARITHMETIC_CLASS_SM_120 \
	120u /* compute capability 12.0 and above: kernels built for sm_120a */
/* A card of class 120 in its own arithmetic, the fastest it has (SWEEP_RESULTS: block-scaled FP4 by FP8 and FP8
 * tensor cores), in place of class 120's emulation of class 80's sums: the kernel files whose sums emulate class 80
 * are taken in their *_native builds (Makefile). The value is no compute capability (sm_121 is another chip): it
 * names "class 120, native". The backend never describes a card so; the caller that chooses the PRO's native
 * arithmetic (main.c, the default) gives this class to the plan in place of 120. Its bits differ from class 80's,
 * so the outputs depend on the layers such a card holds. */
#define ARITHMETIC_CLASS_SM_120_NATIVE 1200u

/* What the planner reads. What only kernels need (grid and shared memory sizes) stays inside the backend. */
struct card_description_record {
	char name[CARD_NAME_CAPACITY]; /* the vendor's name: identical cards share it */
	char bus_address
		[CARD_BUS_ADDRESS_CAPACITY]; /* where the card sits, stable across boots; empty if none */
	uint32_t arithmetic_class; /* one class runs one kernel build; its cards may share a layer if proved equal:
                                       ARITHMETIC_CLASS_*, or 0 for a card no build runs on */
	/* The fastest the link is known to run: the higher of the speed negotiated now (an idle card drops to the
     * lowest) and the highest speed both the card and the port above it state. 0 if unknown. */
	uint32_t link_generation;
	uint32_t link_lane_count; /* as negotiated now; 0 if unknown */
	uint32_t reserved_zero;
	uint64_t memory_total_bytes;
	uint64_t memory_free_bytes; /* at the time of the call */
	uint64_t memory_reserved_bytes; /* what the backend needs on this card outside the arena */
};

struct device_card;

/* One prepared launch. Nothing in it is an address: argument holds a record of a records header as the kernel
 * takes it, argument_byte_count of its bytes, the rest zero. branch (LAUNCH_*) orders it in a graph: CHAIN
 * behind the launch before it; a run of SIDE launches behind the launch before the run (the fork), each behind the
 * one before it; SIDE_MORE behind the fork alone (another side run, of one launch: copies to several peers at once);
 * RESUME behind the fork, not the side runs; JOIN behind the launch before it and every side run's last. A run's
 * first launch may open a side run: its fork is then the graph's root (behind none, as RESUME then is).
 * The list's order stays a serial order (launch_run runs it so); a branch lies inside one run of a card. */
#define LAUNCH_CHAIN 0u
#define LAUNCH_SIDE 1u
#define LAUNCH_RESUME 2u
#define LAUNCH_JOIN 3u
#define LAUNCH_SIDE_MORE 4u
#define LAUNCH_MOST_SIDE_RUNS 4u
struct kernel_launch {
	uint32_t kernel; /* from kernel_prepare, on the card it is run on */
	uint32_t grid[LAUNCH_DIMENSIONS];
	uint32_t block[LAUNCH_DIMENSIONS];
	uint32_t dynamic_shared_byte_count;
	uint32_t argument_byte_count;
	uint32_t branch;
	unsigned char argument[LAUNCH_ARGUMENT_BYTES];
};

struct backend_operations {
	const char *name;
	MUST_CHECK
	result_status (*card_count)(uint32_t *card_count_out);
	MUST_CHECK result_status (*card_describe)(
		uint32_t card_index,
		struct card_description_record *description_out);
	MUST_CHECK
	result_status (*card_open)(uint32_t card_index,
				   uint64_t arena_byte_count,
				   struct device_card **card_out);
	void (*card_close)(struct device_card *card,
			   uint64_t wait_limit_nanoseconds);
	/* First user: the loader's staging slots per card. */
	MUST_CHECK result_status (*staging_allocate)(
		uint64_t byte_count, void **page_locked_memory_out);
	void (*staging_free)(void *page_locked_memory);
	MUST_CHECK result_status (*upload_begin)(struct device_card *card,
						 uint64_t arena_byte_offset,
						 const void *page_locked_source,
						 uint64_t byte_count,
						 uint64_t *upload_ticket_out);
	MUST_CHECK
	result_status (*upload_wait)(struct device_card *card,
				     uint64_t upload_ticket,
				     uint64_t wait_limit_nanoseconds);
	MUST_CHECK result_status (*digest)(struct device_card *card,
					   uint64_t arena_byte_offset,
					   uint64_t byte_count,
					   uint64_t wait_limit_nanoseconds,
					   uint64_t *digest_out);
	/* First users: the card tests of the first kernels. A ticket of a download is waited for with upload_wait. */
	MUST_CHECK result_status (*download_begin)(
		struct device_card *card, uint64_t arena_byte_offset,
		void *page_locked_destination, uint64_t byte_count,
		uint64_t *download_ticket_out);
	MUST_CHECK result_status (*kernel_prepare)(struct device_card *card,
						   const char *kernel_file_name,
						   const char *function_name,
						   uint32_t argument_byte_count,
						   uint32_t *kernel_out);
	MUST_CHECK result_status (*launch_run)(
		struct device_card *card, const struct kernel_launch *launches,
		uint32_t launch_count);
	MUST_CHECK
	result_status (*launch_wait)(struct device_card *card,
				     uint64_t wait_limit_nanoseconds);
	/* A card index at or above the count, or a format outside the enumeration: false. */
	bool (*format_supported)(uint32_t card_index,
				 enum weight_format_code format);
	/* First users: the exchange family and the step driver of the device-driven pipeline (wide_prefill.md 5). */
	MUST_CHECK
	result_status (*exchange_allocate)(uint64_t byte_count,
					   void **exchange_out);
	void (*exchange_free)(void *exchange);
	MUST_CHECK
	result_status (*exchange_attach)(struct device_card *card,
					 const void *exchange,
					 uint64_t address_word_offset);
	MUST_CHECK result_status (*peer_attach)(struct device_card *accessor,
						struct device_card *owner,
						const char *gate_record_path,
						uint64_t address_word_offset);
	MUST_CHECK result_status (*graph_capture)(
		struct device_card *card, const struct kernel_launch *launches,
		uint32_t launch_count, uint32_t *graph_out);
	MUST_CHECK
	result_status (*graph_run)(struct device_card *card, uint32_t graph);
	MUST_CHECK result_status (*mark_record)(struct device_card *card,
						uint32_t mark);
	MUST_CHECK result_status (*mark_follow)(struct device_card *card,
						struct device_card *owner,
						uint32_t mark);
	/* Optional: relinquish a staging block to backend ownership, including outstanding/failed copies.
     * No new work may use it. Backend retains it until every possible reader is proved retired;
     * metadata outlives the destroyed caller. Accepts NULL memory; absent callback retains old policy. */
	void (*staging_retire)(void *page_locked_memory);
	/* Optional load-time direct pointer preparation. It snapshots owner's immutable arena, checks current
	 * directed access and enables that peer context, then queues owner arena+owner_offset at accessor's aligned
	 * 64-bit address word. Self is allowed. owner_offset must name a byte inside the owner arena; the complete
	 * remote extent remains the prepared caller's responsibility. UNSUPPORTED queues no word write. This grants
	 * access only: a current-stack allocation load/store pilot must qualify the route before any collective uses
	 * it. No gate file or credential is read. All arenas and tables remain immutable/live until the caller drains
	 * every participating card before closing any member or rewriting a word. No concurrent close/preparation. */
	MUST_CHECK result_status (*peer_connect)(struct device_card *accessor,
						 struct device_card *owner,
						 uint64_t owner_offset,
						 uint64_t address_word_offset);
	/* Optional diagnostic device timing, allocated only when timing hooks are compiled in; release has NULL
	 * callbacks and no events/code. One interval per card: begin queues a timestamp on its stream, end queues its
	 * timestamp after the measured launches. Neither allocates nor waits. Elapsed polls the end within the limit,
	 * writes same-context device microseconds on success and retires the interval. Timeout keeps it live for retry.
	 * Begin before a previous result is consumed is BUSY; end/elapsed without the proper predecessor is invalid.
	 * Never compare clocks across cards; queue every communicating rank before waiting for any elapsed result.
	 * Calls are outside graph capture; the caller serializes one measurement owner and drains before close. */
	MUST_CHECK
	result_status (*measure_begin)(struct device_card *card);
	MUST_CHECK
	result_status (*measure_end)(struct device_card *card);
	MUST_CHECK result_status (*measure_elapsed_us)(
		struct device_card *card, uint64_t wait_limit_nanoseconds,
		double *microseconds_out);
};

extern const struct backend_operations backend_cuda_value;
/* backend_cuda_value's memory_reserved_bytes on every card: the modules, the driver's own growth */
#define BACKEND_CUDA_MEMORY_RESERVED_BYTES (256ull << 20u)

#endif
