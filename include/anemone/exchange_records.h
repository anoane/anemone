/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * exchange_records.h: what cards and the host hand each other through memory they share (wide_prefill.md 5.2-5.4,
 * tensor_parallel_two_170hx.md 5): places, mailboxes and their lines, the records of the exchange family
 * (source/kernel_exchange.cu) and of a copy (backend_seam.h's BACKEND_COPY). C11 and CUDA, of the engine
 * step_records.h alone.
 *
 * Contract:
 *   - A place is memory named by the arena offset of an address word (backend_seam.h: exchange_attach wrote there the
 *     address of exchange memory, peer_attach that of another card's arena) and an offset from that address;
 *     EXCHANGE_OWN_ARENA in place of the word is the card's own arena. Whoever prepares a launch has checked
 *     every rule here and that every byte a record names lies inside what its word maps.
 *   - A mailbox carries a hop from one sender to one receiver at one site, once a step parity (the parity of the step
 *     block's sequence): a flag line, a credit line and a payload slot, each on a line of its own, parity 1's
 *     parity_stride bytes behind parity 0's. A line is EXCHANGE_LINE_BYTES, its uint64 word at its first byte,
 *     nothing else of it ever written: no two writers share one. The slot holds the payload, then its trailer at the
 *     next multiple of 16. Each lies with the card that polls or reads it (the flag and slot with the receiver, the
 *     credit with the sender): in its arena where peer_attach gave the writer that arena, else in exchange memory.
 *     A launch's error records lie in exchange memory.
 *   - The payload is a hand-off's (model_seam.h) rows of the step or view at step_offset: row_bytes from arena_offset
 *     + r row_bytes for each row r below its row count, or with an entry_ratio q only the entry of a row at p of
 *     session s that emits one, from arena_offset + s session_stride + (p / q) row_bytes; in row order, packed, at
 *     most slot_bytes (the checked modules trap beyond). A receiver puts it at the same places from its arena_offset.
 *   - With s the step's sequence: exchange_send reads the flag of s's parity (the sequence last published
 *     there, 0 at first) and waits for the credit to hold at least that much, so the receiver has consumed it, however
 *     far apart the steps that use the mailbox; then it stores the payload into the slot in 16-byte words, writes the
 *     trailer {s, the payload's bytes, site, sender} and stores s into the flag. A bulk hop opens with a send of no
 *     payload (a slot of its trailer alone), then each piece is a copy and a raise of its flag.
 *     exchange_receive waits for the flag to hold s, checks the trailer against its own step, puts the payload
 *     with volatile 16-byte loads and stores s into the credit. exchange_wait waits for its line to hold s;
 *     exchange_raise stores s into its line (the credit of a slot read in place, behind its last reader, or
 *     the line a launch of several blocks leaves).
 *   - Order (PTX's memory model, at system scope): every store of a flag or credit follows a fence, behind the loads
 *     and stores it publishes or releases. A wait polls with volatile loads, each block of a launch for itself; once
 *     the line holds what it waits for, the polling thread fences, then the block meets a barrier, and no thread
 *     loads the trailer or the payload or stores into the slot before it. The host keeps the same order with its
 *     fences, as a sender or a receiver.
 *   - A wait polls for at most deadline_cycles of the card's clock (clock64). A wait past its deadline, a line above
 *     s, or a trailer that is not the one expected writes the error record of the failing block b at the launch's
 *     error place + b EXCHANGE_ERROR_BYTES (a launch has its own, one a block), failure last behind a
 *     system-scope fence, and traps: the card reports a device error and begins no new work (backend_seam.h); a
 *     record of failure 0 holds nothing. No stale or partial payload is consumed.
 *   - A queued wait is always met: a step's graphs may be queued before its rows are known (wide_prefill.md 5.4), and
 *     a step so queued whose rows the driver will not send (the reply ended, an idle spell, the cards closing) goes
 *     out as a drain step: a step block of row_count 0 (step_records.h), through which no launch writes state or a
 *     result row and each send carries its trailer alone.
 *   - A copy: byte_count bytes from source to destination by the copy engine, in the order of its card's launches;
 *     each place names the arena or a word the backend wrote on that card.
 *   - No atomic operation: every word has one writer. A launch of one block stores its line from one thread behind
 *     the block's other work; one of several blocks stores none (block 0 writes a send's trailer), each block fences
 *     at system scope at its end, and a raise behind it on the card stores the line. Grid and block are otherwise
 *     exchange.h's: no byte depends on them. site and sender lie below EXCHANGE_MAXIMUM_SITES, the trailer's
 *     width, card is the launch's own and a send's sender (checked by whoever prepares a launch; the checked modules
 *     trap). Records have fixed-width members, widest first, no padding: sizes asserted in both languages.
 */
#ifndef EXCHANGE_RECORDS_H
#define EXCHANGE_RECORDS_H

#include <stdint.h>

#include "anemone/step_records.h"

#define KERNEL_FILE_EXCHANGE "kernel_exchange"
#define KERNEL_EXCHANGE_SEND "exchange_send"
#define KERNEL_EXCHANGE_RECEIVE "exchange_receive"
#define KERNEL_EXCHANGE_WAIT "exchange_wait"
#define KERNEL_EXCHANGE_RAISE "exchange_raise"
#define EXCHANGE_OWN_ARENA UINT64_MAX /* in place of an address word */
#define EXCHANGE_LINE_BYTES 128u /* a flag or credit line */
#define EXCHANGE_MAXIMUM_SITES \
	(UINT16_MAX + 1u) /* sites, and cards: the trailer holds 16 bits */
#define EXCHANGE_WORD_BYTES \
	16u /* of a payload's stores and loads; a row_bytes multiple */
#define EXCHANGE_PLACE_BYTES 16u
#define EXCHANGE_TRAILER_BYTES 16u
#define EXCHANGE_ERROR_BYTES 32u
#define EXCHANGE_ARGUMENT_BYTES 128u
#define EXCHANGE_COPY_ARGUMENT_BYTES 40u
enum exchange_failure {
	EXCHANGE_LATE = 1,
	EXCHANGE_AHEAD = 2,
	EXCHANGE_WRONG = 3
};

struct exchange_place {
	uint64_t address_word_offset; /* in the card's arena, or EXCHANGE_OWN_ARENA */
	uint64_t offset; /* from the address the word holds */
};

struct exchange_trailer {
	uint64_t sequence; /* the step's */
	uint32_t byte_count; /* of the payload */
	uint16_t site;
	uint16_t card; /* the sender's */
};

/* What a block that failed an exchange writes into its own record, failure last, before it traps. */
struct exchange_error {
	uint64_t sequence; /* the step's */
	uint64_t seen; /* what the line held last, or the trailer's sequence */
	uint32_t site;
	uint32_t card; /* the card that failed: the record's card */
	uint32_t failure; /* enum exchange_failure */
	uint32_t reserved_zero;
};

/* The record of the four kernels of kernel_exchange.cu. wait and raise read line, error, step_offset, parity_stride,
 * site, card and (wait) deadline_cycles; every other member is 0. A failing block writes card into its error record. */
struct exchange_arguments {
	struct exchange_place slot; /* parity 0's payload slot */
	struct exchange_place
		line; /* parity 0's flag line: send, receive, wait; raise's line */
	struct exchange_place credit; /* parity 0's credit line */
	struct exchange_place error; /* the launch's error records, one a block */
	uint64_t step_offset; /* the step area or a view: its sequence and row count */
	uint64_t arena_offset;
	uint64_t session_stride; /* 0 where entry_ratio is 0 */
	uint64_t parity_stride;
	uint64_t deadline_cycles;
	uint32_t row_bytes; /* a multiple of EXCHANGE_WORD_BYTES */
	uint32_t entry_ratio;
	uint32_t site;
	uint32_t sender; /* the sending card: the trailer's */
	uint32_t slot_bytes; /* the most a payload of the slot may hold */
	uint32_t card; /* the card that runs the launch */
};

/* The record of a launch of BACKEND_COPY (backend_seam.h). */
struct exchange_copy_arguments {
	struct exchange_place source;
	struct exchange_place destination;
	uint64_t byte_count; /* 1 or more */
};

STEP_RECORD_HAS(exchange_place, EXCHANGE_PLACE_BYTES);
STEP_RECORD_HAS(exchange_trailer, EXCHANGE_TRAILER_BYTES);
STEP_RECORD_HAS(exchange_error, EXCHANGE_ERROR_BYTES);
STEP_RECORD_HAS(exchange_arguments, EXCHANGE_ARGUMENT_BYTES);
STEP_RECORD_HAS(exchange_copy_arguments, EXCHANGE_COPY_ARGUMENT_BYTES);

#endif
