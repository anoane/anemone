/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * backend_records.h: the records of the backend's own kernels, as a kernel takes them and as the host lays them into
 * a launch (struct kernel_launch of backend_seam.h). Compiles as C11 and as CUDA: a kernel file may include it.
 *
 * Contract:
 *   - A record is the first argument of its kernel, passed by value; the second is the address of the card's arena.
 *     A record names memory by byte offsets into the arena, never by address.
 *   - Every member has a fixed width and the members fill the record: they stand widest first, a reserved member
 *     that is zero takes the place of padding, and the size, which is the sum of the members, is asserted in both
 *     languages. So the host and every card read the same bytes.
 *   - struct fill_record is the record of fill_words_entry (source/kernel_fill.cu), which writes value
 *     into word_count words of FILL_WORD_BYTES bytes from arena_byte_offset on. The kernel checks nothing:
 *     whoever prepares the launch has checked that arena_byte_offset is a multiple of FILL_WORD_BYTES and
 *     that the words lie inside the arena, without a wrap. A launch of it uses blocks of
 *     FILL_THREADS_PER_BLOCK_VALUE threads in one dimension and a grid of one dimension with a block for every
 *     FILL_THREADS_PER_BLOCK_VALUE words, at least one and at most FILL_MOST_BLOCKS; any grid and block
 *     give the same bytes. The bound only keeps the grid of a large range small: it is not measured.
 */
#ifndef BACKEND_RECORDS_H
#define BACKEND_RECORDS_H

#include <stdint.h>

/* The names kernel_prepare takes: the kernel file without directory and suffix, and the function in it. */
#define KERNEL_FILE_FILL "kernel_fill"
#define KERNEL_FILL_WORDS "fill_words_entry"

#define FILL_WORD_BYTES 4u
#define FILL_RECORD_BYTES 24u
#define FILL_THREADS_PER_BLOCK_VALUE 256u
#define FILL_MOST_BLOCKS 1024u

struct fill_record {
	uint64_t arena_byte_offset; /* of the first word */
	uint64_t word_count; /* 0 writes nothing */
	uint32_t value;
	uint32_t reserved_zero;
};

#ifdef __cplusplus
static_assert(sizeof(struct fill_record) == FILL_RECORD_BYTES,
	      "the fill record has no padding");
#else
_Static_assert(sizeof(struct fill_record) == FILL_RECORD_BYTES,
	       "the fill record has no padding");
#endif

#endif
