/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_fill.cu: one value written into a range of 32-bit words of a card's arena. Device code only, built once per
 * architecture into the module "kernel_fill". The engine clears state on a card with it.
 *
 * Contract:
 *   - fill_words_entry(record, arena) writes record.value into the record.word_count words that begin
 *     record.arena_byte_offset bytes into the arena. It reads nothing of the arena and writes nothing else.
 *   - It checks nothing. The host has checked, before it prepared the launch: record.arena_byte_offset is a multiple
 *     of FILL_WORD_BYTES; the words lie inside the arena, so that neither the offset nor four times the
 *     count nor their sum wraps; arena is the address of the card's arena, which the driver aligns to a word and
 *     more. record.reserved_zero is not read.
 *   - A launch uses a block of (FILL_THREADS_PER_BLOCK_VALUE, 1, 1) threads and a grid of (blocks, 1, 1), with a
 *     block for every FILL_THREADS_PER_BLOCK_VALUE words, at least one and at most FILL_MOST_BLOCKS
 *     (include/anemone/backend_records.h), and no dynamic shared memory.
 *   - Any grid and block give the same bytes. Only the threads whose own and whose block's second and third index
 *     are zero write; such a thread takes the words whose index is its number in the first dimension plus a
 *     multiple of the number of those threads. So every word is written once, by one thread, whatever the grid.
 *   - Nothing here is arithmetic: no order of the threads changes a bit.
 */
#include <stdint.h>

#include "anemone/backend_records.h"

extern "C" __global__ void fill_words_entry(struct fill_record record,
					    unsigned char *arena)
{
    uint32_t *const words = (uint32_t *)(arena + record.arena_byte_offset);
    const uint64_t thread_count = (uint64_t)gridDim.x * blockDim.x;
    const uint64_t thread_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if ((blockIdx.y | blockIdx.z | threadIdx.y | threadIdx.z) != 0) {
        return;
    }
    for (uint64_t word_index = thread_index; word_index < record.word_count; word_index += thread_count) {
        words[word_index] = record.value;
    }
}
