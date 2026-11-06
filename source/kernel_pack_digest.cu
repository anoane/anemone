/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_pack_digest.cu: the digest of pack_digest.h, computed on a card over words that lie in its memory. Device
 * code only, built once per architecture into the module "kernel_pack_digest".
 *
 * Contract:
 *   - kernel_pack_digest adds to *digest_out the sum modulo 2^64, over word_index from 0 to word_count - 1,
 *     of pack_digest_term(words[word_index], word_index). The host zeroes *digest_out before the launch
 *     and reads it when the kernel has finished: the value is then the digest of those words, bit for bit.
 *   - Any grid of one dimension gives that value. A thread takes the words whose index is its own number plus a
 *     multiple of the number of threads, and adds its sum once: integer additions only, so no order of the
 *     threads changes a bit. A thread with no word adds zero.
 *   - words is aligned to a word. Nothing but those word_count words is read and nothing but *digest_out written.
 *   - A kernel file includes no engine header: the constants and the two functions below are those of
 *     include/anemone/pack_digest.h, repeated word for word. tests/backend_cuda_card_test.c compares this kernel
 *     with the host function on every card.
 */
#include <stdint.h>

#define PACK_DIGEST_POSITION_MULTIPLIER \
	0x9E3779B97F4A7C15ull /* odd: 2^64 divided by the golden ratio */
#define PACK_DIGEST_FIRST_MULTIPLIER \
	0xBB67AE8584CAA73Bull /* odd: the fraction of the square root of 3 */
#define PACK_DIGEST_SECOND_MULTIPLIER \
	0xA54FF53A5F1D36F1ull /* odd: the fraction of the square root of 7 */
#define PACK_DIGEST_FIRST_SHIFT 32
#define PACK_DIGEST_SECOND_SHIFT 29
#define PACK_DIGEST_THIRD_SHIFT 32

/* A bijection of the 64-bit words: a shift-and-XOR and a multiplication by an odd number can each be undone. */
__device__ static inline uint64_t pack_digest_mix(uint64_t value)
{
	value ^= value >> PACK_DIGEST_FIRST_SHIFT;
	value *= PACK_DIGEST_FIRST_MULTIPLIER;
	value ^= value >> PACK_DIGEST_SECOND_SHIFT;
	value *= PACK_DIGEST_SECOND_MULTIPLIER;
	value ^= value >> PACK_DIGEST_THIRD_SHIFT;
	return value;
}

/* What the word at word_index adds to the digest of its extent. */
__device__ static inline uint64_t pack_digest_term(uint64_t word,
						   uint64_t word_index)
{
	return pack_digest_mix(word ^
			       (word_index * PACK_DIGEST_POSITION_MULTIPLIER));
}

extern "C" __global__ void kernel_pack_digest(const unsigned long long *words,
					      unsigned long long word_count,
					      unsigned long long *digest_out)
{
    const unsigned long long thread_count = (unsigned long long)gridDim.x * blockDim.x;
    const unsigned long long thread_index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long thread_sum = 0;
    for (unsigned long long word_index = thread_index; word_index < word_count; word_index += thread_count) {
	    thread_sum += pack_digest_term(words[word_index], word_index);
    }
    atomicAdd(digest_out, thread_sum);
}
