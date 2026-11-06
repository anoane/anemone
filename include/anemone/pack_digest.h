/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * pack_digest.h: the ONE definition of the digest a pack stores for each extent.
 *
 * Contract:
 *   - An extent is a whole number of 64-bit little-endian words. Its digest is the sum modulo 2^64, over every
 *     word of the extent, of pack_digest_term(word, word_index): the mixer below applied to the word XOR
 *     (word_index times PACK_DIGEST_POSITION_MULTIPLIER, modulo 2^64), with word_index counted from 0 at
 *     the start of the extent. The digest of no words is 0.
 *   - A sum of integers has no order. A card computes it with any number of threads in any order, with integer
 *     additions only, and gets this one value; an extent digested in pieces gives it too, when every piece
 *     counts word_index from the start of the extent.
 *   - It detects damage and misplaced blocks. A change confined to one word always changes the digest (the mixer
 *     and the XOR are bijections); changes to several words, and words moved to other positions, change it
 *     except by coincidence. It is NOT cryptographic: two extents with one digest are easy to build on purpose.
 *     It guards against drives, links and card memory, not against an attacker.
 *   - pack_digest_of_host_bytes is the oracle. The converter stores its value in the pack index, the
 *     loader checks host-held extents with it, and the digest kernel of every backend must equal it bit for bit.
 *     A kernel file includes no engine header (the rule is in backend_seam.h): it repeats the two inline
 *     functions below word for word, and its test compares the kernel with the host function.
 *   - Known answer: the sixteen bytes 00 01 02 .. 0F have the digest 0x9F861E2075053703.
 */
#ifndef PACK_DIGEST_H
#define PACK_DIGEST_H

#include <stdint.h>

#include "anemone/status.h"

#define PACK_DIGEST_WORD_BYTES 8
#define PACK_DIGEST_POSITION_MULTIPLIER \
	0x9E3779B97F4A7C15ull /* odd: 2^64 divided by the golden ratio */
#define PACK_DIGEST_FIRST_MULTIPLIER \
	0xBB67AE8584CAA73Bull /* odd: the fraction of the square root of 3 */
#define PACK_DIGEST_SECOND_MULTIPLIER \
	0xA54FF53A5F1D36F1ull /* odd: the fraction of the square root of 7 */
#define PACK_DIGEST_FIRST_SHIFT 32u
#define PACK_DIGEST_SECOND_SHIFT 29u
#define PACK_DIGEST_THIRD_SHIFT 32u

/* A bijection of the 64-bit words: a shift-and-XOR and a multiplication by an odd number can each be undone. */
static inline uint64_t pack_digest_mix(uint64_t value)
{
	value ^= value >> PACK_DIGEST_FIRST_SHIFT;
	value *= PACK_DIGEST_FIRST_MULTIPLIER;
	value ^= value >> PACK_DIGEST_SECOND_SHIFT;
	value *= PACK_DIGEST_SECOND_MULTIPLIER;
	value ^= value >> PACK_DIGEST_THIRD_SHIFT;
	return value;
}

/* What the word at word_index adds to the digest of its extent. */
static inline uint64_t pack_digest_term(uint64_t word, uint64_t word_index)
{
	return pack_digest_mix(word ^
			       (word_index * PACK_DIGEST_POSITION_MULTIPLIER));
}

/* First users: the converter's index writer, the loader's check of host-held extents, and the test of each
 * backend's digest kernel. bytes may have any alignment. Fails with STATUS_INVALID_ARGUMENT when bytes or
 * digest_out is NULL or byte_count is no multiple of PACK_DIGEST_WORD_BYTES. */
MUST_CHECK result_status pack_digest_of_host_bytes(const void *bytes,
						   uint64_t byte_count,
						   uint64_t *digest_out);

/* A piece's share of its extent's digest: the sum of the terms of the byte_count / PACK_DIGEST_WORD_BYTES
 * words at bytes (any alignment; a partial last word is not counted), the first of which lies first_word_index words
 * into the extent. The digest of the bytes above is this with first_word_index 0. Built for AVX-512 (the host build's
 * -march) it takes eight words at a time: integer sums, the same value as word by word. */
uint64_t pack_digest_of_piece(const void *bytes, uint64_t byte_count,
			      uint64_t first_word_index);

#endif
