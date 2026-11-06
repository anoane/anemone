/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "anemone/pack_digest.h"

#include <string.h>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#include <immintrin.h>
#endif

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "a pack holds little-endian words and is read as it is: the host must be little-endian"
#endif

#define VECTOR_WORDS 8u                          /* the 64-bit lanes of an AVX-512 register */

uint64_t pack_digest_of_piece(const void *bytes, uint64_t byte_count,
			      uint64_t first_word_index)
{
    const unsigned char *const first_byte = bytes;
    const uint64_t word_count = byte_count / PACK_DIGEST_WORD_BYTES;
    uint64_t digest = 0;
    uint64_t word_index = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    /* lane l holds the position term of word word_index + l: (first_word_index + word_index + l) times the position
     * multiplier, modulo 2^64; a step of eight words adds eight times the multiplier to every lane */
    const __m512i lanes = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i position_multiplier =
	    _mm512_set1_epi64((long long)PACK_DIGEST_POSITION_MULTIPLIER);
    const __m512i position_step = _mm512_set1_epi64(
	    (long long)(VECTOR_WORDS * PACK_DIGEST_POSITION_MULTIPLIER));
    const __m512i first_multiplier =
	    _mm512_set1_epi64((long long)PACK_DIGEST_FIRST_MULTIPLIER);
    const __m512i second_multiplier =
	    _mm512_set1_epi64((long long)PACK_DIGEST_SECOND_MULTIPLIER);
    __m512i positions = _mm512_mullo_epi64(_mm512_add_epi64(_mm512_set1_epi64((long long)first_word_index), lanes),
                                           position_multiplier);
    __m512i sums = _mm512_setzero_si512();
    for (; word_count - word_index >= VECTOR_WORDS; word_index += VECTOR_WORDS) {
	    __m512i value = _mm512_xor_si512(
		    _mm512_loadu_si512(first_byte +
				       word_index * PACK_DIGEST_WORD_BYTES),
		    positions);
	    value = _mm512_xor_si512(
		    value, _mm512_srli_epi64(value, PACK_DIGEST_FIRST_SHIFT));
	    value = _mm512_mullo_epi64(value, first_multiplier);
	    value = _mm512_xor_si512(
		    value, _mm512_srli_epi64(value, PACK_DIGEST_SECOND_SHIFT));
	    value = _mm512_mullo_epi64(value, second_multiplier);
	    value = _mm512_xor_si512(
		    value, _mm512_srli_epi64(value, PACK_DIGEST_THIRD_SHIFT));
	    sums = _mm512_add_epi64(sums, value);
	    positions = _mm512_add_epi64(positions, position_step);
    }
    /* the eight lanes added in registers with the unsigned vector additions (wrapping modulo 2^64, as defined) */
    const __m256i halves = _mm256_add_epi64(_mm512_castsi512_si256(sums), _mm512_extracti64x4_epi64(sums, 1));
    const __m128i quarters = _mm_add_epi64(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1));
    digest = (uint64_t)_mm_cvtsi128_si64(quarters) + (uint64_t)_mm_extract_epi64(quarters, 1);
#endif
    for (; word_index < word_count; word_index++) {
        uint64_t word;
	memcpy(&word, first_byte + word_index * PACK_DIGEST_WORD_BYTES,
	       sizeof(word)); /* any alignment */
	digest += pack_digest_term(word, first_word_index + word_index);
    }
    return digest;
}

result_status pack_digest_of_host_bytes(const void *bytes, uint64_t byte_count,
					uint64_t *digest_out)
{
	if (bytes == NULL || digest_out == NULL ||
	    byte_count % PACK_DIGEST_WORD_BYTES != 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	*digest_out = pack_digest_of_piece(bytes, byte_count, 0);
	return STATUS_OK;
}
