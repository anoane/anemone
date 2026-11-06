/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* pack_digest_test: the host digest against its known answer and against every property its contract states. */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/pack_digest.h"
#include "anemone/status.h"

#define KNOWN_ANSWER_BYTE_COUNT 16
#define KNOWN_ANSWER_DIGEST 0x9F861E2075053703ull
#define SAMPLE_WORD_COUNT ((size_t)64)
#define SAMPLE_BYTE_COUNT (SAMPLE_WORD_COUNT * PACK_DIGEST_WORD_BYTES)
#define SAMPLE_FILL_MULTIPLIER 37u               /* any odd number: the sample bytes only have to differ */
#define SAMPLE_FILL_SHIFT 8                      /* so that the pattern does not repeat every 256 bytes */
#define FLIPPED_BYTE_INDEX 100
#define FLIPPED_BIT 0x10u
#define SWAPPED_WORD_FIRST ((size_t)3)
#define SWAPPED_WORD_SECOND ((size_t)41)
#define PIECE_WORD_COUNT 24                      /* the sample digested as two pieces of 24 and 40 words */
#define UNALIGNED_OFFSET 1
#define BYTE_COUNT_NOT_A_WORD_MULTIPLE 7
#define UNTOUCHED_OUTPUT_VALUE 0x1234u
#define PIECE_POSITIONS_MOST 40u                 /* first word indices 0, 3, .. 39: both sides of eight-word steps */
#define PIECE_POSITION_STEP 3u
#define LARGE_PIECE_POSITION 0xFFFFFFFFFFFFFFF0ull /* the position terms wrap past 2^64 inside the piece */

static int failures = 0;

static void expect(int condition, const char *what)
{
    if (!condition) {
        (void)fprintf(stderr, "FAILED: %s\n", what);
        failures++;
    }
}

static uint64_t digest_of(const unsigned char *bytes, uint64_t byte_count)
{
    uint64_t digest = 0;
    expect(pack_digest_of_host_bytes(bytes, byte_count, &digest) == STATUS_OK,
	   "a valid call succeeds");
    return digest;
}

int main(void)
{
    unsigned char known_bytes[KNOWN_ANSWER_BYTE_COUNT];
    unsigned char sample[SAMPLE_BYTE_COUNT];
    unsigned char changed[SAMPLE_BYTE_COUNT];
    unsigned char shifted[SAMPLE_BYTE_COUNT + UNALIGNED_OFFSET];
    unsigned char word_copy[PACK_DIGEST_WORD_BYTES];
    uint64_t sample_digest;
    uint64_t piecewise_digest = 0;
    uint64_t untouched = UNTOUCHED_OUTPUT_VALUE;

    for (unsigned int byte_index = 0; byte_index < KNOWN_ANSWER_BYTE_COUNT; byte_index++) {
        known_bytes[byte_index] = (unsigned char)byte_index;
    }
    expect(digest_of(known_bytes, KNOWN_ANSWER_BYTE_COUNT) == KNOWN_ANSWER_DIGEST, "the known answer of the header");
    expect(digest_of(known_bytes, 0) == 0, "the digest of no words is zero");

    for (unsigned int byte_index = 0; byte_index < SAMPLE_BYTE_COUNT; byte_index++) {
        sample[byte_index] = (unsigned char)(byte_index * SAMPLE_FILL_MULTIPLIER + (byte_index >> SAMPLE_FILL_SHIFT));
    }
    sample_digest = digest_of(sample, SAMPLE_BYTE_COUNT);

    memcpy(changed, sample, sizeof(sample));
    changed[FLIPPED_BYTE_INDEX] ^= FLIPPED_BIT;
    expect(digest_of(changed, SAMPLE_BYTE_COUNT) != sample_digest, "one flipped bit changes the digest");

    memcpy(changed, sample, sizeof(sample));
    memcpy(word_copy, changed + SWAPPED_WORD_FIRST * PACK_DIGEST_WORD_BYTES,
	   sizeof(word_copy));
    memcpy(changed + SWAPPED_WORD_FIRST * PACK_DIGEST_WORD_BYTES,
	   changed + SWAPPED_WORD_SECOND * PACK_DIGEST_WORD_BYTES,
	   sizeof(word_copy));
    memcpy(changed + SWAPPED_WORD_SECOND * PACK_DIGEST_WORD_BYTES, word_copy,
	   sizeof(word_copy));
    expect(digest_of(changed, SAMPLE_BYTE_COUNT) != sample_digest, "two words that change places change the digest");

    for (uint64_t word_index = 0; word_index < SAMPLE_WORD_COUNT; word_index++) {
        uint64_t word;
        uint64_t reversed_index = word_index < PIECE_WORD_COUNT ? PIECE_WORD_COUNT - 1 - word_index : word_index;
	memcpy(&word, sample + reversed_index * PACK_DIGEST_WORD_BYTES,
	       sizeof(word));
	piecewise_digest += pack_digest_term(word, reversed_index);
    }
    expect(piecewise_digest == sample_digest, "the terms added in another order give the same digest");

    memcpy(shifted + UNALIGNED_OFFSET, sample, sizeof(sample));
    expect(digest_of(shifted + UNALIGNED_OFFSET, SAMPLE_BYTE_COUNT) == sample_digest,
           "the bytes may have any alignment");

    expect(pack_digest_of_host_bytes(sample, BYTE_COUNT_NOT_A_WORD_MULTIPLE,
				     &untouched) == STATUS_INVALID_ARGUMENT,
	   "a byte count that is no multiple of a word is refused");
    expect(pack_digest_of_host_bytes(NULL, SAMPLE_BYTE_COUNT, &untouched) ==
		   STATUS_INVALID_ARGUMENT,
	   "missing bytes are refused");
    expect(pack_digest_of_host_bytes(sample, SAMPLE_BYTE_COUNT, NULL) ==
		   STATUS_INVALID_ARGUMENT,
	   "a missing output is refused");
    expect(untouched == UNTOUCHED_OUTPUT_VALUE, "a refused call leaves its output untouched");

    /* a piece of every length up to the sample, at every alignment and from positions on both sides of an eight-word
     * step, word by word (the terms of the header) and as pack_digest_of_piece computes it */
    for (uint64_t first_word_index = 0; first_word_index < PIECE_POSITIONS_MOST;
         first_word_index += PIECE_POSITION_STEP) {
        for (size_t word_count = 0; word_count <= SAMPLE_WORD_COUNT; word_count++) {
            uint64_t word_by_word = 0;
            for (size_t word_index = 0; word_index < word_count; word_index++) {
                uint64_t word;
		memcpy(&word, sample + word_index * PACK_DIGEST_WORD_BYTES,
		       sizeof(word));
		word_by_word +=
			pack_digest_term(word, first_word_index + word_index);
	    }
	    const uint64_t piece_bytes =
		    word_count * PACK_DIGEST_WORD_BYTES +
		    BYTE_COUNT_NOT_A_WORD_MULTIPLE % PACK_DIGEST_WORD_BYTES;
	    expect(pack_digest_of_piece(shifted + UNALIGNED_OFFSET, piece_bytes,
					first_word_index) == word_by_word,
		   "a piece's digest is its terms' sum, at any length, alignment and position");
	}
    }
    expect(pack_digest_of_piece(sample, SAMPLE_BYTE_COUNT,
				LARGE_PIECE_POSITION) ==
		   pack_digest_of_piece(
			   sample, PIECE_WORD_COUNT * PACK_DIGEST_WORD_BYTES,
			   LARGE_PIECE_POSITION) +
			   pack_digest_of_piece(
				   sample + PIECE_WORD_COUNT *
						    PACK_DIGEST_WORD_BYTES,
				   SAMPLE_BYTE_COUNT -
					   PIECE_WORD_COUNT *
						   PACK_DIGEST_WORD_BYTES,
				   LARGE_PIECE_POSITION + PIECE_WORD_COUNT),
	   "two pieces far into an extent add up to the whole (positions wrap modulo 2^64)");

    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf("pack_digest_test passed\n");
    return EXIT_SUCCESS;
}
