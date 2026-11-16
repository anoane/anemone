/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * sha256.h: SHA-256 (FIPS 180-4) in the process, for the copier's chunk manifest and the storage profile's keys.
 *
 * Contract:
 *   - The digest of any byte string, as FIPS 180-4 defines it: the same 32 bytes the kernel's "sha256" gives.
 *   - The x86-64 SHA extensions (SHA-NI) do the rounds where this build and the processor have them (chosen once, at
 *     the first call), plain C rounds everywhere else, aarch64 included (no ARMv8 crypto-extension path is built);
 *     both give the same digest for the same bytes.
 *   - sha256_of_two hashes two strings of one length at once: on a processor with the extensions the two streams'
 *     rounds interleave, so two digests cost about the time of one. Each digest equals sha256_of of its string.
 *   - A stream (begin, feed any number of times, finish) gives the digest of the concatenation of what was fed.
 *   - Nothing is allocated; a stream is a plain value the caller owns. A NULL argument (or NULL bytes with a non-zero
 *     count) is STATUS_INVALID_ARGUMENT and leaves the outputs untouched.
 */
#ifndef SHA256_H
#define SHA256_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "anemone/status.h"

#define SHA256_DIGEST_BYTES 32u
#define SHA256_BLOCK_BYTES 64u
#define SHA256_STATE_WORDS 8u

struct sha256_stream {
	uint32_t state[SHA256_STATE_WORDS];
	uint64_t byte_count; /* fed so far */
	unsigned char pending
		[SHA256_BLOCK_BYTES]; /* the start of a block not yet compressed */
	uint32_t pending_bytes;
};

void sha256_begin(struct sha256_stream *stream);
MUST_CHECK result_status sha256_feed(struct sha256_stream *stream,
				     const void *bytes, size_t count);
MUST_CHECK result_status
sha256_finish(struct sha256_stream *stream,
	      unsigned char digest[SHA256_DIGEST_BYTES]);
MUST_CHECK result_status sha256_of(const void *bytes, size_t count,
				   unsigned char digest[SHA256_DIGEST_BYTES]);
MUST_CHECK result_status
sha256_of_two(const void *first, const void *second, size_t count,
	      unsigned char first_digest[SHA256_DIGEST_BYTES],
	      unsigned char second_digest[SHA256_DIGEST_BYTES]);
/* Whether the x86-64 SHA extensions do the rounds (for reports and tests); false on any other processor. */
bool sha256_uses_extensions(void);

#endif
