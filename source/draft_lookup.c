/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * draft_lookup.c: the lookup drafter of draft_lookup.h. An n-gram's bucket is the top bucket_bits of a multiplicative
 * hash of its length and tokens, so every token bit reaches the index; n-grams of any lengths may share a bucket, and
 * the later one then wins it (the comparison refuses the other's proposal). A proposal costs at most a probe and a
 * comparison of ngram_most tokens a length.
 */
#include "draft_lookup.h"

#include <string.h>

#define HASH_MULTIPLIER \
	0x9E3779B97F4A7C15ull /* 2^64 over the golden ratio: Fibonacci hashing */
#define HASH_BITS 64u
#define DRAFT_LOOKUP_LEAST_BUCKET_BITS 10u
#define DRAFT_LOOKUP_MOST_BUCKET_BITS \
	22u /* 16 MiB a session at most: 4M buckets, a 1M context's n-grams of up to 4 lengths (main.c's 2 to 4: 3) */

uint32_t draft_lookup_bucket_bits(uint32_t capacity, uint32_t ngram_least,
				  uint32_t ngram_most)
{
	const uint64_t filed =
		(uint64_t)capacity * (ngram_most - ngram_least + 1u);
	uint32_t bits = DRAFT_LOOKUP_LEAST_BUCKET_BITS;
	while (bits < DRAFT_LOOKUP_MOST_BUCKET_BITS && (1ull << bits) < filed) {
		bits++;
	}
	return bits;
}

void draft_lookup_begin(struct draft_lookup *lookup, uint32_t *tokens,
			uint32_t capacity, uint32_t *buckets,
			uint32_t bucket_bits, uint32_t ngram_least,
			uint32_t ngram_most)
{
	*lookup = (struct draft_lookup){ .tokens = tokens,
					 .buckets = buckets,
					 .capacity = capacity,
					 .bucket_bits = bucket_bits,
					 .ngram_least = ngram_least,
					 .ngram_most = ngram_most };
}

/* The bucket of the n-gram of length tokens at ngram. */
static uint32_t bucket_of(const struct draft_lookup *lookup,
			  const uint32_t *ngram, uint32_t length)
{
	uint64_t hash = length;
	for (uint32_t index = 0; index < length; index++) {
		hash = (hash ^ ngram[index]) * HASH_MULTIPLIER;
	}
	return (uint32_t)(hash >> (HASH_BITS - lookup->bucket_bits));
}

void draft_lookup_keep(struct draft_lookup *lookup, uint32_t kept)
{
	lookup->count = kept < lookup->count ? kept : lookup->count;
}

void draft_lookup_append(struct draft_lookup *lookup, uint32_t token)
{
	const uint32_t position = lookup->count;
	if (position == lookup->capacity) {
		return;
	}
	/* the n-grams that end before the token: it follows their latest occurrence (a position above 0: not empty) */
	for (uint32_t length = lookup->ngram_least;
	     length <= lookup->ngram_most && length <= position; length++) {
		lookup->buckets[bucket_of(
			lookup, lookup->tokens + position - length, length)] =
			position;
	}
	lookup->tokens[position] = token;
	lookup->count = position + 1u;
}

uint32_t draft_lookup_propose(const struct draft_lookup *lookup, uint32_t most,
			      uint32_t *proposals_out)
{
	const uint32_t count = lookup->count;
	/* a suffix with a token before it: ngram_least is at least 1, so the count down ends */
	for (uint32_t length = count > lookup->ngram_most ? lookup->ngram_most :
			       count != 0		  ? count - 1u :
							    0;
	     length >= lookup->ngram_least; length--) {
		const uint32_t *const suffix = lookup->tokens + count - length;
		const uint32_t after =
			lookup->buckets[bucket_of(lookup, suffix, length)];
		if (after >= length && after < count &&
		    memcmp(lookup->tokens + after - length, suffix,
			   length * sizeof(*suffix)) == 0) {
			/* the text goes on as it went on behind the occurrence, count - after tokens back: past the
			 * known tokens, into the proposals themselves (a repeat of a short period proposes most) */
			for (uint32_t index = 0; index < most; index++) {
				proposals_out[index] =
					after + index < count ?
						lookup->tokens[after + index] :
						proposals_out[index -
							      (count - after)];
			}
			return most;
		}
	}
	return 0;
}
