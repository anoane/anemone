/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * draft_lookup.h: the host drafter of speculative decoding (engine.c): prompt lookup over one session's tokens.
 *
 * Contract:
 *   - The record holds a copy of the session's tokens, which come in order (draft_lookup_append), and for each token
 *     files the n-grams of the ngram_least to ngram_most tokens before it under their hash: the bucket holds the
 *     position behind the latest occurrence. A bucket holds one position, so a collision loses the older occurrence,
 *     which costs a proposal and never a token: a proposal is only ever checked by the target (keyed match).
 *   - draft_lookup_propose takes the longest suffix of ngram_most down to ngram_least tokens whose latest earlier
 *     occurrence its bucket still names, compares that occurrence's tokens with the suffix, and proposes the most
 *     tokens that followed it, the text going on as it went on behind the occurrence: past the tokens the copy holds
 *     (the occurrence may run into the suffix itself), it repeats them with the period from the occurrence to the
 *     suffix. None: 0.
 *   - draft_lookup_keep forgets the tokens from kept on (a command keeps the first kept tokens of its session). Their
 *     buckets stay; a bucket that names a position at or past the count, or tokens that differ, is not taken.
 *   - Memory is the caller's, fixed when the record is made: capacity tokens and 1 << draft_lookup_bucket_bits buckets.
 *     A token beyond the capacity is not taken. One thread uses a record. Nothing here can fail.
 */
#ifndef DRAFT_LOOKUP_H
#define DRAFT_LOOKUP_H

#include <stdint.h>

#define DRAFT_LOOKUP_MOST_NGRAM \
	16u /* tokens of the longest suffix looked for */

struct draft_lookup {
	uint32_t *tokens; /* [capacity]: the session's first count tokens */
	uint32_t *buckets; /* [1 << bucket_bits]: the position behind an n-gram's latest occurrence; 0: none */
	uint32_t capacity, count, bucket_bits, ngram_least, ngram_most;
};

/* The buckets of a record for capacity tokens: a power of two, at least the n-grams the tokens file (one of each
 * length a token), within DRAFT_LOOKUP_LEAST_BUCKET_BITS and DRAFT_LOOKUP_MOST_BUCKET_BITS bits (draft_lookup.c). */
uint32_t draft_lookup_bucket_bits(uint32_t capacity, uint32_t ngram_least,
				  uint32_t ngram_most);
/* tokens: capacity of them; buckets: 1 << bucket_bits, all 0. 1 <= ngram_least <= ngram_most <=
 * DRAFT_LOOKUP_MOST_NGRAM (the engine checks it). */
void draft_lookup_begin(struct draft_lookup *lookup, uint32_t *tokens,
			uint32_t capacity, uint32_t *buckets,
			uint32_t bucket_bits, uint32_t ngram_least,
			uint32_t ngram_most);
void draft_lookup_keep(struct draft_lookup *lookup, uint32_t kept);
void draft_lookup_append(struct draft_lookup *lookup, uint32_t token);
uint32_t draft_lookup_propose(const struct draft_lookup *lookup, uint32_t most,
			      uint32_t *proposals_out);

#endif
