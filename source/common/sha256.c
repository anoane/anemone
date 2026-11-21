/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * sha256.c: SHA-256 (FIPS 180-4), contract in sha256.h.
 *
 * Plain C rounds always; the x86-64 SHA extensions (SHA-NI: sha256rnds2 does two rounds, sha256msg1/msg2 the message
 * schedule) when this build and the processor have them, chosen at the first call. There is no ARMv8 path: aarch64
 * runs the plain rounds. The two-stream form runs two messages'
 * rounds side by side: sha256rnds2 has a latency of several cycles and a throughput of one a cycle, so a second
 * stream fills the wait of the first.
 */
#include "sha256.h"

#include <stdatomic.h>
#include <string.h>

#include "hooks.h"

#if defined(__SHA__) && defined(__SSE4_1__) && defined(__SSSE3__)
#include <immintrin.h>
#define EXTENSIONS_BUILT 1
#else
#define EXTENSIONS_BUILT 0
#endif

#define WORD_BITS 32u
#define BYTE_BITS 8u
#define WORD_BYTES 4u
#define MESSAGE_WORDS 16u /* the words of one block */
#define SCHEDULE_WORDS 64u /* one per round */
#define ROUNDS 64u
#define LENGTH_BYTES 8u /* the message's bit count closes the last block */
#define PADDING_MARK 0x80u /* the one bit after the message */
#define BIG_SIGMA0_FIRST 2u
#define BIG_SIGMA0_SECOND 13u
#define BIG_SIGMA0_THIRD 22u
#define BIG_SIGMA1_FIRST 6u
#define BIG_SIGMA1_SECOND 11u
#define BIG_SIGMA1_THIRD 25u
#define SMALL_SIGMA0_FIRST 7u
#define SMALL_SIGMA0_SECOND 18u
#define SMALL_SIGMA0_SHIFT 3u
#define SMALL_SIGMA1_FIRST 17u
#define SMALL_SIGMA1_SECOND 19u
#define SMALL_SIGMA1_SHIFT 10u
#define LAG_SMALL_SIGMA0 \
	15u /* W[t] = s1(W[t-2]) + W[t-7] + s0(W[t-15]) + W[t-16] */
#define LAG_SMALL_SIGMA1 2u
#define LAG_ADDED 7u
#define LAG_OLDEST 16u

enum state_word {
	WORD_A,
	WORD_B,
	WORD_C,
	WORD_D,
	WORD_E,
	WORD_F,
	WORD_G,
	WORD_H
};

static const uint32_t round_constants[ROUNDS] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
	0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
	0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
	0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
	0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
	0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
	0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
	0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};
static const uint32_t initial_state[SHA256_STATE_WORDS] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

static uint32_t rotate_right(uint32_t value, uint32_t bits)
{
	return value >> bits | value << (WORD_BITS - bits);
}

static uint32_t big_endian_word(const unsigned char *bytes)
{
	uint32_t value = 0;
	for (uint32_t index = 0; index < WORD_BYTES; index++)
		value = value << BYTE_BITS | bytes[index];
	return value;
}

static void blocks_plain(uint32_t state[SHA256_STATE_WORDS],
			 const unsigned char *data, size_t blocks)
{
	for (size_t block = 0; block < blocks; block++) {
		const unsigned char *const bytes =
			data + block * SHA256_BLOCK_BYTES;
		uint32_t schedule[SCHEDULE_WORDS];
		for (uint32_t word = 0; word < MESSAGE_WORDS; word++)
			schedule[word] = big_endian_word(
				bytes + (size_t)word * WORD_BYTES);
		for (uint32_t word = MESSAGE_WORDS; word < SCHEDULE_WORDS;
		     word++) {
			const uint32_t early = schedule[word - LAG_SMALL_SIGMA0],
				       late = schedule[word - LAG_SMALL_SIGMA1];
			schedule[word] =
				(rotate_right(late, SMALL_SIGMA1_FIRST) ^
				 rotate_right(late, SMALL_SIGMA1_SECOND) ^
				 late >> SMALL_SIGMA1_SHIFT) +
				schedule[word - LAG_ADDED] +
				(rotate_right(early, SMALL_SIGMA0_FIRST) ^
				 rotate_right(early, SMALL_SIGMA0_SECOND) ^
				 early >> SMALL_SIGMA0_SHIFT) +
				schedule[word - LAG_OLDEST];
		}
		uint32_t work[SHA256_STATE_WORDS];
		memcpy(work, state, sizeof(work));
		for (uint32_t round = 0; round < ROUNDS; round++) {
			const uint32_t e = work[WORD_E], a = work[WORD_A];
			const uint32_t first =
				work[WORD_H] +
				(rotate_right(e, BIG_SIGMA1_FIRST) ^
				 rotate_right(e, BIG_SIGMA1_SECOND) ^
				 rotate_right(e, BIG_SIGMA1_THIRD)) +
				((e & work[WORD_F]) ^ (~e & work[WORD_G])) +
				round_constants[round] + schedule[round];
			const uint32_t second =
				(rotate_right(a, BIG_SIGMA0_FIRST) ^
				 rotate_right(a, BIG_SIGMA0_SECOND) ^
				 rotate_right(a, BIG_SIGMA0_THIRD)) +
				((a & work[WORD_B]) ^ (a & work[WORD_C]) ^
				 (work[WORD_B] & work[WORD_C]));
			memmove(work + WORD_B, work + WORD_A,
				(SHA256_STATE_WORDS - 1u) * sizeof(*work));
			work[WORD_E] += first;
			work[WORD_A] = first + second;
		}
		for (uint32_t word = 0; word < SHA256_STATE_WORDS; word++)
			state[word] += work[word];
	}
}

#if EXTENSIONS_BUILT
#define LANE_BYTES 16u
#define LANE_WORDS 4u
#define LANE_GROUPS 4u /* a block is four lanes of four words */
#define ROUND_GROUPS 16u /* four rounds per group */
#define SHUFFLE_SWAP_PAIRS 0xB1 /* a b c d -> b a d c */
#define SHUFFLE_REVERSE 0x1B /* a b c d -> d c b a */
#define SHUFFLE_HIGH_PAIR \
	0x0E /* the group's second two words, for the second two rounds */
#define BLEND_HIGH_HALF 0xF0
#define ALIGN_HALF 8
#define ALIGN_WORD 4
#define BYTE_ORDER_HIGH \
	0x0c0d0e0f08090a0bll /* each word of a lane read big-endian */
#define BYTE_ORDER_LOW 0x0405060700010203ll
#define NEXT_LANE 1u
#define BEFORE_LANE 2u /* (group + 2) % 4: two groups back */
#define PREVIOUS_LANE 3u /* (group + 3) % 4: one group back */

struct lanes {
	__m128i abef, cdgh;
};

static __m128i lane_load(const void *bytes)
{
	return _mm_loadu_si128((const __m128i_u *)bytes);
}

static struct lanes lanes_of(const uint32_t state[SHA256_STATE_WORDS])
{
	const __m128i low =
		_mm_shuffle_epi32(lane_load(state), SHUFFLE_SWAP_PAIRS);
	const __m128i high = _mm_shuffle_epi32(lane_load(state + LANE_WORDS),
					       SHUFFLE_REVERSE);
	return (struct lanes){ .abef = _mm_alignr_epi8(low, high, ALIGN_HALF),
			       .cdgh = _mm_blend_epi16(high, low,
						       BLEND_HIGH_HALF) };
}

static void lanes_store(uint32_t state[SHA256_STATE_WORDS], struct lanes lanes)
{
	const __m128i feba = _mm_shuffle_epi32(lanes.abef, SHUFFLE_REVERSE);
	const __m128i dchg = _mm_shuffle_epi32(lanes.cdgh, SHUFFLE_SWAP_PAIRS);
	_mm_storeu_si128((__m128i_u *)(void *)state,
			 _mm_blend_epi16(feba, dchg, BLEND_HIGH_HALF));
	_mm_storeu_si128((__m128i_u *)(void *)(state + LANE_WORDS),
			 _mm_alignr_epi8(dchg, feba, ALIGN_HALF));
}

/* One group of four rounds: the group's message words (from the schedule after the first four groups), then two
 * sha256rnds2. words holds the last four groups of the schedule, the current one at group % 4. */
static void group_rounds(struct lanes *lanes, __m128i words[LANE_GROUPS],
			 uint32_t group)
{
	__m128i *const current = &words[group % LANE_GROUPS];
	if (group >= LANE_GROUPS) {
		const __m128i previous =
			words[(group + PREVIOUS_LANE) % LANE_GROUPS];
		*current = _mm_sha256msg2_epu32(
			_mm_add_epi32(
				_mm_sha256msg1_epu32(*current,
						     words[(group + NEXT_LANE) %
							   LANE_GROUPS]),
				_mm_alignr_epi8(previous,
						words[(group + BEFORE_LANE) %
						      LANE_GROUPS],
						ALIGN_WORD)),
			previous);
	}
	const __m128i sum = _mm_add_epi32(
		*current, lane_load(&round_constants[group * LANE_WORDS]));
	lanes->cdgh = _mm_sha256rnds2_epu32(lanes->cdgh, lanes->abef, sum);
	lanes->abef = _mm_sha256rnds2_epu32(
		lanes->abef, lanes->cdgh,
		_mm_shuffle_epi32(sum, SHUFFLE_HIGH_PAIR));
}

static void words_load(__m128i words[LANE_GROUPS], const unsigned char *bytes)
{
	const __m128i order = _mm_set_epi64x(BYTE_ORDER_HIGH, BYTE_ORDER_LOW);
	for (uint32_t lane = 0; lane < LANE_GROUPS; lane++)
		words[lane] = _mm_shuffle_epi8(
			lane_load(bytes + lane * LANE_BYTES), order);
}

static void blocks_extensions(uint32_t state[SHA256_STATE_WORDS],
			      const unsigned char *data, size_t blocks)
{
	struct lanes lanes = lanes_of(state);
	for (size_t block = 0; block < blocks; block++) {
		const struct lanes saved = lanes;
		__m128i words[LANE_GROUPS];
		words_load(words, data + block * SHA256_BLOCK_BYTES);
		for (uint32_t group = 0; group < ROUND_GROUPS; group++)
			group_rounds(&lanes, words, group);
		lanes.abef = _mm_add_epi32(lanes.abef, saved.abef);
		lanes.cdgh = _mm_add_epi32(lanes.cdgh, saved.cdgh);
	}
	lanes_store(state, lanes);
}

static void blocks_extensions_two(uint32_t first_state[SHA256_STATE_WORDS],
				  uint32_t second_state[SHA256_STATE_WORDS],
				  const unsigned char *first,
				  const unsigned char *second, size_t blocks)
{
	struct lanes one = lanes_of(first_state), two = lanes_of(second_state);
	for (size_t block = 0; block < blocks; block++) {
		const struct lanes saved_one = one, saved_two = two;
		__m128i words_one[LANE_GROUPS], words_two[LANE_GROUPS];
		words_load(words_one, first + block * SHA256_BLOCK_BYTES);
		words_load(words_two, second + block * SHA256_BLOCK_BYTES);
		for (uint32_t group = 0; group < ROUND_GROUPS; group++) {
			group_rounds(&one, words_one, group);
			group_rounds(&two, words_two, group);
		}
		one.abef = _mm_add_epi32(one.abef, saved_one.abef);
		one.cdgh = _mm_add_epi32(one.cdgh, saved_one.cdgh);
		two.abef = _mm_add_epi32(two.abef, saved_two.abef);
		two.cdgh = _mm_add_epi32(two.cdgh, saved_two.cdgh);
	}
	lanes_store(first_state, one);
	lanes_store(second_state, two);
}
#endif

/* -1 not known yet, 0 plain, 1 the extensions; racing first calls both store the same answer. */
static _Atomic int extensions_known = -1;

bool sha256_uses_extensions(void)
{
	int known =
		atomic_load_explicit(&extensions_known, memory_order_relaxed);
	if (known < 0) {
#if EXTENSIONS_BUILT
		__builtin_cpu_init();
		known = __builtin_cpu_supports("sha") &&
					__builtin_cpu_supports("sse4.1") &&
					__builtin_cpu_supports("ssse3") ?
				1 :
				0;
#else
		known = 0;
#endif
		atomic_store_explicit(&extensions_known, known,
				      memory_order_relaxed);
	}
	return known != 0;
}

static void blocks(uint32_t state[SHA256_STATE_WORDS],
		   const unsigned char *data, size_t count)
{
#if EXTENSIONS_BUILT
	/* a test arms sha256_plain_rounds to take the plain rounds for one call: both forms share the state */
	if (sha256_uses_extensions() && !FAULT_POINT(sha256_plain_rounds)) {
		blocks_extensions(state, data, count);
		return;
	}
#endif
	blocks_plain(state, data, count);
}

void sha256_begin(struct sha256_stream *stream)
{
	memcpy(stream->state, initial_state, sizeof(stream->state));
	stream->byte_count = 0;
	stream->pending_bytes = 0;
}

result_status sha256_feed(struct sha256_stream *stream, const void *bytes,
			  size_t count)
{
	if (stream == NULL || (bytes == NULL && count != 0) ||
	    stream->pending_bytes >= SHA256_BLOCK_BYTES)
		return STATUS_INVALID_ARGUMENT;
	if (count == 0)
		return STATUS_OK; /* nothing to copy: no pointer arithmetic on what may be NULL */
	const unsigned char *input = bytes;
	stream->byte_count += count;
	if (stream->pending_bytes != 0) {
		const size_t room = SHA256_BLOCK_BYTES - stream->pending_bytes;
		const size_t taken = count < room ? count : room;
		memcpy(stream->pending + stream->pending_bytes, input, taken);
		stream->pending_bytes += (uint32_t)taken;
		input += taken;
		count -= taken;
		if (stream->pending_bytes < SHA256_BLOCK_BYTES)
			return STATUS_OK;
		blocks(stream->state, stream->pending, 1);
		stream->pending_bytes = 0;
	}
	const size_t whole = count / SHA256_BLOCK_BYTES;
	blocks(stream->state, input, whole);
	input += whole * SHA256_BLOCK_BYTES;
	count -= whole * SHA256_BLOCK_BYTES;
	memcpy(stream->pending, input, count);
	stream->pending_bytes = (uint32_t)count;
	return STATUS_OK;
}

result_status sha256_finish(struct sha256_stream *stream,
			    unsigned char digest[SHA256_DIGEST_BYTES])
{
	if (stream == NULL || digest == NULL ||
	    stream->pending_bytes >= SHA256_BLOCK_BYTES)
		return STATUS_INVALID_ARGUMENT;
	const uint64_t bits = stream->byte_count * BYTE_BITS;
	uint32_t used = stream->pending_bytes;
	stream->pending[used++] = PADDING_MARK;
	if (used > SHA256_BLOCK_BYTES - LENGTH_BYTES) {
		memset(stream->pending + used, 0, SHA256_BLOCK_BYTES - used);
		blocks(stream->state, stream->pending, 1);
		used = 0;
	}
	memset(stream->pending + used, 0,
	       SHA256_BLOCK_BYTES - LENGTH_BYTES - used);
	for (uint32_t index = 0; index < LENGTH_BYTES; index++)
		stream->pending[SHA256_BLOCK_BYTES - 1u - index] =
			(unsigned char)(bits >> (index * BYTE_BITS));
	blocks(stream->state, stream->pending, 1);
	for (uint32_t word = 0; word < SHA256_STATE_WORDS; word++)
		for (uint32_t index = 0; index < WORD_BYTES; index++)
			digest[word * WORD_BYTES + index] =
				(unsigned char)(stream->state[word] >>
						((WORD_BYTES - 1u - index) *
						 BYTE_BITS));
	stream->pending_bytes = 0;
	return STATUS_OK;
}

result_status sha256_of(const void *bytes, size_t count,
			unsigned char digest[SHA256_DIGEST_BYTES])
{
	struct sha256_stream stream;
	sha256_begin(&stream);
	const result_status status = sha256_feed(&stream, bytes, count);
	return status == STATUS_OK ? sha256_finish(&stream, digest) : status;
}

result_status sha256_of_two(const void *first, const void *second, size_t count,
			    unsigned char first_digest[SHA256_DIGEST_BYTES],
			    unsigned char second_digest[SHA256_DIGEST_BYTES])
{
	if (((first == NULL || second == NULL) && count != 0) ||
	    first_digest == NULL || second_digest == NULL)
		return STATUS_INVALID_ARGUMENT;
	struct sha256_stream one, two;
	sha256_begin(&one);
	sha256_begin(&two);
	size_t done = 0;
#if EXTENSIONS_BUILT
	if (sha256_uses_extensions()) {
		const size_t whole = count / SHA256_BLOCK_BYTES;
		blocks_extensions_two(one.state, two.state, first, second,
				      whole);
		done = whole * SHA256_BLOCK_BYTES;
		one.byte_count = two.byte_count = done;
	}
#endif
	const unsigned char *const first_bytes = first, *const second_bytes =
								second;
	result_status status = sha256_feed(
		&one, count != 0 ? first_bytes + done : first, count - done);
	status = status == STATUS_OK ?
			 sha256_feed(&two,
				     count != 0 ? second_bytes + done : second,
				     count - done) :
			 status;
	unsigned char first_result[SHA256_DIGEST_BYTES],
		second_result[SHA256_DIGEST_BYTES];
	status = status == STATUS_OK ? sha256_finish(&one, first_result) :
				       status;
	status = status == STATUS_OK ? sha256_finish(&two, second_result) :
				       status;
	if (status == STATUS_OK) {
		memcpy(first_digest, first_result, sizeof(first_result));
		memcpy(second_digest, second_result, sizeof(second_result));
	}
	return status;
}
