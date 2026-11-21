/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* sha256_test: the in-process SHA-256 against the FIPS 180-4 known answers, the kernel's "sha256" (AF_ALG) as an
 * oracle over many lengths, streams fed in pieces, the two-stream form, its refusals, and (with checking hooks) the
 * plain rounds forced at every block of a message. Prints the rate of each form on a large buffer. */
#include <errno.h>
#include <linux/if_alg.h>
#include <stdbool.h>
#include <stdint.h>
#if defined(__x86_64__)
#include <cpuid.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "common/hooks.h"
#include "common/sha256.h"

#define HEX_CHARACTERS 2u
#define HEX_DIGITS 16u
#define TEXT_BYTES (SHA256_DIGEST_BYTES * HEX_CHARACTERS + 1u)
#define MILLION_BYTES 1000000u
#define MILLION_FILL 'a'
#define ORACLE_LENGTHS_MOST \
	300u /* every length 0 .. this, both against the kernel and in two streams */
#define LARGE_BYTES ((64u << 20u) + 13u) /* not a whole number of blocks */
#define RATE_BYTES (256u << 20u)
#define PIECE_ROUNDS 64u
#define PIECE_MOST 777u
#define PLAIN_MESSAGE_BYTES \
	1000u /* fifteen whole blocks and a tail of 40 bytes, fed a block at a time */
#define PLAIN_CALLS_MOST \
	17u /* every reach of the rounds: fifteen blocks, the tail's feed (no whole block), the padded last block */
#define CPUID_FEATURES_LEAF 7u
#define CPUID_SHA_BIT 29u /* EBX of leaf 7: the SHA extensions */
#define FILL_MULTIPLIER 2654435761u
#define FILL_SHIFT 13u
#define RANDOM_MULTIPLIER 6364136223846793005ull
#define RANDOM_INCREMENT 1442695040888963407ull
#define RANDOM_SHIFT 33u
#define NS_PER_SECOND 1000000000.0
#define UNTOUCHED_BYTE 0x5Au
#define SEED_LARGE_FIRST 1u
#define SEED_LARGE_SECOND 2u
#define SEED_RATE_FIRST 3u
#define SEED_RATE_SECOND 4u
#define SEED_PLAIN 5u
#define STREAMS_TWO 2.0 /* the two-stream form digests twice the bytes */

static int failures = 0;

static void expect(bool condition, const char *what)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: %s\n", what);
		failures++;
	}
}

static void text_of(const unsigned char digest[SHA256_DIGEST_BYTES],
		    char text[TEXT_BYTES])
{
	static const char hex[] = "0123456789abcdef";
	for (uint32_t index = 0; index < SHA256_DIGEST_BYTES; index++) {
		text[(size_t)index * HEX_CHARACTERS] =
			hex[digest[index] / HEX_DIGITS];
		text[(size_t)index * HEX_CHARACTERS + 1u] =
			hex[digest[index] % HEX_DIGITS];
	}
	text[TEXT_BYTES - 1u] = '\0';
}

static bool digest_is(const void *bytes, size_t count, const char *expected)
{
	unsigned char digest[SHA256_DIGEST_BYTES];
	char text[TEXT_BYTES];
	if (sha256_of(bytes, count, digest) != STATUS_OK)
		return false;
	text_of(digest, text);
	return strcmp(text, expected) == 0;
}

/* The kernel's digest of bytes; false when this machine offers no AF_ALG sha256 (then the oracle is skipped). */
static bool kernel_digest(const void *bytes, size_t count,
			  unsigned char digest[SHA256_DIGEST_BYTES])
{
	const int socket_descriptor =
		socket(AF_ALG, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
	if (socket_descriptor < 0)
		return false;
	struct sockaddr_alg address = { .salg_family = AF_ALG,
					.salg_type = "hash",
					.salg_name = "sha256" };
	const int operation =
		bind(socket_descriptor, (const struct sockaddr *)&address,
		     sizeof(address)) == 0 ?
			accept4(socket_descriptor, NULL, NULL, SOCK_CLOEXEC) :
			-1;
	(void)close(socket_descriptor);
	if (operation < 0)
		return false;
	const unsigned char *cursor = bytes;
	bool sent = true;
	while (sent && count != 0) {
		const ssize_t amount =
			send(operation, cursor, count, MSG_MORE | MSG_NOSIGNAL);
		if (amount < 0 && errno == EINTR)
			continue;
		sent = amount > 0;
		cursor += sent ? (size_t)amount : 0u;
		count -= sent ? (size_t)amount : 0u;
	}
	const bool read_back = sent &&
			       send(operation, "", 0, MSG_NOSIGNAL) == 0 &&
			       read(operation, digest, SHA256_DIGEST_BYTES) ==
				       (ssize_t)SHA256_DIGEST_BYTES;
	(void)close(operation);
	return read_back;
}

static void fill(unsigned char *bytes, size_t count, uint32_t seed)
{
	for (size_t index = 0; index < count; index++)
		bytes[index] =
			(unsigned char)(((uint32_t)index * FILL_MULTIPLIER +
					 seed) >>
					FILL_SHIFT);
}

static uint64_t random_next(uint64_t *state)
{
	*state = *state * RANDOM_MULTIPLIER + RANDOM_INCREMENT;
	return *state >> RANDOM_SHIFT;
}

static double seconds_now(void)
{
	struct timespec now;
	(void)clock_gettime(CLOCK_MONOTONIC, &now);
	return (double)now.tv_sec + (double)now.tv_nsec / NS_PER_SECOND;
}

static void known_answers(void)
{
	static const struct {
		const char *message, *digest;
	} answers[] = {
		{ "",
		  "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
		{ "abc",
		  "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
		{ "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
		  "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" },
		{ "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
		  "hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
		  "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1" },
	};
	for (size_t index = 0; index < sizeof(answers) / sizeof(answers[0]);
	     index++)
		expect(digest_is(answers[index].message,
				 strlen(answers[index].message),
				 answers[index].digest),
		       "a FIPS 180-4 known answer");
	unsigned char *const million = malloc(MILLION_BYTES);
	expect(million != NULL, "memory for the million-byte answer");
	if (million != NULL) {
		memset(million, MILLION_FILL, MILLION_BYTES);
		expect(digest_is(
			       million, MILLION_BYTES,
			       "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"),
		       "the million-'a' known answer");
		free(million);
	}
}

static void oracle_and_two_streams(unsigned char *buffer, unsigned char *other)
{
	unsigned char kernel[SHA256_DIGEST_BYTES], own[SHA256_DIGEST_BYTES],
		first[SHA256_DIGEST_BYTES], second[SHA256_DIGEST_BYTES],
		other_own[SHA256_DIGEST_BYTES];
	bool kernel_present = kernel_digest("abc", strlen("abc"), kernel);
	if (!kernel_present)
		(void)fprintf(
			stderr,
			"sha256_test: no AF_ALG sha256 on this machine: the kernel oracle is skipped\n");
	for (uint32_t length = 0; length <= ORACLE_LENGTHS_MOST; length++) {
		fill(buffer, length, length);
		fill(other, length, ~length);
		expect(sha256_of(buffer, length, own) == STATUS_OK &&
			       sha256_of(other, length, other_own) == STATUS_OK,
		       "a digest of every short length");
		if (kernel_present)
			expect(kernel_digest(buffer, length, kernel) &&
				       memcmp(kernel, own, sizeof(own)) == 0,
			       "the kernel's digest at every short length");
		expect(sha256_of_two(buffer, other, length, first, second) ==
				       STATUS_OK &&
			       memcmp(first, own, sizeof(own)) == 0 &&
			       memcmp(second, other_own, sizeof(own)) == 0,
		       "two streams give each its own digest at every short length");
	}
	fill(buffer, LARGE_BYTES, SEED_LARGE_FIRST);
	fill(other, LARGE_BYTES, SEED_LARGE_SECOND);
	expect(sha256_of(buffer, LARGE_BYTES, own) == STATUS_OK &&
		       sha256_of(other, LARGE_BYTES, other_own) == STATUS_OK,
	       "a large digest");
	if (kernel_present)
		expect(kernel_digest(buffer, LARGE_BYTES, kernel) &&
			       memcmp(kernel, own, sizeof(own)) == 0,
		       "the kernel's digest of a large buffer");
	expect(sha256_of_two(buffer, other, LARGE_BYTES, first, second) ==
			       STATUS_OK &&
		       memcmp(first, own, sizeof(own)) == 0 &&
		       memcmp(second, other_own, sizeof(own)) == 0,
	       "two large streams");
}

static void pieces(const unsigned char *buffer)
{
	unsigned char whole[SHA256_DIGEST_BYTES], streamed[SHA256_DIGEST_BYTES];
	uint64_t state = 1u;
	for (uint32_t round = 0; round < PIECE_ROUNDS; round++) {
		const size_t length = (size_t)random_next(&state) %
				      (LARGE_BYTES / PIECE_ROUNDS);
		struct sha256_stream stream;
		sha256_begin(&stream);
		bool fed = true;
		for (size_t done = 0; done < length;) {
			size_t piece = (size_t)random_next(&state) % PIECE_MOST;
			piece = piece < length - done ? piece : length - done;
			fed = fed && sha256_feed(&stream, buffer + done,
						 piece) == STATUS_OK;
			done += piece;
		}
		expect(fed && sha256_finish(&stream, streamed) == STATUS_OK &&
			       sha256_of(buffer, length, whole) == STATUS_OK &&
			       memcmp(whole, streamed, sizeof(whole)) == 0,
		       "a stream fed in pieces gives the one-shot digest");
	}
}

static void refusals(void)
{
	unsigned char digest[SHA256_DIGEST_BYTES];
	memset(digest, UNTOUCHED_BYTE, sizeof(digest));
	expect(sha256_of(NULL, 1, digest) == STATUS_INVALID_ARGUMENT,
	       "NULL bytes with a count are refused");
	expect(sha256_of("a", 1, NULL) == STATUS_INVALID_ARGUMENT,
	       "a NULL digest is refused");
	expect(sha256_of_two("a", NULL, 1, digest, digest) ==
		       STATUS_INVALID_ARGUMENT,
	       "a NULL second stream is refused");
	expect(sha256_feed(NULL, "a", 1) == STATUS_INVALID_ARGUMENT,
	       "a NULL stream is refused");
	bool untouched = true;
	for (uint32_t index = 0; index < SHA256_DIGEST_BYTES; index++)
		untouched = untouched && digest[index] == UNTOUCHED_BYTE;
	expect(untouched, "a refusal leaves the digest untouched");
	expect(digest_is(
		       NULL, 0,
		       "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
	       "NULL bytes with no count are the empty message");
}

#if CHECKING_HOOKS_ENABLED
/* The message fed a block at a time, so that each block is one compression (one reach of the fault point). */
static bool blockwise_digest(const unsigned char *buffer, size_t bytes,
			     unsigned char digest[SHA256_DIGEST_BYTES])
{
	struct sha256_stream stream;
	sha256_begin(&stream);
	bool fed = true;
	for (size_t offset = 0; fed && offset < bytes;
	     offset += SHA256_BLOCK_BYTES)
		fed = sha256_feed(&stream, buffer + offset,
				  bytes - offset < SHA256_BLOCK_BYTES ?
					  bytes - offset :
					  SHA256_BLOCK_BYTES) == STATUS_OK;
	return fed && sha256_finish(&stream, digest) == STATUS_OK;
}
#endif

/* Whether the processor reports the SHA extensions (CPUID leaf 7, EBX bit 29); false off x86-64. */
static bool processor_has_sha(void)
{
#if defined(__x86_64__)
	unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
	return __get_cpuid_count(CPUID_FEATURES_LEAF, 0, &eax, &ebx, &ecx,
				 &edx) != 0 &&
	       (ebx >> CPUID_SHA_BIT & 1u) != 0;
#else
	return false;
#endif
}

/* The path that ran: a build with the extensions on a processor that has them uses them (review finding 18). */
static void sha_path(void)
{
#if defined(__SHA__) && defined(__SSE4_1__) && defined(__SSSE3__)
	expect(sha256_uses_extensions() == processor_has_sha(),
	       "the extension rounds run exactly where the processor has them");
#else
	expect(!sha256_uses_extensions(),
	       "a build without the extensions runs the plain rounds");
#endif
}

static void plain_rounds(const unsigned char *buffer)
{
#if CHECKING_HOOKS_ENABLED
	unsigned char expected[SHA256_DIGEST_BYTES],
		forced[SHA256_DIGEST_BYTES];
	expect(sha256_of(buffer, PLAIN_MESSAGE_BYTES, expected) == STATUS_OK,
	       "the reference of the forced runs");
	/* each compression in turn takes the plain rounds; the fault point is reached at least that often */
	for (uint64_t call = 1; call <= PLAIN_CALLS_MOST; call++) {
		expect(fault_arm("sha256_plain_rounds", call),
		       "the plain rounds can be armed");
		expect(blockwise_digest(buffer, PLAIN_MESSAGE_BYTES, forced) &&
			       memcmp(forced, expected, sizeof(forced)) == 0,
		       "the plain rounds at one block give the same digest");
		const uint64_t reached = fault_disarm();
		expect(reached >= call || !sha256_uses_extensions(),
		       "the armed block is reached: the plain rounds ran on it");
	}
	expect(fault_arm("sha256_plain_rounds", 1),
	       "the plain rounds can be armed");
	expect(sha256_of(buffer, 1, forced) == STATUS_OK,
	       "a one-byte message under the armed fault");
	expect(fault_disarm() != 0 || !sha256_uses_extensions(),
	       "the fault point is reached");
#else
	(void)buffer;
#endif
}

static void rates(unsigned char *buffer, unsigned char *other)
{
	unsigned char first[SHA256_DIGEST_BYTES], second[SHA256_DIGEST_BYTES];
	fill(buffer, RATE_BYTES, SEED_RATE_FIRST);
	fill(other, RATE_BYTES, SEED_RATE_SECOND);
	double started = seconds_now();
	expect(sha256_of(buffer, RATE_BYTES, first) == STATUS_OK,
	       "the rate's one stream");
	const double one = seconds_now() - started;
	started = seconds_now();
	expect(sha256_of_two(buffer, other, RATE_BYTES, first, second) ==
		       STATUS_OK,
	       "the rate's two streams");
	const double two = seconds_now() - started;
	unsigned char kernel[SHA256_DIGEST_BYTES];
	started = seconds_now();
	const bool kernel_present = kernel_digest(buffer, RATE_BYTES, kernel);
	const double kernel_seconds = seconds_now() - started;
	(void)printf(
		"sha256_test: extensions %s; one stream %.3f GB/s, two streams %.3f GB/s (of input), kernel %s %.3f "
		"GB/s\n",
		sha256_uses_extensions() ? "used" : "absent",
		(double)RATE_BYTES / one / NS_PER_SECOND,
		STREAMS_TWO * (double)RATE_BYTES / two / NS_PER_SECOND,
		kernel_present ? "AF_ALG" : "absent",
		kernel_present ?
			(double)RATE_BYTES / kernel_seconds / NS_PER_SECOND :
			0.0);
}

int main(void)
{
	const size_t bytes = RATE_BYTES > LARGE_BYTES ? RATE_BYTES :
							LARGE_BYTES;
	unsigned char *const buffer = malloc(bytes), *const other =
							     malloc(bytes);
	if (buffer == NULL || other == NULL) {
		(void)fprintf(stderr, "sha256_test: no memory\n");
		free(buffer);
		free(other);
		return EXIT_FAILURE;
	}
	known_answers();
	oracle_and_two_streams(buffer, other);
	pieces(buffer);
	refusals();
	fill(buffer, PLAIN_MESSAGE_BYTES, SEED_PLAIN);
	sha_path();
	plain_rounds(buffer);
	rates(buffer, other);
	free(buffer);
	free(other);
	if (failures != 0) {
		(void)fprintf(stderr, "sha256_test: %d failures\n", failures);
		return EXIT_FAILURE;
	}
	(void)printf("sha256_test: passed\n");
	return EXIT_SUCCESS;
}
