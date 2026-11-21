/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * sha256_rate.c: how fast the engine's SHA-256 (sha256.c) hashes the copier's chunks on this host, one stream a
 * call (sha256_of, as the copier hashes each chunk today) against two streams side by side (sha256_of_two), on 1 to N
 * threads at once.
 *
 *   sha256_rate [--threads=1,2,4,8,16,32] [--seconds=3] [--chunk_mib=64]
 *
 * Each thread owns two chunks of distinct bytes, written once and larger than the caches, so the rounds read them from
 * memory as a copy's freshly read chunk is. For the seconds, `one` hashes the chunks one at a time and `two` hashes
 * both at once. Every thread starts together and the rate is the bytes over the slowest thread's time. One JSON line a
 * (form, threads): bytes, seconds, GB/s in all and a thread, and whether the SHA extensions ran. Each pass's digests
 * are checked against the first `one` digests, so a mismatch fails the run (exit 1).
 */
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common/sha256.h"

#define NS_SECOND 1000000000ull
#define DECIMAL_GB 1e9
#define MIB_BYTES (1ull << 20u)
#define CHUNK_MIB_DEFAULT 64u
#define CHUNK_MIB_MOST 1024u
#define SECONDS_DEFAULT 3u
#define SECONDS_MOST 600u
#define THREADS_MOST 192u
#define THREAD_COUNTS_MOST 16u
#define DECIMAL_BASE 10
#define CHUNKS_A_THREAD 2u
#define PATTERN_MULTIPLIER 0x9E3779B97F4A7C15ull
#define PATTERN_SHIFT 29u
static const uint32_t default_threads[] = { 1u, 2u, 4u, 8u, 16u, 32u };

enum form { FORM_ONE, FORM_TWO, FORMS };
static const char *const form_names[FORMS] = { "one", "two" };

struct rate_thread {
	unsigned char *chunks[CHUNKS_A_THREAD];
	unsigned char expected[CHUNKS_A_THREAD][SHA256_DIGEST_BYTES];
	size_t chunk_bytes;
	enum form form;
	uint64_t seconds, bytes, elapsed_ns;
	_Atomic int *
		start; /* 0 wait, 1 go, -1 stop: every thread of a pass starts together */
	bool failed;
	pthread_t thread;
};

static uint64_t now_ns(void)
{
	struct timespec time;
	return clock_gettime(CLOCK_MONOTONIC, &time) == 0 ?
		       (uint64_t)time.tv_sec * NS_SECOND +
			       (uint64_t)time.tv_nsec :
		       0;
}

/* Distinct bytes in every chunk of every thread (a counter mixed by a multiply), so no two chunks hash alike. */
static void pattern_fill(unsigned char *bytes, size_t count, uint64_t seed)
{
	for (size_t index = 0; index < count / sizeof(uint64_t); index++) {
		const uint64_t mixed = (seed + index) * PATTERN_MULTIPLIER,
			       word = mixed ^ (mixed >> PATTERN_SHIFT);
		memcpy(bytes + index * sizeof(word), &word, sizeof(word));
	}
}

static bool
digests_match(const struct rate_thread *thread,
	      unsigned char got[CHUNKS_A_THREAD][SHA256_DIGEST_BYTES])
{
	for (uint32_t chunk = 0; chunk < CHUNKS_A_THREAD; chunk++)
		if (memcmp(got[chunk], thread->expected[chunk],
			   SHA256_DIGEST_BYTES) != 0)
			return false;
	return true;
}

static void *rate_run(void *argument)
{
	struct rate_thread *const thread = argument;
	unsigned char got[CHUNKS_A_THREAD][SHA256_DIGEST_BYTES];
	int start = 0;
	while ((start = atomic_load(thread->start)) == 0)
		(void)sched_yield();
	if (start < 0)
		return NULL;
	const uint64_t started = now_ns(),
		       deadline = started + thread->seconds * NS_SECOND;
	uint64_t ended = started;
	while (!thread->failed && ended < deadline) {
		bool hashed = true;
		if (thread->form == FORM_ONE)
			for (uint32_t chunk = 0; chunk < CHUNKS_A_THREAD;
			     chunk++)
				hashed = hashed &&
					 sha256_of(thread->chunks[chunk],
						   thread->chunk_bytes,
						   got[chunk]) == STATUS_OK;
		else
			hashed = sha256_of_two(thread->chunks[0],
					       thread->chunks[1],
					       thread->chunk_bytes, got[0],
					       got[1]) == STATUS_OK;
		thread->failed = !hashed || !digests_match(thread, got);
		thread->bytes += CHUNKS_A_THREAD * thread->chunk_bytes;
		ended = now_ns();
	}
	thread->elapsed_ns = ended - started;
	return NULL;
}

static bool number_parse(const char *text, uint32_t most, uint32_t *output)
{
	char *after = NULL;
	errno = 0;
	const unsigned long value = strtoul(text, &after, DECIMAL_BASE);
	if (errno != 0 || after == text || (*after != '\0' && *after != ',') ||
	    value == 0 || value > most)
		return false;
	*output = (uint32_t)value;
	return true;
}

static uint32_t threads_parse(const char *text,
			      uint32_t output[THREAD_COUNTS_MOST])
{
	uint32_t count = 0;
	for (const char *cursor = text; *cursor != '\0';) {
		if (count == THREAD_COUNTS_MOST ||
		    !number_parse(cursor, THREADS_MOST, &output[count]))
			return 0;
		count++;
		const char *const comma = strchr(cursor, ',');
		cursor = comma != NULL ? comma + 1 : cursor + strlen(cursor);
	}
	return count;
}

/* One pass: count threads of one form, started together. False on a failed hash, a mismatch or no thread. */
static bool rate_pass(struct rate_thread *threads, uint32_t count,
		      enum form form, uint64_t seconds)
{
	_Atomic int start = 0;
	uint32_t launched = 0;
	for (; launched < count; launched++) {
		threads[launched].form = form;
		threads[launched].seconds = seconds;
		threads[launched].bytes = 0;
		threads[launched].start = &start;
		if (pthread_create(&threads[launched].thread, NULL, rate_run,
				   &threads[launched]) != 0)
			break;
	}
	bool passed = launched == count;
	atomic_store(&start, passed ? 1 : -1);
	uint64_t bytes = 0, slowest = 0;
	for (uint32_t index = 0; index < launched; index++) {
		passed = pthread_join(threads[index].thread, NULL) == 0 &&
			 !threads[index].failed && passed;
		bytes += threads[index].bytes;
		if (threads[index].elapsed_ns > slowest)
			slowest = threads[index].elapsed_ns;
	}
	if (!passed || slowest == 0)
		return false;
	const double seconds_taken = (double)slowest / (double)NS_SECOND,
		     rate = (double)bytes / DECIMAL_GB / seconds_taken;
	(void)printf(
		"{\"schema\":1,\"kind\":\"sha256_rate\",\"form\":\"%s\",\"threads\":%" PRIu32
		",\"chunk_bytes\":%zu,\"bytes\":%" PRIu64
		",\"seconds\":%.3f,\"gb_per_second\":%.3f,\"gb_per_second_a_thread\":%.3f,\"extensions\":%s}\n",
		form_names[form], count, threads[0].chunk_bytes, bytes,
		seconds_taken, rate, rate / count,
		sha256_uses_extensions() ? "true" : "false");
	(void)fflush(stdout);
	return true;
}

int main(int argument_count, char **arguments)
{
	uint32_t counts[THREAD_COUNTS_MOST], count_total = 0,
					     seconds = SECONDS_DEFAULT,
					     chunk_mib = CHUNK_MIB_DEFAULT;
	for (uint32_t index = 0;
	     index < sizeof(default_threads) / sizeof(default_threads[0]);
	     index++)
		counts[count_total++] = default_threads[index];
	bool valid = true;
	for (int index = 1; valid && index < argument_count; index++) {
		const char *const argument = arguments[index];
		if (strncmp(argument, "--threads=", strlen("--threads=")) ==
		    0) {
			count_total = threads_parse(
				argument + strlen("--threads="), counts);
			valid = count_total != 0;
		} else if (strncmp(argument,
				   "--seconds=", strlen("--seconds=")) == 0)
			valid = number_parse(argument + strlen("--seconds="),
					     SECONDS_MOST, &seconds);
		else if (strncmp(argument,
				 "--chunk_mib=", strlen("--chunk_mib=")) == 0)
			valid = number_parse(argument + strlen("--chunk_mib="),
					     CHUNK_MIB_MOST, &chunk_mib);
		else
			valid = false;
	}
	if (!valid) {
		(void)fputs(
			"usage: sha256_rate [--threads=1,2,4,8,16,32] [--seconds=3] [--chunk_mib=64]\n",
			stderr);
		return EXIT_FAILURE;
	}
	uint32_t most = 0;
	for (uint32_t index = 0; index < count_total; index++)
		most = counts[index] > most ? counts[index] : most;
	const size_t chunk_bytes = (size_t)chunk_mib * MIB_BYTES;
	struct rate_thread *const threads = calloc(most, sizeof(*threads));
	bool passed = threads != NULL;
	for (uint32_t index = 0; passed && index < most; index++) {
		threads[index].chunk_bytes = chunk_bytes;
		for (uint32_t chunk = 0; passed && chunk < CHUNKS_A_THREAD;
		     chunk++) {
			unsigned char *const bytes = malloc(chunk_bytes);
			threads[index].chunks[chunk] = bytes;
			passed = bytes != NULL;
			if (passed) {
				pattern_fill(
					bytes, chunk_bytes,
					((uint64_t)index * CHUNKS_A_THREAD +
					 chunk) *
						chunk_bytes);
				passed = sha256_of(bytes, chunk_bytes,
						   threads[index]
							   .expected[chunk]) ==
					 STATUS_OK;
			}
		}
	}
	for (uint32_t index = 0; passed && index < count_total; index++)
		for (uint32_t form = 0; passed && form < FORMS; form++)
			passed = rate_pass(threads, counts[index],
					   (enum form)form, seconds);
	for (uint32_t index = 0; threads != NULL && index < most; index++)
		for (uint32_t chunk = 0; chunk < CHUNKS_A_THREAD; chunk++)
			free(threads[index].chunks[chunk]);
	free(threads);
	if (!passed)
		(void)fputs(
			"sha256_rate: FAILED (an allocation, a thread, a hash or a digest)\n",
			stderr);
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
