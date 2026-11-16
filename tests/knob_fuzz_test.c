/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* CPU-only argv fuzzing. Each complete parse inherits untouched startup state.
 * Only the parser is called: no model, file credential, socket or card opens. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include "step_driver.h"
#include "commands.h"

#define TEST_FUZZ_INPUT_BYTES 4096u
#define TEST_FUZZ_ARGUMENTS 32u
#define TEST_FUZZ_OPTION_BYTES 64u
#define TEST_FUZZ_NUMBERS 8u
#define TEST_FUZZ_CHILD_SECONDS 2
#define TEST_FUZZ_POLL_NS 1000000L
#define TEST_FUZZ_PATH_BYTES 4096u

/* libFuzzer's entry point: the symbol its main calls is LLVMFuzzerTestOneInput */
int knob_fuzz_one_input(const uint8_t *data,
			size_t size) __asm__("LLVMFuzzerTestOneInput");
int knob_fuzz_one_input(const uint8_t *data, size_t size)
{
	if (size > TEST_FUZZ_INPUT_BYTES) {
		return 0;
	}
	char storage[TEST_FUZZ_INPUT_BYTES + 1u];
	memcpy(storage, data, size);
	storage[size] = '\0';
	char program_name[] = "anemone", command_name[] = "serve";
	char *arguments[TEST_FUZZ_ARGUMENTS + 1u] = { program_name,
						      command_name };
	int count = ARGUMENTS_BEFORE_THE_OPTIONS;
	for (size_t index = 0;
	     index < size && (unsigned)count < TEST_FUZZ_ARGUMENTS;) {
		arguments[count] = storage + index;
		count++;
		while (index < size && storage[index] != '\0' &&
		       storage[index] != '\n') {
			index++;
		}
		storage[index] = '\0';
		index++;
	}
	/* Direct primitive calls keep coverage feedback in the parent process. */
	uint32_t numbers[TEST_FUZZ_NUMBERS] = { 0 }, number_count = 0;
	float fraction = 0.0f;
	uint32_t given = 0;
	const bool number_read = take_option(storage, "--rows=", 0, UINT32_MAX,
					     TEST_FUZZ_NUMBERS, false, numbers,
					     &number_count);
	const bool fraction_read = take_fraction(
		storage, "--temperature=", 0.0f, 1.0f, &fraction, &given);
	if ((number_read && number_count > TEST_FUZZ_NUMBERS) ||
	    (fraction_read && !(fraction >= 0.0f && fraction <= 1.0f))) {
		abort();
	}
	struct axis_arguments_record expanded = { 0 };
	const bool expanded_ok = axis_expand(count, arguments, &expanded);
	axis_arguments_free(&expanded);
	(void)expanded_ok;

	const pid_t child = fork();
	if (child < 0) {
		abort();
	}
	if (child == 0) {
		/* Fuzzed paths must never cause any file to be read: the child can open no descriptor at all. */
		const struct rlimit no_descriptors = { 0 };
		if (setrlimit(RLIMIT_NOFILE, &no_descriptors) != 0) {
			_exit(1);
		}
		struct axis_arguments_record selected = { 0 };
		if (axis_expand(count, arguments, &selected)) {
			const bool parsed = parse_arguments(selected.count,
							    selected.values);
			(void)parsed;
		}
		axis_arguments_free(&selected);
		_exit(0);
	}
	struct timespec start = { 0 }, now = { 0 };
	if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
		abort();
	}
	int status = 0;
	for (;;) {
		const pid_t waited = waitpid(child, &status, WNOHANG);
		if (waited == child) {
			break;
		}
		if (waited < 0 || clock_gettime(CLOCK_MONOTONIC, &now) != 0 ||
		    now.tv_sec - start.tv_sec >= TEST_FUZZ_CHILD_SECONDS) {
			if (kill(child, SIGKILL) == 0) {
				const pid_t reaped = waitpid(child, &status, 0);
				(void)reaped;
			}
			abort();
		}
		const struct timespec delay = { .tv_nsec = TEST_FUZZ_POLL_NS };
		const int slept = nanosleep(&delay, NULL);
		(void)slept;
	}
	if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
		abort();
	}
	return 0;
}

#ifndef PARSER_FUZZING
int main(int argument_count, char **arguments)
{
	static const char *const seeds[] = {
		"--rows=0,1,4294967295",
		"--temperature=0.5",
		"--category=code\n--exactness=balanced\n--concurrency=single\n--focus=decode",
		"--card_order=0,1,3,4,2\n--context_tokens=4096\nplaceholder.pack",
		"--draft=dspark\n--speculative=0",
		"--batch_rows=8,16,32",
		"--temperature=nan",
		"--seed=18446744073709551616"
	};
	for (size_t index = 0; index < sizeof(seeds) / sizeof(seeds[0]);
	     index++) {
		if (knob_fuzz_one_input((const uint8_t *)seeds[index],
					strlen(seeds[index])) != 0) {
			return 1;
		}
		if (argument_count > 1) {
			char path[TEST_FUZZ_PATH_BYTES];
			const int length = snprintf(path, sizeof(path),
						    "%s/seed_%zu", arguments[1],
						    index);
			if (length < 0 || (size_t)length >= sizeof(path)) {
				return 1;
			}
			FILE *const stream = fopen(path, "wb");
			if (stream == NULL) {
				return 1;
			}
			const bool written =
				fwrite(seeds[index], 1, strlen(seeds[index]),
				       stream) == strlen(seeds[index]);
			const int closed = fclose(stream);
			if (!written || closed != 0) {
				return 1;
			}
		}
	}
	return 0;
}
#endif
