/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * golden_capture_test: golden_capture.h over serving_fixture.h's function model in a speculative engine. A synthetic
 * golden (two cases whose continuations are not the model's draws, an end token among them; their logits recorded
 * by a forced run of the test's own) is captured at four rows a step and gathered from one, two and four parts: the
 * candidate's logits.f32 is the golden's byte for byte, its IDs files canonical (SHA-256 constants from Python's
 * json.dumps), native_drawn_ids.json the model's draws, conditioning the prompt and the golden tokens but the last;
 * a second run reads no row, and a case whose cases.jsonl line is lost (its folder's logits spoilt) and a .partial
 * folder are captured again; another plan, a golden hash that is not its IDs' and free space a byte below the
 * remaining rows and the floor are refused (at the floor it runs); each witness failure (a row's position, its input
 * token, a logit not finite, a draw not the argmax, a logit's bits, the conditioning) fails its case, and every
 * refusal and failure is named on stderr; --capture_positions keeps the first positions (truncated) and
 * --capture_free=1 writes an ordinary command's tokens and logit bits. CAPTURE_MANIFEST.json and a case's
 * metadata.json are held byte for byte to the texts the test builds from its own digests (under ASan their buffers
 * start as its malloc pattern, malloc_fill_byte, so a text left unterminated is read past its end and reported in the
 * log), witness.json to each row's position and input
 * token; each digest that cannot be made (the executable, the keys file, the pack header, the golden's MANIFEST.json),
 * a key listed twice and a set or case folder that is a link are refused with their line, before any text is built
 * or anything is removed (the link's target keeps its files). The last capture, of case c0 alone (the greedy case),
 * stays in $TMPDIR/golden_capture_test.PID/ for tools/golden_score.py and tools/golden/golden_capture_score.py (the
 * batch scores it).
 */
#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "anemone/front_end_seam.h"
#include "anemone/sampler.h"
#include "anemone/step.h"
#include "anemone/step_records.h"
#include "engine_internal.h"
#include "golden_capture.h"
#include "serving_fixture.h"
#include "sha256.h"

#define CONTEXT 256u
#define PROMPT 5u
#define GOLDEN 12u
#define CASES 2u
#define V TEST_SERVING_VOCABULARY
#define ROWS 8u
#define DEPTH 4u
#define PATH_BYTES 4096u
#define ROOT_BYTES 512u /* the test's folder: room for every path below it */
#define FOLDER_BYTES (2u * ROOT_BYTES)
#define CASE_NAME_BYTES 32u
#define MOST_PARTS 4u
#define DRAIN_PAUSE_US 1000u
#define TEXT_BYTES 8192u
#define OBJECT_BYTES 1024u /* a capture object's text */
#define DIGEST_HEX 65u
#define HEX_PER_BYTE 2u
#define DIRECTORY_MODE 0755
#define END_AT 3u
#define FAULT_ROW \
	2u /* the case's position a wrapped read or a seen row spoils */
#define WRONG_TOKEN 99u
#define NOT_ARGMAX_LOGIT 1.0e6f
#define PROMPT_STRIDE 13u
#define PROMPT_CASE_STRIDE 7u
#define PROMPT_RANGE 250u
#define GOLDEN_STRIDE 37u
#define GOLDEN_FIRST 11u
#define TOKEN_RANGE 256u /* the byte tokens */
#define PART_OFFSET_SHIFT \
	20u /* part p's rows at (p + 1) MiB: offsets of no meaning, distinct */
#define SEEN_ROWS 64u /* the most rows of a step rows_seen changes */
#define TRUNCATED 5u /* --capture_positions */
#define REMAINING_BYTES ((uint64_t)CASES * GOLDEN * V * sizeof(float))
#define PYTHON_IDS_HEX \
	"7370eb4bd6bffb7b8d5fa9de4d7d3043cfef85a388ba8b8e9b33a539c89c3bf5" /* [1,22,333] */
#define PACK_HEADER "header"
#define DESCRIPTOR_TEXT "{\"schema_version\": 1}"
#define ARGV_BYTES \
	"anemone\0capture" /* the capture's argv, each word NUL-ended */
#define ARGV_BYTE_COUNT (sizeof(ARGV_BYTES))

enum fault {
	NO_FAULT,
	POSITION,
	INPUT_TOKEN,
	NOT_FINITE,
	NOT_ARGMAX,
	BITS,
	CONDITIONING,
	FAULTS
};

static int failures = 0;
static const char *context_text = "";
static char root[ROOT_BYTES], golden[FOLDER_BYTES], out[FOLDER_BYTES];
static uint32_t prompts[CASES][PROMPT], continuations[CASES][GOLDEN];
static float golden_logits[CASES][GOLDEN][V];
static uint32_t golden_draws[CASES][GOLDEN];
static struct inference_engine *engine;
static uint32_t recorded;
static uint32_t recording_case;
/* the wrapped io: parts of the fixture's one head row, a fault at FAULT_ROW, the free space OUT's disk reports */
static struct sampler_logits_part parts[MOST_PARTS];
static uint32_t part_count;
static enum fault fault;
static uint32_t landed_reads;
static uint64_t free_space = UINT64_MAX;
/* the options of the next capture beyond the required three and rows, and what it said on stderr */
static uint32_t positions_option, free_option;
static const char *map_option; /* the capture's EXL3 map (NULL: none) */
static const char *executable_option; /* NULL: /proc/self/exe */
static bool header_missing;
/* the next captures leave stderr where it is, so that a sanitizer's report reaches the log (said is not kept) */
static bool stderr_open;
static uint32_t held_cases =
	CASES; /* the cases captured_held holds: the first ones */
static char said[TEXT_BYTES];

static void expect(bool condition, const char *what)
{
	if (!condition) {
		(void)printf("FAILED: %s: %s\n", context_text, what);
		failures++;
	}
}

static void ids_text(const uint32_t *ids, uint32_t count, char *text)
{
	size_t length = (size_t)sprintf(text, "[");
	for (uint32_t index = 0; index < count; index++)
		length += (size_t)sprintf(
			text + length, index == 0 ? "%u" : ",%u", ids[index]);
	(void)sprintf(text + length, "]");
}

static void hex_of_text(const char *text, char hex[DIGEST_HEX])
{
	unsigned char digest[SHA256_DIGEST_BYTES];
	expect(sha256_of(text, strlen(text), digest) == STATUS_OK, "a digest");
	for (size_t index = 0; index < SHA256_DIGEST_BYTES; index++)
		(void)sprintf(hex + HEX_PER_BYTE * index, "%02x",
			      digest[index]);
}

static bool written(const char *path, const void *bytes, size_t count)
{
	FILE *const file = fopen(path, "wb");
	const bool is_written = file != NULL &&
				fwrite(bytes, 1, count, file) == count;
	return file != NULL && fclose(file) == 0 && is_written;
}

static char *read_all(const char *path, size_t *count_out)
{
	FILE *const file = fopen(path, "rb");
	if (file == NULL)
		return NULL;
	char *bytes = NULL;
	long size = -1;
	if (fseek(file, 0, SEEK_END) == 0)
		size = ftell(file);
	if (size >= 0 && fseek(file, 0, SEEK_SET) == 0)
		bytes = malloc((size_t)size + 1u);
	if (bytes != NULL &&
	    fread(bytes, 1, (size_t)size, file) == (size_t)size) {
		bytes[size] = '\0';
		*count_out = (size_t)size;
	} else {
		free(bytes);
		bytes = NULL;
	}
	(void)fclose(file);
	return bytes;
}

/* The test's own recorder: the golden's rows, read where the fixture's head leaves them. */
MUST_CHECK static result_status recorder(void *context,
					 const struct step_row_record *rows,
					 const struct step_row_result *results,
					 uint32_t step_row_count,
					 uint32_t first_row, uint32_t row_count)
{
	(void)context;
	(void)rows;
	(void)step_row_count;
	for (uint32_t row = 0; row < row_count; row++) {
		golden_draws[recording_case][recorded] =
			(uint32_t)results[first_row + row].token_id;
		const result_status status = step_arena_read(
			engine->driver, 0,
			TEST_SERVING_LOGITS_OFFSET +
				(uint64_t)(first_row + row) * V * sizeof(float),
			V * sizeof(float),
			golden_logits[recording_case][recorded]);
		if (status != STATUS_OK)
			return status;
		recorded++;
	}
	return STATUS_OK;
}

static void events_drained(void)
{
	struct generation_event event = { .event_kind =
						  GENERATION_EVENT_TOKEN };
	while (event.event_kind == GENERATION_EVENT_TOKEN) {
		bool is_available = false;
		expect(engine_next_event(engine, &event, &is_available) ==
			       STATUS_OK,
		       "next_event");
		if (!is_available) {
			event.event_kind = GENERATION_EVENT_TOKEN;
			(void)usleep(DRAIN_PAUSE_US);
		}
	}
}

/* The golden dataset: per case its IDs, metadata (hashes of the canonical IDs) and the recorded rows. */
static void golden_made(void)
{
	char folder[FOLDER_BYTES + CASE_NAME_BYTES], path[PATH_BYTES],
		text[TEXT_BYTES], prompt_text[TEXT_BYTES],
		continuation_text[TEXT_BYTES], prompt_hex[DIGEST_HEX],
		continuation_hex[DIGEST_HEX];
	(void)snprintf(golden, sizeof(golden), "%s/golden", root);
	(void)snprintf(path, sizeof(path), "%s/cases", golden);
	expect(mkdir(golden, DIRECTORY_MODE) == 0 &&
		       mkdir(path, DIRECTORY_MODE) == 0,
	       "the golden's folders");
	(void)snprintf(path, sizeof(path), "%s/cases/set", golden);
	expect(mkdir(path, DIRECTORY_MODE) == 0, "the golden's set");
	for (uint32_t index = 0; index < CASES; index++) {
		for (uint32_t k = 0; k < PROMPT; k++)
			prompts[index][k] = (k * PROMPT_STRIDE +
					     PROMPT_CASE_STRIDE * index + 1u) %
					    PROMPT_RANGE;
		for (uint32_t k = 0; k < GOLDEN; k++)
			continuations[index][k] =
				(k * GOLDEN_STRIDE + GOLDEN_FIRST + index) %
				TOKEN_RANGE;
		continuations[index][END_AT] = TEST_SERVING_END_TOKEN;
		const struct generation_command_record command = {
			.new_token_ids = prompts[index],
			.new_token_count = PROMPT,
			.generated_token_limit = GOLDEN,
			.sampler_top_p = 1.0f,
			.request_tag = 1u
		};
		const struct engine_forced_continuation forced = {
			continuations[index], GOLDEN, 1u, recorder, NULL
		};
		/* case 0 greedy (its continuation the model's own draws, as the scorer wants a golden: each pass fixes
		 * one more position, the draw at k following the tokens before it), case 1 not (its draws differ) */
		for (uint32_t pass = 0; pass <= (index == 0 ? GOLDEN : 0u);
		     pass++) {
			recorded = 0;
			recording_case = index;
			expect(engine_submit_forced(engine, &command,
						    &forced) == STATUS_OK,
			       "the recorder's forced run");
			events_drained(); /* its final event follows the last rows_landed */
			expect(recorded == GOLDEN,
			       "the recorder saw every row");
			if (index == 0)
				memcpy(continuations[0], golden_draws[0],
				       sizeof(continuations[0]));
		}
		(void)snprintf(folder, sizeof(folder), "%s/cases/set/c%u",
			       golden, index);
		expect(mkdir(folder, DIRECTORY_MODE) == 0, "a case folder");
		ids_text(prompts[index], PROMPT, prompt_text);
		ids_text(continuations[index], GOLDEN, continuation_text);
		hex_of_text(prompt_text, prompt_hex);
		hex_of_text(continuation_text, continuation_hex);
		uint32_t conditioning[PROMPT + GOLDEN - 1u];
		char conditioning_text[TEXT_BYTES],
			conditioning_hex[DIGEST_HEX];
		memcpy(conditioning, prompts[index], sizeof(prompts[index]));
		memcpy(conditioning + PROMPT, continuations[index],
		       (GOLDEN - 1u) * sizeof(uint32_t));
		ids_text(conditioning, PROMPT + GOLDEN - 1u, conditioning_text);
		hex_of_text(conditioning_text, conditioning_hex);
		(void)snprintf(
			text, sizeof(text),
			"{\"schema_version\": 1, \"vocabulary_size\": %u, \"generated_positions\": %u, "
			"\"temperature\": 0.0, \"alignment\": {\"mode\": \"autoregressive_greedy\", "
			"\"prompt_ids_sha256\": \"%s\", \"continuation_ids_sha256\": \"%s\", "
			"\"conditioning_ids_sha256\": \"%s\"}}",
			V, GOLDEN, prompt_hex, continuation_hex,
			conditioning_hex);
		static const char *const names[] = { "prompt_ids.json",
						     "token_ids.json",
						     "metadata.json",
						     "conditioning_ids.json" };
		const char *const texts[] = { prompt_text, continuation_text,
					      text, conditioning_text };
		for (size_t file = 0; file < sizeof(names) / sizeof(names[0]);
		     file++) {
			(void)snprintf(path, sizeof(path), "%s/%s", folder,
				       names[file]);
			expect(written(path, texts[file], strlen(texts[file])),
			       "a golden file");
		}
		(void)snprintf(path, sizeof(path), "%s/logits.f32", folder);
		expect(written(path, golden_logits[index],
			       sizeof(golden_logits[index])),
		       "the golden's logits");
	}
	(void)snprintf(path, sizeof(path), "%s/MANIFEST.json", golden);
	expect(written(path, "{}", strlen("{}")), "the golden's manifest");
}

/* The wrapped arena read: each part's columns of the fixture's [rows][V] head rows, the fault at FAULT_ROW. */
MUST_CHECK static result_status parts_read(void *context, uint32_t card,
					   uint64_t offset, uint64_t byte_count,
					   void *bytes_out)
{
	(void)context;
	static float head[ROWS][V];
	for (uint32_t part = 0; part < part_count; part++) {
		const uint64_t row_bytes =
			(uint64_t)parts[part].token_count * sizeof(float);
		if (card != parts[part].card || offset < parts[part].offset ||
		    (offset - parts[part].offset) % row_bytes != 0)
			continue;
		const uint32_t first = (uint32_t)((offset -
						   parts[part].offset) /
						  row_bytes),
			       count = (uint32_t)(byte_count / row_bytes);
		const result_status status = step_arena_read(
			engine->driver, 0,
			TEST_SERVING_LOGITS_OFFSET +
				(uint64_t)first * V * sizeof(float),
			(uint64_t)count * V * sizeof(float), head);
		for (uint32_t row = 0; row < count && status == STATUS_OK;
		     row++) {
			float *const own =
				(float *)bytes_out +
				(size_t)row * parts[part].token_count;
			memcpy(own, &head[row][parts[part].first_token],
			       row_bytes);
			const bool faulty = landed_reads + row == FAULT_ROW &&
					    parts[part].first_token == 0;
			if (faulty && fault == NOT_FINITE)
				own[V - 1u < parts[part].token_count ? V - 1u :
								       1u] =
					NAN;
			if (faulty && fault == NOT_ARGMAX)
				own[0] = NOT_ARGMAX_LOGIT;
		}
		if (part + 1u == part_count)
			landed_reads += count;
		return status;
	}
	return STATUS_INVALID_ARGUMENT;
}

static uint32_t history_read(void *context, uint32_t position)
{
	(void)context;
	const uint32_t token = step_history_token(engine->driver, 0, position);
	return fault == CONDITIONING && position == PROMPT + 1u ? WRONG_TOKEN :
								  token;
}

MUST_CHECK static result_status bits_read(void *context, uint32_t card,
					  uint64_t offset, uint64_t byte_count,
					  void *bytes_out)
{
	const result_status status =
		parts_read(context, card, offset, byte_count, bytes_out);
	if (fault == BITS && status == STATUS_OK) {
		/* the draw's logit (the row's position) one unit of the last place off, at every row */
		float *const rows = bytes_out;
		for (uint32_t row = 0; row < byte_count / (V * sizeof(float));
		     row++)
			for (uint32_t column = 0; column < V; column++)
				if (rows[(size_t)row * V + column] >= 0.0f)
					rows[(size_t)row * V + column] =
						nextafterf(
							rows[(size_t)row * V +
							     column],
							INFINITY);
	}
	return status;
}

static bool free_read(void *context, const char *path, uint64_t *bytes_out)
{
	(void)context;
	(void)path;
	*bytes_out = free_space;
	return true;
}

/* The rows the witnesses see: the engine's, with the position or the input token of case position FAULT_ROW's row
 * spoilt. */
static const struct step_row_record *
rows_spoilt(void *context, const struct step_row_record *rows,
	    uint32_t step_row_count)
{
	(void)context;
	static struct step_row_record seen[SEEN_ROWS];
	if ((fault != POSITION && fault != INPUT_TOKEN) ||
	    step_row_count > SEEN_ROWS)
		return rows;
	memcpy(seen, rows, step_row_count * sizeof(*rows));
	for (uint32_t row = 0; row < step_row_count; row++) {
		if (seen[row].position != PROMPT - 1u + FAULT_ROW ||
		    (seen[row].flags & STEP_ROW_LOGITS) == 0)
			continue;
		if (fault == POSITION)
			seen[row].position++;
		else
			seen[row].token_id++;
	}
	return seen;
}

/* A capture of the golden's cases in keys at rows a step through the test's io; what it says on stderr in said. */
MUST_CHECK static result_status capture(uint32_t rows, const char *keys)
{
	const struct golden_capture_io io = { bits_read, history_read,
					      free_read, rows_spoilt, NULL };
	static char program[] = "anemone", command[] = "capture";
	static char *arguments[] = { program, command };
	golden_capture_options =
		(struct golden_capture_options){ .golden = golden,
						 .cases = keys,
						 .out = out,
						 .rows = rows,
						 .positions = positions_option,
						 .free_run = free_option,
						 .given = CAPTURE_REQUIRED };
	landed_reads = 0;
	const struct golden_capture_run run = {
		.engine = engine,
		.driver = engine->driver,
		.io = &io,
		.argument_count =
			(int)(sizeof(arguments) / sizeof(arguments[0])),
		.arguments = arguments,
		.version = "test",
		.placement = "test",
		.weights = "original",
		.map = "test",
		.pack_header = header_missing ? NULL : PACK_HEADER,
		.pack_header_bytes = strlen(PACK_HEADER),
		.executable = executable_option,
		.map_path = map_option,
		.parts = parts,
		.part_count = part_count
	};
	said[0] = '\0';
	if (stderr_open)
		return golden_capture_run(&run);
	FILE *const kept = tmpfile();
	const int saved = dup(STDERR_FILENO);
	expect(kept != NULL && saved >= 0 && fflush(stderr) == 0 &&
		       dup2(fileno(kept), STDERR_FILENO) == STDERR_FILENO,
	       "stderr is kept");
	const result_status status = golden_capture_run(&run);
	expect(fflush(stderr) == 0 && saved >= 0 &&
		       dup2(saved, STDERR_FILENO) == STDERR_FILENO,
	       "stderr is restored");
	if (kept != NULL && fseek(kept, 0L, SEEK_SET) == 0) {
		clearerr(kept);
		said[fread(said, 1, sizeof(said) - 1u, kept)] = '\0';
	}
	if (kept != NULL)
		(void)fclose(kept);
	if (saved >= 0)
		(void)close(saved);
	(void)fputs(said, stderr);
	return status;
}

static bool same_file(const char *one, const char *other_bytes, size_t count)
{
	size_t got = 0;
	char *const bytes = read_all(one, &got);
	const bool same = bytes != NULL && got == count &&
			  memcmp(bytes, other_bytes, count) == 0;
	free(bytes);
	return same;
}

/* Every file of each captured case against the golden and the model, its first positions of them. */
static void captured_held(uint32_t positions)
{
	char path[PATH_BYTES], text[TEXT_BYTES];
	for (uint32_t index = 0; index < held_cases; index++) {
		(void)snprintf(path, sizeof(path),
			       "%s/cases/set/c%u/logits.f32", out, index);
		expect(same_file(path, (const char *)golden_logits[index],
				 positions * sizeof(golden_logits[0][0])),
		       "logits.f32 is the golden's, byte for byte");
		(void)snprintf(path, sizeof(path),
			       "%s/cases/set/c%u/native_drawn_ids.json", out,
			       index);
		ids_text(golden_draws[index], positions, text);
		expect(same_file(path, text, strlen(text)),
		       "native_drawn_ids.json holds the model's draws");
		uint32_t conditioning[PROMPT + GOLDEN - 1u];
		memcpy(conditioning, prompts[index], sizeof(prompts[index]));
		memcpy(conditioning + PROMPT, continuations[index],
		       (positions - 1u) * sizeof(uint32_t));
		ids_text(conditioning, PROMPT + positions - 1u, text);
		(void)snprintf(path, sizeof(path),
			       "%s/cases/set/c%u/conditioning_ids.json", out,
			       index);
		expect(same_file(path, text, strlen(text)),
		       "conditioning_ids.json: the prompt and the golden tokens but the last");
		ids_text(continuations[index], positions, text);
		(void)snprintf(path, sizeof(path),
			       "%s/cases/set/c%u/token_ids.json", out, index);
		expect(same_file(path, text, strlen(text)),
		       "token_ids.json canonical");
	}
}

/* Whether a case's file of OUT holds text. */
static bool file_holds(const char *key, const char *name, const char *text)
{
	char path[PATH_BYTES];
	size_t count = 0;
	(void)snprintf(path, sizeof(path), "%s/cases/%s/%s", out, key, name);
	char *const bytes = read_all(path, &count);
	const bool holds = bytes != NULL && strstr(bytes, text) != NULL;
	free(bytes);
	return holds;
}

static void hex_of_bytes(const void *bytes, size_t count, char hex[DIGEST_HEX])
{
	unsigned char digest[SHA256_DIGEST_BYTES];
	expect(sha256_of(bytes, count, digest) == STATUS_OK, "a digest");
	for (size_t index = 0; index < SHA256_DIGEST_BYTES; index++)
		(void)sprintf(hex + HEX_PER_BYTE * index, "%02x",
			      digest[index]);
}

static void hex_of_file(const char *path, char hex[DIGEST_HEX])
{
	size_t count = 0;
	char *const bytes = read_all(path, &count);
	expect(bytes != NULL, "a file to digest");
	hex_of_bytes(bytes != NULL ? bytes : "", count, hex);
	free(bytes);
}

/* The capture object of a capture at rows a step (no map: no descriptor), as the test's own digests give it. */
static void capture_object(char *text, size_t size, uint32_t rows)
{
	char executable[DIGEST_HEX], argv[DIGEST_HEX], header[DIGEST_HEX];
	hex_of_file("/proc/self/exe", executable);
	hex_of_bytes(ARGV_BYTES, ARGV_BYTE_COUNT, argv);
	hex_of_bytes(PACK_HEADER, strlen(PACK_HEADER), header);
	(void)snprintf(
		text, size,
		"\"executable_sha256\":\"%s\",\"argv_sha256\":\"%s\",\"pack_header_sha256\":\"%s\","
		"\"descriptor_sha256\":null,"
		"\"rows\":%u,\"context_tokens\":%u,\"version\":\"test\",\"placement\":\"test\","
		"\"weights\":\"original\",\"map\":\"test\"",
		executable, argv, header, rows, CONTEXT);
}

/* OUT's CAPTURE_MANIFEST.json and case set/c0's metadata.json, byte for byte, of a capture of keys at rows a step. */
static void texts_held(const char *keys, uint32_t rows)
{
	char object[OBJECT_BYTES], text[TEXT_BYTES], path[PATH_BYTES],
		executable[DIGEST_HEX], argv[DIGEST_HEX], keys_hex[DIGEST_HEX],
		golden_hex[DIGEST_HEX], plan[DIGEST_HEX], ids[TEXT_BYTES],
		prompt_hex[DIGEST_HEX], continuation_hex[DIGEST_HEX],
		conditioning_hex[DIGEST_HEX], logits_hex[DIGEST_HEX];
	capture_object(object, sizeof(object), rows);
	hex_of_file("/proc/self/exe", executable);
	hex_of_bytes(ARGV_BYTES, ARGV_BYTE_COUNT, argv);
	hex_of_file(keys, keys_hex);
	(void)snprintf(path, sizeof(path), "%s/MANIFEST.json", golden);
	hex_of_file(path, golden_hex);
	(void)snprintf(text, sizeof(text),
		       "executable=%s\nargv=%s\nkeys=%s\ngolden_manifest=%s\n",
		       executable, argv, keys_hex, golden_hex);
	hex_of_text(text, plan);
	(void)snprintf(
		text, sizeof(text),
		"{\"plan_sha256\":\"%s\",\"executable_sha256\":\"%s\",\"argv_sha256\":\"%s\","
		"\"keys_sha256\":\"%s\",\"golden_manifest_sha256\":\"%s\",\"cases\":%u,\"positions\":%u,"
		"\"bytes\":%llu,\"argv\":[\"anemone\",\"capture\"],\"capture\":{%s}}\n",
		plan, executable, argv, keys_hex, golden_hex, CASES,
		CASES * GOLDEN, (unsigned long long)REMAINING_BYTES, object);
	(void)snprintf(path, sizeof(path), "%s/CAPTURE_MANIFEST.json", out);
	expect(same_file(path, text, strlen(text)),
	       "CAPTURE_MANIFEST.json, byte for byte");
	uint32_t conditioning[PROMPT + GOLDEN - 1u];
	memcpy(conditioning, prompts[0], sizeof(prompts[0]));
	memcpy(conditioning + PROMPT, continuations[0],
	       (GOLDEN - 1u) * sizeof(uint32_t));
	ids_text(prompts[0], PROMPT, ids);
	hex_of_text(ids, prompt_hex);
	ids_text(continuations[0], GOLDEN, ids);
	hex_of_text(ids, continuation_hex);
	ids_text(conditioning, PROMPT + GOLDEN - 1u, ids);
	hex_of_text(ids, conditioning_hex);
	hex_of_bytes(golden_logits[0], sizeof(golden_logits[0]), logits_hex);
	(void)snprintf(
		text, sizeof(text),
		"{\"schema_version\":1,\"vocabulary_size\":%u,\"generated_positions\":%u,\"temperature\":0,"
		"\"alignment\":{\"mode\":\"teacher_forced\",\"prompt_ids_sha256\":\"%s\","
		"\"continuation_ids_sha256\":\"%s\",\"conditioning_ids_sha256\":\"%s\"},\"capture\":{%s,"
		"\"truncated\":false,\"logits_sha256\":\"%s\",\"witnessed_positions\":%u}}\n",
		V, GOLDEN, prompt_hex, continuation_hex, conditioning_hex,
		object, logits_hex, GOLDEN);
	(void)snprintf(path, sizeof(path), "%s/cases/set/c0/metadata.json",
		       out);
	expect(same_file(path, text, strlen(text)),
	       "metadata.json, byte for byte");
	/* witness.json: each row's position and input token, as it landed */
	size_t length = (size_t)sprintf(text, "\"position\":[");
	for (uint32_t k = 0; k < GOLDEN; k++)
		length += (size_t)sprintf(text + length, k == 0 ? "%u" : ",%u",
					  PROMPT - 1u + k);
	length += (size_t)sprintf(text + length, "],\"input_token\":[%u",
				  prompts[0][PROMPT - 1u]);
	for (uint32_t k = 1; k < GOLDEN; k++)
		length += (size_t)sprintf(text + length, ",%u",
					  continuations[0][k - 1u]);
	(void)sprintf(text + length, "],");
	expect(file_holds("set/c0", "witness.json", text),
	       "witness.json: each row's position and input token");
}

/* A refusal before any text: its line on stderr, and no OUT made. */
static void refused(const char *keys, const char *line, const char *what)
{
	struct stat facts;
	(void)snprintf(out, sizeof(out), "%s/out_refused", root);
	expect(capture(DEPTH, keys) == STATUS_INVALID_ARGUMENT &&
		       strstr(said, line) != NULL && stat(out, &facts) != 0,
	       what);
}

/* A capture again over OUT, of which link_path (a set or case folder) is a link to a folder of target_path holding
 * a capture file: refused, the file still there. */
static void link_refused(const char *keys, const char *link_path,
			 const char *target_path, const char *file_path,
			 const char *what)
{
	struct stat facts;
	expect(mkdir(target_path, DIRECTORY_MODE) == 0 &&
		       written(file_path, "x", 1u) &&
		       symlink(target_path, link_path) == 0,
	       "a link to a folder of a capture file");
	expect(capture(DEPTH, keys) == STATUS_INVALID_ARGUMENT &&
		       strstr(said,
			      "capture refused: a set or case folder is a link or no folder") !=
			       NULL,
	       what);
	expect(stat(file_path, &facts) == 0,
	       "the link's target keeps its file");
	expect(unlink(link_path) == 0, "the link removed");
}

/* --capture_free=1 on case 0, which is greedy: the ordinary command's tokens are its continuation up to and with its
 * first end token, each event's logit its row's position (the fixture's draw logit). */
static void free_run_held(void)
{
	char text[TEXT_BYTES], path[PATH_BYTES];
	uint32_t count = 0;
	while (count < GOLDEN && (count == 0 || continuations[0][count - 1u] !=
							TEST_SERVING_END_TOKEN))
		count++;
	ids_text(continuations[0], count, text);
	(void)snprintf(path, sizeof(path),
		       "%s/cases/set/c0/free_token_ids.json", out);
	expect(same_file(path, text, strlen(text)),
	       "free_token_ids.json: the ordinary command's tokens");
	size_t length = (size_t)sprintf(text, "[");
	for (uint32_t k = 0; k < count; k++) {
		const float logit = (float)(PROMPT - 1u + k);
		uint32_t bits = 0;
		memcpy(&bits, &logit, sizeof(bits));
		length += (size_t)sprintf(text + length,
					  k == 0 ? "\"0x%08x\"" : ",\"0x%08x\"",
					  bits);
	}
	(void)sprintf(text + length, "]\n");
	(void)snprintf(path, sizeof(path), "%s/cases/set/c0/free_logits.json",
		       out);
	expect(same_file(path, text, strlen(text)),
	       "free_logits.json: its events' logit bits");
	expect(file_holds("set/c1", "free_token_ids.json", "["),
	       "case 1's free run");
}

int main(void)
{
	(void)setvbuf(stdout, NULL, _IOLBF, 0);
	const char *const temporary = getenv("TMPDIR");
	(void)snprintf(root, sizeof(root), "%s/golden_capture_test.%ld",
		       temporary != NULL ? temporary : "/var/tmp",
		       (long)getpid());
	expect(mkdir(root, DIRECTORY_MODE) == 0, "the test's folder");
	context_text = "canonical IDs";
	char text[TEXT_BYTES], hex[DIGEST_HEX], path[PATH_BYTES],
		keys[PATH_BYTES];
	static const uint32_t python_ids[] = { 1, 22, 333 };
	ids_text(python_ids, sizeof(python_ids) / sizeof(python_ids[0]), text);
	hex_of_text(text, hex);
	expect(strcmp(hex, PYTHON_IDS_HEX) == 0,
	       "the test's canonical IDs digest is Python's");
	static const uint32_t none[] = { TEST_SERVING_END_TOKEN };
	static const uint32_t *const scripts[] = { none };
	test_serving_set_function_model(true, 0);
	test_serving_set_engine(1, 1, ROWS, DEPTH);
	expect(test_serving_open(CONTEXT, scripts, 1, &engine) == STATUS_OK,
	       "the engine opens");
	if (engine == NULL)
		return EXIT_FAILURE;
	golden_made();
	(void)snprintf(keys, sizeof(keys), "%s/keys.txt", root);
	expect(written(keys, "set/c0\nset/c1\n", strlen("set/c0\nset/c1\n")),
	       "the keys");
	/* one, two and four parts of the head; their texts held byte for byte, stderr open (a sanitizer's report of a
	 * text read past its end reaches the log) */
	stderr_open = true;
	static const uint32_t splits[] = { 1, 2, 4 };
	for (uint32_t split = 0; split < sizeof(splits) / sizeof(splits[0]);
	     split++) {
		context_text = "a capture gathered from parts";
		part_count = splits[split];
		for (uint32_t part = 0; part < part_count; part++)
			parts[part] = (struct sampler_logits_part){
				(uint64_t)(part + 1u) << PART_OFFSET_SHIFT,
				part, part * (V / part_count), V / part_count, 0
			};
		(void)snprintf(out, sizeof(out), "%s/out%u", root, split);
		expect(capture(DEPTH, keys) == STATUS_OK, "the capture runs");
		captured_held(GOLDEN);
		texts_held(keys, DEPTH);
	}
	stderr_open = false;
	context_text = "resume";
	expect(capture(DEPTH, keys) == STATUS_OK && landed_reads == 0,
	       "a second run reads no row");
	/* c1's line is lost and its folder's logits spoilt: the case runs again */
	(void)snprintf(path, sizeof(path), "%s/cases.jsonl", out);
	size_t count = 0;
	char *const lines = read_all(path, &count);
	char *const second = lines != NULL ? strchr(lines, '\n') : NULL;
	expect(second != NULL && unlink(path) == 0 &&
		       written(path, lines, (size_t)(second - lines) + 1u),
	       "case c1's line is lost");
	free(lines);
	(void)snprintf(path, sizeof(path), "%s/cases/set/c1/logits.f32", out);
	expect(unlink(path) == 0 && written(path, "x", 1u),
	       "case c1's logits are spoilt");
	(void)snprintf(path, sizeof(path), "%s/cases/set/c0.partial", out);
	expect(mkdir(path, DIRECTORY_MODE) == 0, "a partial folder of c0");
	(void)snprintf(path, sizeof(path), "%s/cases/set/c0.partial/logits.f32",
		       out);
	expect(written(path, "x", 1u), "a partial logits file of c0");
	expect(capture(DEPTH, keys) == STATUS_OK, "the run again");
	captured_held(GOLDEN);
	(void)snprintf(path, sizeof(path), "%s/cases.jsonl", out);
	char *const again = read_all(path, &count);
	expect(again != NULL &&
		       strncmp(again, "{\"case\":\"set/c0\",",
			       strlen("{\"case\":\"set/c0\",")) == 0 &&
		       strchr(again, '\n') != NULL &&
		       strncmp(strchr(again, '\n') + 1, "{\"case\":\"set/c1\",",
			       strlen("{\"case\":\"set/c1\",")) == 0,
	       "cases.jsonl: c0's line, then c1's again");
	free(again);
	(void)snprintf(path, sizeof(path), "%s/cases/set/c0.partial", out);
	struct stat facts;
	expect(stat(path, &facts) != 0, "the partial folder is gone");
	context_text = "links";
	char target[PATH_BYTES], file[PATH_BYTES], link[PATH_BYTES],
		moved[PATH_BYTES];
	(void)snprintf(link, sizeof(link), "%s/cases/set/c0.partial", out);
	(void)snprintf(target, sizeof(target), "%s/elsewhere", root);
	(void)snprintf(file, sizeof(file), "%s/elsewhere/logits.f32", root);
	link_refused(keys, link, target, file,
		     "a .partial folder that is a link");
	(void)snprintf(link, sizeof(link), "%s/cases/set", out);
	(void)snprintf(moved, sizeof(moved), "%s/set_moved", root);
	(void)snprintf(target, sizeof(target), "%s/elsewhere_set", root);
	(void)snprintf(file, sizeof(file), "%s/elsewhere_set/c0.partial", root);
	expect(rename(link, moved) == 0, "the set folder moved away");
	expect(mkdir(target, DIRECTORY_MODE) == 0, "a folder for the link");
	(void)snprintf(target, sizeof(target), "%s/elsewhere_set/c0.partial",
		       root);
	(void)snprintf(file, sizeof(file),
		       "%s/elsewhere_set/c0.partial/logits.f32", root);
	expect(mkdir(target, DIRECTORY_MODE) == 0 && written(file, "x", 1u),
	       "the link's target holds a partial folder of a capture file");
	(void)snprintf(target, sizeof(target), "%s/elsewhere_set", root);
	struct stat kept_facts;
	expect(symlink(target, link) == 0 &&
		       capture(DEPTH, keys) == STATUS_INVALID_ARGUMENT &&
		       strstr(said,
			      "capture refused: a set or case folder is a link or no folder") !=
			       NULL,
	       "a set folder that is a link");
	expect(stat(file, &kept_facts) == 0,
	       "the link's target keeps its file");
	expect(unlink(link) == 0 && rename(moved, link) == 0,
	       "the set folder back");
	expect(capture(DEPTH, keys) == STATUS_OK && landed_reads == 0,
	       "then the run reads no row");
	context_text = "refusals";
	(void)snprintf(path, sizeof(path), "%s/keys_twice.txt", root);
	expect(written(path, "set/c0\nset/c1\nset/c0\n",
		       strlen("set/c0\nset/c1\nset/c0\n")),
	       "a keys file of a key twice");
	refused(path, "capture: case set/c0 refused: a key listed twice",
		"a key listed twice");
	(void)snprintf(path, sizeof(path), "%s/no_keys.txt", root);
	refused(path, "no_keys.txt: the keys file is not readable",
		"a keys file that does not read");
	char missing[PATH_BYTES];
	(void)snprintf(missing, sizeof(missing), "%s/no_executable", root);
	executable_option = missing;
	refused(keys, "capture refused: the executable could not be digested",
		"an executable that does not read");
	executable_option = NULL;
	header_missing = true;
	refused(keys, "capture refused: the pack header could not be digested",
		"a pack header of no bytes");
	header_missing = false;
	char manifest[PATH_BYTES];
	(void)snprintf(manifest, sizeof(manifest), "%s/MANIFEST.json", golden);
	(void)snprintf(moved, sizeof(moved), "%s/MANIFEST.json.kept", golden);
	expect(rename(manifest, moved) == 0 &&
		       mkdir(manifest, DIRECTORY_MODE) == 0,
	       "the golden's MANIFEST.json a folder");
	refused(keys,
		"capture refused: the golden's MANIFEST.json could not be digested",
		"a golden MANIFEST.json that does not read");
	expect(rmdir(manifest) == 0 && rename(moved, manifest) == 0,
	       "the golden's MANIFEST.json back");
	char map[PATH_BYTES], sidecar[PATH_BYTES];
	(void)snprintf(map, sizeof(map), "%s/test.exl3map", root);
	(void)snprintf(sidecar, sizeof(sidecar),
		       "%s/test.exl3map.descriptor.json", root);
	expect(written(map, PACK_HEADER, strlen(PACK_HEADER)) &&
		       mkdir(sidecar, DIRECTORY_MODE) == 0,
	       "a map whose descriptor is a folder");
	map_option = map;
	refused(keys,
		"capture refused: the map's descriptor could not be digested",
		"a map's descriptor that does not read");
	map_option = NULL;
	expect(rmdir(sidecar) == 0 && written(sidecar, DESCRIPTOR_TEXT,
					      strlen(DESCRIPTOR_TEXT)),
	       "the map's descriptor");
	(void)snprintf(out, sizeof(out), "%s/out2", root);
	(void)snprintf(path, sizeof(path), "%s/keys_one.txt", root);
	expect(written(path, "set/c0\n", strlen("set/c0\n")) &&
		       capture(DEPTH, path) == STATUS_INVALID_ARGUMENT &&
		       strstr(said,
			      "capture refused: OUT holds something else, or a capture of another plan") !=
			       NULL,
	       "another plan on the same OUT");
	(void)snprintf(path, sizeof(path), "%s/cases/set/c1/metadata.json",
		       golden);
	char *const metadata = read_all(path, &count);
	char *const hash = metadata != NULL ?
				   strstr(metadata, "prompt_ids_sha256\": \"") :
				   NULL;
	if (hash != NULL)
		hash[strlen("prompt_ids_sha256\": \"")] ^= 1;
	expect(hash != NULL && unlink(path) == 0 &&
		       written(path, metadata, count),
	       "case c1's prompt hash is broken");
	(void)snprintf(out, sizeof(out), "%s/out_hash", root);
	expect(capture(DEPTH, keys) == STATUS_INVALID_ARGUMENT &&
		       strstr(said,
			      "capture: case set/c1 refused: a hash is not the SHA-256 of its IDs' "
			      "canonical JSON") != NULL,
	       "a golden hash that is not its IDs' is refused");
	if (hash != NULL)
		hash[strlen("prompt_ids_sha256\": \"")] ^= 1;
	expect(unlink(path) == 0 && written(path, metadata, count),
	       "case c1 restored");
	free(metadata);
	(void)snprintf(out, sizeof(out), "%s/out_space", root);
	free_space = REMAINING_BYTES + GOLDEN_CAPTURE_FLOOR_BYTES - 1u;
	expect(capture(DEPTH, keys) == STATUS_INVALID_ARGUMENT &&
		       landed_reads == 0 &&
		       strstr(said,
			      "capture refused: OUT's disk has less free space than the remaining rows and "
			      "1 GiB") != NULL,
	       "free space a byte below the remaining rows and the floor");
	context_text = "free space at the floor";
	free_space = REMAINING_BYTES + GOLDEN_CAPTURE_FLOOR_BYTES;
	expect(capture(DEPTH, keys) == STATUS_OK, "the capture runs");
	captured_held(GOLDEN);
	free_space = UINT64_MAX;
	static const char *const fault_names[FAULTS] = {
		"",
		"position 2: a row's position is not its case position's",
		"position 2: a row's input token is not the prompt's last or the golden token before it",
		"position 2: a logit is not finite",
		"position 2: the draw is not the row's lowest-ID argmax",
		"position 0: the draw's logit is not the row's at the draw, bit for bit",
		"position 12: the conditioning is not the prompt and the golden tokens but the last"
	};
	part_count = 1;
	parts[0] = (struct sampler_logits_part){ 1u << PART_OFFSET_SHIFT, 0, 0,
						 V, 0 };
	for (fault = POSITION; fault < FAULTS; fault++) {
		context_text = fault_names[fault];
		(void)snprintf(out, sizeof(out), "%s/out_fault%u", root,
			       (unsigned)fault);
		expect(capture(1u, keys) != STATUS_OK,
		       "the witness fails the case");
		expect(strstr(said, "capture: case set/c0 failed (") != NULL &&
			       strstr(said, fault_names[fault]) != NULL,
		       "the failure is named on stderr");
	}
	fault = NO_FAULT;
	context_text = "--capture_positions";
	positions_option = TRUNCATED;
	(void)snprintf(out, sizeof(out), "%s/out_positions", root);
	expect(capture(DEPTH, keys) == STATUS_OK, "the capture runs");
	captured_held(TRUNCATED);
	(void)snprintf(text, sizeof(text), "\"generated_positions\":%u,",
		       TRUNCATED);
	expect(file_holds("set/c0", "metadata.json", text) &&
		       file_holds("set/c0", "metadata.json",
				  "\"truncated\":true,"),
	       "metadata.json: the positions kept, truncated");
	positions_option = 0;
	context_text = "--capture_free=1";
	free_option = 1;
	(void)snprintf(out, sizeof(out), "%s/out_free", root);
	expect(capture(DEPTH, keys) == STATUS_OK, "the capture runs");
	captured_held(GOLDEN);
	free_run_held();
	free_option = 0;
	/* The map descriptor: the kept capture's map has one beside it, the free run's capture no map */
	char descriptor_hex[DIGEST_HEX];
	hex_of_text(DESCRIPTOR_TEXT, descriptor_hex);
	expect(file_holds("set/c0", "metadata.json",
			  "\"descriptor_sha256\":null,"),
	       "the free run's capture: no map, no descriptor");
	map_option = map;
	/* the kept capture: case c0 alone, the greedy one (a T=0 golden the scorer takes; c1's draws are not its
	 * tokens) */
	char keys_one[PATH_BYTES];
	(void)snprintf(keys_one, sizeof(keys_one), "%s/keys_one.txt", root);
	(void)snprintf(out, sizeof(out), "%s/out_kept", root);
	context_text = "the kept capture";
	held_cases = 1;
	expect(capture(DEPTH, keys_one) == STATUS_OK,
	       "the capture the batch scores");
	captured_held(GOLDEN);
	expect(file_holds("set/c0", "metadata.json", "\"truncated\":false,"),
	       "metadata.json: not truncated");
	(void)snprintf(text, sizeof(text), "\"descriptor_sha256\":\"%s\",",
		       descriptor_hex);
	expect(file_holds("set/c0", "metadata.json", text),
	       "metadata.json: the map's descriptor's SHA-256");
	map_option = keys; /* a file with no descriptor beside it */
	(void)snprintf(out, sizeof(out), "%s/out_no_descriptor", root);
	expect(capture(DEPTH, keys_one) == STATUS_OK &&
		       file_holds("set/c0", "metadata.json",
				  "\"descriptor_sha256\":null,"),
	       "a map without a descriptor: null");
	map_option = map;
	(void)snprintf(out, sizeof(out), "%s/out_kept", root);
	test_serving_close();
	test_serving_set_function_model(false, 0);
	test_serving_set_engine(1, 1, TEST_SERVING_ROWS, 0);
	(void)printf("golden_capture_test: kept %s and %s\n", golden, out);
	if (failures != 0) {
		(void)printf("golden_capture_test: %d failures\n", failures);
		return EXIT_FAILURE;
	}
	(void)printf("golden_capture_test: passed\n");
	return EXIT_SUCCESS;
}
