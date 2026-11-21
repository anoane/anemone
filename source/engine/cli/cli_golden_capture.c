/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * cli_golden_capture.c: anemone capture's front end. Contract: cli_golden_capture.h.
 */
#include "cli_golden_capture.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <inttypes.h>
#include <math.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include "common/hooks.h"
#include "engine/library/block_scaled_linear/linear.h"
#include "engine/library/sampling/sampler.h"
#include "engine/runtime/step/step.h"
#include "engine/program/step_records.h"
#include "commands.h"
#include "engine/formats/exl3/exl3_descriptor.h"
#include "common/json.h"
#include "common/sha256.h"

#define OK STATUS_OK
#define HEX_PER_BYTE 2u
#define HEX_BYTES (HEX_PER_BYTE * SHA256_DIGEST_BYTES + 1u)
#define PATH_BYTES 4096u
#define KEY_BYTES 256u
#define FAILURE_BYTES 256u
#define ID_TEXT_BYTES 11u /* a uint32's digits and a comma */
#define ID_TEXT_BYTES_LEAST \
	2u /* a digit and a comma: a file of n bytes holds n / 2 IDs at most */
#define ARRAY_FRAME_BYTES 3u /* the brackets and the NUL */
#define BITS_TEXT_BYTES 13u /* "0x%08x", quoted, and a comma */
#define WITNESS_ARRAYS 6u
#define JSON_BYTES_PER_BYTE 6u /* json_write_string's most for a byte */
#define QUOTES_AND_COMMA 3u
#define LINE_BYTES 512u
#define TEXT_BYTES_LEAST ((size_t)4u * LINE_BYTES) /* metadata.json */
#define MANIFEST_BYTES ((size_t)4u * LINE_BYTES)
#define CAPTURE_JSON_BYTES ((size_t)2u * LINE_BYTES)
#define ALIGNMENT_HASHES 2u /* the prompt's and the continuation's */
#define CASES_FIRST_CAPACITY 64u
#define CASES_GROWTH 2u
#define RUNS 2u /* the draws and bits of the forced run, then the free run's */
#define SCHEMA_VERSION 1u
#define READ_PIECE_BYTES (1u << 20u)
#define DIRECTORY_MODE 0755
#define FILE_MODE 0644
#define DECIMAL 10
#define HEX_DIGITS "0123456789abcdef"
#define NIBBLE_BITS 4u
#define NIBBLE_MASK 0xFu
#define FLOAT_BITS_TEXT "0x%08" PRIx32
#define NULL_TEXT "null"
#define EXECUTABLE "/proc/self/exe"

struct golden_capture_options golden_capture_options;

/* One case of the keys file: its key and IDs (prompt then continuation, the positions run) and the bytes it writes. */
struct capture_case {
	char key[KEY_BYTES];
	uint32_t *ids; /* prompt_count + continuation */
	uint32_t prompt_count, positions, golden_positions;
	bool is_done;
};

/* The case in flight: what rows_landed checks and streams (the stepping thread's while the command runs). */
struct capture_sink {
	const struct capture_case *own;
	const struct golden_capture_io *io;
	const struct sampler_logits_part *parts;
	uint32_t part_count, vocabulary_size, landed;
	int logits_descriptor;
	struct sha256_stream digest;
	float *rows, *part; /* [LINEAR_MAXIMUM_ROWS][V], a part's rows */
	uint32_t *drawn, *step_rows, *row_in_step, *logit_bits, *conditioning;
	uint32_t *position, *input_token; /* each row's, as it landed */
	char failure[FAILURE_BYTES];
};

/* The run's digests as hex, each "null" until made. */
struct capture_digests {
	char executable[HEX_BYTES], argv[HEX_BYTES], keys[HEX_BYTES],
		golden[HEX_BYTES], header[HEX_BYTES], plan[HEX_BYTES],
		descriptor[HEX_BYTES];
};

static bool option_text(const char *argument, const char *name,
			const char **value_out)
{
	const size_t length = strlen(name);
	if (strncmp(argument, name, length) != 0)
		return false;
	*value_out = argument + length;
	return true;
}

bool golden_capture_option_taken(const char *argument, bool *is_valid_out)
{
	static const struct {
		const char *name;
		uint32_t *value, least, most, bit;
	} numbers[] = { { "--capture_rows=", &golden_capture_options.rows, 1,
			  LINEAR_MAXIMUM_ROWS, CAPTURE_ROWS_GIVEN },
			{ "--capture_positions=",
			  &golden_capture_options.positions, 1, UINT32_MAX - 1u,
			  CAPTURE_POSITIONS_GIVEN },
			{ "--capture_free=", &golden_capture_options.free_run,
			  0, 1, CAPTURE_FREE_GIVEN } };
	struct golden_capture_options *const own = &golden_capture_options;
	const char **const texts[] = { &own->golden, &own->cases, &own->out };
	static const char *const names[] = {
		"--capture_golden=", "--capture_cases=", "--capture_out="
	};
	*is_valid_out = true;
	for (uint32_t index = 0; index < sizeof(names) / sizeof(names[0]);
	     index++)
		if (option_text(argument, names[index], texts[index])) {
			*is_valid_out = *texts[index][0] != '\0' &&
					(own->given & (1u << index)) == 0;
			own->given |= 1u << index;
			return true;
		}
	for (uint32_t index = 0; index < sizeof(numbers) / sizeof(numbers[0]);
	     index++) {
		const char *text = NULL;
		if (!option_text(argument, numbers[index].name, &text))
			continue;
		char *end = NULL;
		errno = 0;
		const unsigned long long value = strtoull(text, &end, DECIMAL);
		*is_valid_out = errno == 0 && end != text && *end == '\0' &&
				text[0] >= '0' && text[0] <= '9' &&
				value >= numbers[index].least &&
				value <= numbers[index].most &&
				(own->given & numbers[index].bit) == 0;
		*numbers[index].value = (uint32_t)value;
		own->given |= numbers[index].bit;
		return true;
	}
	return false;
}

static void hex_of(const unsigned char digest[SHA256_DIGEST_BYTES],
		   char hex[HEX_BYTES])
{
	for (uint32_t index = 0; index < SHA256_DIGEST_BYTES; index++) {
		hex[(size_t)HEX_PER_BYTE * index] =
			HEX_DIGITS[digest[index] >> NIBBLE_BITS];
		hex[(size_t)HEX_PER_BYTE * index + 1u] =
			HEX_DIGITS[digest[index] & NIBBLE_MASK];
	}
	hex[HEX_BYTES - 1u] = '\0';
}

/* json.dumps(ids, separators=(',', ':')) into a new buffer; its length in *length_out. */
static char *ids_json(const uint32_t *ids, uint32_t count, size_t *length_out)
{
	char *const text =
		malloc((size_t)count * ID_TEXT_BYTES + ARRAY_FRAME_BYTES);
	size_t length = 0;
	if (text == NULL)
		return NULL;
	text[length++] = '[';
	for (uint32_t index = 0; index < count; index++)
		length += (size_t)sprintf(text + length,
					  index == 0 ? "%" PRIu32 : ",%" PRIu32,
					  ids[index]);
	text[length++] = ']';
	text[length] = '\0';
	*length_out = length;
	return text;
}

static bool ids_digest(const uint32_t *ids, uint32_t count, char hex[HEX_BYTES])
{
	size_t length = 0;
	unsigned char digest[SHA256_DIGEST_BYTES];
	char *const text = ids_json(ids, count, &length);
	const bool is_done = text != NULL &&
			     sha256_of(text, length, digest) == OK;
	free(text);
	if (is_done)
		hex_of(digest, hex);
	return is_done;
}

/* A whole file, NUL-terminated, or NULL. */
static char *file_bytes(const char *path, size_t *count_out)
{
	const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
	struct stat facts;
	char *bytes = NULL;
	size_t count = 0;
	if (descriptor >= 0 && fstat(descriptor, &facts) == 0 &&
	    S_ISREG(facts.st_mode) && facts.st_size >= 0)
		bytes = malloc((size_t)facts.st_size + 1u);
	while (bytes != NULL && count < (size_t)facts.st_size) {
		const ssize_t got = read(descriptor, bytes + count,
					 (size_t)facts.st_size - count);
		if (got <= 0) {
			free(bytes);
			bytes = NULL;
		} else {
			count += (size_t)got;
		}
	}
	if (descriptor >= 0)
		(void)close(descriptor);
	if (bytes != NULL) {
		bytes[count] = '\0';
		*count_out = count;
	}
	return bytes;
}

static bool file_digest(const char *path, char hex[HEX_BYTES])
{
	static unsigned char piece[READ_PIECE_BYTES];
	unsigned char digest[SHA256_DIGEST_BYTES];
	struct sha256_stream stream;
	const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
	ssize_t got = descriptor >= 0 ? 1 : -1;
	sha256_begin(&stream);
	while (got > 0) {
		got = read(descriptor, piece, sizeof(piece));
		got = got > 0 && sha256_feed(&stream, piece, (size_t)got) !=
					      OK ?
			      -1 :
			      got;
	}
	if (descriptor >= 0)
		(void)close(descriptor);
	const bool is_done = got == 0 && sha256_finish(&stream, digest) == OK;
	if (is_done)
		hex_of(digest, hex);
	return is_done;
}

/* The SHA-256 of an EXL3 map's descriptor, as exl3_descriptor_check finds it (beside the map's real path; nothing
 * there, as there, is none), left "null" without a map or a descriptor; false when one is there and does not read. */
static bool descriptor_digest(const char *map_path, char text[HEX_BYTES])
{
	char real[PATH_MAX], sidecar[PATH_MAX];
	struct stat facts;
	if (map_path == NULL)
		return true;
	const char *const map = realpath(map_path, real) != NULL ? real :
								   map_path;
	const int length = snprintf(sidecar, sizeof(sidecar),
				    "%s" EXL3_DESCRIPTOR_SUFFIX, map);
	return length >= 0 && (size_t)length < sizeof(sidecar) &&
	       ((lstat(sidecar, &facts) != 0 &&
		 (errno == ENOENT || errno == ENAMETOOLONG)) ||
		file_digest(sidecar, text));
}

/* A new file of these bytes, written whole and synced. */
static bool file_written(const char *path, const void *bytes, size_t count)
{
	const int descriptor =
		open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, FILE_MODE);
	size_t done = 0;
	while (descriptor >= 0 && done < count) {
		const ssize_t put = write(
			descriptor, (const char *)bytes + done, count - done);
		if (put <= 0)
			break;
		done += (size_t)put;
	}
	const bool is_written = descriptor >= 0 && done == count &&
				fsync(descriptor) == 0;
	return descriptor >= 0 && close(descriptor) == 0 && is_written;
}

static bool directory_synced(const char *path)
{
	const int descriptor = open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
	const bool is_synced = descriptor >= 0 && fsync(descriptor) == 0;
	return descriptor >= 0 && close(descriptor) == 0 && is_synced;
}

static bool ids_written(const char *folder, const char *name,
			const uint32_t *ids, uint32_t count)
{
	char path[PATH_BYTES];
	size_t length = 0;
	char *const text = ids_json(ids, count, &length);
	const bool is_written = text != NULL &&
				(size_t)snprintf(path, sizeof(path), "%s/%s",
						 folder, name) < sizeof(path) &&
				file_written(path, text, length);
	free(text);
	return is_written;
}

/* A string member's span is a hex digest equal to hex. */
static bool member_is(struct json_value value, const char *text)
{
	const size_t length = strlen(text);
	return value.text != NULL &&
	       value.end - value.begin == length + strlen("\"\"") &&
	       value.text[value.begin] == '"' &&
	       memcmp(value.text + value.begin + 1u, text, length) == 0;
}

/* A number member that is zero as json.loads reads it (0, 0.0, -0.0, 0e0). */
static bool member_zero(struct json_value value)
{
	char text[LINE_BYTES];
	const size_t length = value.end - value.begin;
	char *end = NULL;
	if (value.text == NULL || length == 0 || length >= sizeof(text) ||
	    value.text[value.begin] == '"')
		return false;
	memcpy(text, value.text + value.begin, length);
	text[length] = '\0';
	return strtod(text, &end) == 0.0 && *end == '\0';
}

static bool member_number(struct json_value value, uint64_t *number_out)
{
	uint64_t number = 0;
	for (size_t offset = value.begin;
	     value.text != NULL && offset < value.end; offset++) {
		if (value.text[offset] < '0' || value.text[offset] > '9' ||
		    number > UINT32_MAX)
			return false;
		number =
			number * DECIMAL + (uint64_t)(value.text[offset] - '0');
	}
	*number_out = number;
	return value.text != NULL && value.end > value.begin;
}

/* The golden metadata's rules (cli_golden_capture.h), the IDs already parsed. */
static const char *metadata_refusal(const char *text, size_t count,
				    const struct capture_case *own,
				    uint32_t *vocabulary_out)
{
	struct json_value document = { 0 }, key = { 0 }, value = { 0 };
	char prompt_hex[HEX_BYTES], continuation_hex[HEX_BYTES];
	uint64_t schema = 0, vocabulary = 0, positions = 0,
		 nonzero_temperature = 1;
	bool hashes = false, mode = false;
	if (json_read_document(text, count, &document) != JSON_DOCUMENT_VALID ||
	    document.text[document.begin] != '{' ||
	    !ids_digest(own->ids, own->prompt_count, prompt_hex) ||
	    !ids_digest(own->ids + own->prompt_count, own->golden_positions,
			continuation_hex))
		return "metadata.json is no JSON object";
	size_t cursor = document.begin;
	while (json_next(document, &cursor, &key, &value)) {
		if (member_is(key, "schema_version"))
			(void)member_number(value, &schema);
		else if (member_is(key, "vocabulary_size"))
			(void)member_number(value, &vocabulary);
		else if (member_is(key, "generated_positions"))
			(void)member_number(value, &positions);
		else if (member_is(key, "temperature"))
			nonzero_temperature = member_zero(value) ? 0u : 1u;
		else if (member_is(key, "alignment") &&
			 value.text[value.begin] == '{') {
			struct json_value inner_key, inner;
			size_t inner_cursor = value.begin;
			uint32_t matched = 0;
			while (json_next(value, &inner_cursor, &inner_key,
					 &inner)) {
				mode = mode ||
				       (member_is(inner_key, "mode") &&
					member_is(inner,
						  "autoregressive_greedy"));
				matched += member_is(inner_key,
						     "prompt_ids_sha256") &&
					   member_is(inner, prompt_hex);
				matched +=
					member_is(inner_key,
						  "continuation_ids_sha256") &&
					member_is(inner, continuation_hex);
			}
			hashes = matched == ALIGNMENT_HASHES;
		}
	}
	*vocabulary_out = (uint32_t)vocabulary;
	return schema != SCHEMA_VERSION ? "schema_version is not 1" :
	       nonzero_temperature != 0 ? "temperature is not 0" :
	       !mode			? "mode is not autoregressive_greedy" :
	       positions != own->golden_positions ?
		       "generated_positions is not the continuation's count" :
	       !hashes ?
		       "a hash is not the SHA-256 of its IDs' canonical JSON" :
	       vocabulary == 0 ? "no vocabulary_size" :
				 NULL;
}

static bool key_is_valid(const char *key)
{
	const char *const slash = strchr(key, '/');
	bool is_valid = slash != NULL && slash != key && slash[1] != '\0' &&
			strchr(slash + 1, '/') == NULL &&
			strstr(key, "..") == NULL;
	for (const char *byte = key; is_valid && *byte != '\0'; byte++)
		is_valid = byte == slash || (*byte >= 'a' && *byte <= 'z') ||
			   (*byte >= 'A' && *byte <= 'Z') ||
			   (*byte >= '0' && *byte <= '9') || *byte == '_' ||
			   *byte == '.' || *byte == '-';
	return is_valid;
}

/* The IDs of a case's file, parsed as --prompt_token_ids is (any vocabulary below 2^32 here; checked at run). */
static uint32_t *ids_read(const char *folder, const char *name,
			  uint32_t *count_out)
{
	char path[PATH_BYTES];
	size_t count = 0;
	char *const text = (size_t)snprintf(path, sizeof(path), "%s/%s", folder,
					    name) < sizeof(path) ?
				   file_bytes(path, &count) :
				   NULL;
	const uint32_t capacity = (uint32_t)(count / ID_TEXT_BYTES_LEAST + 1u);
	uint32_t *ids = text != NULL ? calloc(capacity, sizeof(*ids)) : NULL;
	if (ids != NULL && prompt_token_ids_parse(text, count, UINT32_MAX, ids,
						  capacity, count_out) != OK) {
		free(ids);
		ids = NULL;
	}
	free(text);
	return ids;
}

static void cases_free(struct capture_case *cases, uint32_t count)
{
	for (uint32_t index = 0; cases != NULL && index < count; index++)
		free(cases[index].ids);
	free(cases);
}

/* Every case of the keys file, read and checked, and the SHA-256 of the bytes read in keys_hex; NULL (one error line)
 * on a refusal. */
static struct capture_case *cases_read(uint32_t *count_out,
				       uint32_t *vocabulary_out,
				       char keys_hex[HEX_BYTES])
{
	const struct golden_capture_options *const own =
		&golden_capture_options;
	size_t byte_count = 0;
	unsigned char digest[SHA256_DIGEST_BYTES];
	char *const keys = file_bytes(own->cases, &byte_count);
	uint32_t count = 0, capacity = 0;
	struct capture_case *cases = NULL;
	const char *refusal =
		keys == NULL || sha256_of(keys, byte_count, digest) != OK ?
			"the keys file is not readable" :
			NULL;
	if (refusal == NULL)
		hex_of(digest, keys_hex);
	for (char *line = keys, *next = NULL;
	     refusal == NULL && line != NULL && *line != '\0'; line = next) {
		next = strchr(line, '\n');
		if (next != NULL)
			*next++ = '\0';
		if (*line == '\0')
			continue;
		if (count == capacity) {
			capacity = capacity == 0 ? CASES_FIRST_CAPACITY :
						   CASES_GROWTH * capacity;
			struct capture_case *const grown =
				realloc(cases, capacity * sizeof(*cases));
			if (grown == NULL) {
				refusal = "out of memory";
				break;
			}
			cases = grown;
		}
		struct capture_case *const own_case = &cases[count];
		*own_case = (struct capture_case){ 0 };
		char folder[PATH_BYTES], *metadata = NULL;
		uint32_t continuation_count = 0, vocabulary = 0;
		size_t metadata_bytes = 0;
		uint32_t *prompt = NULL, *continuation = NULL;
		bool is_listed = false;
		for (uint32_t index = 0; index < count && !is_listed; index++)
			is_listed = strcmp(cases[index].key, line) == 0;
		if (strlen(line) >= KEY_BYTES || !key_is_valid(line) ||
		    (size_t)snprintf(folder, sizeof(folder), "%s/cases/%s",
				     own->golden, line) >= sizeof(folder)) {
			refusal = "a key that is not set/id of [A-Za-z0-9_.-]";
		} else if (is_listed) {
			refusal = "a key listed twice";
		} else {
			(void)snprintf(own_case->key, sizeof(own_case->key),
				       "%s", line);
			count++;
			prompt = ids_read(folder, "prompt_ids.json",
					  &own_case->prompt_count);
			continuation = ids_read(folder, "token_ids.json",
						&continuation_count);
			char path[PATH_BYTES];
			metadata = (size_t)snprintf(path, sizeof(path),
						    "%s/metadata.json",
						    folder) < sizeof(path) ?
					   file_bytes(path, &metadata_bytes) :
					   NULL;
		}
		if (refusal == NULL &&
		    (prompt == NULL || continuation == NULL ||
		     metadata == NULL))
			refusal =
				"an IDs or metadata file is missing or malformed";
		if (refusal == NULL) {
			own_case->ids = calloc((size_t)own_case->prompt_count +
						       continuation_count,
					       sizeof(uint32_t));
			if (own_case->ids == NULL)
				refusal = "out of memory";
		}
		if (refusal == NULL) {
			memcpy(own_case->ids, prompt,
			       own_case->prompt_count * sizeof(uint32_t));
			memcpy(own_case->ids + own_case->prompt_count,
			       continuation,
			       continuation_count * sizeof(uint32_t));
			own_case->golden_positions = continuation_count;
			own_case->positions =
				own->positions != 0 &&
						own->positions <
							continuation_count ?
					own->positions :
					continuation_count;
			refusal = metadata_refusal(metadata, metadata_bytes,
						   own_case, &vocabulary);
			refusal = refusal == NULL && *vocabulary_out != 0 &&
						  vocabulary !=
							  *vocabulary_out ?
					  "cases of different vocabularies" :
					  refusal;
			*vocabulary_out = vocabulary;
		}
		free(prompt);
		free(continuation);
		free(metadata);
		if (refusal != NULL)
			(void)fprintf(stderr, "capture: case %s refused: %s\n",
				      line, refusal);
	}
	if (refusal == NULL && count == 0)
		refusal = "the keys file names no case";
	if (refusal != NULL && keys == NULL)
		(void)fprintf(stderr, "capture: %s: %s\n", own->cases, refusal);
	free(keys);
	if (refusal != NULL) {
		cases_free(cases, count);
		return NULL;
	}
	*count_out = count;
	return cases;
}

/* Each part's rows of the step, gathered into rows [row_count][V] at their columns, then each row checked and kept. */
MUST_CHECK static result_status
rows_landed(void *context, const struct step_row_record *rows,
	    const struct step_row_result *results, uint32_t step_row_count,
	    uint32_t first_row, uint32_t row_count)
{
	struct capture_sink *const sink = context;
	if (sink->io->rows_seen != NULL)
		rows = sink->io->rows_seen(sink->io->context, rows,
					   step_row_count);
	const struct capture_case *const own = sink->own;
	const uint32_t vocabulary = sink->vocabulary_size;
	result_status status =
		row_count == 0 || row_count > LINEAR_MAXIMUM_ROWS ||
				first_row + row_count > step_row_count ||
				sink->landed + row_count > own->positions ?
			STATUS_INVALID_ARGUMENT :
			OK;
	for (uint32_t part = 0; part < sink->part_count && status == OK;
	     part++) {
		const struct sampler_logits_part *const own_part =
			&sink->parts[part];
		const uint64_t row_bytes =
			(uint64_t)own_part->token_count * sizeof(float);
		status = sink->io->arena_read(
			sink->io->context, own_part->card,
			own_part->offset + first_row * row_bytes,
			row_count * row_bytes, sink->part);
		for (uint32_t row = 0; row < row_count && status == OK; row++)
			memcpy(sink->rows + (size_t)row * vocabulary +
				       own_part->first_token,
			       sink->part + (size_t)row * own_part->token_count,
			       row_bytes);
	}
	const char *failure = status == OK ? NULL :
					     "the rows could not be read";
	for (uint32_t row = 0; row < row_count && failure == NULL; row++) {
		const uint32_t k = sink->landed;
		const struct step_row_record *const step_row =
			&rows[first_row + row];
		const struct step_row_result *const result =
			&results[first_row + row];
		const float *const logits =
			sink->rows + (size_t)row * vocabulary;
		uint32_t argmax = 0, bits = 0, drawn_bits = 0;
		bool finite = true;
		for (uint32_t column = 0; column < vocabulary; column++) {
			finite = finite && isfinite(logits[column]);
			argmax = logits[column] > logits[argmax] ? column :
								   argmax;
		}
		const uint32_t drawn = (uint32_t)result->token_id;
		memcpy(&drawn_bits, &result->logit, sizeof(drawn_bits));
		if (drawn < vocabulary)
			memcpy(&bits, &logits[drawn], sizeof(bits));
		failure =
			(uint32_t)step_row->position !=
					own->prompt_count - 1u + k ?
				"a row's position is not its case position's" :
			(uint32_t)step_row->token_id !=
					own->ids[own->prompt_count - 1u + k] ?
				"a row's input token is not the prompt's last or the golden token before it" :
			!finite ?
				"a logit is not finite" :
			drawn != argmax ?
				"the draw is not the row's lowest-ID argmax" :
			bits != drawn_bits ?
				"the draw's logit is not the row's at the draw, bit for bit" :
				NULL;
		const size_t bytes = (size_t)vocabulary * sizeof(float);
		failure = failure == NULL && (write(sink->logits_descriptor,
						    logits,
						    bytes) != (ssize_t)bytes ||
					      sha256_feed(&sink->digest, logits,
							  bytes) != OK) ?
				  "logits.f32 could not be written" :
				  failure;
		sink->drawn[k] = drawn;
		sink->position[k] = (uint32_t)step_row->position;
		sink->input_token[k] = (uint32_t)step_row->token_id;
		sink->step_rows[k] = step_row_count;
		sink->row_in_step[k] = first_row + row;
		sink->logit_bits[k] = drawn_bits;
		sink->landed += failure == NULL ? 1u : 0u;
	}
	/* the driver's committed history at the case's last landing: what the rows read */
	const uint32_t conditioned = own->prompt_count + own->positions - 1u;
	for (uint32_t position = 0;
	     failure == NULL && sink->landed == own->positions &&
	     position < conditioned;
	     position++) {
		sink->conditioning[position] =
			sink->io->history_token(sink->io->context, position);
		failure =
			sink->conditioning[position] != own->ids[position] ?
				"the conditioning is not the prompt and the golden tokens but the last" :
				NULL;
	}
	if (failure != NULL) {
		(void)snprintf(sink->failure, sizeof(sink->failure),
			       "position %" PRIu32 ": %s", sink->landed,
			       failure);
		return status != OK ? status : STATUS_INVALID_ARGUMENT;
	}
	return OK;
}

MUST_CHECK static result_status driver_arena_read(void *context, uint32_t card,
						  uint64_t offset,
						  uint64_t byte_count,
						  void *bytes_out)
{
	return step_arena_read(context, card, offset, byte_count, bytes_out);
}

static uint32_t driver_history_token(void *context, uint32_t position)
{
	return step_history_token(context, 0, position);
}

static bool disk_free_bytes(void *context, const char *path,
			    uint64_t *bytes_out)
{
	(void)context;
	struct statvfs space;
	if (statvfs(path, &space) != 0)
		return false;
	*bytes_out = (uint64_t)space.f_bavail * space.f_frsize;
	return true;
}

/* The events of the command just submitted until its final; FINISHED_AT_LIMIT after count TOKEN events is OK. */
MUST_CHECK static result_status command_events(struct inference_engine *engine,
					       uint32_t count,
					       uint32_t *tokens_out,
					       uint32_t *logit_bits_out)
{
	int descriptor = -1;
	result_status status = engine_event_descriptor(engine, &descriptor);
	struct generation_event event = { .event_kind =
						  GENERATION_EVENT_TOKEN };
	uint32_t taken = 0;
	while (status == OK && event.event_kind == GENERATION_EVENT_TOKEN) {
		struct pollfd entry = { descriptor, POLLIN, 0 };
		bool is_available = false;
		status = poll(&entry, 1, -1) < 0 && errno != EINTR ?
				 STATUS_INPUT_OUTPUT_ERROR :
				 engine_next_event(engine, &event,
						   &is_available);
		if (status != OK || !is_available ||
		    event.event_kind != GENERATION_EVENT_TOKEN)
			continue;
		if (taken < count && tokens_out != NULL)
			tokens_out[taken] = event.token_id;
		if (taken < count && logit_bits_out != NULL)
			memcpy(&logit_bits_out[taken], &event.logit,
			       sizeof(*logit_bits_out));
		taken++;
	}
	return status != OK ? status :
	       event.event_kind == GENERATION_EVENT_FAILED ?
			      (result_status)event.status :
	       event.event_kind != GENERATION_EVENT_FINISHED_AT_LIMIT ||
			       taken != count ?
			      STATUS_DEVICE_ERROR :
			      OK;
}

/* A JSON array of these numbers ("0x%08x" bit patterns as strings with bits), written as a file of the folder. */
static bool numbers_text(char *text, size_t capacity, size_t *length,
			 const char *name, const uint32_t *values,
			 uint32_t count, bool bits)
{
	*length += (size_t)snprintf(text + *length, capacity - *length,
				    "%s\"%s\":[", *length > 1u ? "," : "",
				    name);
	for (uint32_t index = 0; index < count && *length < capacity; index++)
		*length += (size_t)snprintf(
			text + *length, capacity - *length,
			bits ? "%s\"" FLOAT_BITS_TEXT "\"" : "%s%" PRIu32,
			index == 0 ? "" : ",", values[index]);
	*length += (size_t)snprintf(text + *length, capacity - *length, "]");
	return *length < capacity;
}

/* The case's JSON files beside its logits, in folder. */
static bool case_files_written(const char *folder,
			       const struct capture_case *own,
			       const struct capture_sink *sink,
			       const char *capture_json, const char *logits_hex)
{
	char prompt_hex[HEX_BYTES], continuation_hex[HEX_BYTES],
		conditioning_hex[HEX_BYTES], path[PATH_BYTES];
	const uint32_t conditioned = own->prompt_count + own->positions - 1u;
	size_t capacity =
		(size_t)own->positions * WITNESS_ARRAYS * BITS_TEXT_BYTES +
		LINE_BYTES;
	capacity = capacity > TEXT_BYTES_LEAST ? capacity : TEXT_BYTES_LEAST;
	char *const text = malloc(capacity);
	size_t length = 1;
	bool is_written =
		text != NULL &&
		ids_digest(own->ids, own->prompt_count, prompt_hex) &&
		ids_digest(own->ids + own->prompt_count, own->positions,
			   continuation_hex) &&
		ids_digest(sink->conditioning, conditioned, conditioning_hex) &&
		ids_written(folder, "prompt_ids.json", own->ids,
			    own->prompt_count) &&
		ids_written(folder, "token_ids.json",
			    own->ids + own->prompt_count, own->positions) &&
		ids_written(folder, "conditioning_ids.json", sink->conditioning,
			    conditioned) &&
		ids_written(folder, "native_drawn_ids.json", sink->drawn,
			    own->positions);
	if (is_written) {
		text[0] = '{';
		is_written =
			numbers_text(text, capacity, &length,
				     "positions_in_step", sink->row_in_step,
				     own->positions, false) &&
			numbers_text(text, capacity, &length, "step_rows",
				     sink->step_rows, own->positions, false) &&
			numbers_text(text, capacity, &length, "drawn",
				     sink->drawn, own->positions, false) &&
			numbers_text(text, capacity, &length,
				     "drawn_logit_bits", sink->logit_bits,
				     own->positions, true) &&
			numbers_text(text, capacity, &length, "position",
				     sink->position, own->positions, false) &&
			numbers_text(text, capacity, &length, "input_token",
				     sink->input_token, own->positions, false);
		length += (size_t)snprintf(text + length, capacity - length,
					   ",\"first_position\":%" PRIu32 "}\n",
					   own->prompt_count - 1u);
		is_written = is_written && length < capacity &&
			     (size_t)snprintf(path, sizeof(path),
					      "%s/witness.json",
					      folder) < sizeof(path) &&
			     file_written(path, text, length);
	}
	if (is_written) {
		length = (size_t)snprintf(
			text, capacity,
			"{\"schema_version\":1,\"vocabulary_size\":%" PRIu32
			",\"generated_positions\":%" PRIu32
			",\"temperature\":0,\"alignment\":{\"mode\":\"teacher_forced\","
			"\"prompt_ids_sha256\":\"%s\",\"continuation_ids_sha256\":\"%s\","
			"\"conditioning_ids_sha256\":\"%s\"},\"capture\":{%s,"
			"\"truncated\":%s,\"logits_sha256\":\"%s\",\"witnessed_positions\":%" PRIu32
			"}}\n",
			sink->vocabulary_size, own->positions, prompt_hex,
			continuation_hex, conditioning_hex, capture_json,
			own->positions < own->golden_positions ? "true" :
								 "false",
			logits_hex, sink->landed);
		is_written = length < capacity &&
			     (size_t)snprintf(path, sizeof(path),
					      "%s/metadata.json",
					      folder) < sizeof(path) &&
			     file_written(path, text, length);
	}
	free(text);
	return is_written;
}

static const char *const capture_files[] = {
	"logits.f32",	  "metadata.json",	   "prompt_ids.json",
	"token_ids.json", "conditioning_ids.json", "native_drawn_ids.json",
	"witness.json",	  "free_token_ids.json",   "free_logits.json"
};

/* Whether path is a folder itself (lstat: a link is not), or nothing. */
static bool folder_or_none(const char *path)
{
	struct stat facts;
	return lstat(path, &facts) != 0 ? errno == ENOENT :
					  S_ISDIR(facts.st_mode);
}

/* A case folder's own files removed, then the folder; false when anything else stays in it or when it is a link (it
 * is opened O_NOFOLLOW and its files removed through that descriptor: nothing outside OUT is touched). */
static bool folder_cleared(const char *folder)
{
	struct stat facts;
	if (lstat(folder, &facts) != 0)
		return errno == ENOENT;
	const int descriptor =
		S_ISDIR(facts.st_mode) ?
			open(folder,
			     O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC) :
			-1;
	bool is_cleared = descriptor >= 0;
	for (uint32_t index = 0;
	     is_cleared &&
	     index < sizeof(capture_files) / sizeof(capture_files[0]);
	     index++)
		is_cleared = unlinkat(descriptor, capture_files[index], 0) ==
				     0 ||
			     errno == ENOENT;
	if (descriptor >= 0)
		(void)close(descriptor);
	return is_cleared && rmdir(folder) == 0;
}

/* The JSON member name: text's bytes up to a newline, quoted, behind length in buffer, which stays NUL-terminated;
 * length becomes capacity when they do not fit. */
static void quoted(char *buffer, size_t capacity, size_t *length,
		   const char *name, const char *text)
{
	struct json_writer writer = { buffer + *length, capacity - *length, 0 };
	const char *const line_end = strchr(text, '\n');
	const char *const opening = *length > 1u ? ",\"" : "\"";
	json_write_bytes(&writer, opening, strlen(opening));
	json_write_bytes(&writer, name, strlen(name));
	json_write_bytes(&writer, "\":", strlen("\":"));
	json_write_string(
		&writer,
		(struct json_bytes){ text, line_end != NULL ?
						   (size_t)(line_end - text) :
						   strlen(text) });
	*length = writer.length < writer.capacity ? *length + writer.length :
						    capacity;
	if (*length < capacity)
		buffer[*length] = '\0';
}

/* The words as a JSON array of strings (Python's json.dumps), NUL-terminated, in a new buffer; NULL without memory. */
static char *words_json(char *const *words, int count)
{
	size_t capacity = ARRAY_FRAME_BYTES;
	for (int index = 0; index < count; index++)
		capacity += JSON_BYTES_PER_BYTE * strlen(words[index]) +
			    QUOTES_AND_COMMA;
	char *const text = malloc(capacity);
	struct json_writer writer = { text, capacity, 0 };
	if (text == NULL)
		return NULL;
	json_write_bytes(&writer, "[", 1u);
	for (int index = 0; index < count; index++) {
		if (index != 0)
			json_write_bytes(&writer, ",", 1u);
		json_write_string(&writer,
				  (struct json_bytes){ words[index],
						       strlen(words[index]) });
	}
	json_write_bytes(&writer, "]", 1u);
	if (writer.length >= capacity) {
		free(text);
		return NULL;
	}
	text[writer.length] = '\0';
	return text;
}

MUST_CHECK static result_status
ordinary_written(const struct golden_capture_run *run,
		 const struct capture_case *own, const char *folder,
		 uint32_t *tokens, uint32_t *bits)
{
	const struct generation_command_record command = {
		.new_token_ids = own->ids,
		.new_token_count = own->prompt_count,
		.generated_token_limit = own->positions,
		.sampler_top_p = 1.0f,
		.request_tag = 1u
	};
	uint32_t taken = 0;
	result_status status = engine_submit(run->engine, &command);
	int descriptor = -1;
	status = status != OK ?
			 status :
			 engine_event_descriptor(run->engine, &descriptor);
	struct generation_event event = { .event_kind =
						  GENERATION_EVENT_TOKEN };
	while (status == OK && event.event_kind == GENERATION_EVENT_TOKEN) {
		struct pollfd entry = { descriptor, POLLIN, 0 };
		bool is_available = false;
		status = poll(&entry, 1, -1) < 0 && errno != EINTR ?
				 STATUS_INPUT_OUTPUT_ERROR :
				 engine_next_event(run->engine, &event,
						   &is_available);
		if (status == OK && is_available &&
		    event.event_kind == GENERATION_EVENT_TOKEN &&
		    taken < own->positions) {
			tokens[taken] = event.token_id;
			memcpy(&bits[taken++], &event.logit, sizeof(*bits));
		}
	}
	status = status != OK ? status :
		 event.event_kind == GENERATION_EVENT_FAILED ?
				(result_status)event.status :
				OK;
	char path[PATH_BYTES],
		*text = malloc((size_t)taken * BITS_TEXT_BYTES + LINE_BYTES);
	size_t length = 0;
	if (status == OK && text != NULL) {
		length = (size_t)sprintf(text, "[");
		for (uint32_t index = 0; index < taken; index++)
			length += (size_t)sprintf(text + length,
						  "%s\"" FLOAT_BITS_TEXT "\"",
						  index == 0 ? "" : ",",
						  bits[index]);
		length += (size_t)sprintf(text + length, "]\n");
	}
	status = status != OK ? status :
		 text == NULL ? STATUS_OUT_OF_MEMORY :
		 !ids_written(folder, "free_token_ids.json", tokens, taken) ||
				 (size_t)snprintf(path, sizeof(path),
						  "%s/free_logits.json",
						  folder) >= sizeof(path) ||
				 !file_written(path, text, length) ?
				STATUS_INPUT_OUTPUT_ERROR :
				OK;
	free(text);
	return status;
}

/* One case: the forced command streaming into OUT/cases/KEY.partial, then its files, the rename and its line. */
MUST_CHECK static result_status
case_captured(const struct golden_capture_run *run,
	      const struct capture_case *own, struct capture_sink *sink,
	      uint32_t rows, const char *capture_json)
{
	const struct golden_capture_options *const options =
		&golden_capture_options;
	char partial[PATH_BYTES], final[PATH_BYTES], set[PATH_BYTES],
		path[PATH_BYTES], logits_hex[HEX_BYTES] = "", line[LINE_BYTES];
	unsigned char digest[SHA256_DIGEST_BYTES];
	const char *const slash = strchr(own->key, '/');
	const bool named =
		(size_t)snprintf(partial, sizeof(partial),
				 "%s/cases/%s.partial", options->out,
				 own->key) < sizeof(partial) &&
		(size_t)snprintf(final, sizeof(final), "%s/cases/%s",
				 options->out, own->key) < sizeof(final) &&
		(size_t)snprintf(set, sizeof(set), "%s/cases/%.*s",
				 options->out, (int)(slash - own->key),
				 own->key) < sizeof(set) &&
		(size_t)snprintf(path, sizeof(path), "%s/logits.f32", partial) <
			sizeof(path);
	result_status status =
		named &&
				(mkdir(set, DIRECTORY_MODE) == 0 ||
				 (errno == EEXIST && folder_or_none(set))) &&
				mkdir(partial, DIRECTORY_MODE) == 0 ?
			OK :
			STATUS_INPUT_OUTPUT_ERROR;
	sink->own = own;
	sink->landed = 0;
	sink->failure[0] = '\0';
	sink->logits_descriptor =
		status == OK ?
			open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
			     FILE_MODE) :
			-1;
	sha256_begin(&sink->digest);
	const struct generation_command_record command = {
		.new_token_ids = own->ids,
		.new_token_count = own->prompt_count,
		.generated_token_limit = own->positions,
		.sampler_top_p = 1.0f,
		.request_tag = 1u
	};
	const struct engine_forced_continuation forced = {
		own->ids + own->prompt_count, own->positions, rows, rows_landed,
		sink
	};
	status = status == OK && sink->logits_descriptor < 0 ?
			 STATUS_INPUT_OUTPUT_ERROR :
			 status;
	status = status != OK ?
			 status :
			 engine_submit_forced(run->engine, &command, &forced);
	status = status != OK ? status :
				command_events(run->engine, own->positions,
					       NULL, NULL);
	status = status == OK && sink->landed != own->positions ?
			 STATUS_DEVICE_ERROR :
			 status;
	status = status == OK && (fsync(sink->logits_descriptor) != 0 ||
				  sha256_finish(&sink->digest, digest) != OK) ?
			 STATUS_INPUT_OUTPUT_ERROR :
			 status;
	if (sink->logits_descriptor >= 0 && close(sink->logits_descriptor) != 0)
		status = status == OK ? STATUS_INPUT_OUTPUT_ERROR : status;
	if (status == OK)
		hex_of(digest, logits_hex);
	status = status == OK && options->free_run != 0 ?
			 ordinary_written(run, own, partial,
					  sink->drawn + own->positions,
					  sink->logit_bits + own->positions) :
			 status;
	status = status == OK && (!case_files_written(partial, own, sink,
						      capture_json,
						      logits_hex) ||
				  rename(partial, final) != 0 ||
				  !directory_synced(set)) ?
			 STATUS_INPUT_OUTPUT_ERROR :
			 status;
	const int lines =
		status == OK ?
			open(options->out, O_RDONLY | O_DIRECTORY | O_CLOEXEC) :
			-1;
	const int jsonl =
		lines >= 0 ? openat(lines, "cases.jsonl",
				    O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC,
				    FILE_MODE) :
			     -1;
	const int length = snprintf(line, sizeof(line),
				    "{\"case\":\"%s\",\"positions\":%" PRIu32
				    ",\"logits_sha256\":\"%s\"}\n",
				    own->key, own->positions,
				    status == OK ? logits_hex : "");
	status =
		status == OK && (jsonl < 0 || length <= 0 ||
				 write(jsonl, line, (size_t)length) != length ||
				 fsync(jsonl) != 0) ?
			STATUS_INPUT_OUTPUT_ERROR :
			status;
	if (jsonl >= 0)
		(void)close(jsonl);
	if (lines >= 0)
		(void)close(lines);
	if (status !=
	    OK) /* a product output: release compiles the log hooks out */
		(void)fprintf(stderr, "capture: case %s failed (%s)%s%s\n",
			      own->key, status_text(status),
			      sink->failure[0] != '\0' ? ": " : "",
			      sink->failure);
	else
		(void)printf("capture %s: %" PRIu32 " positions, logits %s\n",
			     own->key, own->positions, logits_hex);
	return status;
}

/* OUT's plan: new (its manifest written), or the same plan's (its done cases marked, other folders cleared). */
static const char *root_refusal(const char *manifest, size_t manifest_length,
				const char *plan_hex,
				struct capture_case *cases, uint32_t count)
{
	const char *const out = golden_capture_options.out;
	char path[PATH_BYTES], *jsonl = NULL, *held = NULL;
	size_t held_length = 0, jsonl_length = 0;
	if ((size_t)snprintf(path, sizeof(path), "%s/CAPTURE_MANIFEST.json",
			     out) >= sizeof(path))
		return "OUT's path is too long";
	if (mkdir(out, DIRECTORY_MODE) == 0 ||
	    (errno == EEXIST && rmdir(out) == 0 &&
	     mkdir(out, DIRECTORY_MODE) == 0)) {
		char cases_path[PATH_BYTES];
		return (size_t)snprintf(cases_path, sizeof(cases_path),
					"%s/cases", out) < sizeof(cases_path) &&
				       mkdir(cases_path, DIRECTORY_MODE) == 0 &&
				       file_written(path, manifest,
						    manifest_length) &&
				       directory_synced(out) ?
			       NULL :
			       "OUT could not be made";
	}
	held = file_bytes(path, &held_length);
	char wanted[LINE_BYTES];
	(void)snprintf(wanted, sizeof(wanted), "\"plan_sha256\":\"%s\"",
		       plan_hex);
	const bool same = held != NULL && strstr(held, wanted) != NULL;
	free(held);
	if (!same)
		return "OUT holds something else, or a capture of another plan";
	if ((size_t)snprintf(path, sizeof(path), "%s/cases", out) >=
		    sizeof(path) ||
	    !folder_or_none(path))
		return "OUT's cases folder is a link";
	if ((size_t)snprintf(path, sizeof(path), "%s/cases.jsonl", out) <
	    sizeof(path))
		jsonl = file_bytes(path, &jsonl_length);
	for (uint32_t index = 0; index < count; index++) {
		char final[PATH_BYTES], partial[PATH_BYTES], set[PATH_BYTES],
			prefix[LINE_BYTES];
		struct stat facts;
		const char *const slash = strchr(cases[index].key, '/');
		(void)snprintf(prefix, sizeof(prefix), "{\"case\":\"%s\",",
			       cases[index].key);
		const bool named =
			(size_t)snprintf(final, sizeof(final), "%s/cases/%s",
					 out,
					 cases[index].key) < sizeof(final) &&
			(size_t)snprintf(partial, sizeof(partial),
					 "%s/cases/%s.partial", out,
					 cases[index].key) < sizeof(partial) &&
			(size_t)snprintf(set, sizeof(set), "%s/cases/%.*s", out,
					 (int)(slash - cases[index].key),
					 cases[index].key) < sizeof(set);
		if (named && (!folder_or_none(set) || !folder_or_none(final) ||
			      !folder_or_none(partial))) {
			free(jsonl);
			return "a set or case folder is a link or no folder";
		}
		bool listed = false;
		for (const char *line = jsonl; line != NULL && !listed;
		     line = strchr(line, '\n') != NULL ?
				    strchr(line, '\n') + 1 :
				    NULL)
			listed = strncmp(line, prefix, strlen(prefix)) == 0;
		cases[index].is_done = named && listed &&
				       lstat(final, &facts) == 0 &&
				       S_ISDIR(facts.st_mode);
		if (!named || !folder_cleared(partial) ||
		    (!cases[index].is_done && !folder_cleared(final))) {
			free(jsonl);
			return "a case folder holds files the capture does not write";
		}
	}
	free(jsonl);
	return NULL;
}

/* The run's digests (the plan's: the executable, the argv, the keys file, the dataset's MANIFEST.json, null when it has
 * none; the pack header's; the EXL3 map's descriptor's, null without one), each named when it cannot be made: NULL, or
 * the refusal. The descriptor joins no plan digest: it describes the map, whose pack header digest is recorded. */
static const char *digests_refusal(const struct golden_capture_run *run,
				   struct capture_digests *own)
{
	char path[PATH_BYTES], plan_text[LINE_BYTES];
	unsigned char digest[SHA256_DIGEST_BYTES];
	struct sha256_stream stream;
	struct stat facts;
	bool fed = run->argument_count >= 0 &&
		   (run->arguments != NULL || run->argument_count == 0);
	sha256_begin(&stream);
	for (int index = 0; fed && index < run->argument_count; index++)
		fed = sha256_feed(&stream, run->arguments[index],
				  strlen(run->arguments[index]) + 1u) == OK;
	if (!fed || sha256_finish(&stream, digest) != OK)
		return "the argv could not be digested";
	hex_of(digest, own->argv);
	if (!file_digest(run->executable != NULL ? run->executable : EXECUTABLE,
			 own->executable))
		return "the executable could not be digested";
	if (sha256_of(run->pack_header, (size_t)run->pack_header_bytes,
		      digest) != OK)
		return "the pack header could not be digested";
	hex_of(digest, own->header);
	if (!descriptor_digest(run->map_path, own->descriptor))
		return "the map's descriptor could not be digested";
	if ((size_t)snprintf(path, sizeof(path), "%s/MANIFEST.json",
			     golden_capture_options.golden) >= sizeof(path) ||
	    ((lstat(path, &facts) == 0 || errno != ENOENT) &&
	     !file_digest(path, own->golden)))
		return "the golden's MANIFEST.json could not be digested";
	const int length = snprintf(
		plan_text, sizeof(plan_text),
		"executable=%s\nargv=%s\nkeys=%s\ngolden_manifest=%s\n",
		own->executable, own->argv, own->keys, own->golden);
	if (length <= 0 || (size_t)length >= sizeof(plan_text) ||
	    sha256_of(plan_text, (size_t)length, digest) != OK)
		return "the plan could not be digested";
	hex_of(digest, own->plan);
	return NULL;
}

/* A JSON string member of a digest: "hex", or null. */
static const char *digest_quote(const char *hex)
{
	return strcmp(hex, NULL_TEXT) == 0 ? "" : "\"";
}

result_status golden_capture_run(const struct golden_capture_run *run)
{
	extern char **environ;
	const struct golden_capture_options *const options =
		&golden_capture_options;
	struct engine_description_record description = { 0 };
	struct sampler_logits_part parts[BACKEND_MAXIMUM_CARDS];
	struct capture_sink sink = { .logits_descriptor = -1 };
	uint32_t count = 0, vocabulary = 0, part_count = 0, most_positions = 0,
		 most_tokens = 0, most_part = 0;
	uint64_t remaining = 0;
	const struct golden_capture_io driver_io = { driver_arena_read,
						     driver_history_token,
						     disk_free_bytes, NULL,
						     run->driver };
	sink.io = run->io != NULL ? run->io : &driver_io;
	const char *refusal = NULL;
	for (char **variable = environ;
	     variable != NULL && *variable != NULL && refusal == NULL;
	     variable++)
		refusal =
			strncmp(*variable, "DIAG_", strlen("DIAG_")) == 0 ?
				"a DIAG_ variable is set (out of capture's scope)" :
				NULL;
	refusal =
		refusal != NULL ?
			refusal :
		engine_describe(run->engine, &description) != OK ?
			"the engine is not described" :
		run->parts != NULL ?
			(run->part_count == 0 || run->part_count >
							 BACKEND_MAXIMUM_CARDS ?
				 "no logits parts given" :
				 NULL) :
		sampler_logits_parts(run->program, description.vocabulary_size,
				     parts, BACKEND_MAXIMUM_CARDS,
				     &part_count) != OK ?
			"no logits parts in the driver's thin program" :
			NULL;
	if (refusal == NULL && run->parts != NULL) {
		memcpy(parts, run->parts, run->part_count * sizeof(*parts));
		part_count = run->part_count;
	}
	const uint32_t rows = options->rows != 0 ?
				      options->rows :
				      description.speculative_depth + 1u;
	if (refusal == NULL && rows > description.speculative_depth + 1u)
		refusal = "--capture_rows is above speculative_depth + 1";
	struct capture_digests digests = { NULL_TEXT, NULL_TEXT, NULL_TEXT,
					   NULL_TEXT, NULL_TEXT, NULL_TEXT,
					   NULL_TEXT };
	struct capture_case *const cases =
		refusal == NULL ?
			cases_read(&count, &vocabulary, digests.keys) :
			NULL;
	refusal = refusal != NULL ?
			  refusal :
		  cases == NULL ?
			  "the cases" :
		  vocabulary != description.vocabulary_size ?
			  "the cases' vocabulary_size is not the model's" :
			  NULL;
	for (uint32_t index = 0; refusal == NULL && index < count; index++) {
		const struct capture_case *const own = &cases[index];
		const uint32_t tokens = own->prompt_count + own->positions;
		for (uint32_t id = 0; refusal == NULL && id < tokens; id++)
			refusal = own->ids[id] >= vocabulary ?
					  "a token ID outside the vocabulary" :
					  NULL;
		refusal =
			refusal == NULL &&
					(own->prompt_count == 0 ||
					 tokens - 1u >
						 description
							 .context_token_limit) ?
				"a case's prompt and positions but the last exceed --context_tokens" :
				refusal;
		most_positions = own->positions > most_positions ?
					 own->positions :
					 most_positions;
		most_tokens = tokens > most_tokens ? tokens : most_tokens;
	}
	for (uint32_t part = 0; part < part_count; part++)
		most_part = parts[part].token_count > most_part ?
				    parts[part].token_count :
				    most_part;
	/* the digests, each named when it fails, before any text is built */
	refusal = refusal != NULL ? refusal : digests_refusal(run, &digests);
	uint64_t positions = 0;
	for (uint32_t index = 0; refusal == NULL && index < count; index++) {
		positions += cases[index].positions;
		remaining += (uint64_t)cases[index].positions * vocabulary *
			     sizeof(float);
	}
	/* the capture object every metadata.json holds, and CAPTURE_MANIFEST.json: NUL-terminated in new buffers */
	char *const capture_json =
		refusal == NULL ? malloc(CAPTURE_JSON_BYTES) : NULL;
	char *const argv_json =
		refusal == NULL ?
			words_json(run->arguments, run->argument_count) :
			NULL;
	const size_t manifest_capacity =
		MANIFEST_BYTES + CAPTURE_JSON_BYTES +
		(argv_json != NULL ? strlen(argv_json) : 0);
	char *const manifest = argv_json != NULL && capture_json != NULL ?
				       malloc(manifest_capacity) :
				       NULL;
	refusal = refusal == NULL && manifest == NULL ? "out of memory" :
							refusal;
	size_t capture_length = 0;
	if (refusal == NULL) {
		const int length = snprintf(
			capture_json, CAPTURE_JSON_BYTES,
			"\"executable_sha256\":\"%s\",\"argv_sha256\":\"%s\","
			"\"pack_header_sha256\":\"%s\",\"descriptor_sha256\":%s%s%s,"
			"\"rows\":%" PRIu32 ",\"context_tokens\":%" PRIu32,
			digests.executable, digests.argv, digests.header,
			digest_quote(digests.descriptor), digests.descriptor,
			digest_quote(digests.descriptor), rows,
			description.context_token_limit);
		capture_length = length > 0 && (size_t)length <
							 CAPTURE_JSON_BYTES ?
					 (size_t)length :
					 CAPTURE_JSON_BYTES;
		quoted(capture_json, CAPTURE_JSON_BYTES, &capture_length,
		       "version", run->version != NULL ? run->version : "");
		quoted(capture_json, CAPTURE_JSON_BYTES, &capture_length,
		       "placement",
		       run->placement != NULL ? run->placement : "");
		quoted(capture_json, CAPTURE_JSON_BYTES, &capture_length,
		       "weights", run->weights != NULL ? run->weights : "");
		quoted(capture_json, CAPTURE_JSON_BYTES, &capture_length, "map",
		       run->map != NULL ? run->map : "");
		refusal = capture_length >= CAPTURE_JSON_BYTES ?
				  "the recorded options are too long" :
				  NULL;
	}
	int manifest_length = 0;
	if (refusal == NULL)
		manifest_length = snprintf(
			manifest, manifest_capacity,
			"{\"plan_sha256\":\"%s\",\"executable_sha256\":\"%s\",\"argv_sha256\":\"%s\","
			"\"keys_sha256\":\"%s\",\"golden_manifest_sha256\":%s%s%s,\"cases\":%" PRIu32
			",\"positions\":%" PRIu64 ",\"bytes\":%" PRIu64
			",\"argv\":%s,\"capture\":{%s}}\n",
			digests.plan, digests.executable, digests.argv,
			digests.keys, digest_quote(digests.golden),
			digests.golden, digest_quote(digests.golden), count,
			positions, remaining, argv_json, capture_json);
	refusal = refusal != NULL ?
			  refusal :
		  manifest_length <= 0 ||
				  (size_t)manifest_length >= manifest_capacity ?
			  "the manifest is too long" :
			  root_refusal(manifest, (size_t)manifest_length,
				       digests.plan, cases, count);
	free(argv_json);
	free(manifest);
	remaining = 0;
	for (uint32_t index = 0; refusal == NULL && index < count; index++)
		remaining += cases[index].is_done ?
				     0u :
				     (uint64_t)cases[index].positions *
					     vocabulary * sizeof(float);
	uint64_t free_bytes = 0;
	refusal =
		refusal == NULL &&
				(!sink.io->free_bytes(sink.io->context,
						      options->out,
						      &free_bytes) ||
				 free_bytes <
					 remaining +
						 GOLDEN_CAPTURE_FLOOR_BYTES) ?
			"OUT's disk has less free space than the remaining rows and 1 GiB" :
			refusal;
	if (refusal != NULL) {
		(void)fprintf(stderr, "capture refused: %s\n", refusal);
		free(capture_json);
		cases_free(cases, count);
		return STATUS_INVALID_ARGUMENT;
	}
	sink.parts = parts;
	sink.part_count = part_count;
	sink.vocabulary_size = vocabulary;
	sink.rows = malloc((size_t)LINEAR_MAXIMUM_ROWS * vocabulary *
			   sizeof(float));
	sink.part =
		malloc((size_t)LINEAR_MAXIMUM_ROWS * most_part * sizeof(float));
	/* the draws and bits hold a free run's behind the forced ones */
	sink.drawn = calloc(RUNS * (size_t)most_positions, sizeof(uint32_t));
	sink.logit_bits =
		calloc(RUNS * (size_t)most_positions, sizeof(uint32_t));
	sink.step_rows = calloc(most_positions, sizeof(uint32_t));
	sink.row_in_step = calloc(most_positions, sizeof(uint32_t));
	sink.position = calloc(most_positions, sizeof(uint32_t));
	sink.input_token = calloc(most_positions, sizeof(uint32_t));
	sink.conditioning = calloc(most_tokens, sizeof(uint32_t));
	result_status status = sink.rows == NULL || sink.part == NULL ||
					       sink.drawn == NULL ||
					       sink.logit_bits == NULL ||
					       sink.step_rows == NULL ||
					       sink.row_in_step == NULL ||
					       sink.position == NULL ||
					       sink.input_token == NULL ||
					       sink.conditioning == NULL ?
				       STATUS_OUT_OF_MEMORY :
				       OK;
	uint32_t captured = 0, skipped = 0;
	for (uint32_t index = 0; status == OK && index < count; index++) {
		skipped += cases[index].is_done ? 1u : 0u;
		if (cases[index].is_done)
			continue;
		status = case_captured(run, &cases[index], &sink, rows,
				       capture_json);
		captured += status == OK ? 1u : 0u;
	}
	if (status == OK)
		(void)printf("capture: %" PRIu32 " cases captured, %" PRIu32
			     " already done, rows %" PRIu32 ", plan %s\n",
			     captured, skipped, rows, digests.plan);
	free(sink.rows);
	free(sink.part);
	free(sink.drawn);
	free(sink.logit_bits);
	free(sink.step_rows);
	free(sink.row_in_step);
	free(sink.position);
	free(sink.input_token);
	free(capture_json);
	free(sink.conditioning);
	cases_free(cases, count);
	return status;
}
