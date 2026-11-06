/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * front_end_golden_test: the tokenizer and the chat template against the model's own tokenizer and encoder, on the
 * golden sets of tools/golden/make_front_end_golden.py.
 *
 *     front_end_golden_test <tables file> <tokenizer golden file> <template golden file>
 *
 * The tables file is what tools/convert/front_end_tables.py writes from the checkpoint's tokenizer.json; the golden
 * files' text states their format. Every text must give the reference's ids, and the ids its reference text back,
 * token by token; every conversation the encoder renders must give its prompt byte for byte, and the prompt its ids;
 * every one the encoder or the contract refuses must be refused. Then the longest text is encoded, and the longest
 * conversation rendered and its prompt encoded, TIMED_RUNS times each, and the medians printed. Without arguments,
 * as `make check` runs every test, it says that it compared nothing. The gate, on the workstation:
 *     front_end_tables.py write <checkpoint>/tokenizer.json <tables>
 *     make_front_end_golden.py --checkpoint <checkpoint> --output-directory <golden> --corrupt --text <files>
 *     build/<each build>/front_end_golden_test <tables> <golden>/tokenizer_golden.bin <golden>/template_golden.bin
 * and once on the *_corrupt.bin files, whose changed id it must report with the status 1.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "anemone/chat_template.h"
#include "anemone/status.h"
#include "anemone/tokenizer.h"

#define TOKENIZER_MAGIC "ANEMTOKG"
#define TEMPLATE_MAGIC "ANEMCHTG"
#define MAGIC_BYTES ((size_t)8)
#define FORMAT_VERSION 1u
#define THINKING 1u                                   /* the flags of a conversation */
#define DROPS_THINKING 2u
#define CONTINUES 4u
#define HAS_TOOLS 8u
#define REFUSED 16u
#define REFUSED_BY_CONTRACT 32u                      /* the encoder renders it, the template refuses it */
#define ID_SLACK 64u                                  /* room for more ids than expected, to see how many came */
#define PROMPT_SLACK ((size_t)65536)
#define TIMED_RUNS 5
#define MEDIAN_DIVISOR 2
#define NANOSECONDS_PER_SECOND 1e9
#define MISMATCHES_PRINTED 8
#define SHOWN_BYTES 48
#define DETAIL_CAPACITY 256
#define PROMPT_ROOM_FACTOR 2                          /* a prompt twice as long as expected still shows how */
#define MARKER_TOKEN_CAPACITY 256u
#define DSML "｜DSML｜"

enum test_status { TEST_PASSED = 0, TEST_FAILED = 1, TEST_WRONG_USAGE = 2 };
enum argument { PROGRAM_NAME, TABLES_FILE, TOKENIZER_GOLDEN_FILE, TEMPLATE_GOLDEN_FILE, ARGUMENT_COUNT };

struct reader {                                       /* a golden file in memory, read from the front */
    const unsigned char *bytes;
    size_t byte_count;
    size_t position;
    bool is_short;                                    /* a read went past the end: the file is not whole */
};

static int mismatches;

static void *read_file(const char *path, size_t *byte_count_out)
{
    unsigned char *bytes = NULL;
    long end_position = -1;
    FILE *const file = fopen(path, "rb");
    if (file == NULL) {
        (void)fprintf(stderr, "FAILED: %s cannot be opened\n", path);
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) == 0) {
        end_position = ftell(file);
    }
    if (end_position >= 0 && fseek(file, 0, SEEK_SET) == 0) {
        bytes = malloc((size_t)end_position + 1);
    }
    if (bytes != NULL && fread(bytes, 1, (size_t)end_position, file) != (size_t)end_position) {
        free(bytes);
        bytes = NULL;
    }
    (void)fclose(file);
    if (bytes == NULL) {
        (void)fprintf(stderr, "FAILED: %s cannot be read\n", path);
        return NULL;
    }
    *byte_count_out = (size_t)end_position;
    return bytes;
}

static const unsigned char *take(struct reader *reader, size_t byte_count)
{
    if (reader->is_short || byte_count > reader->byte_count - reader->position) {
        reader->is_short = true;
        return NULL;
    }
    reader->position += byte_count;
    return reader->bytes + reader->position - byte_count;
}

static uint64_t take_number(struct reader *reader, size_t byte_count)
{
    uint64_t value = 0;
    const unsigned char *const bytes = take(reader, byte_count);
    if (bytes != NULL) {
        memcpy(&value, bytes, byte_count);            /* little-endian, as the host */
    }
    return value;
}

/* A run of bytes after its count of count_bytes bytes. */
static const char *take_sized(struct reader *reader, size_t count_bytes, size_t *byte_count_out)
{
    *byte_count_out = (size_t)take_number(reader, count_bytes);
    const unsigned char *const bytes = take(reader, *byte_count_out);
    return bytes != NULL ? (const char *)bytes : "";
}

/* Ids after their uint32 count, copied out (they need not be aligned in the file). NULL when the file is short. */
static uint32_t *take_ids(struct reader *reader, uint32_t *count_out)
{
    *count_out = (uint32_t)take_number(reader, sizeof(uint32_t));
    const unsigned char *const bytes = take(reader, (size_t)*count_out * sizeof(uint32_t));
    uint32_t *const ids = bytes == NULL ? NULL : malloc(((size_t)*count_out + 1) * sizeof(uint32_t));
    if (ids != NULL) {
        memcpy(ids, bytes, (size_t)*count_out * sizeof(uint32_t));
    }
    return ids;
}

static bool open_reader(struct reader *reader, const char *magic)
{
    const unsigned char *const found = take(reader, MAGIC_BYTES);
    return found != NULL && memcmp(found, magic, MAGIC_BYTES) == 0 &&
           take_number(reader, sizeof(uint32_t)) == FORMAT_VERSION;
}

static double monotonic_seconds(void)
{
    struct timespec now = {0};
    const int clock_status = clock_gettime(CLOCK_MONOTONIC, &now);
    return clock_status == 0 ? (double)now.tv_sec + (double)now.tv_nsec / NANOSECONDS_PER_SECOND : 0.0;
}

static int compare_doubles(const void *first, const void *second)
{
    const double first_value = *(const double *)first;
    const double second_value = *(const double *)second;
    return (first_value > second_value) - (first_value < second_value);
}

static void report(const char *what, const char *name, size_t name_bytes, size_t place, const char *detail)
{
    if (mismatches++ < MISMATCHES_PRINTED) {
        (void)fprintf(stderr, "FAILED: %s of \"%.*s\" at %zu: %s\n", what, (int)name_bytes, name, place, detail);
    }
}

/* The first place where two runs of ids or bytes differ, their common length when one ends first. */
static size_t first_difference(const void *first, size_t first_count, const void *second, size_t second_count,
                               size_t element_bytes)
{
    size_t place = 0;
    while (place < first_count && place < second_count &&
           memcmp((const char *)first + place * element_bytes, (const char *)second + place * element_bytes,
                  element_bytes) == 0) {
        place++;
    }
    return place;
}

/* The ids of a text against the expected ones, and their bytes against the expected text. */
static void compare_encoding(const struct text_tokenizer *tokenizer,
			     const char *what, const char *name,
			     size_t name_bytes, const char *text,
			     size_t text_bytes, const uint32_t *expected,
			     uint32_t expected_count, const char *expected_text,
			     size_t expected_text_bytes)
{
    uint32_t count = 0;
    char detail[DETAIL_CAPACITY] = {0};
    uint32_t *const ids = malloc(((size_t)expected_count + ID_SLACK) * sizeof(uint32_t));
    const result_status status =
	    ids == NULL ? STATUS_OUT_OF_MEMORY :
			  tokenizer_encode(tokenizer, text, text_bytes, ids,
					   expected_count + ID_SLACK, &count);
    const size_t place = status == STATUS_OK ?
				 first_difference(ids, count, expected,
						  expected_count,
						  sizeof(uint32_t)) :
				 0;
    if (status != STATUS_OK || count != expected_count || place != count) {
	    (void)snprintf(detail, sizeof(detail),
			   "status %d, %u ids for %u, id %u for %u",
			   (int)status, count, expected_count,
			   ids != NULL && place < count ? ids[place] : 0u,
			   place < expected_count ? expected[place] : 0u);
	    report(what, name, name_bytes, place, detail);
    }
    size_t decoded_bytes = 0;
    bool is_decoded = status == STATUS_OK;
    for (uint32_t index = 0; is_decoded && index < count; index++) {
        const char *bytes = NULL;
        uint32_t byte_count = 0;
	is_decoded = tokenizer_token_bytes(tokenizer, ids[index], &bytes,
					   &byte_count) == STATUS_OK &&
		     byte_count <= expected_text_bytes - decoded_bytes &&
		     memcmp(bytes, expected_text + decoded_bytes, byte_count) ==
			     0;
	decoded_bytes += byte_count;
    }
    if (status == STATUS_OK &&
	(!is_decoded || decoded_bytes != expected_text_bytes)) {
	    report(what, name, name_bytes, decoded_bytes,
		   "the ids do not decode to the reference text");
    }
    free(ids);
}

static int compare_tokenizer_cases(const struct text_tokenizer *tokenizer,
				   const char *path)
{
    size_t byte_count = 0;
    void *const bytes = read_file(path, &byte_count);
    struct reader reader = {bytes, byte_count, 0, false};
    if (bytes == NULL || !open_reader(&reader, TOKENIZER_MAGIC)) {
        free(bytes);
        (void)fprintf(stderr, "FAILED: %s is not a tokenizer golden file\n", path);
        return TEST_FAILED;
    }
    const uint32_t case_count = (uint32_t)take_number(&reader, sizeof(uint32_t));
    const char *longest_text = "";
    size_t longest_bytes = 0, id_total = 0;
    uint32_t longest_count = 0;
    const int mismatches_before = mismatches;
    for (uint32_t index = 0; index < case_count && !reader.is_short; index++) {
        size_t name_bytes = 0, text_bytes = 0, reference_bytes = 0;
        uint32_t id_count = 0;
        const char *const name = take_sized(&reader, sizeof(uint32_t), &name_bytes);
        const char *const text = take_sized(&reader, sizeof(uint64_t), &text_bytes);
        const char *const reference = take_sized(&reader, sizeof(uint64_t), &reference_bytes);
        uint32_t *const ids = take_ids(&reader, &id_count);
        if (ids != NULL) {
            compare_encoding(tokenizer, "the ids", name, name_bytes, text, text_bytes, ids, id_count, reference,
                             reference_bytes);
        }
        longest_text = text_bytes > longest_bytes ? text : longest_text;
        longest_count = text_bytes > longest_bytes ? id_count : longest_count;
        longest_bytes = text_bytes > longest_bytes ? text_bytes : longest_bytes;
        id_total += id_count;
        free(ids);
    }
    double seconds[TIMED_RUNS] = {0};
    uint32_t *const ids = malloc(((size_t)longest_count + ID_SLACK) * sizeof(uint32_t));
    uint32_t count = 0;
    for (int run = 0; ids != NULL && run < TIMED_RUNS; run++) {
        const double started = monotonic_seconds();
	const result_status status =
		tokenizer_encode(tokenizer, longest_text, longest_bytes, ids,
				 longest_count + ID_SLACK, &count);
	seconds[run] = status == STATUS_OK ? monotonic_seconds() - started :
					     0.0;
    }
    qsort(seconds, TIMED_RUNS, sizeof(seconds[0]), compare_doubles);
    (void)printf("tokenizer: %u texts, %zu ids, %d differ%s; the longest, %zu bytes and %u ids, encoded in %.3f s "
                 "(median of %d)\n", case_count, id_total, mismatches - mismatches_before,
                 reader.is_short ? " (the file is cut short)" : "", longest_bytes, longest_count,
                 seconds[TIMED_RUNS / MEDIAN_DIVISOR], TIMED_RUNS);
    free(ids);
    free(bytes);
    return reader.is_short || mismatches != mismatches_before ? TEST_FAILED : TEST_PASSED;
}

/* The old golden path still proves raw reference parity. The chat path additionally proves identical ids for
 * every marker-free payload; records containing payload control spellings are classified as intentional changes. */
static bool compare_spanned_prompt(
	const struct text_tokenizer *tokenizer, const bool *added,
	const char *messages, size_t messages_bytes, const char *tools,
	size_t tools_bytes, const struct chat_template_options *options,
	char *prompt, size_t capacity, const char *expected,
	size_t expected_bytes, const uint32_t *expected_ids,
	uint32_t expected_count, bool *has_payload_marker)
{
	struct chat_template_spans trusted = { 0 };
	size_t length = 0;
	const result_status rendered = chat_template_render_spans(
		messages, messages_bytes, tools, tools_bytes, options, prompt,
		capacity, &length, &trusted);
	bool matches = rendered == STATUS_OK && length == expected_bytes &&
		       memcmp(prompt, expected, length) == 0;
	*has_payload_marker = false;
	size_t position = 0;
	uint32_t span = 0;
	for (uint32_t index = 0; matches && index < expected_count; index++) {
		const char *bytes = NULL;
		uint32_t byte_count = 0;
		const uint32_t token = expected_ids[index];
		matches = tokenizer_token_bytes(tokenizer, token, &bytes,
						&byte_count) == STATUS_OK &&
			  byte_count <= length - position;
		if (!matches) {
			break;
		}
		while (span < trusted.count &&
		       trusted.spans[span].end <= position) {
			span++;
		}
		if (added[token] &&
		    !(span < trusted.count &&
		      trusted.spans[span].begin <= position &&
		      position + byte_count <= trusted.spans[span].end)) {
			*has_payload_marker = true;
		}
		position += byte_count;
	}
	uint32_t *const actual =
		expected_bytes > TOKENIZER_MAXIMUM_TEXT_BYTES ?
			NULL :
			malloc((expected_bytes + 1u) * sizeof(uint32_t));
	uint32_t count = 0;
	matches = matches && position == length && actual != NULL &&
		  tokenizer_encode_spans(tokenizer, prompt, length,
					 trusted.spans, trusted.count, actual,
					 (uint32_t)expected_bytes,
					 &count) == STATUS_OK;
	if (matches && !*has_payload_marker) {
		matches = count == expected_count &&
			  memcmp(actual, expected_ids,
				 (size_t)count * sizeof(*actual)) == 0;
	}
    free(actual);
    free(trusted.spans);
    return matches;
}

static int compare_template_cases(const struct text_tokenizer *tokenizer,
				  const bool *added, const char *path)
{
    size_t byte_count = 0;
    void *const bytes = read_file(path, &byte_count);
    struct reader reader = {bytes, byte_count, 0, false};
    if (bytes == NULL || !open_reader(&reader, TEMPLATE_MAGIC)) {
        free(bytes);
        (void)fprintf(stderr, "FAILED: %s is not a template golden file\n", path);
        return TEST_FAILED;
    }
    const uint32_t case_count = (uint32_t)take_number(&reader, sizeof(uint32_t));
    uint32_t refused = 0, longest_count = 0, marker_free = 0, literal_payloads = 0;
    size_t prompt_total = 0, longest_messages_bytes = 0;
    const char *longest_messages = "";
    const int mismatches_before = mismatches;
    struct chat_template_options longest_options = { 0 };
    for (uint32_t index = 0; index < case_count && !reader.is_short; index++) {
        size_t name_bytes = 0, messages_bytes = 0, tools_bytes = 0, expected_bytes = 0, prompt_bytes = 0;
        uint32_t id_count = 0;
        const char *const name = take_sized(&reader, sizeof(uint32_t), &name_bytes);
        const uint32_t flags = (uint32_t)take_number(&reader, sizeof(uint32_t));
	const struct chat_template_options options = {
		.reasoning_effort =
			(uint32_t)take_number(&reader, sizeof(uint32_t)),
		.is_thinking = (flags & THINKING) != 0,
		.drops_thinking = (flags & DROPS_THINKING) != 0,
		.continues_final_message = (flags & CONTINUES) != 0
	};
	const char *const messages = take_sized(&reader, sizeof(uint64_t), &messages_bytes);
        const char *const tools = take_sized(&reader, sizeof(uint64_t), &tools_bytes);
        const char *const expected = take_sized(&reader, sizeof(uint64_t), &expected_bytes);
        uint32_t *const ids = take_ids(&reader, &id_count);
        char *const prompt = ids == NULL ? NULL : malloc(expected_bytes * PROMPT_ROOM_FACTOR + PROMPT_SLACK);
	const result_status status =
		prompt == NULL ?
			STATUS_OUT_OF_MEMORY :
			chat_template_render(
				messages, messages_bytes,
				(flags & HAS_TOOLS) != 0 ? tools : NULL,
				tools_bytes, &options, prompt,
				expected_bytes * PROMPT_ROOM_FACTOR +
					PROMPT_SLACK,
				&prompt_bytes);
	const bool is_refused = (flags & (REFUSED | REFUSED_BY_CONTRACT)) != 0;
        refused += is_refused ? 1u : 0u;
        if (!is_refused && messages_bytes > longest_messages_bytes && (flags & HAS_TOOLS) == 0) {
            longest_messages = messages;
            longest_messages_bytes = messages_bytes;
            longest_count = id_count;
            longest_options = options;
        }
        prompt_total += expected_bytes;
	if (is_refused && status != STATUS_INVALID_ARGUMENT) {
		report("the refusal", name, name_bytes, 0,
		       "the encoder or the contract refuses it, the template does not");
	} else if (!is_refused && status != STATUS_OK) {
		report("the prompt", name, name_bytes, 0,
		       "the template refuses what the encoder renders");
	} else if (!is_refused) {
		const size_t place = first_difference(
			prompt, prompt_bytes, expected, expected_bytes, 1);
		if (place != prompt_bytes || prompt_bytes != expected_bytes) {
			char detail[DETAIL_CAPACITY] = { 0 };
			(void)snprintf(
				detail, sizeof(detail), "\"%.*s\" for \"%.*s\"",
				(int)(prompt_bytes - place < SHOWN_BYTES ?
					      prompt_bytes - place :
					      SHOWN_BYTES),
				prompt + place,
				(int)(expected_bytes - place < SHOWN_BYTES ?
					      expected_bytes - place :
					      SHOWN_BYTES),
				expected + place);
			report("the prompt", name, name_bytes, place, detail);
		} else if (tokenizer != NULL) {
			compare_encoding(tokenizer, "the prompt's ids", name,
					 name_bytes, prompt, prompt_bytes, ids,
					 id_count, expected, expected_bytes);
			bool has_payload_marker = false;
			if (!compare_spanned_prompt(
				    tokenizer, added, messages, messages_bytes,
				    (flags & HAS_TOOLS) != 0 ? tools : NULL,
				    tools_bytes, &options, prompt,
				    expected_bytes * PROMPT_ROOM_FACTOR +
					    PROMPT_SLACK,
				    expected, expected_bytes, ids, id_count,
				    &has_payload_marker)) {
				report("the spanned prompt", name, name_bytes,
				       0,
				       "rendered bytes, valid literal encoding or marker-free reference ids differ");
			}
			marker_free += has_payload_marker ? 0u : 1u;
			literal_payloads += has_payload_marker ? 1u : 0u;
		}
	}
	free(prompt);
        free(ids);
    }
    double render_seconds[TIMED_RUNS] = {0}, encode_seconds[TIMED_RUNS] = {0};
    const size_t capacity = longest_messages_bytes * PROMPT_ROOM_FACTOR + PROMPT_SLACK;
    char *const prompt = malloc(capacity);
    uint32_t *const ids = malloc(((size_t)longest_count + ID_SLACK) * sizeof(uint32_t));
    size_t prompt_bytes = 0;
    uint32_t count = 0;
    for (int run = 0; prompt != NULL && ids != NULL && run < TIMED_RUNS; run++) {
        const double started = monotonic_seconds();
	const result_status status = chat_template_render(
		longest_messages, longest_messages_bytes, NULL, 0,
		&longest_options, prompt, capacity, &prompt_bytes);
	const double rendered = monotonic_seconds();
	const result_status encoded =
		status != STATUS_OK || tokenizer == NULL ?
			status :
			tokenizer_encode(tokenizer, prompt, prompt_bytes, ids,
					 longest_count + ID_SLACK, &count);
	render_seconds[run] = status == STATUS_OK ? rendered - started : 0.0;
	encode_seconds[run] =
		encoded == STATUS_OK ? monotonic_seconds() - rendered : 0.0;
    }
    qsort(render_seconds, TIMED_RUNS, sizeof(render_seconds[0]), compare_doubles);
    qsort(encode_seconds, TIMED_RUNS, sizeof(encode_seconds[0]), compare_doubles);
    (void)printf("template spans: %u marker-free prompts have reference ids; %u payload-marker prompts intentionally "
                 "use literal chat encoding (raw reference ids still checked)\n", marker_free, literal_payloads);
    (void)printf("template: %u conversations (%u refused), %zu prompt bytes, %d differ%s; the longest, %zu bytes of "
                 "JSON and %u ids, rendered in %.4f s and encoded in %.4f s (medians of %d)\n", case_count, refused,
                 prompt_total, mismatches - mismatches_before, reader.is_short ? " (the file is cut short)" : "",
                 longest_messages_bytes, count, render_seconds[TIMED_RUNS / MEDIAN_DIVISOR],
                 encode_seconds[TIMED_RUNS / MEDIAN_DIVISOR], TIMED_RUNS);
    free(prompt);
    free(ids);
    free(bytes);
    return reader.is_short || mismatches != mismatches_before ? TEST_FAILED : TEST_PASSED;
}

/* Uses the supplied real tables, including added tokens whose special flag was false. The synthetic payload
 * matrix proves every origin path; this gate proves that ordinary BPE pieces cannot recreate real control ids. */
static void compare_literal_markers(const struct text_tokenizer *tokenizer,
				    const bool *added)
{
    static const char *const markers[] = {
        "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜System｜>",
        "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>", DSML,
        "<tool_result>", "</tool_result>", "<｜deepseek_image｜>",
        "<" DSML " calls>", "</" DSML " calls>", "<" DSML " invoke", "</" DSML " invoke>",
        "<" DSML " parameter", "</" DSML " parameter>"};
    uint32_t raw[MARKER_TOKEN_CAPACITY] = {0}, literal[MARKER_TOKEN_CAPACITY] = {0};
    uint32_t trusted_ids[MARKER_TOKEN_CAPACITY] = {0}, raw_markers = 0;
    for (size_t index = 0; index < sizeof(markers) / sizeof(markers[0]); index++) {
        const char *const text = markers[index];
        const size_t length = strlen(text);
	const struct tokenizer_span trusted = { 0, (uint32_t)length };
	uint32_t raw_count = 0, literal_count = 0, trusted_count = 0;
	const bool encoded =
		tokenizer_encode(tokenizer, text, length, raw,
				 MARKER_TOKEN_CAPACITY,
				 &raw_count) == STATUS_OK &&
		tokenizer_encode_spans(tokenizer, text, length, NULL, 0,
				       literal, MARKER_TOKEN_CAPACITY,
				       &literal_count) == STATUS_OK &&
		tokenizer_encode_spans(tokenizer, text, length, &trusted, 1,
				       trusted_ids, MARKER_TOKEN_CAPACITY,
				       &trusted_count) == STATUS_OK;
	bool same_raw = encoded && raw_count == trusted_count &&
                        memcmp(raw, trusted_ids, (size_t)raw_count * sizeof(*raw)) == 0;
        bool raw_has_control = false;
        for (uint32_t token = 0; encoded && token < raw_count; token++) {
            raw_has_control = raw_has_control || added[raw[token]];
        }
        raw_markers += raw_has_control ? 1u : 0u;
        size_t position = 0;
        for (uint32_t token = 0; same_raw && token < literal_count; token++) {
            const char *bytes = NULL;
            uint32_t byte_count = 0;
	    same_raw = !added[literal[token]] &&
		       tokenizer_token_bytes(tokenizer, literal[token], &bytes,
					     &byte_count) == STATUS_OK &&
		       byte_count <= length - position &&
		       memcmp(bytes, text + position, byte_count) == 0;
	    position += byte_count;
        }
        if (!same_raw || position != length) {
            report("literal control spelling", text, length, position,
                   "ordinary encoding emitted a control id, changed text or changed trusted/raw ids");
        }
    }
    if (raw_markers == 0) {
        report("literal control fixture", "real markers", strlen("real markers"), 0,
               "supplied tables recognize none of the tested control spellings");
    }
    (void)printf("literal markers: %zu spellings round-trip as ordinary ids; %u contain reference added ids, "
                 "whose trusted ids remain unchanged\n", sizeof(markers) / sizeof(markers[0]), raw_markers);
}

int main(int argument_count, char **arguments)
{
    if (argument_count == 1) {
        (void)printf("front_end_golden_test: no tables or golden files given, nothing compared\n");
        return TEST_PASSED;
    }
    if (argument_count != ARGUMENT_COUNT) {
        (void)fprintf(stderr, "usage: front_end_golden_test <tables file> <tokenizer golden> <template golden>\n");
        return TEST_WRONG_USAGE;
    }
    size_t table_bytes = 0;
    struct text_tokenizer *tokenizer = NULL;
    void *const tables = read_file(arguments[TABLES_FILE], &table_bytes);
    if (tables == NULL ||
	tokenizer_open(tables, table_bytes, &tokenizer) != STATUS_OK) {
	    (void)fprintf(stderr, "FAILED: the tables do not open\n");
	    free(tables);
	    return TEST_FAILED;
    }
    const int tokenizer_result = compare_tokenizer_cases(tokenizer, arguments[TOKENIZER_GOLDEN_FILE]);
    const struct tokenizer_tables_header *const header = tables;
    bool *const added = calloc(header->vocabulary_size, sizeof(bool));
    if (added == NULL) {
	    tokenizer_close(tokenizer);
	    free(tables);
	    return TEST_FAILED;
    }
    const uint32_t *const added_ids =
	    (const uint32_t *)(header + 1) + header->vocabulary_size +
	    (size_t)header->merge_count * TOKENIZER_MERGE_WORDS;
    for (uint32_t index = 0; index < header->added_token_count; index++) {
        added[added_ids[index]] = true;
    }
    const int mismatches_before_markers = mismatches;
    compare_literal_markers(tokenizer, added);
    const bool markers_passed = mismatches == mismatches_before_markers;
    const int template_result = compare_template_cases(tokenizer, added, arguments[TEMPLATE_GOLDEN_FILE]);
    free(added);
    tokenizer_close(tokenizer);
    free(tables);
    if (tokenizer_result != TEST_PASSED || template_result != TEST_PASSED || !markers_passed) {
        return TEST_FAILED;
    }
    (void)printf("front_end_golden_test passed\n");
    return TEST_PASSED;
}
