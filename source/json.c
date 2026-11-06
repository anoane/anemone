/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * json.c: JSON as Python's json module reads and writes it. Contract: json.h.
 *
 * The check is a recursive descent bounded by its depth; the readings of checked JSON find the end of a value by
 * what follows it inside its array or object (a scalar ends before ",]}" or a space), so they need no bound but
 * their container's. The canonical writer and the search for a key twice go forward once through the text: in
 * checked JSON a string is a key when ':' follows it. A float is written as Python's repr: the closest decimal of
 * the fewest digits that reads back as it, found by printing with ever more digits, and for a power of two (twice
 * as near its lower neighbour as its upper one) the next decimal up when the closest does not read back.
 */
#include "json.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/tokenizer.h"

#define SPACES " \t\n\r"
#define WORD_ENDS ",]} \t\r\n"                    /* what follows a number or a word inside an array or object */
#define SHORT_ESCAPE_BYTES 2
#define SEPARATOR_BYTES 2                         /* ", " and ": " */
#define UNICODE_ESCAPE_BYTES 6                    /* \uXXXX */
#define HEX_DIGITS "0123456789abcdef"
#define HEX_BASE 16
#define HEX_DIGIT_BITS 4
#define HEX_DIGIT_MASK 0xFu
#define NUMBER_TEXT_CAPACITY 48
#define MOST_SIGNIFICANT_DIGITS 17                /* enough for any double to read back */
#define LOWEST_FIXED_POINT (-3)                   /* Python's repr writes 0.0001 but 1e-05, 10^16 - 1 but 1e+16 */
#define HIGHEST_FIXED_POINT 16
#define ZEROS "0000000000000000"                 /* as many as a fixed float may need between digits and point */
#define DECIMAL_BASE 10

enum code_point_limit {
    HIGH_SURROGATE_FIRST = 0xD800, LOW_SURROGATE_FIRST = 0xDC00, SURROGATES_END = 0xE000, SUPPLEMENTARY_FIRST = 0x10000,
    SURROGATE_PAYLOAD_BITS = 10, ASCII_END = 0x80, TWO_UTF8_BYTES_END = 0x800, CONTROL_END = 0x20,
    CONTINUATION_TAG = 0x80, CONTINUATION_PAYLOAD = 0x3F, CONTINUATION_BITS = 6
};
enum utf8_bytes { TWO_UTF8_BYTES = 2, THREE_UTF8_BYTES, FOUR_UTF8_BYTES };
static const unsigned char lead_tags[FOUR_UTF8_BYTES + 1] = {0, 0, 0xC0, 0xE0, 0xF0};   /* by a sequence's bytes */

struct document_check {                           /* what makes a text beyond this reader, though json.loads reads it */
    bool is_too_deep;
    bool has_lone_surrogate;
};

void json_write_bytes(struct json_writer *writer, const void *bytes,
		      size_t count)
{
    if (count > 0 && writer->length <= writer->capacity && count <= writer->capacity - writer->length) {
        memcpy(writer->bytes + writer->length, bytes, count);
    }
    writer->length += count;
}

static void write_code_point(struct json_writer *writer, uint32_t code_point)
{
    unsigned char bytes[FOUR_UTF8_BYTES] = {0};
    const size_t count = code_point < ASCII_END ? 1 : code_point < TWO_UTF8_BYTES_END ? TWO_UTF8_BYTES
                       : code_point < SUPPLEMENTARY_FIRST ? THREE_UTF8_BYTES : FOUR_UTF8_BYTES;
    for (size_t index = count - 1; index > 0; index--, code_point >>= CONTINUATION_BITS) {
        bytes[index] = (unsigned char)(CONTINUATION_TAG | (code_point & CONTINUATION_PAYLOAD));
    }
    bytes[0] = (unsigned char)(lead_tags[count] | code_point);
    json_write_bytes(writer, bytes, count);
}

static size_t skip_space(const char *text, size_t position, size_t end)
{
    while (position < end && text[position] != '\0' && strchr(SPACES, text[position]) != NULL) {
        position++;
    }
    return position;
}

/* The four hex digits of a \u escape at text[position], which has them all. */
static uint32_t hex_value(const char *text, size_t position, bool *is_hex_out)
{
    char digits[UNICODE_ESCAPE_BYTES - SHORT_ESCAPE_BYTES + 1] = {0};
    memcpy(digits, text + position, sizeof(digits) - 1);
    *is_hex_out = *is_hex_out && strspn(digits, HEX_DIGITS "ABCDEF") == sizeof(digits) - 1;
    return (uint32_t)strtoul(digits, NULL, HEX_BASE);
}

/* The code point of the escape at text[position], a backslash, and its bytes (a pair of escapes for one code point
 * is one escape); whether Python reads it goes to is_valid_out. A lone half of a pair is its own code point. */
static uint32_t escape_at(const char *text, size_t position, size_t end, size_t *byte_count_out, bool *is_valid_out)
{
    static const char letters[] = "\"\\/bfnrt";
    static const char meanings[] = "\"\\/\b\f\n\r\t";
    *byte_count_out = SHORT_ESCAPE_BYTES;
    if (position + 1 < end && text[position + 1] != 'u') {
        const char *const letter = memchr(letters, text[position + 1], sizeof(letters) - 1);
        *is_valid_out = letter != NULL;
        return letter != NULL ? (unsigned char)meanings[letter - letters] : 0;
    }
    bool is_valid = end - position >= UNICODE_ESCAPE_BYTES;
    uint32_t code_point = is_valid ? hex_value(text, position + SHORT_ESCAPE_BYTES, &is_valid) : 0;
    const size_t low = position + UNICODE_ESCAPE_BYTES;
    *byte_count_out = UNICODE_ESCAPE_BYTES;
    if (is_valid && code_point >= HIGH_SURROGATE_FIRST && code_point < LOW_SURROGATE_FIRST &&
        end - low >= UNICODE_ESCAPE_BYTES && text[low] == '\\' && text[low + 1] == 'u') {
        bool is_low_hex = true;
        const uint32_t low_half = hex_value(text, low + SHORT_ESCAPE_BYTES, &is_low_hex);
        if (is_low_hex && low_half >= LOW_SURROGATE_FIRST && low_half < SURROGATES_END) {
            code_point = SUPPLEMENTARY_FIRST + ((code_point - HIGH_SURROGATE_FIRST) << SURROGATE_PAYLOAD_BITS |
                                                (low_half - LOW_SURROGATE_FIRST));
            *byte_count_out = UNICODE_ESCAPE_BYTES + UNICODE_ESCAPE_BYTES;
        }
    }
    *is_valid_out = is_valid;
    return code_point;
}

/* The end of the string at text[position], a quote, as json.loads reads one, or 0 when there is none. */
static size_t valid_string_end(const char *text, size_t position, size_t end, struct document_check *check)
{
    for (position++; position < end;) {
        while (position < end && (unsigned char)text[position] >= CONTROL_END &&      /* plain ASCII, the most */
               (unsigned char)text[position] < ASCII_END && text[position] != '"' && text[position] != '\\') {
            position++;
        }
        const unsigned char byte = position < end ? (unsigned char)text[position] : 0;
        bool is_valid = byte >= CONTROL_END;
        size_t byte_count = 1;
        if (byte == '"') {
            return position + 1;
        }
        if (byte == '\\') {
            const uint32_t code_point = escape_at(text, position, end, &byte_count, &is_valid);
            check->has_lone_surrogate = check->has_lone_surrogate ||
                                        (code_point >= HIGH_SURROGATE_FIRST && code_point < SURROGATES_END);
        } else if (byte >= ASCII_END) {
		byte_count = tokenizer_utf8_sequence_bytes(text, position, end,
							   &is_valid);
	}
        if (!is_valid) {
            return 0;
        }
        position += byte_count;
    }
    return 0;
}

static size_t digits_end(const char *text, size_t position, size_t end)
{
    while (position < end && text[position] >= '0' && text[position] <= '9') {
        position++;
    }
    return position;
}

/* The end of the number at text[position] as json.loads reads one, -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][-+]?[0-9]+)?,
 * an integer of at most JSON_MOST_INTEGER_DIGITS digits; 0 when there is none. */
static size_t valid_number_end(const char *text, size_t position, size_t end)
{
    const size_t digits_begin = position + (position < end && text[position] == '-' ? 1 : 0);
    const size_t integer_end = digits_end(text, digits_begin, end);
    size_t number_end = integer_end;
    if (integer_end == digits_begin || (integer_end - digits_begin > 1 && text[digits_begin] == '0')) {
        return 0;
    }
    if (number_end < end && text[number_end] == '.' && digits_end(text, number_end + 1, end) > number_end + 1) {
        number_end = digits_end(text, number_end + 1, end);
    }
    const size_t exponent = number_end + 1 + (number_end + 1 < end && (text[number_end + 1] == '+' ||
                                                                        text[number_end + 1] == '-') ? 1 : 0);
    if (number_end < end && (text[number_end] == 'e' || text[number_end] == 'E') &&
        digits_end(text, exponent, end) > exponent) {
        number_end = digits_end(text, exponent, end);
    }
    return number_end == integer_end && integer_end - digits_begin >
						JSON_MOST_INTEGER_DIGITS ?
		   0 :
		   number_end;
}

/* The end of the JSON value at text[position], as json.loads reads one, or 0 when there is none. */
static size_t valid_value_end(const char *text, size_t position, size_t end, unsigned depth,
                              struct document_check *check)
{
    static const char *const words[] = {"true", "false", "null", "NaN", "Infinity", "-Infinity"};
    if (position >= end) {
        return 0;
    }
    if (text[position] == '"') {
        return valid_string_end(text, position, end, check);
    }
    check->is_too_deep = check->is_too_deep ||
			 ((text[position] == '[' || text[position] == '{') &&
			  depth == JSON_MAXIMUM_DEPTH);
    if ((text[position] == '[' || text[position] == '{') &&
	depth < JSON_MAXIMUM_DEPTH) {
	    const bool is_object = text[position] == '{';
	    const char closing = is_object ? '}' : ']';
	    position = skip_space(text, position + 1, end);
	    if (position < end && text[position] == closing) {
		    return position + 1;
	    }
	    for (;;) {
		    if (is_object) {
			    position = position < end && text[position] == '"' ?
					       valid_string_end(text, position,
								end, check) :
					       0;
			    position = position == 0 ?
					       0 :
					       skip_space(text, position, end);
			    if (position == 0 || position >= end ||
				text[position] != ':') {
				    return 0;
			    }
			    position = skip_space(text, position + 1, end);
		    }
		    position = valid_value_end(text, position, end, depth + 1,
					       check);
		    position = position == 0 ? 0 :
					       skip_space(text, position, end);
		    if (position == 0 || position >= end ||
			(text[position] != ',' && text[position] != closing)) {
			    return 0;
		    }
		    if (text[position] == closing) {
			    return position + 1;
		    }
		    position = skip_space(text, position + 1, end);
	    }
    }
    for (size_t index = 0; index < sizeof(words) / sizeof(words[0]); index++) {
        const size_t length = strlen(words[index]);
        if (end - position >= length && memcmp(text + position, words[index], length) == 0) {
            return position + length;
        }
    }
    return valid_number_end(text, position, end);
}

enum json_document json_read_document(const char *text, size_t length,
				      struct json_value *value_out)
{
    struct document_check check = {false, false};
    const size_t begin = skip_space(text, 0, length);
    size_t end = valid_value_end(text, begin, length, 0, &check);
    end = end != 0 && skip_space(text, end, length) == length ? end : 0;
    *value_out = (struct json_value){text, begin, end};
    return check.is_too_deep ? JSON_DOCUMENT_BEYOND_THIS_READER : end == 0 ? JSON_DOCUMENT_NOT_JSON
         : check.has_lone_surrogate ? JSON_DOCUMENT_BEYOND_THIS_READER : JSON_DOCUMENT_VALID;
}

/* The closing quote of a checked string that opens at text[position]: the first quote after an even number of
 * backslashes (the opening quote stops the count). */
static size_t closing_quote(const char *text, size_t position, size_t end)
{
    for (;;) {
        const char *const quote = memchr(text + position + 1, '"', end - position - 1);
        size_t backslashes = 0;
        position = (size_t)(quote - text);
        while (text[position - backslashes - 1] == '\\') {
            backslashes++;
        }
        if ((backslashes & 1u) == 0) {
            return position;
        }
    }
}

/* The end of a value inside an array or object of checked JSON that ends at end. */
static size_t value_end(const char *text, size_t position, size_t end)
{
    for (unsigned depth = 0;; position++) {
        if (text[position] == '"') {
            position = closing_quote(text, position, end);
        }
        depth += text[position] == '[' || text[position] == '{' ? 1 : 0;
        depth -= text[position] == ']' || text[position] == '}' ? 1 : 0;
        if (depth == 0 && (strchr("\"]}", text[position]) != NULL || strchr(WORD_ENDS, text[position + 1]) != NULL)) {
            return position + 1;                  /* the end of a string, a container, a number or a word */
        }
    }
}

bool json_next(struct json_value container, size_t *cursor,
	       struct json_value *key_out, struct json_value *value_out)
{
    const char *const text = container.text;
    struct json_value key = {0};
    size_t position = *cursor + 1 < container.end ? skip_space(text, *cursor + 1, container.end) : container.end;
    if (position + 1 >= container.end) {
        *cursor = container.end - 1;
        return false;
    }
    if (text[container.begin] == '{') {
        key = (struct json_value){text, position, value_end(text, position, container.end)};
        position = skip_space(text, skip_space(text, key.end, container.end) + 1, container.end);
    }
    if (key_out != NULL) {
        *key_out = key;
    }
    *value_out = (struct json_value){text, position, value_end(text, position, container.end)};
    *cursor = skip_space(text, value_out->end, container.end);
    return true;
}

void json_write_unescaped(struct json_writer *writer, struct json_value string)
{
    for (size_t position = string.begin + 1, end = string.end - 1, byte_count = 0;
         string.text != NULL && position < end;) {
        const char *const backslash = memchr(string.text + position, '\\', end - position);
        const size_t run_end = backslash == NULL ? end : (size_t)(backslash - string.text);
        bool is_valid = true;
	json_write_bytes(writer, string.text + position, run_end - position);
	if (run_end < end) {
            write_code_point(writer, escape_at(string.text, run_end, end, &byte_count, &is_valid));
        }
        position = run_end + (run_end < end ? byte_count : 0);
    }
}

struct json_bytes json_unescape(struct json_scratch *scratch,
				struct json_value string)
{
    struct json_writer writer = {scratch->bytes + scratch->used, scratch->capacity - scratch->used, 0};
    json_write_unescaped(&writer, string);
    scratch->is_short = scratch->is_short || writer.length > writer.capacity;
    writer.length = writer.length > writer.capacity ? 0 : writer.length;
    scratch->used += writer.length;
    return (struct json_bytes){writer.bytes, writer.length};
}

bool json_string_is(struct json_scratch *scratch, struct json_value value,
		    const char *word)
{
    const size_t mark = scratch->used;
    const bool is_string = value.text != NULL && value.text[value.begin] == '"';
    const struct json_bytes bytes = is_string ? json_unescape(scratch, value) :
						(struct json_bytes){ NULL, 0 };
    scratch->used = mark;
    return is_string && bytes.count == strlen(word) && memcmp(bytes.data, word, bytes.count) == 0;
}

int json_compare_bytes(const void *first, const void *second)
{
    const struct json_bytes *const left = first, *const right = second;
    const int order = memcmp(left->data, right->data, left->count < right->count ? left->count : right->count);
    return order != 0 ? order : (left->count > right->count) - (left->count < right->count);
}

bool json_has_repeated_key(struct json_scratch *scratch,
			   struct json_value value)
{
    const size_t misalignment = (uintptr_t)(scratch->bytes + scratch->used) % _Alignof(struct json_bytes);
    const size_t first = scratch->used + (misalignment == 0 ? 0 : _Alignof(struct json_bytes) - misalignment);
    size_t key_count = 0, copies_begin = scratch->capacity,
	   opened[JSON_MAXIMUM_DEPTH] = { 0 };
    unsigned depth = 0;
    bool is_repeated = false;
    scratch->is_short = scratch->is_short || first > scratch->capacity;
    struct json_bytes *const keys = scratch->is_short ? NULL : (void *)(scratch->bytes + first);   /* then copies */
    for (size_t position = value.begin; !is_repeated && !scratch->is_short && position < value.end; position++) {
        const char byte = value.text[position];
	if ((byte == '{' || byte == '[') && depth < JSON_MAXIMUM_DEPTH) {
		opened[depth++] = key_count;
	} else if ((byte == '}' || byte == ']') && depth > 0) {
		const size_t base = opened[--depth];
		qsort(keys + base, key_count - base, sizeof(keys[0]),
		      json_compare_bytes);
		for (size_t index = base + 1; index < key_count; index++) {
			is_repeated = is_repeated ||
				      json_compare_bytes(&keys[index - 1],
							 &keys[index]) == 0;
		}
		key_count = base;
	} else if (byte == '"') {
		const size_t quote = closing_quote(value.text, position,
						   value.end),
			     raw_count = quote - position - 1;
		const size_t after =
			skip_space(value.text, quote + 1, value.end);
		const size_t records_end =
			first + (key_count + 1) * sizeof(keys[0]);
		const bool is_key = after < value.end &&
				    value.text[after] == ':';
		const size_t copied =
			is_key && memchr(value.text + position + 1, '\\',
					 raw_count) != NULL ?
				raw_count :
				0;
		scratch->is_short = is_key &&
				    (records_end > copies_begin ||
				     copies_begin - records_end < copied);
		if (is_key && !scratch->is_short) {
			struct json_writer copy = {
				scratch->bytes + copies_begin - copied, copied,
				0
			};
			if (copied > 0) {
				json_write_unescaped(
					&copy, (struct json_value){
						       value.text, position,
						       quote + 1 });
			}
			copies_begin -= copied;
			keys[key_count++] =
				copied > 0 ?
					(struct json_bytes){ copy.bytes,
							     copy.length } :
					(struct json_bytes){
						value.text + position + 1,
						raw_count
					};
		}
		position = quote;
	}
    }
    return is_repeated;
}

/* A number's value as Python's float() reads its text. */
static double number_value(struct json_scratch *scratch, struct json_value number)
{
    const size_t length = number.end - number.begin;
    if (length >= scratch->capacity - scratch->used) {
        scratch->is_short = true;
        return 0.0;
    }
    memcpy(scratch->bytes + scratch->used, number.text + number.begin, length);
    scratch->bytes[scratch->used + length] = '\0';
    return strtod(scratch->bytes + scratch->used, NULL);
}

bool json_is_truthy(struct json_scratch *scratch, struct json_value value)
{
    const char first = (char)(value.text != NULL ? value.text[value.begin] : 'n');
    if (first == '[' || first == '{') {
        return value.text[skip_space(value.text, value.begin + 1, value.end)] != (first == '[' ? ']' : '}');
    }
    return first == '"' ? value.end - value.begin > SHORT_ESCAPE_BYTES
         : first == 'f' || first == 'n' ? false
         : first == 't' || first == 'N' || first == 'I' ? true
         : (value.end - value.begin > 1 && value.text[value.begin + 1] == 'I') || number_value(scratch, value) != 0.0;
}

void json_write_string(struct json_writer *writer, struct json_bytes bytes)
{
    size_t run_begin = 0;
    json_write_bytes(writer, "\"", 1);
    for (size_t index = 0; index < bytes.count; index++) {
        const unsigned char byte = (unsigned char)bytes.data[index];
        if (byte >= CONTROL_END && byte != '"' && byte != '\\') {
            continue;
        }
	json_write_bytes(writer, bytes.data + run_begin, index - run_begin);
	run_begin = index + 1;
        const char *const short_escape = byte == '"' ? "\\\"" : byte == '\\' ? "\\\\" : byte == '\b' ? "\\b"
                                       : byte == '\f' ? "\\f" : byte == '\n' ? "\\n" : byte == '\r' ? "\\r"
                                       : byte == '\t' ? "\\t" : NULL;
        const char long_escape[UNICODE_ESCAPE_BYTES] = {'\\', 'u', '0', '0', HEX_DIGITS[byte >> HEX_DIGIT_BITS],
                                                        HEX_DIGITS[byte & HEX_DIGIT_MASK]};
	json_write_bytes(writer,
			 short_escape != NULL ? short_escape : long_escape,
			 short_escape != NULL ? SHORT_ESCAPE_BYTES :
						UNICODE_ESCAPE_BYTES);
    }
    json_write_bytes(writer, bytes.data + run_begin, bytes.count - run_begin);
    json_write_bytes(writer, "\"", 1);
}

/* Whether the digits, with a point after the first and the exponent, read back as the magnitude. */
static bool reads_back(const char *digits, int digit_count, int exponent, double magnitude)
{
    char text[NUMBER_TEXT_CAPACITY] = {0};
    const int length = snprintf(text, sizeof(text), "%c.%.*se%d", digits[0], digit_count - 1, digits + 1, exponent);
    return length > 0 && (size_t)length < sizeof(text) && strtod(text, NULL) == magnitude;
}

/* Python's repr of a float, as json.dumps writes it. */
static void write_float(struct json_writer *writer, double value)
{
    char text[NUMBER_TEXT_CAPACITY] = {0}, digits[NUMBER_TEXT_CAPACITY] = {0};
    if (isnan(value) || isinf(value)) {
        const char *const word = isnan(value) ? "NaN" : value < 0.0 ? "-Infinity" : "Infinity";
	json_write_bytes(writer, word, strlen(word));
	return;
    }
    const double magnitude = signbit(value) ? -value : value;
    int digit_count = 0, exponent = 0;
    for (bool reads = false; !reads && digit_count < MOST_SIGNIFICANT_DIGITS;) {
        digit_count++;
        const int length = snprintf(text, sizeof(text), "%.*e", digit_count - 1, magnitude);
        const char *const exponent_text = length > 0 ? strchr(text, 'e') : NULL;
        exponent = exponent_text != NULL ? (int)strtol(exponent_text + 1, NULL, DECIMAL_BASE) : 0;
        digits[0] = text[0];
        memcpy(digits + 1, text + SHORT_ESCAPE_BYTES, (size_t)digit_count - 1);     /* the digits after "d." */
        reads = reads_back(digits, digit_count, exponent, magnitude);
        int index = digit_count - 1;
        while (!reads && index >= 0 && digits[index] == '9') {
            digits[index--] = '0';
        }
        if (!reads) {                             /* the next decimal up */
            exponent += index < 0 ? 1 : 0;
            digits[index < 0 ? 0 : index] = (char)(index < 0 ? '1' : digits[index] + 1);
            reads = reads_back(digits, digit_count, exponent, magnitude);
        }
    }
    const int point = exponent + 1;               /* where the decimal point stands after the first digit */
    const int whole = point <= 0 ? 0 : point < digit_count ? point : digit_count;      /* digits before the point */
    const int length = point < LOWEST_FIXED_POINT || point > HIGHEST_FIXED_POINT
        ? snprintf(text, sizeof(text), "%s%c%s%.*se%c%02d", signbit(value) ? "-" : "", digits[0],
                   digit_count > 1 ? "." : "", digit_count - 1, digits + 1, exponent < 0 ? '-' : '+', abs(exponent))
        : snprintf(text, sizeof(text), "%s%s%.*s%.*s.%.*s%.*s%s", signbit(value) ? "-" : "", point <= 0 ? "0" : "",
                   whole, digits, point > digit_count ? point - digit_count : 0, ZEROS, point < 0 ? -point : 0, ZEROS,
                   digit_count - whole, digits + whole, whole == digit_count ? "0" : "");
    json_write_bytes(writer, text, length > 0 ? (size_t)length : 0);
}

/* A number or a word of checked JSON as json.dumps writes it. */
static void write_scalar(struct json_writer *writer, struct json_scratch *scratch, struct json_value value)
{
    const char first = value.text[value.begin];
    bool is_integer = true, is_zero = true;
    if (first != '-' && (first < '0' || first > '9')) {
	    json_write_bytes(writer, value.text + value.begin,
			     value.end - value.begin); /* a word as it is */
	    return;
    }
    for (size_t index = value.begin; index < value.end; index++) {
        is_integer = is_integer && strchr(".eEI", value.text[index]) == NULL;
        is_zero = is_zero && (value.text[index] == '-' || value.text[index] == '0');
    }
    if (!is_integer) {
        write_float(writer, number_value(scratch, value));
    } else {
	    json_write_bytes(writer, is_zero ? "0" : value.text + value.begin,
			     is_zero ? 1 : value.end - value.begin);
    }
}

void json_write_canonical(struct json_writer *writer,
			  struct json_scratch *scratch, struct json_value value)
{
    const size_t mark = scratch->used;
    for (size_t position = value.begin; position < value.end;) {
        const char byte = value.text[position];
        size_t next = position + 1;
        if (byte == '"') {
            next = closing_quote(value.text, position, value.end) + 1;
            const struct json_value string = {value.text, position, next};
	    json_write_string(writer, json_unescape(scratch, string));
	} else if (byte == ',' || byte == ':') {
		json_write_bytes(writer, byte == ',' ? ", " : ": ",
				 SEPARATOR_BYTES);
	} else if (byte == '[' || byte == ']' || byte == '{' || byte == '}') {
		json_write_bytes(writer, &byte, 1);
	} else if (strchr(SPACES, byte) == NULL) {
            while (next < value.end && strchr(WORD_ENDS, value.text[next]) == NULL) {
                next++;
            }
            write_scalar(writer, scratch, (struct json_value){value.text, position, next});
        }
        position = next;
        scratch->used = mark;
    }
}
