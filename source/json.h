/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * json.h: JSON as Python's json module reads and writes it. Private to the engine's sources: the chat template's
 * and the server's.
 *
 * Contract:
 *   - json_read_document checks a whole text as json.loads reads one: spaces, then one value, then spaces;
 *     objects, arrays, strings with their escapes, the numbers of json.loads (an integer of at most
 *     JSON_MOST_INTEGER_DIGITS digits, Python's int() limit), true, false, null, NaN, Infinity, -Infinity.
 *     It refuses besides what makes no UTF-8 text: invalid UTF-8 in a string and an escaped half of a surrogate
 *     pair without the other, and nesting deeper than JSON_MAXIMUM_DEPTH. Every other function reads only
 *     JSON so checked, and never past the value it is given.
 *   - A value is a span of its text without the spaces around it; text NULL stands for a value that is absent.
 *   - A writer writes what fits and counts all: once its length is above its capacity, what it holds is cut and
 *     must not be used. Scratch memory takes strings unescaped behind what it holds; whoever unescapes gives the
 *     memory back by setting used again. is_short is set, and an empty string given, when one does not fit.
 *   - json_write_canonical writes json.dumps(json.loads(value), ensure_ascii=False): ", " and ": " between
 *     items, strings with '"', '\' and control characters escaped (the short escapes, else \u00xx), an integer as it
 *     is written but "-0" as "0", a float as Python's repr, NaN, Infinity and -Infinity as they are. Work is linear
 *     in the value; an object with a key twice is written with every member (json_has_repeated_key finds
 *     it: json.loads keeps the last value, in the place of the first).
 *   - Numbers are read and written in the C locale, which the process keeps: nothing in the engine calls setlocale.
 */
#ifndef JSON_H
#define JSON_H

#include <stdbool.h>
#include <stddef.h>

#include "anemone/status.h"

#define JSON_MAXIMUM_DEPTH 128 /* arrays and objects, one inside the other */
#define JSON_MOST_INTEGER_DIGITS 4300

struct json_value {
    const char *text;
    size_t begin;
    size_t end;
};

struct json_bytes {                               /* a string unescaped into scratch memory */
    const char *data;
    size_t count;
};

struct json_writer {
    char *bytes;
    size_t capacity;
    size_t length;
};

struct json_scratch {
    char *bytes;
    size_t capacity;
    size_t used;
    bool is_short;
};

enum json_document {
    JSON_DOCUMENT_VALID,
    JSON_DOCUMENT_NOT_JSON,                       /* what json.loads refuses too */
    JSON_DOCUMENT_BEYOND_THIS_READER              /* json.loads may read it: too deep, or a lone half of a pair */
};

/* The one value of a whole text, its span in value_out; the span's end is 0 when it is not JSON. A text nested too
 * deep is beyond this reader whatever follows the place where it is. */
MUST_CHECK enum json_document json_read_document(const char *text,
						 size_t length,
						 struct json_value *value_out);
/* The next element of an array, or member of an object (its key then in key_out, which may be NULL), with the
 * cursor first on the opening bracket; false after the last. */
bool json_next(struct json_value container, size_t *cursor,
	       struct json_value *key_out, struct json_value *value_out);
/* Python's truth of a value: false for null, false, 0, 0.0, "", [], {} and a value that is absent. */
bool json_is_truthy(struct json_scratch *scratch, struct json_value value);
struct json_bytes json_unescape(struct json_scratch *scratch,
				struct json_value string);
/* Whether a value is a string that unescapes to word. */
bool json_string_is(struct json_scratch *scratch, struct json_value value,
		    const char *word);
/* The order of two struct json_bytes by their bytes, as qsort and bsearch take it. */
int json_compare_bytes(const void *first, const void *second);
/* Whether an object in a value holds a key twice. It sorts the keys of each object in the scratch memory, 16 bytes
 * for each key of the objects open at a place and the bytes of those that hold an escape, and gives it back. */
MUST_CHECK bool json_has_repeated_key(struct json_scratch *scratch,
				      struct json_value value);
void json_write_bytes(struct json_writer *writer, const void *bytes,
		      size_t count);
void json_write_unescaped(struct json_writer *writer, struct json_value string);
/* Python's json.dumps of a string's bytes, with ensure_ascii=False: at most six bytes for each, and the quotes. */
void json_write_string(struct json_writer *writer, struct json_bytes bytes);
void json_write_canonical(struct json_writer *writer,
			  struct json_scratch *scratch,
			  struct json_value value);

#endif
