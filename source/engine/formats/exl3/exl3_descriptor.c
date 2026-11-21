/* SPDX-License-Identifier: AGPL-3.0-or-later */
#include "exl3_descriptor.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "engine/library/exl3/exl3_layout.h"
#include "engine/library/exl3/exl3_records.h"
#include "engine/library/exl3/exl3_tc.h"
#include "common/hooks.h"
#include "engine/formats/weights/weight_format_seam.h"
#include "common/json.h"
#include "common/sha256.h"

#define FORMAT_VERSION 1u /* the converter's DESCRIPTOR_FORMAT_VERSION */
#define HEX_DIGITS "0123456789abcdef"
#define HEX_PER_BYTE 2u
#define NIBBLE_BITS 4u
#define NIBBLE_MASK 15u
#define DIGEST_DIGITS ((size_t)HEX_PER_BYTE * SHA256_DIGEST_BYTES)
#define QUOTE_BYTES 2u /* a string's two quotes */
#define DECIMAL_BASE 10u
#define SPACES \
	" \t\n\v\f\r" /* a text of only these is none (the converter's SPACES) */
/* the reader's scratch for a text of n bytes: its strings unescaped (at most n), and json_has_repeated_key's 16 bytes
 * for each key open at once (a member takes at least 4 bytes, "":0,) and its copies of escaped keys (at most n) */
#define SCRATCH_PER_TEXT_BYTE 6u
#define LINE_BYTES 1024u
#define FIELD_BYTES 128u
#define NAME_BYTES 128u
#define CODEBOOK_BYTES 16u /* "0x83DCD12D" and its NUL */
#define CAPTURE_PIECE_BYTES \
	1048576u /* a recorded capture is hashed a MiB at a time */
#define CAPTURE_MOST_BYTES ((uint64_t)INT64_MAX) /* any regular file's size */
#define SHAPE_DIMENSIONS 2u /* [outputs, inputs] */
/* a count for each bits_x2: the format's widths are K1 (2) to K8 (16); slot 0, no width, counts "mixed" layers */
#define WIDTH_SLOTS (EXL3_HALF_BITS_PER_BIT * EXL3_WHOLE_WIDTH_MOST + 1u)
#define NARROWEST_BITS_X2 EXL3_HALF_BITS_PER_BIT /* K1 */
#define MIXED_SLOT 0u
#define WIDTH_TEXT_BYTES 8u /* "K7.5" and its NUL */
#define WIDTHS_TEXT_BYTES ((size_t)WIDTH_SLOTS * FIELD_BYTES)
#define QUALITY_WORDS_BYTES (PATH_MAX + FIELD_BYTES) /* "recorded (<path>)" */
#define REFUSAL_FORMAT_PARAMETER 3
#define REFUSAL_VALUES_PARAMETER 4
#define TC_REFUSAL \
	"exl3 descriptor: --exl3_tc_moe=1: the model %s's " /* exl3_descriptor_tc_check's line */
#define TC_SPLIT_WIDTH \
	EXL3_K3 /* any width the build decodes: the split is the same */

/* Each object's members by place, as the converter writes them; only those that bind are read */
enum root_member {
	ROOT_FORMAT_VERSION,
	ROOT_MAP,
	ROOT_ARCHITECTURE,
	ROOT_QUANTISER,
	ROOT_CODEBOOK,
	ROOT_TABLES,
	ROOT_EXPERTS,
	ROOT_WIDTHS,
	ROOT_SUMMARY,
	ROOT_QUALITY,
	ROOT_MEMBERS
};
static const char *const root_names[ROOT_MEMBERS] = {
	"format_version", "map",      "architecture",
	"quantiser",	  "codebook", "front_end_tables_sha256",
	"experts",	  "widths",   "widths_summary",
	"quality"
};
enum map_member {
	MAP_LOGICAL_BYTES,
	MAP_HEADER_DIGEST,
	MAP_FILE,
	MAP_SHA256,
	MAP_MANIFEST_BYTES,
	MAP_NATIVE_HEADER_DIGEST,
	MAP_MEMBERS
};
static const char *const map_names[MAP_MEMBERS] = {
	"logical_bytes", "pack_header_digest", "file",
	"sha256",	 "manifest_bytes",     "native_pack_header_digest"
};
enum architecture_member {
	ARCHITECTURE_MODEL_TYPE,
	ARCHITECTURE_ARCHITECTURES,
	ARCHITECTURE_CONFIG,
	ARCHITECTURE_MEMBERS
};
static const char *const architecture_names[ARCHITECTURE_MEMBERS] = {
	"model_type", "architectures", "config_sha256"
};
enum quantiser_member {
	QUANTISER_SOURCE,
	QUANTISER_INDEX,
	QUANTISER_CONFIG,
	QUANTISER_MEMBERS
};
static const char *const quantiser_names[QUANTISER_MEMBERS] = {
	"source", "index_sha256", "quantization_config_sha256"
};
enum codebook_member { CODEBOOK_MULTIPLIER, CODEBOOK_MEMBERS };
static const char *const codebook_names[CODEBOOK_MEMBERS] = {
	"mul1_multiplier"
};
enum expert_member {
	EXPERTS_LAYERS,
	EXPERTS_PER_LAYER,
	EXPERTS_HIDDEN,
	EXPERTS_INTERMEDIATE,
	EXPERTS_MATRICES,
	EXPERTS_MEMBERS
};
static const char *const expert_names[EXPERTS_MEMBERS] = {
	"layers", "experts_per_layer", "hidden", "intermediate", "matrices"
};
enum summary_member { SUMMARY_MATRICES, SUMMARY_LAYERS, SUMMARY_MEMBERS };
static const char *const summary_names[SUMMARY_MEMBERS] = { "matrices",
							    "layers" };
/* quality by its status, each status a member more: none's status, established's reference, recorded's record and
 * sha256 */
enum quality_status {
	QUALITY_NONE,
	QUALITY_ESTABLISHED,
	QUALITY_RECORDED,
	QUALITY_STATUSES
};
enum quality_member {
	QUALITY_STATUS,
	QUALITY_DETAIL,
	QUALITY_SHA256,
	QUALITY_MEMBERS
};
static const char *const quality_statuses[QUALITY_STATUSES] = { "none",
								"established",
								"recorded" };
static const char *const quality_names[QUALITY_STATUSES][QUALITY_MEMBERS] = {
	{ "status" },
	{ "status", "reference" },
	{ "status", "record", "sha256" }
};

struct reading {
	const char *path; /* the sidecar, named by each refusal */
	struct json_scratch scratch;
	result_status status; /* the first refusal's */
};

/* A file read once: its bytes when kept (a NUL behind them), their count and digest */
struct file_bytes {
	char *text;
	size_t count;
	unsigned char digest[SHA256_DIGEST_BYTES];
};

static bool refuse(struct reading *reading, result_status status,
		   const char *format, ...)
	__attribute__((format(printf, REFUSAL_FORMAT_PARAMETER,
			      REFUSAL_VALUES_PARAMETER)));
/* One error line naming the sidecar; the first refusal's status is kept. Always false. */
static bool refuse(struct reading *reading, result_status status,
		   const char *format, ...)
{
	char line[LINE_BYTES];
	va_list values;
	va_start(values, format);
	const int written = vsnprintf(line, sizeof(line), format, values);
	va_end(values);
	(void)fprintf(stderr, "exl3 descriptor: %s: %s\n", reading->path,
		      written >= 0 ? line : "refused");
	if (reading->status == STATUS_OK)
		reading->status = status;
	return false;
}

/* A value's bytes in the file, at most a line's, for a refusal line */
static int shown_bytes(struct json_value value)
{
	const size_t count = value.text != NULL ? value.end - value.begin : 0;
	return (int)(count < LINE_BYTES ? count : LINE_BYTES);
}

/* Where a value's bytes begin, "" for an absent value */
static const char *shown(struct json_value value)
{
	return value.text != NULL ? value.text + value.begin : "";
}

static bool is_kind(struct json_value value, char opening)
{
	return value.text != NULL && value.begin < value.end &&
	       value.text[value.begin] == opening;
}

static bool is_null(struct json_value value)
{
	return value.text != NULL &&
	       value.end - value.begin == strlen("null") &&
	       memcmp(value.text + value.begin, "null", strlen("null")) == 0;
}

static void field_name(char field[FIELD_BYTES], const char *parent,
		       const char *name)
{
	if (snprintf(field, FIELD_BYTES, "%s.%s", parent, name) < 0)
		field[0] = '\0';
}

/* The decimal digits from begin to end, as Python's str() of an int writes them: no sign, no leading zero */
static bool decimal_of(const char *text, size_t begin, size_t end,
		       uint64_t *number_out)
{
	uint64_t number = 0;
	bool is_decimal = begin < end &&
			  (text[begin] != '0' || end - begin == 1u);
	for (size_t position = begin; is_decimal && position < end;
	     position++) {
		const char digit = text[position];
		is_decimal = digit >= '0' && digit <= '9' &&
			     number <= (UINT64_MAX - (uint64_t)(digit - '0')) /
					       DECIMAL_BASE;
		if (is_decimal)
			number =
				number * DECIMAL_BASE + (uint64_t)(digit - '0');
	}
	if (is_decimal)
		*number_out = number;
	return is_decimal;
}

static bool whole_of(struct reading *reading, struct json_value value,
		     const char *field, uint64_t *number_out)
{
	return (value.text != NULL &&
		decimal_of(value.text, value.begin, value.end, number_out)) ||
	       refuse(reading, STATUS_INVALID_ARGUMENT,
		      "%s is %.*s, not a whole number", field,
		      shown_bytes(value), shown(value));
}

static bool whole_is(struct reading *reading, struct json_value value,
		     const char *field, uint64_t expected, const char *whose)
{
	uint64_t number = 0;
	return whole_of(reading, value, field, &number) &&
	       (number == expected || refuse(reading, STATUS_INVALID_ARGUMENT,
					     "%s is %" PRIu64 ", %s %" PRIu64,
					     field, number, whose, expected));
}

/* A string unescaped into the scratch memory: a byte in it other than SPACES, and no NUL */
static bool text_of(struct reading *reading, struct json_value value,
		    const char *field, struct json_bytes *text_out)
{
	const struct json_bytes text =
		is_kind(value, '"') ? json_unescape(&reading->scratch, value) :
				      (struct json_bytes){ NULL, 0 };
	size_t spaces = 0;
	while (text.data != NULL && spaces < text.count &&
	       strchr(SPACES, text.data[spaces]) != NULL)
		spaces++;
	if (text.data == NULL || spaces == text.count ||
	    reading->scratch.is_short ||
	    memchr(text.data, '\0', text.count) != NULL)
		return refuse(reading, STATUS_INVALID_ARGUMENT,
			      "%s is %.*s, not a text", field,
			      shown_bytes(value), shown(value));
	*text_out = text;
	return true;
}

static void digest_text(const unsigned char digest[SHA256_DIGEST_BYTES],
			char text[DIGEST_DIGITS + 1u])
{
	for (uint32_t index = 0; index < SHA256_DIGEST_BYTES; index++) {
		text[(size_t)HEX_PER_BYTE * index] =
			HEX_DIGITS[digest[index] >> NIBBLE_BITS];
		text[(size_t)HEX_PER_BYTE * index + 1u] =
			HEX_DIGITS[digest[index] & NIBBLE_MASK];
	}
	text[DIGEST_DIGITS] = '\0';
}

/* A SHA-256 as the converter writes it (64 lower-case hexadecimal digits) against bytes' digest */
static bool digest_holds(struct reading *reading, struct json_value value,
			 const char *field,
			 const unsigned char digest[SHA256_DIGEST_BYTES],
			 const char *whose)
{
	char text[DIGEST_DIGITS + 1u];
	digest_text(digest, text);
	return (is_kind(value, '"') &&
		value.end - value.begin == DIGEST_DIGITS + QUOTE_BYTES &&
		memcmp(value.text + value.begin + 1u, text, DIGEST_DIGITS) ==
			0) ||
	       refuse(reading, STATUS_INVALID_ARGUMENT, "%s is %.*s, %s %s",
		      field, shown_bytes(value), shown(value), whose, text);
}

/* The members of an object: exactly the named ones, each once (a repeated key is refused before), into values in
 * their order */
static bool members_of(struct reading *reading, struct json_value object,
		       const char *field, const char *const *names,
		       uint32_t name_count, struct json_value *values)
{
	if (!is_kind(object, '{'))
		return refuse(reading, STATUS_INVALID_ARGUMENT,
			      "%s is %.*s, not an object", field,
			      shown_bytes(object), shown(object));
	for (uint32_t index = 0; index < name_count; index++)
		values[index] = (struct json_value){ NULL, 0, 0 };
	struct json_value key = { NULL, 0, 0 }, value = { NULL, 0, 0 };
	for (size_t cursor = object.begin;
	     json_next(object, &cursor, &key, &value);) {
		uint32_t index = 0;
		while (index < name_count &&
		       !json_string_is(&reading->scratch, key, names[index]))
			index++;
		if (index == name_count)
			return refuse(reading, STATUS_INVALID_ARGUMENT,
				      "%s holds %.*s, which it does not define",
				      field, shown_bytes(key),
				      key.text + key.begin);
		values[index] = value;
	}
	for (uint32_t index = 0; index < name_count; index++)
		if (values[index].text == NULL)
			return refuse(reading, STATUS_INVALID_ARGUMENT,
				      "%s has no %s", field, names[index]);
	return true;
}

/* The converter's name of a width (width_name): K<whole>, or K<whole>.5 for a half K */
static size_t width_text(uint32_t bits_x2, char text[WIDTH_TEXT_BYTES])
{
	const int written =
		snprintf(text, WIDTH_TEXT_BYTES, "K%" PRIu32 "%s",
			 bits_x2 / EXL3_HALF_BITS_PER_BIT,
			 bits_x2 % EXL3_HALF_BITS_PER_BIT != 0 ? ".5" : "");
	return written > 0 ? (size_t)written : 0;
}

/* The bits_x2 of a width's name (raw, inside its quotes), MIXED_SLOT for "mixed", WIDTH_SLOTS for neither */
static uint32_t width_slot(const char *name, size_t count)
{
	char text[WIDTH_TEXT_BYTES];
	for (uint32_t bits_x2 = NARROWEST_BITS_X2; bits_x2 < WIDTH_SLOTS;
	     bits_x2++)
		if (width_text(bits_x2, text) == count &&
		    memcmp(text, name, count) == 0)
			return bits_x2;
	return count == strlen("mixed") && memcmp(name, "mixed", count) == 0 ?
		       MIXED_SLOT :
		       WIDTH_SLOTS;
}

/* An object of {width: count}, each count at least 1, into counts by bits_x2 ("mixed" counts in MIXED_SLOT, which
 * only a summary's layers may hold: the comparison refuses it elsewhere) */
static bool width_counts(struct reading *reading, struct json_value object,
			 const char *field, uint64_t counts[WIDTH_SLOTS])
{
	if (!is_kind(object, '{'))
		return refuse(reading, STATUS_INVALID_ARGUMENT,
			      "%s is %.*s, not an object", field,
			      shown_bytes(object), shown(object));
	struct json_value key = { NULL, 0, 0 }, value = { NULL, 0, 0 };
	for (size_t cursor = object.begin;
	     json_next(object, &cursor, &key, &value);) {
		const char *const name = key.text + key.begin + 1u;
		const size_t count = key.end - key.begin - QUOTE_BYTES;
		const uint32_t slot = width_slot(name, count);
		char counted[FIELD_BYTES];
		(void)snprintf(counted, sizeof(counted), "%s.%.*s", field,
			       (int)count, name);
		if (slot == WIDTH_SLOTS)
			return refuse(reading, STATUS_INVALID_ARGUMENT,
				      "%s: %.*s names no EXL3 width", field,
				      shown_bytes(key), shown(key));
		if (!whole_of(reading, value, counted, &counts[slot]))
			return false;
		if (counts[slot] == 0)
			return refuse(reading, STATUS_INVALID_ARGUMENT,
				      "%s is 0", counted);
	}
	return true;
}

/* Two tables of counts by bits_x2, the first difference named */
static bool counts_equal(struct reading *reading, const char *field,
			 const uint64_t given[WIDTH_SLOTS],
			 const uint64_t expected[WIDTH_SLOTS],
			 const char *whose)
{
	for (uint32_t slot = 0; slot < WIDTH_SLOTS; slot++) {
		char name[WIDTH_TEXT_BYTES] = "mixed";
		if (slot != MIXED_SLOT)
			(void)width_text(slot, name);
		if (given[slot] != expected[slot])
			return refuse(reading, STATUS_INVALID_ARGUMENT,
				      "%s gives %" PRIu64 " of %s, %s %" PRIu64,
				      field, given[slot], name, whose,
				      expected[slot]);
	}
	return true;
}

/* The pack's routed matrices of one layer and kind, by bits_x2: each must be an EXL3 matrix of the kind's shape */
static bool pack_widths(struct reading *reading, const struct pack_record *pack,
			const struct exl3_descriptor_matrix *matrix,
			uint32_t layer, uint32_t experts,
			uint64_t counts[WIDTH_SLOTS])
{
	const struct pack_tensor *const tensors = pack_tensors_of(pack);
	for (uint32_t expert = 0; expert < experts; expert++) {
		char name[NAME_BYTES];
		const int length = snprintf(
			name, sizeof(name),
			"layers.%" PRIu32 ".ffn.experts.%" PRIu32 ".%s.weight",
			layer, expert, matrix->kind);
		const uint32_t found =
			length > 0 && (size_t)length < sizeof(name) ?
				pack_find_tensor(pack, name, (uint32_t)length) :
				PACK_NOT_FOUND;
		const struct pack_tensor *const tensor =
			found != PACK_NOT_FOUND ? &tensors[found] : NULL;
		const uint32_t bits_x2 =
			tensor != NULL ? exl3_bits_x2(tensor->bits_per_weight) :
					 0u;
		if (tensor == NULL ||
		    tensor->weight_format != WEIGHT_FORMAT_EXL3 ||
		    tensor->dimension_count != SHAPE_DIMENSIONS ||
		    tensor->dimensions[0] != matrix->outputs ||
		    tensor->dimensions[1] != matrix->inputs || bits_x2 == 0 ||
		    bits_x2 >= WIDTH_SLOTS)
			return refuse(
				reading, STATUS_INVALID_ARGUMENT,
				"the pack holds no EXL3 matrix %s of (%" PRIu32
				", %" PRIu32 ")",
				name, matrix->outputs, matrix->inputs);
		counts[bits_x2]++;
	}
	return true;
}

/* widths against the pack: each layer once, each of its kinds' counts the pack's; the summary's counts recomputed,
 * the matrices by width and the layers by their one width (MIXED_SLOT for more) */
static bool widths_hold(struct reading *reading, struct json_value widths,
			const struct pack_record *pack,
			const struct exl3_descriptor_model *model,
			uint64_t matrices[WIDTH_SLOTS],
			uint64_t layers[WIDTH_SLOTS])
{
	const char *kinds[EXL3_DESCRIPTOR_MATRIX_KINDS];
	for (uint32_t kind = 0; kind < EXL3_DESCRIPTOR_MATRIX_KINDS; kind++)
		kinds[kind] = model->matrices[kind].kind;
	if (!is_kind(widths, '{'))
		return refuse(reading, STATUS_INVALID_ARGUMENT,
			      "widths is not an object");
	uint32_t layer_count = 0;
	struct json_value key = { NULL, 0, 0 }, value = { NULL, 0, 0 };
	for (size_t cursor = widths.begin;
	     json_next(widths, &cursor, &key, &value); layer_count++) {
		uint64_t layer = 0;
		if (!decimal_of(key.text, key.begin + 1u, key.end - 1u,
				&layer) ||
		    layer >= model->layers)
			return refuse(
				reading, STATUS_INVALID_ARGUMENT,
				"widths holds the layer %.*s: the model's are 0 to %" PRIu32,
				shown_bytes(key), shown(key),
				model->layers - 1u);
		char field[FIELD_BYTES];
		(void)snprintf(field, sizeof(field), "widths.%" PRIu64, layer);
		struct json_value counted[EXL3_DESCRIPTOR_MATRIX_KINDS] = {
			{ NULL, 0, 0 }
		};
		uint64_t in_layer[WIDTH_SLOTS] = { 0 };
		if (!members_of(reading, value, field, kinds,
				EXL3_DESCRIPTOR_MATRIX_KINDS, counted))
			return false;
		for (uint32_t kind = 0; kind < EXL3_DESCRIPTOR_MATRIX_KINDS;
		     kind++) {
			char kind_field[FIELD_BYTES];
			field_name(kind_field, field, kinds[kind]);
			uint64_t given[WIDTH_SLOTS] = { 0 },
				 held[WIDTH_SLOTS] = { 0 };
			if (!width_counts(reading, counted[kind], kind_field,
					  given) ||
			    !pack_widths(reading, pack, &model->matrices[kind],
					 (uint32_t)layer,
					 model->experts_per_layer, held) ||
			    !counts_equal(reading, kind_field, given, held,
					  "the pack holds"))
				return false;
			for (uint32_t slot = 0; slot < WIDTH_SLOTS; slot++)
				in_layer[slot] += held[slot];
		}
		uint32_t layer_width = MIXED_SLOT, layer_widths = 0;
		for (uint32_t slot = NARROWEST_BITS_X2; slot < WIDTH_SLOTS;
		     slot++) {
			matrices[slot] += in_layer[slot];
			layer_widths += in_layer[slot] != 0 ? 1u : 0u;
			layer_width = in_layer[slot] != 0 ? slot : layer_width;
		}
		layers[layer_widths == 1u ? layer_width : MIXED_SLOT]++;
	}
	return layer_count == model->layers ||
	       refuse(reading, STATUS_INVALID_ARGUMENT,
		      "widths gives %" PRIu32 " layers, the model %" PRIu32,
		      layer_count, model->layers);
}

/* experts: the model's geometry and each kind's [outputs, inputs] */
static bool experts_hold(struct reading *reading, struct json_value experts,
			 const struct exl3_descriptor_model *model)
{
	struct json_value values[EXPERTS_MEMBERS] = { { NULL, 0, 0 } },
			  shapes[EXL3_DESCRIPTOR_MATRIX_KINDS] = { { NULL, 0,
								     0 } };
	const char *kinds[EXL3_DESCRIPTOR_MATRIX_KINDS];
	for (uint32_t kind = 0; kind < EXL3_DESCRIPTOR_MATRIX_KINDS; kind++)
		kinds[kind] = model->matrices[kind].kind;
	if (!members_of(reading, experts, "experts", expert_names,
			EXPERTS_MEMBERS, values) ||
	    !whole_is(reading, values[EXPERTS_LAYERS], "experts.layers",
		      model->layers, "the model's") ||
	    !whole_is(reading, values[EXPERTS_PER_LAYER],
		      "experts.experts_per_layer", model->experts_per_layer,
		      "the model's") ||
	    !whole_is(reading, values[EXPERTS_HIDDEN], "experts.hidden",
		      model->hidden, "the model's") ||
	    !whole_is(reading, values[EXPERTS_INTERMEDIATE],
		      "experts.intermediate", model->intermediate,
		      "the model's") ||
	    !members_of(reading, values[EXPERTS_MATRICES], "experts.matrices",
			kinds, EXL3_DESCRIPTOR_MATRIX_KINDS, shapes))
		return false;
	for (uint32_t kind = 0; kind < EXL3_DESCRIPTOR_MATRIX_KINDS; kind++) {
		const uint32_t expected[SHAPE_DIMENSIONS] = {
			model->matrices[kind].outputs,
			model->matrices[kind].inputs
		};
		char field[FIELD_BYTES];
		field_name(field, "experts.matrices", kinds[kind]);
		uint32_t dimension = 0;
		struct json_value size = { NULL, 0, 0 };
		for (size_t cursor = shapes[kind].begin;
		     is_kind(shapes[kind], '[') &&
		     json_next(shapes[kind], &cursor, NULL, &size);
		     dimension++)
			if (dimension < SHAPE_DIMENSIONS &&
			    !whole_is(reading, size, field, expected[dimension],
				      "the model's"))
				return false;
		if (!is_kind(shapes[kind], '[') ||
		    dimension != SHAPE_DIMENSIONS)
			return refuse(reading, STATUS_INVALID_ARGUMENT,
				      "%s is not [outputs, inputs]", field);
	}
	return true;
}

/* A regular file of at most most_bytes bytes (named in a refusal by what), read once and hashed: kept whole, a NUL
 * behind it, or else a piece at a time */
MUST_CHECK static bool file_read(struct reading *reading, const char *path,
				 const char *what, uint64_t most_bytes,
				 bool keeps, struct file_bytes *file_out)
{
	/* O_NONBLOCK: a FIFO at the path must not hold the open before the regular-file check */
	const int descriptor =
		open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOCTTY);
	struct stat file;
	errno = descriptor >= 0 ? 0 : errno;
	if (descriptor < 0 || FAULT_POINT(exl3_descriptor_stat) ||
	    fstat(descriptor, &file) != 0) {
		const int error = errno;
		if (descriptor >= 0)
			(void)close(descriptor);
		return refuse(reading, STATUS_INPUT_OUTPUT_ERROR,
			      "%s cannot be opened: %s", what,
			      error != 0 ? strerror(error) :
					   "its status fails");
	}
	const uint64_t size = (uint64_t)file.st_size;
	const bool fits = S_ISREG(file.st_mode) && size <= most_bytes;
	char *const bytes =
		!fits || FAULT_POINT(exl3_descriptor_memory) ?
			NULL :
			malloc(keeps ? (size_t)size + 1u : CAPTURE_PIECE_BYTES);
	struct sha256_stream stream;
	sha256_begin(&stream);
	uint64_t done = 0;
	bool is_read = bytes != NULL;
	while (is_read && done < size) {
		char *const place = keeps ? bytes + done : bytes;
		const size_t asked =
			keeps || size - done < CAPTURE_PIECE_BYTES ?
				(size_t)(size - done) :
				CAPTURE_PIECE_BYTES;
		const ssize_t got =
			FAULT_POINT(exl3_descriptor_read) ?
				-1 :
				pread(descriptor, place, asked, (off_t)done);
		is_read = got > 0 &&
			  sha256_feed(&stream, place, (size_t)got) == STATUS_OK;
		done += is_read ? (uint64_t)got : 0u;
	}
	(void)close(descriptor);
	const bool has_memory = bytes != NULL;
	if (is_read && sha256_finish(&stream, file_out->digest) == STATUS_OK) {
		if (keeps)
			bytes[size] = '\0';
		else
			free(bytes);
		file_out->text = keeps ? bytes : NULL;
		file_out->count = (size_t)size;
		return true;
	}
	free(bytes);
	return !fits ? refuse(reading, STATUS_INVALID_ARGUMENT,
			      "%s is not a regular file of at most %" PRIu64
			      " bytes",
			      what, most_bytes) :
	       !has_memory ? refuse(reading, STATUS_OUT_OF_MEMORY,
				    "%s cannot be held in memory", what) :
			     refuse(reading, STATUS_INPUT_OUTPUT_ERROR,
				    "%s cannot be read whole", what);
}

/* quality: none (only with the opt-in), established by a reference, or recorded by a capture whose digest holds; its
 * words for the startup line */
static bool quality_holds(struct reading *reading, struct json_value quality,
			  const char *map, bool admits_unrated,
			  char words[QUALITY_WORDS_BYTES])
{
	struct json_value values[QUALITY_MEMBERS] = { { NULL, 0, 0 } },
			  key = { NULL, 0, 0 }, value = { NULL, 0, 0 };
	if (!is_kind(quality, '{'))
		return refuse(reading, STATUS_INVALID_ARGUMENT,
			      "quality is %.*s, not an object",
			      shown_bytes(quality), shown(quality));
	for (size_t cursor = quality.begin;
	     json_next(quality, &cursor, &key, &value);)
		if (json_string_is(&reading->scratch, key, "status"))
			values[QUALITY_STATUS] = value;
	if (values[QUALITY_STATUS].text == NULL)
		return refuse(reading, STATUS_INVALID_ARGUMENT,
			      "quality has no status");
	uint32_t status = 0;
	while (status < QUALITY_STATUSES &&
	       !json_string_is(&reading->scratch, values[QUALITY_STATUS],
			       quality_statuses[status]))
		status++;
	if (status == QUALITY_STATUSES)
		return refuse(
			reading, STATUS_UNSUPPORTED,
			"quality.status is %.*s: this build serves none, established and recorded",
			shown_bytes(values[QUALITY_STATUS]),
			shown(values[QUALITY_STATUS]));
	if (!members_of(reading, quality, "quality", quality_names[status],
			status + 1u, values))
		return false;
	if (status == QUALITY_NONE && !admits_unrated)
		return refuse(
			reading, STATUS_UNSUPPORTED,
			"quality none: no NLL/PPL capture or qualification covers the map %s, which "
			"is served only on " EXL3_DESCRIPTOR_OPT_IN
			" (no quality claim)",
			map);
	if (status == QUALITY_NONE) {
		(void)snprintf(words, QUALITY_WORDS_BYTES,
			       "none (" EXL3_DESCRIPTOR_OPT_IN
			       ": no quality claim)");
		return true;
	}
	char field[FIELD_BYTES], record[PATH_MAX], named[QUALITY_WORDS_BYTES];
	struct json_bytes text = { NULL, 0 };
	field_name(field, "quality", quality_names[status][QUALITY_DETAIL]);
	if (!text_of(reading, values[QUALITY_DETAIL], field, &text))
		return false;
	if (status == QUALITY_ESTABLISHED) {
		(void)snprintf(words, QUALITY_WORDS_BYTES, "established (%.*s)",
			       (int)(text.count < LINE_BYTES ? text.count :
							       LINE_BYTES),
			       text.data);
		return true;
	}
	if (text.data[0] != '/' || text.count >= sizeof(record))
		return refuse(reading, STATUS_INVALID_ARGUMENT,
			      "quality.record is %.*s, not an absolute path",
			      shown_bytes(values[QUALITY_DETAIL]),
			      shown(values[QUALITY_DETAIL]));
	memcpy(record, text.data, text.count);
	record[text.count] = '\0';
	struct file_bytes capture = { NULL, 0, { 0 } };
	(void)snprintf(named, sizeof(named), "quality.record %s", record);
	if (!file_read(reading, record, named, CAPTURE_MOST_BYTES, false,
		       &capture) ||
	    !digest_holds(reading, values[QUALITY_SHA256], "quality.sha256",
			  capture.digest, "the record's"))
		return false;
	(void)snprintf(words, QUALITY_WORDS_BYTES, "recorded (%s)", record);
	return true;
}

/* Every rule of the contract, in the order of the file's members; the startup line once all hold */
static bool descriptor_holds(struct reading *reading,
			     const struct file_bytes *file, const char *map,
			     const struct pack_record *pack,
			     const struct exl3_descriptor_model *model,
			     const char *tables, uint64_t table_byte_count,
			     bool admits_unrated)
{
	struct json_value document = { NULL, 0, 0 }, key = { NULL, 0, 0 },
			  value = { NULL, 0, 0 };
	if (json_read_document(file->text, file->count, &document) !=
		    JSON_DOCUMENT_VALID ||
	    !is_kind(document, '{'))
		return refuse(reading, STATUS_INVALID_ARGUMENT,
			      "not one JSON object");
	if (json_has_repeated_key(&reading->scratch, document) ||
	    reading->scratch.is_short)
		return refuse(reading, STATUS_INVALID_ARGUMENT,
			      "an object holds a key twice");
	/* the version first: a later format is named as such, not by the members it adds */
	struct json_value version = { NULL, 0, 0 };
	for (size_t cursor = document.begin;
	     json_next(document, &cursor, &key, &value);)
		if (json_string_is(&reading->scratch, key, "format_version"))
			version = value;
	uint64_t number = 0;
	if (version.text == NULL)
		return refuse(reading, STATUS_INVALID_ARGUMENT,
			      "the descriptor has no format_version");
	if (!decimal_of(version.text, version.begin, version.end, &number) ||
	    number != FORMAT_VERSION)
		return refuse(reading, STATUS_UNSUPPORTED,
			      "format_version is %.*s: this build reads %u",
			      shown_bytes(version), shown(version),
			      FORMAT_VERSION);
	const struct pack_header *const header = pack_header_of(pack);
	struct json_value root[ROOT_MEMBERS] = { { NULL, 0, 0 } },
			  map_values[MAP_MEMBERS] = { { NULL, 0, 0 } },
			  architecture[ARCHITECTURE_MEMBERS] = { { NULL, 0,
								   0 } },
			  quantiser[QUANTISER_MEMBERS] = { { NULL, 0, 0 } },
			  codebook[CODEBOOK_MEMBERS] = { { NULL, 0, 0 } },
			  summary[SUMMARY_MEMBERS] = { { NULL, 0, 0 } };
	struct json_bytes source = { NULL, 0 };
	char codebook_text[CODEBOOK_BYTES];
	(void)snprintf(codebook_text, sizeof(codebook_text), "0x%08" PRIX32,
		       (uint32_t)EXL3_MUL1_MULTIPLIER);
	if (!members_of(reading, document, "the descriptor", root_names,
			ROOT_MEMBERS, root) ||
	    !members_of(reading, root[ROOT_MAP], "map", map_names, MAP_MEMBERS,
			map_values) ||
	    !whole_is(reading, map_values[MAP_LOGICAL_BYTES],
		      "map.logical_bytes", header->file_byte_count,
		      "the pack's") ||
	    !whole_is(reading, map_values[MAP_HEADER_DIGEST],
		      "map.pack_header_digest", header->header_digest,
		      "the pack's") ||
	    !members_of(reading, root[ROOT_ARCHITECTURE], "architecture",
			architecture_names, ARCHITECTURE_MEMBERS, architecture))
		return false;
	if (!json_string_is(&reading->scratch,
			    architecture[ARCHITECTURE_MODEL_TYPE],
			    model->model_type))
		return refuse(
			reading, STATUS_UNSUPPORTED,
			"architecture.model_type is %.*s, the model's \"%s\"",
			shown_bytes(architecture[ARCHITECTURE_MODEL_TYPE]),
			shown(architecture[ARCHITECTURE_MODEL_TYPE]),
			model->model_type);
	if (!members_of(reading, root[ROOT_QUANTISER], "quantiser",
			quantiser_names, QUANTISER_MEMBERS, quantiser) ||
	    !text_of(reading, quantiser[QUANTISER_SOURCE], "quantiser.source",
		     &source) ||
	    !members_of(reading, root[ROOT_CODEBOOK], "codebook",
			codebook_names, CODEBOOK_MEMBERS, codebook))
		return false;
	if (!json_string_is(&reading->scratch, codebook[CODEBOOK_MULTIPLIER],
			    codebook_text))
		return refuse(
			reading, STATUS_UNSUPPORTED,
			"codebook.mul1_multiplier is %.*s: this build decodes \"%s\"",
			shown_bytes(codebook[CODEBOOK_MULTIPLIER]),
			shown(codebook[CODEBOOK_MULTIPLIER]), codebook_text);
	const struct json_value tables_digest = root[ROOT_TABLES];
	unsigned char digest[SHA256_DIGEST_BYTES];
	if (tables != NULL && !is_null(tables_digest) &&
	    (sha256_of(tables, (size_t)table_byte_count, digest) != STATUS_OK ||
	     !digest_holds(reading, tables_digest, "front_end_tables_sha256",
			   digest, "the tables'")))
		return false;
	uint64_t matrices[WIDTH_SLOTS] = { 0 }, layers[WIDTH_SLOTS] = { 0 },
		 given[WIDTH_SLOTS] = { 0 }, given_layers[WIDTH_SLOTS] = { 0 };
	uint64_t exl3_matrices = 0;
	for (uint32_t tensor = 0; tensor < header->tensor_count; tensor++)
		exl3_matrices += pack_tensors_of(pack)[tensor].weight_format ==
						 WEIGHT_FORMAT_EXL3 ?
					 1u :
					 0u;
	const uint64_t routed = (uint64_t)model->layers *
				model->experts_per_layer *
				EXL3_DESCRIPTOR_MATRIX_KINDS;
	char quality_words[QUALITY_WORDS_BYTES];
	if (!experts_hold(reading, root[ROOT_EXPERTS], model) ||
	    !widths_hold(reading, root[ROOT_WIDTHS], pack, model, matrices,
			 layers) ||
	    (exl3_matrices != routed &&
	     !refuse(reading, STATUS_INVALID_ARGUMENT,
		     "the pack holds %" PRIu64
		     " EXL3 matrices, the model's routed experts %" PRIu64,
		     exl3_matrices, routed)) ||
	    !members_of(reading, root[ROOT_SUMMARY], "widths_summary",
			summary_names, SUMMARY_MEMBERS, summary) ||
	    !width_counts(reading, summary[SUMMARY_MATRICES],
			  "widths_summary.matrices", given) ||
	    !counts_equal(reading, "widths_summary.matrices", given, matrices,
			  "widths") ||
	    !width_counts(reading, summary[SUMMARY_LAYERS],
			  "widths_summary.layers", given_layers) ||
	    !counts_equal(reading, "widths_summary.layers", given_layers,
			  layers, "widths") ||
	    !quality_holds(reading, root[ROOT_QUALITY], map, admits_unrated,
			   quality_words))
		return false;
	char widths_words[WIDTHS_TEXT_BYTES] = "",
	     sidecar_digest[DIGEST_DIGITS + 1u];
	size_t used = 0;
	for (uint32_t slot = NARROWEST_BITS_X2; slot < WIDTH_SLOTS; slot++) {
		char name[WIDTH_TEXT_BYTES];
		const int written =
			matrices[slot] == 0 || width_text(slot, name) == 0 ?
				0 :
				snprintf(widths_words + used,
					 sizeof(widths_words) - used,
					 "%s%s %" PRIu64, used != 0 ? ", " : "",
					 name, matrices[slot]);
		used += written > 0 && (size_t)written <
						sizeof(widths_words) - used ?
				(size_t)written :
				0u;
	}
	digest_text(file->digest, sidecar_digest);
	(void)fprintf(
		stderr,
		"exl3 descriptor: %s sha256 %s: %s, %.*s, widths %s, quality %s\n",
		reading->path, sidecar_digest, model->model_type,
		(int)source.count, source.data, widths_words, quality_words);
	return true;
}

/* No sidecar: the open pack served by default when the model lists it as established (its header digest and size),
 * else only on the opt-in */
static bool absent_holds(struct reading *reading,
			 const struct pack_header *header,
			 const struct exl3_descriptor_model *model,
			 bool admits_unrated)
{
	const char *name = NULL;
	for (uint32_t index = 0;
	     name == NULL && index < model->established_count; index++)
		if (model->established[index].header_digest ==
			    header->header_digest &&
		    model->established[index].file_byte_count ==
			    header->file_byte_count)
			name = model->established[index].name;
	if (name == NULL && !admits_unrated)
		return refuse(
			reading, STATUS_UNSUPPORTED,
			"none here, and the pack (header digest %" PRIu64
			", %" PRIu64
			" bytes) is not listed as established: it is served only on " EXL3_DESCRIPTOR_OPT_IN
			" (no quality claim)",
			header->header_digest, header->file_byte_count);
	(void)fprintf(
		stderr,
		"exl3 descriptor: none at %s: %s%s (header digest %" PRIu64
		", %" PRIu64 " bytes), served %s\n",
		reading->path,
		name != NULL ? "the established pack " :
			       "a pack the model does not list as established",
		name != NULL ? name : "", header->header_digest,
		header->file_byte_count,
		name != NULL ? "by default" :
			       "on " EXL3_DESCRIPTOR_OPT_IN
			       " (no quality claim)");
	return true;
}

result_status exl3_descriptor_check(const char *map_path,
				    const struct pack_record *pack,
				    const struct exl3_descriptor_model *model,
				    const char *tables,
				    uint64_t table_byte_count,
				    bool admits_unrated)
{
	if (map_path == NULL || pack == NULL || model == NULL ||
	    model->model_type == NULL || model->layers == 0 ||
	    model->experts_per_layer == 0 ||
	    (model->established == NULL && model->established_count != 0) ||
	    (tables == NULL && table_byte_count != 0))
		return STATUS_INVALID_ARGUMENT;
	char real[PATH_MAX], sidecar[PATH_MAX];
	const char *const map = realpath(map_path, real) != NULL ? real :
								   map_path;
	const int length = snprintf(sidecar, sizeof(sidecar),
				    "%s" EXL3_DESCRIPTOR_SUFFIX, map);
	if (length < 0 || (size_t)length >= sizeof(sidecar)) {
		(void)fprintf(
			stderr,
			"exl3 descriptor: the map's path %s is too long for its sidecar\n",
			map);
		return STATUS_INVALID_ARGUMENT;
	}
	struct reading reading = { .path = sidecar, .status = STATUS_OK };
	struct stat link;
	/* nothing at the path, not even a link (one to nothing is refused at its open), or nothing can be: a map name
	 * over NAME_MAX - strlen(EXL3_DESCRIPTOR_SUFFIX) bytes leaves no room for its sidecar's */
	if (lstat(sidecar, &link) != 0 &&
	    (errno == ENOENT || errno == ENAMETOOLONG))
		return absent_holds(&reading, pack_header_of(pack), model,
				    admits_unrated) ?
			       STATUS_OK :
			       reading.status;
	struct file_bytes file = { NULL, 0, { 0 } };
	if (!file_read(&reading, sidecar, "the file",
		       EXL3_DESCRIPTOR_MOST_BYTES, true, &file))
		return reading.status;
	reading.scratch = (struct json_scratch){
		.bytes = FAULT_POINT(exl3_descriptor_memory) ?
				 NULL :
				 malloc(SCRATCH_PER_TEXT_BYTE *
					(file.count + 1u)),
		.capacity = SCRATCH_PER_TEXT_BYTE * (file.count + 1u)
	};
	if (reading.scratch.bytes == NULL)
		(void)refuse(&reading, STATUS_OUT_OF_MEMORY,
			     "has no memory to be checked in");
	else if (!descriptor_holds(&reading, &file, map, pack, model, tables,
				   table_byte_count, admits_unrated) &&
		 reading.status == STATUS_OK)
		reading.status = STATUS_INVALID_ARGUMENT;
	free(reading.scratch.bytes);
	free(file.text);
	return reading.status;
}

result_status
exl3_descriptor_tc_check(const char *model_name,
			 const struct exl3_descriptor_model *model,
			 uint32_t ranks, uint32_t rotation)
{
	if (model_name == NULL || model == NULL || ranks == 0)
		return STATUS_INVALID_ARGUMENT;
	const struct {
		const char *field;
		uint32_t value, built;
	} fields[] = { { "experts.hidden", model->hidden,
			 DS41F_EXL3_TC_HIDDEN },
		       { "routes_per_row", model->routes_per_row,
			 DS41F_EXL3_TC_PAIRS_PER_ROW },
		       { "experts.experts_per_layer", model->experts_per_layer,
			 DS41F_EXL3_TCP_EXPERTS },
		       { "experts.layers", model->layers, EXL3_TC_CONE_LAYERS },
		       { "cone_first_layer", model->cone_first_layer,
			 EXL3_TC_CONE_FIRST_LAYER } };
	for (size_t index = 0; index < sizeof(fields) / sizeof(fields[0]);
	     index++) {
		if (fields[index].value != fields[index].built) {
			(void)fprintf(stderr,
				      TC_REFUSAL
				      "%s is %" PRIu32
				      ", the tensor-core MoE's kernels %" PRIu32
				      " (exl3_tc.h)\n",
				      model_name, fields[index].field,
				      fields[index].value, fields[index].built);
			return STATUS_UNSUPPORTED;
		}
	}
	/* each rank's gate/up columns as the plan splits them (exl3_layout.h): chunk c, rank c's unrotated */
	for (uint32_t rank = 0; rank < ranks; rank++) {
		struct bb_exl3_output_shard_plan plan;
		if (bb_exl3_output_shard_prepare(
			    model->hidden, model->intermediate, TC_SPLIT_WIDTH,
			    ranks, rank, BB_EXL3_REFERENCE_ORDER,
			    &plan) != STATUS_OK) {
			(void)fprintf(stderr,
				      TC_REFUSAL "experts.intermediate %" PRIu32
						 " has no split at %" PRIu32
						 " ranks (exl3_layout.h)\n",
				      model_name, model->intermediate, ranks);
			return STATUS_UNSUPPORTED;
		}
		const uint32_t columns = plan.gate_up.output_count;
		const bool narrow = (rank & 1u) == 0u;
		const uint32_t mixed = narrow ? DS41F_EXL3_TC_MIXED_WIDTH -
							DS41F_EXL3_TC_NARROWER :
						DS41F_EXL3_TC_MIXED_WIDTH;
		if (rotation == 0 && columns != EXL3_TC_WIDE_INTERMEDIATE &&
		    columns != EXL3_TC_NARROW_INTERMEDIATE) {
			(void)fprintf(
				stderr,
				TC_REFUSAL
				"experts.intermediate %" PRIu32 " at %" PRIu32
				" ranks gives rank %" PRIu32 " %" PRIu32
				" columns, the tensor-core MoE's kernels %" PRIu32
				" or %" PRIu32 " (exl3_tc.h)\n",
				model_name, model->intermediate, ranks, rank,
				columns, EXL3_TC_WIDE_INTERMEDIATE,
				EXL3_TC_NARROW_INTERMEDIATE);
			return STATUS_UNSUPPORTED;
		}
		if (rotation != 0 && columns != mixed) {
			(void)fprintf(
				stderr,
				TC_REFUSAL
				"experts.intermediate %" PRIu32 " at %" PRIu32
				" ranks gives chunk %" PRIu32 " (rank %" PRIu32
				"'s unrotated) %" PRIu32
				" columns, the tensor-core MoE's mixed forms %" PRIu32
				" at an %s chunk (--exl3_expert_rotation=%" PRIu32
				", exl3_tc_records.h)\n",
				model_name, model->intermediate, ranks, rank,
				rank, columns, mixed, narrow ? "even" : "odd",
				rotation);
			return STATUS_UNSUPPORTED;
		}
	}
	return STATUS_OK;
}
