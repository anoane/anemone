/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* The EXL3 map descriptor (exl3_descriptor.h) against the model's EXL3 pack (model_v41_pack.h: layers 18 to 22 K2, the
 * others K3, as the 2.9 bpw map), written as the converter writes it beside a map file in a fresh folder: it holds with
 * each quality (none only with the opt-in), through a link to the map, with the tables or without them; each rule's
 * refusal is seen with its status and its line; with nothing beside the map, a pack the model lists as established is
 * served and any other only on the opt-in, whatever path opened it, while a descriptor beside it decides (none
 * refused, established served with its own reference, as tools/model_data.sh writes it); and a pack with one
 * matrix of another width in a layer (a synthetic mixed pack) is described as such and refused by the uniform
 * descriptor. */
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "anemone/exl3_records.h"
#include "anemone/exl3_tc_records.h"
#include "anemone/hooks.h"
#include "exl3_descriptor.h"
#include "model_registry.h"
#include "model_v41.h"
#include "model_v41_pack.h"
#include "model_v41_tp2.h"
#include "sha256.h"

#define TEST_TEXT_BYTES 262144u
#define TEST_LOG_BYTES 8192u
#define TEST_PATH_BYTES 1024u
#define K2_LAYERS \
	UINT64_C( \
		0x7C0000) /* model_v41_pack.c's: layers 18 to 22 hold K2, the others K3 */
#define MIXED_LAYER 0u /* the mixed pack's: one w2 matrix K2 */
#define NO_MIXED_LAYER MODEL_LAYERS
#define SOME_DIGEST \
	"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
#define HEX_PER_BYTE 2u
#define DIGEST_DIGITS ((size_t)HEX_PER_BYTE * SHA256_DIGEST_BYTES)
#define QUOTE_BYTES 2u
#define TEST_QUALITY_BYTES 4096u
#define TEST_MAP_SUFFIX ".exl3map"
/* map names at the sidecar's edge: the longest with room for its sidecar's name, one byte more, and NAME_MAX */
#define TEST_EDGE_NAMES 3u
#define TEST_MISMATCHED_PACKS \
	2u /* the listed pack's digest at another size, and its size with another digest */
#define NIBBLE_BITS 4u
#define NIBBLE_MASK 15u
#define PUT_FORMAT_PARAMETER 2
#define PUT_VALUES_PARAMETER 3
#define TABLES "the front end's tables"
/* the tensor-core MoE's check: one field of the model's another value each, and its line */
#define TC_LINE \
	"exl3 descriptor: --exl3_tc_moe=1: the model " V41_DATA_MODEL_NAME "'s "
#define TC_OTHER_HIDDEN 4096u
#define TC_OTHER_ROUTES 8u
#define TC_OTHER_EXPERTS 256u
#define TC_OTHER_LAYERS 61u
#define TC_OTHER_CONE_FIRST 3u
#define TC_NARROW_CHUNKS 1536u /* 12 blocks of 128: 384 columns a rank */
#define TC_WIDE_CHUNKS \
	2560u /* 20 blocks: 640 columns at every rank, the i640 form, not the mixed alternation */
#define TC_THREE_WIDE_CHUNKS 2432u /* 19 blocks: 512, 640, 640, 640 */
#define TC_OFF_GRID 2300u /* no whole 128-column block split */
#define ESTABLISHED                                                                 \
	"{\"status\": \"established\", \"reference\": \"TP4_EXL3's default route, " \
	"qualified by identity\"}"
/* the reference tools/model_data.sh gives the 2.9 bpw map's descriptor, the map the model lists as established */
#define MODEL_DATA_REFERENCE                                                                                     \
	"LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw @ 0be63c33, TP4_EXL3's default map: qualified by the " \
	"release gates' TP4_EXL3 identity checks (RELEASE_NOTES.md) and its golden metrics "                     \
	"(docs/results/GOLDEN_METRICS.md)"

struct text {
	char bytes[TEST_TEXT_BYTES];
	size_t used;
	bool is_short;
};

static const struct exl3_descriptor_model model = {
	.model_type = V41_DATA_CONFIG_MODEL_TYPE,
	.layers = MODEL_LAYERS,
	.experts_per_layer = MODEL_EXPERTS,
	.hidden = MODEL_HIDDEN,
	.intermediate = V41_DATA_EXPERT_INNER,
	.matrices = { { "w1", V41_DATA_EXPERT_INNER, MODEL_HIDDEN },
		      { "w2", MODEL_HIDDEN, V41_DATA_EXPERT_INNER },
		      { "w3", V41_DATA_EXPERT_INNER, MODEL_HIDDEN } },
	.routes_per_row = MODEL_EXPERTS_PER_ROW,
	.cone_first_layer = MODEL_DECODER_FIRST_LAYER
};
static const char *const kinds[EXL3_DESCRIPTOR_MATRIX_KINDS] = { "w1", "w2",
								 "w3" };

/* The first old_text of the descriptor written as the converter writes it becomes new_text: one rule broken */
struct refusal_case {
	const char *old_text, *new_text;
	result_status status;
	const char *line;
};
static const struct refusal_case refusals[] = {
	{ "\"format_version\": 1", "\"format_version\": 2", STATUS_UNSUPPORTED,
	  "format_version is 2: this build reads 1" },
	{ "\"format_version\": 1,", "", STATUS_INVALID_ARGUMENT,
	  "the descriptor has no format_version" },
	{ "{", "[", STATUS_INVALID_ARGUMENT, "not one JSON object" },
	{ "\"map\": {", "\"map\": {\"file\": \"twice\", ",
	  STATUS_INVALID_ARGUMENT, "an object holds a key twice" },
	{ "\"codebook\": {", "\"codebook\": {\"other\": 1, ",
	  STATUS_INVALID_ARGUMENT,
	  "codebook holds \"other\", which it does not define" },
	{ "\"native_pack_header_digest\"", "\"native_header_digest\"",
	  STATUS_INVALID_ARGUMENT, "map holds \"native_header_digest\"" },
	{ "\"quantiser\": {\"source\": \"test @ 0\", ", "\"quantiser\": {",
	  STATUS_INVALID_ARGUMENT, "quantiser has no source" },
	{ "\"source\": \"test @ 0\"", "\"source\": \" \\t\"",
	  STATUS_INVALID_ARGUMENT, "quantiser.source is \" \\t\", not a text" },
	{ "\"model_type\": \"deepseek_v41\"", "\"model_type\": \"deepseek_v4\"",
	  STATUS_UNSUPPORTED,
	  "architecture.model_type is \"deepseek_v4\", the model's \"deepseek_v41\"" },
	{ "\"model_type\": \"deepseek_v41\"", "\"model_type\": null",
	  STATUS_UNSUPPORTED, "architecture.model_type is null" },
	{ "\"0x83DCD12D\"", "\"0x83DCD12E\"", STATUS_UNSUPPORTED,
	  "codebook.mul1_multiplier is \"0x83DCD12E\": this build decodes \"0x83DCD12D\"" },
	{ "\"layers\": 40", "\"layers\": 41", STATUS_INVALID_ARGUMENT,
	  "experts.layers is 41, the model's 40" },
	{ "\"layers\": 40", "\"layers\": \"40\"", STATUS_INVALID_ARGUMENT,
	  "experts.layers is \"40\", not a whole number" },
	{ "\"experts_per_layer\": 384", "\"experts_per_layer\": 256",
	  STATUS_INVALID_ARGUMENT,
	  "experts.experts_per_layer is 256, the model's 384" },
	{ "\"hidden\": 5120", "\"hidden\": 4096", STATUS_INVALID_ARGUMENT,
	  "experts.hidden is 4096, the model's 5120" },
	{ "\"intermediate\": 2304", "\"intermediate\": 2048",
	  STATUS_INVALID_ARGUMENT,
	  "experts.intermediate is 2048, the model's 2304" },
	{ "\"w2\": [5120, 2304]", "\"w2\": [2304, 5120]",
	  STATUS_INVALID_ARGUMENT,
	  "experts.matrices.w2 is 2304, the model's 5120" },
	{ "\"w2\": [5120, 2304]", "\"w2\": [5120, 2304, 1]",
	  STATUS_INVALID_ARGUMENT,
	  "experts.matrices.w2 is not [outputs, inputs]" },
	{ "\"w2\": [5120, 2304]", "\"w2\": [5120]", STATUS_INVALID_ARGUMENT,
	  "experts.matrices.w2 is not [outputs, inputs]" },
	{ "\"0\": {\"w1\": {\"K3\": 384}", "\"0\": {\"w1\": {\"K3\": 383}",
	  STATUS_INVALID_ARGUMENT,
	  "widths.0.w1 gives 383 of K3, the pack holds 384" },
	{ "\"0\": {\"w1\": {\"K3\": 384}", "\"0\": {\"w1\": {\"K2\": 384}",
	  STATUS_INVALID_ARGUMENT,
	  "widths.0.w1 gives 384 of K2, the pack holds 0" },
	{ "\"0\": {\"w1\": {\"K3\": 384}", "\"0\": {\"w1\": {\"K9\": 384}",
	  STATUS_INVALID_ARGUMENT, "widths.0.w1: \"K9\" names no EXL3 width" },
	{ "\"0\": {\"w1\": {\"K3\": 384}", "\"0\": {\"w1\": {\"K3\": 0}",
	  STATUS_INVALID_ARGUMENT, "widths.0.w1.K3 is 0" },
	{ "\"0\": {", "\"00\": {", STATUS_INVALID_ARGUMENT,
	  "widths holds the layer \"00\": the model's are 0 to 39" },
	{ "\"39\": {", "\"40\": {", STATUS_INVALID_ARGUMENT,
	  "widths holds the layer \"40\": the model's are 0 to 39" },
	{ "\"w2\": {\"K3\"", "\"w4\": {\"K3\"", STATUS_INVALID_ARGUMENT,
	  "widths.0 holds \"w4\", which it does not define" },
	{ "\"matrices\": {\"K2\": 5760", "\"matrices\": {\"K2\": 5761",
	  STATUS_INVALID_ARGUMENT,
	  "widths_summary.matrices gives 5761 of K2, widths 5760" },
	{ "\"matrices\": {\"K2\"", "\"matrices\": {\"mixed\": 1, \"K2\"",
	  STATUS_INVALID_ARGUMENT,
	  "widths_summary.matrices gives 1 of mixed, widths 0" },
	{ "\"layers\": {\"K2\": 5", "\"layers\": {\"K2\": 5, \"mixed\": 1",
	  STATUS_INVALID_ARGUMENT,
	  "widths_summary.layers gives 1 of mixed, widths 0" },
	{ "\"codebook\": {\"mul1_multiplier\": \"0x83DCD12D\"}",
	  "\"codebook\": \"0x83DCD12D\"", STATUS_INVALID_ARGUMENT,
	  "codebook is \"0x83DCD12D\", not an object" },
	{ "\"format_version\": 1,", "\"format_version\": 1, \"extra\": 1,",
	  STATUS_INVALID_ARGUMENT,
	  "the descriptor holds \"extra\", which it does not define" },
	{ "{\"w1\": {\"K3\": 384}", "{\"w1\": 384", STATUS_INVALID_ARGUMENT,
	  "widths.0.w1 is 384, not an object" },
	{ "\"w2\": [5120, 2304]", "\"w2\": 5120", STATUS_INVALID_ARGUMENT,
	  "experts.matrices.w2 is not [outputs, inputs]" },
	{ "\"status\": \"established\"", "\"status\": \"measured\"",
	  STATUS_UNSUPPORTED,
	  "quality.status is \"measured\": this build serves none, established and recorded" },
	{ "\"quality\": {\"status\": \"established\", ", "\"quality\": {",
	  STATUS_INVALID_ARGUMENT, "quality has no status" },
	{ "\"reference\": \"TP4_EXL3",
	  "\"reference\": \"\", \"was\": \"TP4_EXL3", STATUS_INVALID_ARGUMENT,
	  "quality holds \"was\", which it does not define" },
};

static char folder[TEST_PATH_BYTES], map[TEST_PATH_BYTES],
	sidecar[TEST_PATH_BYTES], linked_map[TEST_PATH_BYTES],
	capture[TEST_PATH_BYTES];
static unsigned cases_run, failures;

static void put(struct text *text, const char *format, ...)
	__attribute__((format(printf, PUT_FORMAT_PARAMETER,
			      PUT_VALUES_PARAMETER)));
static void put(struct text *text, const char *format, ...)
{
	va_list values;
	va_start(values, format);
	const int written = vsnprintf(text->bytes + text->used,
				      sizeof(text->bytes) - text->used, format,
				      values);
	va_end(values);
	if (written < 0 || (size_t)written >= sizeof(text->bytes) - text->used)
		text->is_short = true;
	else
		text->used += (size_t)written;
}

static void digest_text(const void *bytes, size_t count,
			char text[DIGEST_DIGITS + 1u])
{
	static const char digits[] = "0123456789abcdef";
	unsigned char digest[SHA256_DIGEST_BYTES] = { 0 };
	if (sha256_of(bytes, count, digest) != STATUS_OK)
		failures++;
	for (uint32_t index = 0; index < SHA256_DIGEST_BYTES; index++) {
		text[(size_t)HEX_PER_BYTE * index] =
			digits[digest[index] >> NIBBLE_BITS];
		text[(size_t)HEX_PER_BYTE * index + 1u] =
			digits[digest[index] & NIBBLE_MASK];
	}
	text[DIGEST_DIGITS] = '\0';
}

static const char *layer_width(uint32_t layer)
{
	return (K2_LAYERS >> layer & 1u) != 0 ? "K2" : "K3";
}

/* The descriptor of the model's EXL3 pack as the converter writes it, its members in the converter's order: its first
 * layers (all but for one case), and mixed_layer's w2 with one K2 matrix (none at NO_MIXED_LAYER); widths, when not
 * NULL, in place of the widths object */
static void descriptor_write_widths(struct text *text,
				    const struct pack_header *header,
				    uint32_t layers, uint32_t mixed_layer,
				    const char *tables_digest,
				    const char *quality, const char *widths)
{
	uint64_t k2_matrices = 0, k2_layers = 0, k3_layers = 0;
	text->used = 0;
	text->is_short = false;
	put(text,
	    "{\n \"format_version\": 1,\n \"map\": {\"file\": \"map.exl3map\", \"sha256\": \"%s\", "
	    "\"manifest_bytes\": %" PRIu64 ", \"logical_bytes\": %" PRIu64
	    ", \"pack_header_digest\": %" PRIu64
	    ", \"native_pack_header_digest\": %" PRIu64 "},\n",
	    SOME_DIGEST, header->tables_byte_count, header->file_byte_count,
	    header->header_digest, header->tables_digest);
	put(text,
	    " \"architecture\": {\"model_type\": \"deepseek_v41\", \"architectures\": [\"DeepseekV41ForCausalLM\"], "
	    "\"config_sha256\": \"%s\"},\n \"quantiser\": {\"source\": \"test @ 0\", \"index_sha256\": \"%s\", "
	    "\"quantization_config_sha256\": \"%s\"},\n \"codebook\": {\"mul1_multiplier\": \"0x83DCD12D\"},\n"
	    " \"front_end_tables_sha256\": %s,\n",
	    SOME_DIGEST, SOME_DIGEST, SOME_DIGEST, tables_digest);
	put(text,
	    " \"experts\": {\"layers\": %" PRIu32
	    ", \"experts_per_layer\": %u, \"hidden\": %u, \"intermediate\": %u, "
	    "\"matrices\": {\"w1\": [%u, %u], \"w2\": [%u, %u], \"w3\": [%u, %u]}},\n \"widths\": %s",
	    layers, MODEL_EXPERTS, MODEL_HIDDEN, V41_DATA_EXPERT_INNER,
	    V41_DATA_EXPERT_INNER, MODEL_HIDDEN, MODEL_HIDDEN,
	    V41_DATA_EXPERT_INNER, V41_DATA_EXPERT_INNER, MODEL_HIDDEN,
	    widths != NULL ? widths : "{");
	for (uint32_t layer = 0; layer < layers; layer++) {
		if (widths == NULL)
			put(text, "%s\"%" PRIu32 "\": {",
			    layer != 0 ? ", " : "", layer);
		for (uint32_t kind = 0;
		     widths == NULL && kind < EXL3_DESCRIPTOR_MATRIX_KINDS;
		     kind++)
			if (layer == mixed_layer && kind == 1u)
				put(text, ", \"%s\": {\"%s\": %u, \"K2\": 1}",
				    kinds[kind], layer_width(layer),
				    MODEL_EXPERTS - 1u);
			else
				put(text, "%s\"%s\": {\"%s\": %u}",
				    kind != 0 ? ", " : "", kinds[kind],
				    layer_width(layer), MODEL_EXPERTS);
		if (widths == NULL)
			put(text, "}");
		const bool is_k2 = (K2_LAYERS >> layer & 1u) != 0;
		k2_matrices += is_k2 ? (uint64_t)MODEL_EXPERTS *
					       EXL3_DESCRIPTOR_MATRIX_KINDS :
				       0u;
		k2_layers += is_k2 ? 1u : 0u;
		k3_layers += !is_k2 && layer != mixed_layer ? 1u : 0u;
	}
	const uint64_t mixed = mixed_layer < layers ? 1u : 0u;
	const uint64_t k3_matrices = (uint64_t)layers * MODEL_EXPERTS *
					     EXL3_DESCRIPTOR_MATRIX_KINDS -
				     k2_matrices - mixed;
	put(text,
	    "%s,\n \"widths_summary\": {\"matrices\": {\"K2\": %" PRIu64
	    ", \"K3\": %" PRIu64 "}, \"layers\": {\"K2\": %" PRIu64
	    ", \"K3\": %" PRIu64 "%s}},\n \"quality\": %s\n}\n",
	    widths != NULL ? "" : "}", k2_matrices + mixed, k3_matrices,
	    k2_layers, k3_layers, mixed != 0 ? ", \"mixed\": 1" : "", quality);
}

static void descriptor_write(struct text *text,
			     const struct pack_header *header, uint32_t layers,
			     uint32_t mixed_layer, const char *tables_digest,
			     const char *quality)
{
	descriptor_write_widths(text, header, layers, mixed_layer,
				tables_digest, quality, NULL);
}

static bool file_put(const char *path, const void *bytes, size_t count)
{
	(void)unlink(path);
	FILE *const file = fopen(path, "wb");
	if (file == NULL)
		return false;
	const bool wrote = fwrite(bytes, 1u, count, file) == count;
	return fclose(file) == 0 && wrote;
}

/* The first old_text in text becomes new_text; false when it is not there or does not fit */
static bool replaced(struct text *text, const char *old_text,
		     const char *new_text)
{
	static char rewritten[TEST_TEXT_BYTES];
	const char *const found = strstr(text->bytes, old_text);
	if (found == NULL)
		return false;
	const int written = snprintf(rewritten, sizeof(rewritten), "%.*s%s%s",
				     (int)(found - text->bytes), text->bytes,
				     new_text, found + strlen(old_text));
	if (written < 0 || (size_t)written >= sizeof(rewritten))
		return false;
	(void)snprintf(text->bytes, sizeof(text->bytes), "%s", rewritten);
	text->used = (size_t)written;
	return true;
}

/* stderr into a file while a check runs, then its lines in log */
struct stderr_capture {
	FILE *diagnostic;
	int saved;
};
static bool capture_start(struct stderr_capture *redirect,
			  char log[TEST_LOG_BYTES])
{
	redirect->diagnostic = tmpfile();
	(void)fflush(stderr);
	redirect->saved = dup(STDERR_FILENO);
	log[0] = '\0';
	if (redirect->diagnostic != NULL && redirect->saved >= 0 &&
	    dup2(fileno(redirect->diagnostic), STDERR_FILENO) == STDERR_FILENO)
		return true;
	failures++;
	if (redirect->diagnostic != NULL)
		(void)fclose(redirect->diagnostic);
	if (redirect->saved >= 0)
		(void)close(redirect->saved);
	return false;
}
static void capture_end(struct stderr_capture *redirect,
			char log[TEST_LOG_BYTES])
{
	(void)fflush(stderr);
	if (dup2(redirect->saved, STDERR_FILENO) != STDERR_FILENO)
		failures++;
	(void)close(redirect->saved);
	if (fseek(redirect->diagnostic, 0L, SEEK_SET) != 0)
		failures++;
	clearerr(redirect->diagnostic);
	log[fread(log, 1u, TEST_LOG_BYTES - 1u, redirect->diagnostic)] = '\0';
	(void)fclose(redirect->diagnostic);
}

/* One check of map's descriptor, its stderr in log */
MUST_CHECK static result_status
checked(const char *map_path, const struct pack_record *pack,
	const struct exl3_descriptor_model *checked_model, bool with_tables,
	bool admits_unrated, char log[TEST_LOG_BYTES])
{
	struct stderr_capture redirect;
	if (!capture_start(&redirect, log))
		return STATUS_INPUT_OUTPUT_ERROR;
	const result_status status = exl3_descriptor_check(
		map_path, pack, checked_model, with_tables ? TABLES : NULL,
		with_tables ? strlen(TABLES) : 0u, admits_unrated);
	capture_end(&redirect, log);
	return status;
}

/* The descriptor in text beside the map (none written for NULL), checked: the status and a line in the log */
static void expect(const char *label, const struct text *text,
		   const char *map_path, const struct pack_record *pack,
		   const struct exl3_descriptor_model *checked_model,
		   bool with_tables, bool admits_unrated, result_status status,
		   const char *line)
{
	char log[TEST_LOG_BYTES];
	cases_run++;
	const bool written =
		text == NULL ||
		(!text->is_short && file_put(sidecar, text->bytes, text->used));
	const result_status found =
		written ? checked(map_path, pack, checked_model, with_tables,
				  admits_unrated, log) :
			  STATUS_INPUT_OUTPUT_ERROR;
	if (!written || found != status || strstr(log, line) == NULL) {
		(void)fprintf(
			stderr,
			"EXL3 descriptor failure (%s): status %d, expected %d; wanted \"%s\" in: %s\n",
			label, (int)found, (int)status, line, log);
		failures++;
	}
}

static void refusal_cases(const struct pack_record *pack)
{
	const struct pack_header *const header = pack_header_of(pack);
	char tables[DIGEST_DIGITS + 1u];
	char tables_digest[sizeof(tables) + QUOTE_BYTES];
	char old_text[TEST_PATH_BYTES], new_text[TEST_PATH_BYTES];
	struct text *const text = malloc(sizeof(*text));
	if (text == NULL) {
		failures++;
		return;
	}
	digest_text(TABLES, strlen(TABLES), tables);
	(void)snprintf(tables_digest, sizeof(tables_digest), "\"%s\"", tables);
	for (size_t index = 0; index < sizeof(refusals) / sizeof(refusals[0]);
	     index++) {
		descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER,
				 tables_digest, ESTABLISHED);
		if (!replaced(text, refusals[index].old_text,
			      refusals[index].new_text)) {
			(void)fprintf(
				stderr,
				"EXL3 descriptor failure: case %zu's text is not in the descriptor\n",
				index);
			failures++;
			continue;
		}
		expect(refusals[index].line, text, map, pack, &model, true,
		       false, refusals[index].status, refusals[index].line);
	}
	/* the binding to the open pack: its size and header digest */
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER,
			 tables_digest, ESTABLISHED);
	(void)snprintf(old_text, sizeof(old_text),
		       "\"logical_bytes\": %" PRIu64, header->file_byte_count);
	(void)snprintf(new_text, sizeof(new_text),
		       "\"logical_bytes\": %" PRIu64,
		       header->file_byte_count + PACK_BLOCK_BYTES);
	if (!replaced(text, old_text, new_text))
		failures++;
	expect("another size", text, map, pack, &model, true, false,
	       STATUS_INVALID_ARGUMENT, "map.logical_bytes is");
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER,
			 tables_digest, ESTABLISHED);
	(void)snprintf(old_text, sizeof(old_text),
		       "\"pack_header_digest\": %" PRIu64,
		       header->header_digest);
	(void)snprintf(new_text, sizeof(new_text),
		       "\"pack_header_digest\": %" PRIu64,
		       header->header_digest ^ 1u);
	if (!replaced(text, old_text, new_text))
		failures++;
	expect("another pack", text, map, pack, &model, true, false,
	       STATUS_INVALID_ARGUMENT, "map.pack_header_digest is");
	/* the tables: another digest is refused where the tables were read, and not checked where they were not */
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER,
			 "\"" SOME_DIGEST "\"", ESTABLISHED);
	expect("other tables", text, map, pack, &model, true, false,
	       STATUS_INVALID_ARGUMENT,
	       "front_end_tables_sha256 is \"" SOME_DIGEST "\", the tables'");
	expect("other tables, not read", text, map, pack, &model, false, false,
	       STATUS_OK, "quality established");
	/* a layer short */
	descriptor_write(text, header, MODEL_LAYERS - 1u, NO_MIXED_LAYER,
			 tables_digest, ESTABLISHED);
	(void)snprintf(old_text, sizeof(old_text), "\"layers\": %u",
		       MODEL_LAYERS - 1u);
	(void)snprintf(new_text, sizeof(new_text), "\"layers\": %u",
		       MODEL_LAYERS);
	if (!replaced(text, old_text, new_text))
		failures++;
	expect("a layer short", text, map, pack, &model, true, false,
	       STATUS_INVALID_ARGUMENT, "widths gives 39 layers, the model 40");
	/* the model of one layer less: the pack's last layer holds EXL3 matrices no routed expert of it has */
	struct exl3_descriptor_model shorter = model;
	shorter.layers = MODEL_LAYERS - 1u;
	descriptor_write(text, header, MODEL_LAYERS - 1u, NO_MIXED_LAYER,
			 tables_digest, ESTABLISHED);
	expect("EXL3 matrices beyond the model's", text, map, pack, &shorter,
	       true, false, STATUS_INVALID_ARGUMENT,
	       "the pack holds 46080 EXL3 matrices, the model's routed experts 44928");
	descriptor_write_widths(text, header, MODEL_LAYERS, NO_MIXED_LAYER,
				tables_digest, ESTABLISHED, "[]");
	expect("widths not an object", text, map, pack, &model, true, false,
	       STATUS_INVALID_ARGUMENT, "widths is not an object");
	free(text);
}

static void quality_cases(const struct pack_record *pack)
{
	const struct pack_header *const header = pack_header_of(pack);
	struct text *const text = malloc(sizeof(*text));
	char quality[TEST_QUALITY_BYTES], digest[DIGEST_DIGITS + 1u];
	static const char capture_bytes[] =
		"{\"nll\": [], \"scope\": \"test\"}\n";
	if (text == NULL ||
	    !file_put(capture, capture_bytes, strlen(capture_bytes))) {
		failures++;
		free(text);
		return;
	}
	digest_text(capture_bytes, strlen(capture_bytes), digest);
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
			 ESTABLISHED);
	expect("established", text, map, pack, &model, true, false, STATUS_OK,
	       "widths K2 5760, K3 40320, quality established (TP4_EXL3's default route");
	expect("established, through a link", NULL, linked_map, pack, &model,
	       true, false, STATUS_OK, "quality established");
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
			 "{\"status\": \"none\"}");
	expect("none", text, map, pack, &model, true, false, STATUS_UNSUPPORTED,
	       "quality none: no NLL/PPL capture or qualification covers the map");
	expect("none, opted in", text, map, pack, &model, true, true, STATUS_OK,
	       "quality none (" EXL3_DESCRIPTOR_OPT_IN ": no quality claim)");
	expect("none, opted in, the whole line's end", text, map, pack, &model,
	       false, true, STATUS_OK,
	       ": deepseek_v41, test @ 0, widths K2 5760, K3 40320, quality none (" EXL3_DESCRIPTOR_OPT_IN
	       ": no quality claim)\n");
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
			 "{\"status\": \"none\", \"reference\": \"x\"}");
	expect("none with a reference", text, map, pack, &model, true, true,
	       STATUS_INVALID_ARGUMENT,
	       "quality holds \"reference\", which it does not define");
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
			 "{\"status\": \"established\", \"reference\": \"\"}");
	expect("established without a reference", text, map, pack, &model, true,
	       false, STATUS_INVALID_ARGUMENT,
	       "quality.reference is \"\", not a text");
	descriptor_write(
		text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
		"{\"status\": \"established\", \"reference\": \" \\n \"}");
	expect("established by a reference of spaces", text, map, pack, &model,
	       true, false, STATUS_INVALID_ARGUMENT,
	       "quality.reference is \" \\n \", not a text");
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
			 "\"none\"");
	expect("quality not an object", text, map, pack, &model, true, true,
	       STATUS_INVALID_ARGUMENT, "quality is \"none\", not an object");
	(void)snprintf(
		quality, sizeof(quality),
		"{\"status\": \"recorded\", \"record\": \"%s\", \"sha256\": \"%s\"}",
		capture, digest);
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
			 quality);
	expect("recorded", text, map, pack, &model, true, false, STATUS_OK,
	       "quality recorded (");
	(void)snprintf(
		quality, sizeof(quality),
		"{\"status\": \"recorded\", \"record\": \"%s\", \"sha256\": \"" SOME_DIGEST
		"\"}",
		capture);
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
			 quality);
	expect("recorded, another capture", text, map, pack, &model, true,
	       false, STATUS_INVALID_ARGUMENT,
	       "quality.sha256 is \"" SOME_DIGEST "\", the record's");
	(void)snprintf(
		quality, sizeof(quality),
		"{\"status\": \"recorded\", \"record\": \"%s.missing\", \"sha256\": \"%s\"}",
		capture, digest);
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
			 quality);
	expect("recorded, no capture", text, map, pack, &model, true, false,
	       STATUS_INPUT_OUTPUT_ERROR,
	       ".missing cannot be opened: No such file or directory");
	(void)snprintf(
		quality, sizeof(quality),
		"{\"status\": \"recorded\", \"record\": \"%s\", \"sha256\": \"xyz\"}",
		capture);
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
			 quality);
	expect("recorded, no SHA-256", text, map, pack, &model, true, false,
	       STATUS_INVALID_ARGUMENT,
	       "quality.sha256 is \"xyz\", the record's");
	(void)snprintf(quality, sizeof(quality),
		       "{\"status\": \"recorded\", \"record\": \"%s\"}",
		       capture);
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
			 quality);
	expect("recorded without its sha256", text, map, pack, &model, true,
	       false, STATUS_INVALID_ARGUMENT, "quality has no sha256");
	(void)snprintf(
		quality, sizeof(quality),
		"{\"status\": \"recorded\", \"record\": \"capture.json\", \"sha256\": \"%s\"}",
		digest);
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
			 quality);
	expect("recorded, a relative path", text, map, pack, &model, true,
	       false, STATUS_INVALID_ARGUMENT,
	       "quality.record is \"capture.json\", not an absolute path");
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
			 ESTABLISHED);
	if (!replaced(text, "[\"DeepseekV41ForCausalLM\"]", "null"))
		failures++;
	expect("no architectures", text, map, pack, &model, false, false,
	       STATUS_OK, "quality established");
	(void)unlink(capture);
	free(text);
}

/* What lies at the sidecar's place: a link to nothing, a folder, an empty file, one too large */
static void file_cases(const struct pack_record *pack)
{
	char *const large = calloc(EXL3_DESCRIPTOR_MOST_BYTES + 1u, 1u);
	char missing[TEST_PATH_BYTES + sizeof("/missing")];
	char *const long_map = malloc(PATH_MAX);
	if (long_map !=
	    NULL) { /* a map path whose sidecar's would not fit PATH_MAX */
		memset(long_map, 'a', PATH_MAX);
		long_map[0] = '/';
		long_map[PATH_MAX - strlen(EXL3_DESCRIPTOR_SUFFIX)] = '\0';
		expect("a map path too long for its sidecar", NULL, long_map,
		       pack, &model, true, false, STATUS_INVALID_ARGUMENT,
		       "is too long for its sidecar");
	} else
		failures++;
	free(long_map);
	(void)unlink(sidecar);
	(void)snprintf(missing, sizeof(missing), "%s/missing", folder);
	if (symlink(missing, sidecar) != 0)
		failures++;
	expect("a link to nothing", NULL, map, pack, &model, true, false,
	       STATUS_INPUT_OUTPUT_ERROR,
	       "the file cannot be opened: No such file or directory");
	(void)unlink(sidecar);
	if (mkdir(sidecar, S_IRWXU) != 0)
		failures++;
	expect("a folder", NULL, map, pack, &model, true, false,
	       STATUS_INVALID_ARGUMENT,
	       "the file is not a regular file of at most 1048576 bytes");
	(void)rmdir(sidecar);
	if (!file_put(sidecar, "", 0u))
		failures++;
	expect("empty", NULL, map, pack, &model, true, false,
	       STATUS_INVALID_ARGUMENT, "not one JSON object");
	if (large == NULL ||
	    !file_put(sidecar, large, EXL3_DESCRIPTOR_MOST_BYTES + 1u))
		failures++;
	expect("too large", NULL, map, pack, &model, true, false,
	       STATUS_INVALID_ARGUMENT,
	       "the file is not a regular file of at most 1048576 bytes");
	if (large == NULL || !file_put(sidecar, large, PACK_BLOCK_BYTES))
		failures++;
	expect("not JSON", NULL, map, pack, &model, true, false,
	       STATUS_INVALID_ARGUMENT, "not one JSON object");
	(void)unlink(sidecar);
	free(large);
}

/* No sidecar at a map path: listed served, not listed refused, not listed served on the opt-in */
static void absent_at(const char *path, const struct pack_record *pack,
		      const struct exl3_descriptor_model *listed)
{
	expect("absent, listed", NULL, path, pack, listed, true, false,
	       STATUS_OK,
	       ".descriptor.json: the established pack test (header digest");
	expect("absent, not listed", NULL, path, pack, &model, true, false,
	       STATUS_UNSUPPORTED,
	       ".descriptor.json: none here, and the pack (header digest");
	expect("absent, not listed, opted in", NULL, path, pack, &model, true,
	       true, STATUS_OK, "not list as established (header digest");
}

/* No descriptor beside the map: what the open pack carries decides, whatever the path that opened it names: the map,
 * a flat replica's <digest>.anemone, a source no longer there (the storage profile's fallback to its replica), a map
 * whose name leaves no room for its sidecar's (over NAME_MAX - EXL3_DESCRIPTOR_SUFFIX: nothing can be there). A listed
 * pack is served by default, by its header digest and size together, any other only on the opt-in; a descriptor that
 * is there stays authoritative either way. */
static void absent_cases(const struct pack_record *pack)
{
	const struct pack_header *const header = pack_header_of(pack);
	const struct exl3_established_pack packs[] = {
		{ "test", header->header_digest, header->file_byte_count },
		{ "another size", header->header_digest,
		  header->file_byte_count + PACK_BLOCK_BYTES },
		{ "another digest", header->header_digest ^ 1u,
		  header->file_byte_count }
	};
	struct exl3_descriptor_model listed = model, mismatched = model;
	listed.established = packs;
	listed.established_count = 1u;
	mismatched.established = &packs[1];
	mismatched.established_count = TEST_MISMATCHED_PACKS;
	char replica[TEST_PATH_BYTES], gone[TEST_PATH_BYTES],
		edge[TEST_EDGE_NAMES][TEST_PATH_BYTES];
	const size_t edge_bytes[TEST_EDGE_NAMES] = {
		NAME_MAX - strlen(EXL3_DESCRIPTOR_SUFFIX),
		NAME_MAX - strlen(EXL3_DESCRIPTOR_SUFFIX) + 1u, NAME_MAX
	};
	bool made = true;
	for (uint32_t index = 0; index < TEST_EDGE_NAMES; index++) {
		char name[NAME_MAX + 1u];
		const size_t stem = edge_bytes[index] - strlen(TEST_MAP_SUFFIX);
		memset(name, 'm', stem);
		memcpy(name + stem, TEST_MAP_SUFFIX, sizeof(TEST_MAP_SUFFIX));
		made = made &&
		       snprintf(edge[index], sizeof(edge[index]), "%s/%s",
				folder, name) > 0 &&
		       file_put(edge[index], "x", 1u);
	}
	struct text *const text = malloc(sizeof(*text));
	if (text == NULL || !made ||
	    snprintf(replica, sizeof(replica), "%s/%" PRIu64 ".anemone", folder,
		     header->header_digest) <= 0 ||
	    snprintf(gone, sizeof(gone), "%s/gone/map.exl3map", folder) <= 0 ||
	    !file_put(replica, "x", 1u)) {
		failures++;
		free(text);
		return;
	}
	const char *const paths[] = { map, replica, gone };
	(void)unlink(sidecar);
	for (size_t index = 0; index < sizeof(paths) / sizeof(paths[0]);
	     index++)
		absent_at(paths[index], pack, &listed);
	for (uint32_t index = 0; index < TEST_EDGE_NAMES; index++)
		absent_at(edge[index], pack, &listed);
	expect("absent, listed by its digest or its size alone", NULL, map,
	       pack, &mismatched, true, false, STATUS_UNSUPPORTED,
	       "is not listed as established");
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
			 "{\"status\": \"none\"}");
	expect("none beside a listed pack", text, map, pack, &listed, true,
	       false, STATUS_UNSUPPORTED, "quality none: no NLL/PPL capture");
	descriptor_write(text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
			 ESTABLISHED);
	expect("established beside a pack not listed", text, map, pack, &model,
	       true, false, STATUS_OK, "quality established");
	/* the descriptor tools/model_data.sh writes, beside a pack the model lists: the sidecar is read and its own
	 * quality and reference go on the line, not the list's name */
	descriptor_write(
		text, header, MODEL_LAYERS, NO_MIXED_LAYER, "null",
		"{\"status\": \"established\", \"reference\": \"" MODEL_DATA_REFERENCE
		"\"}");
	expect("established beside a listed pack", text, map, pack, &listed,
	       true, false, STATUS_OK,
	       "quality established (" MODEL_DATA_REFERENCE ")\n");
	(void)unlink(sidecar);
	(void)unlink(replica);
	for (uint32_t index = 0; index < TEST_EDGE_NAMES; index++)
		(void)unlink(edge[index]);
	free(text);
}

/* A fault point armed at its call number, against an established or a recorded descriptor */
struct fault_case {
	const char *name;
	uint64_t call;
	bool recorded;
	result_status status;
	const char *line;
};

/* The outside world failing, through the fault points of the checking builds: the sidecar's status, memory and read,
 * the scratch memory, and a recorded capture's status, memory and read */
static void fault_cases(const struct pack_record *pack)
{
#if CHECKING_HOOKS_ENABLED
	static const struct fault_case faults[] = {
		{ "exl3_descriptor_stat", 1, false, STATUS_INPUT_OUTPUT_ERROR,
		  "the file cannot be opened: its status fails" },
		{ "exl3_descriptor_memory", 1, false, STATUS_OUT_OF_MEMORY,
		  "the file cannot be held in memory" },
		{ "exl3_descriptor_read", 1, false, STATUS_INPUT_OUTPUT_ERROR,
		  "the file cannot be read whole" },
		{ "exl3_descriptor_memory", 2, false, STATUS_OUT_OF_MEMORY,
		  "has no memory to be checked in" },
		{ "exl3_descriptor_stat", 2, true, STATUS_INPUT_OUTPUT_ERROR,
		  "capture.json cannot be opened: its status fails" },
		{ "exl3_descriptor_memory", 3, true, STATUS_OUT_OF_MEMORY,
		  "capture.json cannot be held in memory" },
		{ "exl3_descriptor_read", 2, true, STATUS_INPUT_OUTPUT_ERROR,
		  "capture.json cannot be read whole" },
	};
	static const char capture_bytes[] = "{\"nll\": []}\n";
	struct text *const text = malloc(sizeof(*text));
	char quality[TEST_QUALITY_BYTES], digest[DIGEST_DIGITS + 1u];
	if (text == NULL ||
	    !file_put(capture, capture_bytes, strlen(capture_bytes))) {
		failures++;
		free(text);
		return;
	}
	digest_text(capture_bytes, strlen(capture_bytes), digest);
	(void)snprintf(
		quality, sizeof(quality),
		"{\"status\": \"recorded\", \"record\": \"%s\", \"sha256\": \"%s\"}",
		capture, digest);
	for (size_t index = 0; index < sizeof(faults) / sizeof(faults[0]);
	     index++) {
		descriptor_write(text, pack_header_of(pack), MODEL_LAYERS,
				 NO_MIXED_LAYER, "null",
				 faults[index].recorded ? quality :
							  ESTABLISHED);
		if (!fault_arm(faults[index].name, faults[index].call))
			failures++;
		expect(faults[index].line, text, map, pack, &model, true, false,
		       faults[index].status, faults[index].line);
		if (fault_disarm() < faults[index].call) {
			(void)fprintf(
				stderr,
				"EXL3 descriptor failure: fault %s %" PRIu64
				" was not reached\n",
				faults[index].name, faults[index].call);
			failures++;
		}
	}
	(void)unlink(capture);
	free(text);
#else
	(void)pack;
#endif
}

/* The synthetic mixed pack: layer 0's first w2 matrix K2 in a K3 layer. Its own descriptor holds, with a mixed layer;
 * the uniform one is refused at that layer and kind. */
static void mixed_cases(void)
{
	struct pack_record *pack = NULL;
	struct text *const text = malloc(sizeof(*text));
	const struct test_model_v41_change change = {
		TEST_MODEL_V41_CHANGE_EXL3, "layers.0.ffn.experts.0.w2.weight",
		EXL3_K2
	};
	if (text == NULL ||
	    test_model_v41_pack_open(change, &pack) != STATUS_OK) {
		failures++;
		free(text);
		return;
	}
	descriptor_write(text, pack_header_of(pack), MODEL_LAYERS, MIXED_LAYER,
			 "null", ESTABLISHED);
	expect("the mixed pack", text, map, pack, &model, true, false,
	       STATUS_OK, "widths K2 5761, K3 40319");
	descriptor_write(text, pack_header_of(pack), MODEL_LAYERS,
			 NO_MIXED_LAYER, "null", ESTABLISHED);
	expect("the mixed pack, uniformly described", text, map, pack, &model,
	       true, false, STATUS_INVALID_ARGUMENT,
	       "widths.0.w2 gives 0 of K2, the pack holds 1");
	pack_close(pack);
	free(text);
}

/* A routed matrix that is no EXL3 matrix: the pack of the original weights */
static void original_case(void)
{
	struct pack_record *pack = NULL;
	struct text *const text = malloc(sizeof(*text));
	const struct test_model_v41_change change = {
		TEST_MODEL_V41_CHANGE_NONE, NULL, 0
	};
	if (text == NULL ||
	    test_model_v41_pack_open(change, &pack) != STATUS_OK) {
		failures++;
		free(text);
		return;
	}
	descriptor_write(text, pack_header_of(pack), MODEL_LAYERS,
			 NO_MIXED_LAYER, "null", ESTABLISHED);
	expect("an original pack", text, map, pack, &model, true, false,
	       STATUS_INVALID_ARGUMENT,
	       "the pack holds no EXL3 matrix layers.0.ffn.experts.0.w1.weight of (2304, 5120)");
	pack_close(pack);
	free(text);
}

static bool fixture(void)
{
	const char *const base = getenv("TMPDIR") != NULL ? getenv("TMPDIR") :
							    "/tmp";
	const int written = snprintf(folder, sizeof(folder),
				     "%s/exl3_descriptor_test.XXXXXX", base);
	return written > 0 && (size_t)written < sizeof(folder) &&
	       mkdtemp(folder) != NULL &&
	       snprintf(map, sizeof(map), "%s/map.exl3map", folder) > 0 &&
	       snprintf(sidecar, sizeof(sidecar), "%s" EXL3_DESCRIPTOR_SUFFIX,
			map) > 0 &&
	       snprintf(linked_map, sizeof(linked_map), "%s/linked.exl3map",
			folder) > 0 &&
	       snprintf(capture, sizeof(capture), "%s/capture.json", folder) >
		       0 &&
	       file_put(map, "x", 1u) && symlink(map, linked_map) == 0;
}

/* The model against the tensor-core MoE's kernels at TP4 (rotation: --exl3_expert_rotation's): the status, and its
 * line, or no line at all for a pass */
static void tc_expect(const char *label,
		      const struct exl3_descriptor_model *checked_model,
		      uint32_t ranks, uint32_t rotation, result_status status,
		      const char *line)
{
	char log[TEST_LOG_BYTES];
	struct stderr_capture redirect;
	cases_run++;
	result_status found = STATUS_INPUT_OUTPUT_ERROR;
	if (capture_start(&redirect, log)) {
		found = exl3_descriptor_tc_check(
			V41_DATA_MODEL_NAME, checked_model, ranks, rotation);
		capture_end(&redirect, log);
	}
	if (found != status ||
	    (line != NULL ? strstr(log, line) == NULL : log[0] != '\0')) {
		(void)fprintf(
			stderr,
			"EXL3 descriptor failure (%s): status %d, expected %d; wanted \"%s\" in: %s\n",
			label, (int)found, (int)status,
			line != NULL ? line : "no line", log);
		failures++;
	}
}

static void tc_cases(void)
{
	tc_expect("the tensor-core MoE, the model's", &model, TP4_RANKS, 0,
		  STATUS_OK, NULL);
	tc_expect("the tensor-core MoE, the model's, four-way rotation", &model,
		  TP4_RANKS, EXL3_TC_ROTATION_FOUR_WAY, STATUS_OK, NULL);
	tc_expect("the tensor-core MoE, the model's, three-way rotation",
		  &model, TP4_RANKS, EXL3_TC_ROTATION_THREE_WAY, STATUS_OK,
		  NULL);
	struct exl3_descriptor_model other = model;
	other.hidden = TC_OTHER_HIDDEN;
	tc_expect(
		"the tensor-core MoE, another hidden", &other, TP4_RANKS, 0,
		STATUS_UNSUPPORTED,
		TC_LINE
		"experts.hidden is 4096, the tensor-core MoE's kernels 5120 (exl3_tc.h)\n");
	other = model;
	other.routes_per_row = TC_OTHER_ROUTES;
	tc_expect(
		"the tensor-core MoE, other routes", &other, TP4_RANKS, 0,
		STATUS_UNSUPPORTED,
		TC_LINE
		"routes_per_row is 8, the tensor-core MoE's kernels 6 (exl3_tc.h)\n");
	other = model;
	other.experts_per_layer = TC_OTHER_EXPERTS;
	tc_expect(
		"the tensor-core MoE, other experts", &other, TP4_RANKS, 0,
		STATUS_UNSUPPORTED,
		TC_LINE
		"experts.experts_per_layer is 256, the tensor-core MoE's kernels 384 (exl3_tc.h)\n");
	other = model;
	other.layers = TC_OTHER_LAYERS;
	tc_expect(
		"the tensor-core MoE, other layers", &other, TP4_RANKS, 0,
		STATUS_UNSUPPORTED,
		TC_LINE
		"experts.layers is 61, the tensor-core MoE's kernels 40 (exl3_tc.h)\n");
	other = model;
	other.cone_first_layer = TC_OTHER_CONE_FIRST;
	tc_expect(
		"the tensor-core MoE, another cone", &other, TP4_RANKS, 0,
		STATUS_UNSUPPORTED,
		TC_LINE
		"cone_first_layer is 3, the tensor-core MoE's kernels 20 (exl3_tc.h)\n");
	other = model;
	other.intermediate = TC_NARROW_CHUNKS;
	tc_expect(
		"the tensor-core MoE, narrower chunks", &other, TP4_RANKS, 0,
		STATUS_UNSUPPORTED,
		TC_LINE
		"experts.intermediate 1536 at 4 ranks gives rank 0 384 columns, the tensor-core MoE's "
		"kernels 640 or 512 (exl3_tc.h)\n");
	other.intermediate = TC_OFF_GRID;
	tc_expect(
		"the tensor-core MoE, no split", &other, TP4_RANKS, 0,
		STATUS_UNSUPPORTED,
		TC_LINE
		"experts.intermediate 2300 has no split at 4 ranks (exl3_layout.h)\n");
	other.intermediate = TC_WIDE_CHUNKS;
	tc_expect("the tensor-core MoE, every chunk wide", &other, TP4_RANKS, 0,
		  STATUS_OK, NULL);
	tc_expect(
		"the tensor-core MoE, every chunk wide, rotated", &other,
		TP4_RANKS, EXL3_TC_ROTATION_FOUR_WAY, STATUS_UNSUPPORTED,
		TC_LINE
		"experts.intermediate 2560 at 4 ranks gives chunk 0 (rank 0's unrotated) 640 columns, "
		"the tensor-core MoE's mixed forms 512 at an even chunk (--exl3_expert_rotation=1, "
		"exl3_tc_records.h)\n");
	other.intermediate = TC_THREE_WIDE_CHUNKS;
	tc_expect("the tensor-core MoE, three wide chunks", &other, TP4_RANKS,
		  0, STATUS_OK, NULL);
	tc_expect(
		"the tensor-core MoE, three wide chunks, rotated", &other,
		TP4_RANKS, EXL3_TC_ROTATION_THREE_WAY, STATUS_UNSUPPORTED,
		TC_LINE
		"experts.intermediate 2432 at 4 ranks gives chunk 2 (rank 2's unrotated) 640 columns, "
		"the tensor-core MoE's mixed forms 512 at an even chunk (--exl3_expert_rotation=2, "
		"exl3_tc_records.h)\n");
	tc_expect("the tensor-core MoE, no ranks", &model, 0, 0,
		  STATUS_INVALID_ARGUMENT, NULL);
	cases_run++;
	if (exl3_descriptor_tc_check(NULL, &model, TP4_RANKS, 0) !=
		    STATUS_INVALID_ARGUMENT ||
	    exl3_descriptor_tc_check(V41_DATA_MODEL_NAME, NULL, TP4_RANKS, 0) !=
		    STATUS_INVALID_ARGUMENT) {
		(void)fputs(
			"EXL3 descriptor failure (the tensor-core MoE, no model)\n",
			stderr);
		failures++;
	}
}

/* The registry's one entry: the model's table and its EXL3 descriptor, which is the model's (the literal above, its
 * established packs V41_DATA_ESTABLISHED_EXL3_PACKS); no other name finds it */
static void registry_cases(void)
{
	static const struct exl3_established_pack established[] =
		V41_DATA_ESTABLISHED_EXL3_PACKS;
	const struct model_entry *const entry =
		model_registry_find(V41_DATA_MODEL_NAME);
	const struct exl3_descriptor_model *const exl3 =
		entry != NULL ? entry->exl3 : NULL;
	cases_run++;
	bool held = entry == &model_v41_entry && exl3 != NULL &&
		    entry->operations != NULL &&
		    entry->operations ==
			    model_operations_of(V41_DATA_MODEL_NAME) &&
		    strcmp(entry->operations->name, V41_DATA_MODEL_NAME) == 0 &&
		    strcmp(exl3->model_type, model.model_type) == 0 &&
		    exl3->layers == model.layers &&
		    exl3->experts_per_layer == model.experts_per_layer &&
		    exl3->hidden == model.hidden &&
		    exl3->intermediate == model.intermediate &&
		    exl3->routes_per_row == model.routes_per_row &&
		    exl3->cone_first_layer == model.cone_first_layer &&
		    exl3->established_count ==
			    sizeof(established) / sizeof(established[0]);
	for (uint32_t kind = 0; held && kind < EXL3_DESCRIPTOR_MATRIX_KINDS;
	     kind++)
		held = strcmp(exl3->matrices[kind].kind,
			      model.matrices[kind].kind) == 0 &&
		       exl3->matrices[kind].outputs ==
			       model.matrices[kind].outputs &&
		       exl3->matrices[kind].inputs ==
			       model.matrices[kind].inputs;
	for (uint32_t index = 0; held && index < exl3->established_count;
	     index++)
		held = strcmp(exl3->established[index].name,
			      established[index].name) == 0 &&
		       exl3->established[index].header_digest ==
			       established[index].header_digest &&
		       exl3->established[index].file_byte_count ==
			       established[index].file_byte_count;
	char unterminated[MODEL_NAME_CAPACITY];
	memset(unterminated, 'D', sizeof(unterminated));
	held = held && model_registry_find(NULL) == NULL &&
	       model_registry_find(V41_DATA_MODEL_NAME "-Lite") == NULL &&
	       model_registry_find("DeepSeek-V4.1") == NULL &&
	       model_registry_find(unterminated) == NULL &&
	       model_operations_of("DeepSeek-V4.1") == NULL;
	if (!held) {
		(void)fputs("EXL3 descriptor failure (the registry's entry)\n",
			    stderr);
		failures++;
	}
}

/* The step's check of an instance (rotation: --exl3_expert_rotation's): the status and a line in the log */
static void instance_expect(const char *label,
			    const struct model_instance *checked_instance,
			    const struct pack_record *pack, bool admits_unrated,
			    uint32_t tc_ranks, uint32_t rotation,
			    result_status status, const char *line)
{
	char log[TEST_LOG_BYTES];
	struct stderr_capture redirect;
	cases_run++;
	result_status found = STATUS_INPUT_OUTPUT_ERROR;
	if (capture_start(&redirect, log)) {
		found = model_instance_map_check(checked_instance, pack, TABLES,
						 strlen(TABLES), admits_unrated,
						 tc_ranks, rotation);
		capture_end(&redirect, log);
	}
	if (found != status || strstr(log, line) == NULL) {
		(void)fprintf(
			stderr,
			"EXL3 descriptor failure (%s): status %d, expected %d; wanted \"%s\" in: %s\n",
			label, (int)found, (int)status, line, log);
		failures++;
	}
}

/* An instance bound to the model's EXL3 pack (nothing beside the map: not listed as established), to a pack naming
 * another model, to an entry of no EXL3 map and to one the tensor-core MoE does not fit */
static void instance_cases(const struct pack_record *pack)
{
	struct model_instance instance = { NULL, NULL, NULL };
	struct pack_record *other = NULL;
	const struct test_model_v41_change change = {
		TEST_MODEL_V41_CHANGE_MODEL, NULL, 0
	};
	(void)unlink(sidecar);
	model_instance_bind(pack, map, &instance);
	cases_run++;
	if (instance.model != &model_v41_entry || instance.operations != NULL ||
	    instance.map_path != map ||
	    test_model_v41_pack_open(change, &other) != STATUS_OK) {
		(void)fputs("EXL3 descriptor failure (an instance bound)\n",
			    stderr);
		failures++;
		return;
	}
	instance_expect(
		"an instance, not listed", &instance, pack, false, 0, 0,
		STATUS_UNSUPPORTED,
		"is not listed as established: it is served only on --exl3_unrated_map=1");
	instance_expect("an instance, opted in, the tensor-core MoE", &instance,
			pack, true, TP4_RANKS, 0, STATUS_OK,
			"a pack the model does not list as established");
	struct exl3_descriptor_model unfit = *model_v41_entry.exl3,
				     wide_chunks = *model_v41_entry.exl3;
	unfit.hidden = TC_OTHER_HIDDEN;
	wide_chunks.intermediate = TC_WIDE_CHUNKS;
	const struct model_entry unfit_entry = { model_v41_entry.operations,
						 &unfit },
				 wide_entry = { model_v41_entry.operations,
						&wide_chunks },
				 no_exl3 = { model_v41_entry.operations, NULL };
	struct model_instance changed = instance;
	changed.model = &unfit_entry;
	instance_expect(
		"an instance the tensor-core MoE does not fit", &changed, pack,
		true, TP4_RANKS, 0, STATUS_UNSUPPORTED,
		TC_LINE
		"experts.hidden is 4096, the tensor-core MoE's kernels 5120 (exl3_tc.h)\n");
	instance_expect(
		"an instance the tensor-core MoE does not fit, unselected",
		&changed, pack, true, 0, 0, STATUS_OK,
		"a pack the model does not list as established");
	/* the rotation reaches the tensor-core MoE's check: every chunk wide passes only unrotated */
	changed.model = &wide_entry;
	instance_expect("an instance of wide chunks", &changed, pack, true,
			TP4_RANKS, 0, STATUS_OK,
			"a pack the model does not list as established");
	instance_expect(
		"an instance of wide chunks, rotated", &changed, pack, true,
		TP4_RANKS, EXL3_TC_ROTATION_FOUR_WAY, STATUS_UNSUPPORTED,
		TC_LINE
		"experts.intermediate 2560 at 4 ranks gives chunk 0 (rank 0's unrotated) 640 columns, "
		"the tensor-core MoE's mixed forms 512 at an even chunk (--exl3_expert_rotation=1, "
		"exl3_tc_records.h)\n");
	changed.model = &no_exl3;
	instance_expect("an instance of no EXL3 map", &changed, pack, true, 0,
			0, STATUS_UNSUPPORTED,
			"the pack names the model \"" V41_DATA_MODEL_NAME
			"\", of which this build serves no EXL3 map\n");
	model_instance_bind(other, map, &changed);
	instance_expect(
		"a pack naming another model", &changed, other, true, 0, 0,
		STATUS_UNSUPPORTED,
		"the pack names the model \"DeepSeek-V4.1-Lite\", of which this build serves no EXL3 map\n");
	model_instance_bind(NULL, map, &changed);
	changed.map_path = NULL;
	cases_run++;
	if (changed.model != NULL ||
	    model_instance_map_check(&changed, pack, NULL, 0, true, 0, 0) !=
		    STATUS_INVALID_ARGUMENT ||
	    model_instance_map_check(NULL, pack, NULL, 0, true, 0, 0) !=
		    STATUS_INVALID_ARGUMENT ||
	    model_instance_map_check(&instance, NULL, NULL, 0, true, 0, 0) !=
		    STATUS_INVALID_ARGUMENT) {
		(void)fputs(
			"EXL3 descriptor failure (an instance of nothing)\n",
			stderr);
		failures++;
	}
	pack_close(other);
}

int main(void)
{
	struct pack_record *pack = NULL;
	const struct test_model_v41_change change = {
		TEST_MODEL_V41_CHANGE_EXL3, NULL, 0
	};
	if (!fixture() ||
	    test_model_v41_pack_open(change, &pack) != STATUS_OK) {
		(void)fputs("EXL3 descriptor: FAIL (fixture)\n", stderr);
		return EXIT_FAILURE;
	}
	refusal_cases(pack);
	quality_cases(pack);
	file_cases(pack);
	absent_cases(pack);
	fault_cases(pack);
	instance_cases(pack);
	pack_close(pack);
	mixed_cases();
	original_case();
	tc_cases();
	registry_cases();
	(void)unlink(sidecar);
	(void)unlink(linked_map);
	(void)unlink(map);
	(void)rmdir(folder);
	(void)printf(
		"EXL3 descriptor %s: %u cases (each quality, the link rule, each refusal, the synthetic mixed pack, "
		"the tensor-core MoE's geometry, the registry, an instance)\n",
		failures == 0 ? "PASS" : "FAIL", cases_run);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
