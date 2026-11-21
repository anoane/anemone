/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * exl3_descriptor.h: an EXL3 map's descriptor, the sidecar <map>.descriptor.json that
 * make_virtual_pack.py writes beside the map, read once at startup when the pack holds EXL3 matrices,
 * after it opens and before anything loads. Private to the engine's sources.
 *
 * Contract:
 *   - The sidecar is the map's real path (its links followed; the path as given when it does not resolve) with
 *     EXL3_DESCRIPTOR_SUFFIX behind it. When nothing is at that path (nor can be: its name is over NAME_MAX), what the
 *     open pack carries decides, whatever path or reader opened it: a pack on the model's established list (its
 *     header_digest and file_byte_count) is served by default, any other only with admits_unrated
 *     (--exl3_unrated_map=1: no quality claim), refused otherwise, naming its digest; one line says which.
 *   - Otherwise each rule must hold, else one error line names the sidecar and the field, with the status
 *     STATUS_UNSUPPORTED for a format version, model, codebook or quality this build does not serve,
 *     STATUS_INPUT_OUTPUT_ERROR for a file that cannot be read, STATUS_OUT_OF_MEMORY, and STATUS_INVALID_ARGUMENT for
 *     every other rule:
 *       a regular file of at most EXL3_DESCRIPTOR_MOST_BYTES bytes holding one JSON object, no key twice in an
 *       object; format_version 1, then exactly the members the converter writes; the members that bind, each of its
 *       type (a text holds a byte other than a space, a SHA-256 is 64 lower-case hexadecimal digits):
 *       map: logical_bytes and pack_header_digest the open pack's file_byte_count and header_digest;
 *       architecture.model_type the model's; quantiser.source a text; codebook.mul1_multiplier EXL3_MUL1_MULTIPLIER,
 *       as "0x%08X"; front_end_tables_sha256 the tables' sha256 when they were read (tables not NULL) and it is not
 *       null;
 *       experts: the model's layers, experts_per_layer, hidden and intermediate, and each matrix kind's
 *       [outputs, inputs];
 *       widths: for each layer (its decimal, 0 to layers - 1, once each) and each kind, the count of matrices of each
 *       width ("K3", "K2.5": the converter's names), equal to what the pack's records give: every routed matrix
 *       layers.<layer>.ffn.experts.<expert>.<kind>.weight is an EXL3 matrix of its kind's shape, and the pack holds no
 *       other EXL3 matrix. Every width is one the build decodes, as the pack's open has checked (pack.h);
 *       widths_summary: those counts summed by width ("matrices"), and the layers by their one width, "mixed" for a
 *       layer of more than one ("layers");
 *       quality: {"status": "none"}, only with admits_unrated, refused otherwise, naming the map; {"status":
 *       "established", "reference": text}; or {"status": "recorded", "record": an absolute path, "sha256": that
 *       file's, which is read and checked}. A descriptor's quality decides whether or not its pack is listed.
 *   - On success one line gives the sidecar, its sha256, the quantiser's source, the widths and the quality. The
 *     sidecar and a recorded capture are read once, and nothing of them is kept.
 */
#ifndef EXL3_DESCRIPTOR_H
#define EXL3_DESCRIPTOR_H

#include <stdbool.h>
#include <stdint.h>

#include "engine/formats/pack/pack.h"
#include "common/status.h"

#define EXL3_DESCRIPTOR_SUFFIX ".descriptor.json"
#define EXL3_DESCRIPTOR_MOST_BYTES \
	1048576u /* 1 MiB: a map's is about 6 KB, a width table of a (layer, kind) each */
#define EXL3_DESCRIPTOR_MATRIX_KINDS \
	3u /* a routed expert's matrices: gate, up, down (the converter's EXPERT_MATRICES) */
#define EXL3_DESCRIPTOR_OPT_IN "--exl3_unrated_map=1"

/* A routed expert's matrix of one kind: its name's part, layers.<layer>.ffn.experts.<expert>.<kind>.weight */
struct exl3_descriptor_matrix {
	const char *kind;
	uint32_t outputs, inputs;
};

/* A pack whose quality is established without a descriptor beside it, by what it carries; named on the startup line */
struct exl3_established_pack {
	const char *name;
	uint64_t header_digest, file_byte_count;
};

/* What the model requires of a map: config.json's model_type, its routed experts' geometry, its established packs;
 * and, for the tensor-core MoE alone (exl3_descriptor_tc_check), its routes a row and its decoder's first layer */
struct exl3_descriptor_model {
	const char *model_type;
	uint32_t layers, experts_per_layer, hidden, intermediate;
	struct exl3_descriptor_matrix matrices[EXL3_DESCRIPTOR_MATRIX_KINDS];
	const struct exl3_established_pack *established;
	uint32_t established_count;
	uint32_t routes_per_row, cone_first_layer;
};

/* map_path: the map as the run names it; tables: the front end's tables when the command read them, else NULL. */
MUST_CHECK result_status exl3_descriptor_check(
	const char *map_path, const struct pack_record *pack,
	const struct exl3_descriptor_model *model, const char *tables,
	uint64_t table_byte_count, bool admits_unrated);

/* --exl3_tc_moe=1: the model named model_name against the geometry the tensor-core MoE's kernels are built for
 * (exl3_tc.h) at ranks TP ranks (rotation: --exl3_expert_rotation's value): experts.hidden, routes_per_row,
 * experts.experts_per_layer, experts.layers and cone_first_layer each the kernels' value, and every rank's chunk of
 * experts.intermediate (exl3_layout's split, which must take the shape) of the i640 or i512 forms; with a rotation,
 * chunk c the narrow form for an even c and the wide one for an odd c, as the mixed forms read them. Else
 * STATUS_UNSUPPORTED with one line naming the model and the first field that differs. Nothing is printed on success. */
MUST_CHECK result_status exl3_descriptor_tc_check(
	const char *model_name, const struct exl3_descriptor_model *model,
	uint32_t ranks, uint32_t rotation);

#endif
