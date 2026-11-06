/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * pack.h: the file a model is loaded from.
 *
 * Contract:
 *   - A pack is ONE file, the same bytes on every copy, made of blocks of PACK_BLOCK_BYTES, little-endian:
 *       block 0       struct pack_header, zero-filled to the end of the block
 *       the tables    extent records, unit records, tensor records, then the bytes of the tensor names, in that
 *                     order with nothing between them, zero-filled to the end of their last block
 *       the extents   each a whole number of blocks, zero-filled, one after another in the order of their records
 *   - An extent is the smallest run of bytes placement moves: it goes whole into one card's arena or into host
 *     memory, and its record holds its digest (pack_digest.h). A unit is a run of extents that must sit together;
 *     model_seam.h says what the three kinds mean. Units lie in the order they run, each stage unit followed by the
 *     on-demand units it owns: the owner of an on-demand unit is the last stage unit before it. A tensor record
 *     says where a named tensor lies inside an extent; the loader never reads them, the model does, once, at load.
 *     Tensor records are sorted by name, as unsigned bytes, a shorter name before a longer one it begins. The
 *     converter lays the tensors of an extent out in the order of their names with zero fill between them; the
 *     reader does not check that, nor that two tensors share no byte.
 *   - The bytes of an extent are in the form the kernels read: nothing is converted at load. A tensor is either
 *     plain (weight_format WEIGHT_FORMAT_NONE: its elements in order, as the source checkpoint stores
 *     them, or what the converter derives for the host: a model's n-gram tables as rows and the constants of
 *     their hashing, engram.h) or a matrix of a weight format, laid out by the converter with its scales as
 *     weight_format_seam.h says. Format version 1 knew plain tensors only and is no longer read.
 *   - pack_open_entry reads the header and the tables through a storage reader, from the first copy that
 *     delivers them with both digests holding (a copy that does not is logged as a warning), and refuses, with
 *     STATUS_CORRUPT_PACK and one error line that names the rule, a file that breaks any of these:
 *       the magic, the format version and the block size are the ones of this header, and the model name holds a
 *       NUL; the reader's block alignment divides the block size; file_byte_count is the reader's size;
 *       tables_byte_count is at most PACK_MAXIMUM_TABLES_BYTES and exactly the bytes the four counts
 *       need, rounded up to a block; the header's digest (of block 0 with header_digest as zero) and the tables'
 *       digest (of their blocks) hold; block 0 behind the header and the tables behind the name bytes are zero;
 *       there is at least one extent and one unit, and no more units than extents;
 *       every extent is a whole number of blocks and not empty, the first begins right after the tables, each
 *       begins where the one before ends, and the last ends at the end of the file;
 *       unit_index is 0 in the first extent, unit_count - 1 in the last, and the same or one more from one extent
 *       to the next; a unit's kind is known, and an on-demand unit has a stage unit before it;
 *       a tensor's extent exists; it lies inside it and begins on a multiple of
 *       WEIGHT_FORMAT_MATRIX_ALIGNMENT_BYTES; it has at most PACK_MAXIMUM_DIMENSIONS dimensions and
 *       the unused ones are zero; its element type is known; a plain tensor's byte count is the product of its
 *       dimensions times the size of an element, with no overflow, and its bits_per_weight is zero; a matrix of a
 *       weight format has a format this build carries (one it does not is refused like a value outside the
 *       enumeration, until a build exists that carries another), two dimensions (outputs, then inputs), elements of
 *       PACK_ELEMENT_UINT8 and exactly the bytes its format's stored_byte_count gives for that shape and
 *       its bits_per_weight; its name is not empty, lies inside the name bytes and sorts after the name before it;
 *       every reserved field is zero.
 *     Any other failure keeps its own status (input or output, memory, an invalid argument): a reader that
 *     reports no copy or more than STORAGE_MAXIMUM_COPIES, or holds less than one block, is an invalid
 *     argument.
 *   - A pack is immutable once open, reads nothing after open, and may be used from any thread. Its accessors
 *     return pointers into memory the pack owns, valid until pack_close, which accepts NULL.
 * Limits: this file is read on a little-endian host only (the build refuses another).
 */
#ifndef PACK_H
#define PACK_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/model_seam.h"
#include "anemone/status.h"
#include "anemone/storage_seam.h"
#include "anemone/weight_format_seam.h"

#define PACK_MAGIC "ANEMONE" /* seven letters and their NUL: PACK_MAGIC_BYTES */
#define PACK_MAGIC_BYTES 8
#define PACK_FORMAT_VERSION 2
#define PACK_BLOCK_BYTES 4096
#define PACK_MAXIMUM_TABLES_BYTES \
	1073741824ull /* 1 GiB: the tables are held in memory; far above any model */
#define PACK_MAXIMUM_DIMENSIONS 4
#define PACK_TENSOR_RESERVED_WORDS 1
#define PACK_NO_LAYER \
	UINT32_MAX /* layer_index of a unit that belongs to no layer */
#define PACK_NOT_FOUND UINT32_MAX /* in place of a tensor index */
#define PACK_HEADER_BYTES 128
#define PACK_EXTENT_RECORD_BYTES 32
#define PACK_UNIT_RECORD_BYTES 8
#define PACK_TENSOR_RECORD_BYTES 64

struct pack_header {
	char magic[PACK_MAGIC_BYTES];
	uint32_t format_version;
	uint32_t block_byte_count;
	char model_name
		[MODEL_NAME_CAPACITY]; /* names the model table: model_operations_of */
	uint64_t file_byte_count;
	uint64_t tables_byte_count; /* whole blocks, from block 1 on; the extents follow */
	uint64_t tables_digest;
	uint32_t extent_count;
	uint32_t unit_count;
	uint32_t tensor_count;
	uint32_t name_byte_count;
	uint64_t header_digest; /* of block 0 with this field as zero */
};

struct pack_extent {
	uint64_t byte_offset; /* in the file */
	uint64_t byte_count; /* whole blocks */
	uint64_t digest;
	uint32_t unit_index;
	uint32_t reserved_zero;
};

/* A unit as the file stores it. */
struct pack_unit_record {
	uint32_t unit_kind; /* enum placement_unit_kind */
	uint32_t layer_index; /* for logs and placement text only: the converter gives
                                                       * an on-demand unit its stage unit's, the reader checks
                                                       * nothing; PACK_NO_LAYER if none */
};

/* A unit as pack_units_of gives it: the record, and what the order of the file says about it. */
struct pack_unit_record_2 {
	uint32_t unit_kind;
	uint32_t layer_index;
	uint32_t owner_unit_index; /* on-demand units: their stage unit; else the unit itself */
	uint32_t first_extent_index;
	uint32_t extent_count;
	uint32_t reserved_zero;
	uint64_t byte_count; /* the sum of its extents */
};

enum pack_element_type {
	PACK_ELEMENT_FLOAT16 = 0,
	PACK_ELEMENT_BFLOAT16 = 1,
	PACK_ELEMENT_FLOAT32 = 2,
	PACK_ELEMENT_INT8 = 3,
	PACK_ELEMENT_UINT8 = 4,
	PACK_ELEMENT_INT16 = 5,
	PACK_ELEMENT_INT32 = 6,
	PACK_ELEMENT_INT64 = 7,
	PACK_ELEMENT_FLOAT8_E4M3 = 8,
	PACK_ELEMENT_FLOAT8_E8M0 = 9,
	PACK_ELEMENT_TYPE_COUNT = 10 /* number of element types; not a type */
};

struct pack_tensor {
	uint64_t byte_offset_in_extent;
	uint64_t byte_count;
	uint32_t extent_index;
	uint32_t name_byte_offset; /* into the name bytes; names are not NUL-terminated */
	uint32_t name_byte_count;
	uint32_t element_type; /* enum pack_element_type */
	uint32_t dimension_count;
	uint32_t dimensions[PACK_MAXIMUM_DIMENSIONS];
	uint32_t weight_format; /* enum weight_format_code; NONE for a plain tensor */
	uint32_t bits_per_weight; /* of a matrix of a weight format; 0 for a plain tensor */
	uint32_t reserved_zero[PACK_TENSOR_RESERVED_WORDS];
};

_Static_assert(sizeof(struct pack_header) == PACK_HEADER_BYTES,
	       "the header as it is stored");
_Static_assert(sizeof(struct pack_extent) == PACK_EXTENT_RECORD_BYTES,
	       "an extent as it is stored");
_Static_assert(sizeof(struct pack_unit_record) == PACK_UNIT_RECORD_BYTES,
	       "a unit as it is stored");
_Static_assert(sizeof(struct pack_tensor) == PACK_TENSOR_RECORD_BYTES,
	       "a tensor as it is stored");
_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "a pack is read as the host's own integers");

struct pack_record;

/* First user: the program's load. storage and reader are the open copies of the pack; on failure pack_out is
 * untouched. */
MUST_CHECK result_status pack_open_entry(
	const struct storage_operations_record *storage,
	const struct storage_reader *reader, struct pack_record **pack_out);
void pack_close(struct pack_record *pack);

/* For an open pack only (a NULL pack is the caller's error): never NULL, and each table has as many entries as
 * the header says. */
const struct pack_header *pack_header_of(const struct pack_record *pack);
const struct pack_extent *pack_extents_of(const struct pack_record *pack);
const struct pack_unit_record_2 *pack_units_of(const struct pack_record *pack);
const struct pack_tensor *pack_tensors_of(const struct pack_record *pack);
const char *pack_name_bytes_of(const struct pack_record *pack);

/* First users: the model, which looks its tensors up once at load, and the tests. The index of the tensor of that
 * name, by binary search, or PACK_NOT_FOUND (also for a NULL pack or name). */
uint32_t pack_find_tensor(const struct pack_record *pack, const char *name,
			  uint32_t name_byte_count);

/* First users: the default placement, the loader's layout and the model's state sizes. True when card_of_unit
 * holds one entry per unit of the pack and each obeys the kind of its unit: a stage unit on a card below
 * card_count, an on-demand unit on its owner's card or PLACEMENT_IN_HOST_MEMORY, a host-only unit in host
 * memory. Otherwise false, with one error line that names the unit. */
bool pack_placement_is_valid(const struct pack_record *pack,
			     const uint32_t *card_of_unit, uint32_t unit_count,
			     uint32_t card_count);

#endif
