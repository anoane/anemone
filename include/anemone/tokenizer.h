/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * tokenizer.h: text to token ids and back, exactly as the model's tokenizer.json does it.
 *
 * Contract:
 *   - The tables are one run of bytes that tools/convert/front_end_tables.py writes once from tokenizer.json. The
 *     pack carries them as the plain tensor "front_end.tables" (UINT8) of a host-only unit, and
 *     engine_describe hands them over (front_end_seam.h). Little-endian, beginning at a multiple of 8
 *     bytes, in this order with nothing between them:
 *       struct tokenizer_tables_header
 *       uint32 token_byte_ends[vocabulary_size]          token t is the bytes from the end of t - 1 (0 for t = 0)
 *       uint32 merges[merge_count][3]                    left, right, result: a merge's rank is its index
 *       uint32 added_token_ids[added_token_count]        tokens found as they are spelled, before anything else
 *       uint32 byte_token_ids[TOKENIZER_BYTE_VALUES]
 *       uint16 class_block_of[TOKENIZER_CLASS_BLOCK_COUNT]   code point / 256 to its block of classes
 *       uint8  class_blocks[class_block_count][256]      the classes of a code point: enum tokenizer_class
 *       uint8  token_bytes[token_byte_count]
 *     tokenizer_open refuses with STATUS_CORRUPT_PACK, and one error line that names the rule,
 *     tables whose magic, version, reserved words or byte count are not the header's; a count of zero or above its
 *     limit, the added tokens' bytes together among them (the trie that open builds has an edge for each byte); a
 *     token of no byte or of more than TOKENIZER_MAXIMUM_TOKEN_BYTES; a byte's token that is not
 *     that one byte; a merge of a token id out of range, whose result is not its two tokens one after the other,
 *     or whose pair comes twice; an added token out of range or listed twice; a class block out of range, a class
 *     bit it does not know, a code point that is a number and of the second split's range at once.
 *   - Encoding does what the reference does, step by step. The added tokens are found first, leftmost and then
 *     longest. (The reference looks twice, for its raw and its normalized added tokens; the converter proves that
 *     one look finds the same with this vocabulary, and the normalizer changes nothing.) The text between two of
 *     them is cut by the three splits of the pre-tokenizer, each applied to every piece of the one before: numbers
 *     in groups of up to three; runs of the second split's range; the third split's pattern, whose alternatives
 *     are rewritten by hand in tokenizer.c. A match and the text between two matches are pieces alike. Each piece
 *     is then byte-level BPE: of the pairs of neighbours that a merge joins, the one of lowest rank, the leftmost
 *     among equals, is merged, until none is left.
 *   - Text is UTF-8. A maximal subpart of a sequence that is not counts as one U+FFFD, as Python's
 *     bytes.decode(errors="replace") reads it: the ids then spell that text, not the bytes given.
 *   - Work is linear in the text but for the BPE of a piece, O(n log n) in its bytes; memory is the tokenizer's
 *     own (16 bytes for each merge and each byte of an added token, two to four times over: 8 MiB for this model's
 *     tables) and, while encode runs, at most 72 bytes for each byte of the longest piece and three for each byte
 *     of a text that is not UTF-8. A text gives at least its bytes over the longest token's in ids: encode refuses
 *     one too long for token_id_capacity before any work, so the capacity bounds both. Nothing of encode is shared:
 *     a tokenizer is immutable once open and may be used from any number of threads. It reads the tables where they
 *     lie: they must outlive it.
 *   - Fault point: tokenizer_memory, at each allocation of open and encode.
 */
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "anemone/status.h"

#define TOKENIZER_TABLES_MAGIC \
	"ANEMTOKN" /* eight letters, no terminating zero */
#define TOKENIZER_TABLES_MAGIC_BYTES 8
#define TOKENIZER_TABLES_FORMAT_VERSION 1
#define TOKENIZER_TABLES_HEADER_BYTES 40
#define TOKENIZER_TABLES_ALIGNMENT 8
#define TOKENIZER_TABLES_RESERVED_WORDS 2
#define TOKENIZER_BYTE_VALUES 256
#define TOKENIZER_CLASS_BLOCK_CODE_POINTS 256
#define TOKENIZER_CLASS_BLOCK_COUNT \
	4352 /* 0x110000 code points, 256 to a block */
#define TOKENIZER_MAXIMUM_VOCABULARY \
	2097152u /* 2^21: far above any model's; the BPE keys need 32 bits */
#define TOKENIZER_MAXIMUM_TOKEN_BYTES 1024u
#define TOKENIZER_MAXIMUM_ADDED_TOKEN_BYTES \
	1048576u /* 1 MiB of them all: this model's are 39 KiB */
#define TOKENIZER_MAXIMUM_TEXT_BYTES \
	1073741824ull /* 1 GiB: positions of three times as much fit 32 bits */

/* The words of a merge, in this order. */
enum tokenizer_merge_word {
	TOKENIZER_MERGE_LEFT,
	TOKENIZER_MERGE_RIGHT,
	TOKENIZER_MERGE_RESULT,
	TOKENIZER_MERGE_WORDS
};

/* The classes of a code point, as the pre-tokenizer's patterns read them; a code point has any number of them. */
enum tokenizer_class {
	TOKENIZER_CLASS_LETTER = 1, /* \p{L} */
	TOKENIZER_CLASS_MARK = 2, /* \p{M} */
	TOKENIZER_CLASS_NUMBER = 4, /* what the first split groups */
	TOKENIZER_CLASS_PUNCTUATION_OR_SYMBOL = 8, /* \p{P} or \p{S} */
	TOKENIZER_CLASS_WHITESPACE = 16, /* \s */
	TOKENIZER_CLASS_SECOND_SPLIT =
		32, /* what the second split gathers into runs */
	TOKENIZER_CLASS_ALL = 63
};

struct tokenizer_tables_header {
	char magic[TOKENIZER_TABLES_MAGIC_BYTES];
	uint32_t format_version;
	uint32_t vocabulary_size; /* token ids are below it; every one has bytes */
	uint32_t merge_count;
	uint32_t added_token_count;
	uint32_t class_block_count;
	uint32_t token_byte_count;
	uint32_t reserved_zero[TOKENIZER_TABLES_RESERVED_WORDS];
};
_Static_assert(sizeof(struct tokenizer_tables_header) ==
		       TOKENIZER_TABLES_HEADER_BYTES,
	       "the header as it is stored");

struct text_tokenizer;
struct tokenizer_cache;

/* Half-open byte range in the original UTF-8 text. Only a match wholly inside one trusted span may emit an
 * added-token id. Sorted, nonempty spans may touch but never overlap; touching spans remain distinct. */
struct tokenizer_span {
	uint32_t begin, end;
};

#define TOKENIZER_CACHE_SEGMENTS 16384u

/* Optional request cache, private to one caller/thread; tokenizer and its tables must outlive it. Open requires
 * nonzero limits, text_byte_limit <= MAXIMUM_TEXT_BYTES and non-NULL arguments; failure leaves cache_out alone.
 * It holds at most text_byte_limit bytes (rounded to uint32 alignment), token_limit ids, and
 * min(token_limit + 1, CACHE_SEGMENTS) 16-byte segment records, plus a fixed-size header. Storage grows only as
 * needed, never beyond these limits. Oversize or malformed UTF-8 texts and overlapping text/output buffers still
 * use full encoding but evict the cache.
 * Refresh allocation failure also evicts without failing a successful encode. Fault: tokenizer_cache_memory.
 * Common_prefix returns the exact matching byte count (0 for NULL/empty); it grants no KV-prefix reuse.
 * Encode_cached has encode's outputs/refusals; NULL cache is INVALID_ARGUMENT. Any failure invalidates the cache.
 * Close accepts NULL. There are no borrowed request/output buffers and no shared mutable tokenizer state. */
MUST_CHECK result_status tokenizer_cache_open(
	const struct text_tokenizer *tokenizer, uint32_t text_byte_limit,
	uint32_t token_limit, struct tokenizer_cache **cache_out);
void tokenizer_cache_close(struct tokenizer_cache *cache);
uint32_t tokenizer_cache_common_prefix(const struct tokenizer_cache *cache,
				       const char *text,
				       uint64_t text_byte_count);
MUST_CHECK result_status tokenizer_encode_cached(struct tokenizer_cache *cache,
						 const char *text,
						 uint64_t text_byte_count,
						 uint32_t *token_ids_out,
						 uint32_t token_id_capacity,
						 uint32_t *token_count_out);

/* Origin-aware encoding for rendered chat prompts. UTF-8 must be valid; malformed text, unordered/overlapping
 * or out-of-text spans, or NULL trusted with a nonzero count are INVALID_ARGUMENT. NULL trusted and zero count
 * makes the whole text literal. Trust boundaries do not split ordinary pre-tokenizer/BPE segments: those end
 * only at actual added-token matches, exactly as in encode. Cached encoding rescans matches with the NEW spans
 * before reusing any complete ordinary segment; changing trust cannot reuse control-token ids. Its cache holds
 * no borrowed spans. Other outputs, capacities and refusals are encode/encode_cached's. */
MUST_CHECK result_status tokenizer_encode_spans(
	const struct text_tokenizer *tokenizer, const char *text,
	uint64_t text_byte_count, const struct tokenizer_span *trusted,
	uint32_t trusted_count, uint32_t *token_ids_out,
	uint32_t token_id_capacity, uint32_t *token_count_out);
MUST_CHECK result_status tokenizer_encode_cached_spans(
	struct tokenizer_cache *cache, const char *text,
	uint64_t text_byte_count, const struct tokenizer_span *trusted,
	uint32_t trusted_count, uint32_t *token_ids_out,
	uint32_t token_id_capacity, uint32_t *token_count_out);

/* First users: the server at start, from the engine's description, and the tests. On failure tokenizer_out is
 * untouched; a NULL argument or tables not on a multiple of TOKENIZER_TABLES_ALIGNMENT is
 * STATUS_INVALID_ARGUMENT. */
MUST_CHECK result_status tokenizer_open(const void *tables,
					uint64_t table_byte_count,
					struct text_tokenizer **tokenizer_out);
void tokenizer_close(struct text_tokenizer *tokenizer); /* accepts NULL */
uint32_t tokenizer_vocabulary_size(
	const struct text_tokenizer *tokenizer); /* 0 for NULL */

/* The ids of a text, in token_ids_out. More tokens than token_id_capacity (found before any work for a text of more
 * bytes than token_id_capacity times the longest token's), a text above TOKENIZER_MAXIMUM_TEXT_BYTES, or a
 * NULL argument (text may be NULL when it has no byte) is STATUS_INVALID_ARGUMENT. */
MUST_CHECK result_status
tokenizer_encode(const struct text_tokenizer *tokenizer, const char *text,
		 uint64_t text_byte_count, uint32_t *token_ids_out,
		 uint32_t token_id_capacity, uint32_t *token_count_out);

/* The bytes a token stands for, inside the tables: the ids of a text give it back, token after token. A token's
 * bytes need not end on a character: whoever shows them holds a partial character back. An id at or above the
 * vocabulary size is STATUS_INVALID_ARGUMENT. */
MUST_CHECK result_status
tokenizer_token_bytes(const struct text_tokenizer *tokenizer, uint32_t token_id,
		      const char **bytes_out, uint32_t *byte_count_out);

/* The bytes of the UTF-8 sequence at text[position], and whether it is one; when it is not, the bytes of its maximal
 * subpart, which one U+FFFD stands for. 0, not one, for a position at or past end or a NULL argument. First users:
 * encode, and the chat template's check of the strings of its JSON. */
uint32_t tokenizer_utf8_sequence_bytes(const char *text, size_t position,
				       size_t end, bool *is_valid_out);

#endif
