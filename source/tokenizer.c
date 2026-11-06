/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * tokenizer.c: text to token ids and back. Contract: tokenizer.h.
 *
 * Open checks the tables once and builds one hash table of 64-bit keys for the two things encode looks up: a merge
 * by its pair of tokens, and an edge of the trie of the added tokens by its node and byte. Encode then reads the
 * tables without a bound check of its own. The third split is the reference's regular expression run by its
 * engine, which tries the alternatives in order at a position and takes the first that matches, greedy and with
 * backtracking, on a piece whose end is the end of its text:
 *   A  [ASCII punctuation or symbol][A-Za-z]+
 *   B  [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
 *   C   ?[\p{P}\p{S}]+[\r\n]*
 *   D  \s*[\r\n]+          after backtracking: the run of whitespace up to its last line break
 *   E  \s+(?!\S)           the run of whitespace, less its last character unless the piece ends with it
 *   F  \s+
 * B takes its optional character only when letters or marks follow it, and a mark there is the same as the run of
 * letters and marks from the position; so B is that run when the first character is a letter or a mark, else the
 * optional character and the run after it. The BPE is the reference's: a heap of (rank, position) with stale
 * entries skipped when they come up, a pair being stale when its left symbol was merged away or the pair now
 * there is another (a merge only makes the bytes of a position longer, so a pair never comes back).
 */
#include "anemone/tokenizer.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/hooks.h"

#define NO_INDEX UINT32_MAX                       /* no symbol, no token, no edge's token */
#define EMPTY_KEY UINT64_MAX
#define EDGE_KEY_TAG (1ull << 63)                 /* a merge's key is below it: its left id is below 2^21 */
#define EDGE_NODE_SHIFT 8                         /* an edge's key: the tag, its node, its byte */
#define PAIR_LEFT_SHIFT 32
#define HEAP_RANK_SHIFT 32                        /* a heap entry: the rank of the pair, then its position */
#define HASH_MULTIPLIER 0x9E3779B97F4A7C15ull
#define KEY_BITS 64
#define SLOTS_PER_KEY 2                           /* the hash table is at most half full */
#define SMALLEST_SLOT_BITS 1
#define HEAP_ARITY 2
#define WORKSPACE_GROWTH 2                        /* the piece bytes the workspace holds at least double */
#define HEAP_ENTRIES_PER_BYTE 3                   /* a piece's pairs, then two for each merge */
#define DIGITS_PER_GROUP 3                        /* \p{N}{1,3} */
#define CODE_POINT_BLOCK_SHIFT 8
#define CODE_POINT_IN_BLOCK_MASK 0xFFu
#define REPLACEMENT_CHARACTER "\xEF\xBF\xBD"      /* U+FFFD */
#define REPLACEMENT_CHARACTER_BYTES 3
#define PREFIX_COMPARE_BYTES 256u

/* UTF-8 as Unicode's table 3-7 has it: after E0 the second byte is from A0, after F0 from 90 (no longer form of a
 * shorter sequence), after ED up to 9F (no surrogate), after F4 up to 8F (nothing above U+10FFFF). */
enum utf8_byte {
    ASCII_END = 0x80, CONTINUATION_FIRST = 0x80, CONTINUATION_LAST = 0xBF, CONTINUATION_PAYLOAD = 0x3F,
    CONTINUATION_PAYLOAD_BITS = 6, TWO_BYTE_LEAD_FIRST = 0xC2, THREE_BYTE_LEAD_FIRST = 0xE0,
    THREE_BYTE_SECOND_FIRST = 0xA0, SURROGATE_LEAD = 0xED, SURROGATE_LEAD_SECOND_LAST = 0x9F,
    FOUR_BYTE_LEAD_FIRST = 0xF0, FOUR_BYTE_SECOND_FIRST = 0x90, LAST_LEAD = 0xF4, LAST_LEAD_SECOND_LAST = 0x8F,
    LEAD_END = 0xF5
};
enum sequence_bytes { TWO_BYTES = 2, THREE_BYTES, FOUR_BYTES };
static const unsigned char lead_payload_mask[] = {0, 0x7F, 0x1F, 0x0F, 0x07};   /* by the bytes of a sequence */

/* Short names for the classes. */
#define LETTER TOKENIZER_CLASS_LETTER
#define LETTER_OR_MARK (TOKENIZER_CLASS_LETTER | TOKENIZER_CLASS_MARK)
#define NUMBER TOKENIZER_CLASS_NUMBER
#define SYMBOL TOKENIZER_CLASS_PUNCTUATION_OR_SYMBOL
#define WHITESPACE TOKENIZER_CLASS_WHITESPACE
#define SECOND_SPLIT TOKENIZER_CLASS_SECOND_SPLIT

struct slot {
    uint64_t key;                                 /* EMPTY_KEY, a pair or an edge */
    uint32_t value;                               /* a merge: its rank; an edge: the node it leads to */
    uint32_t result;                              /* a merge: its token; an edge: the added token ending there */
};

struct text_tokenizer {
	const uint32_t *token_byte_ends;
	const uint32_t *merges;
	const uint32_t *byte_token_ids;
	const uint16_t *class_block_of;
	const uint8_t *class_blocks;
	const char *token_bytes;
	uint32_t vocabulary_size;
	uint32_t longest_token_bytes;
	uint32_t slot_shift; /* a key's slot is the top bits of its product */
	uint64_t slot_mask;
	bool may_start_added_token[TOKENIZER_BYTE_VALUES];
	const uint8_t *ascii_classes; /* the classes of code points below 128 */
	struct slot slots[];
};

struct character {
    uint32_t byte_count;                          /* 0 at the end of a piece */
    uint32_t classes;
};

struct symbol {
    uint32_t token_id;                            /* NO_INDEX once merged into the symbol before it */
    uint32_t previous;
    uint32_t next;
};

struct cached_segment {
    uint32_t begin, end;
    uint32_t token_begin, token_end;
};

struct tokenizer_cache {
	const struct text_tokenizer *tokenizer;
	void *storage; /* text, padding to uint32_t, then token ids */
	size_t storage_capacity;
	uint32_t *tokens;
	uint32_t text_count, text_limit, token_limit;
	uint32_t segment_count, segment_capacity;
	struct cached_segment segments[];
};

struct encoder {
	const struct text_tokenizer *tokenizer;
	const unsigned char *text;
	uint32_t *token_ids;
	uint32_t token_id_capacity;
	uint32_t token_count;
	void *workspace; /* the heap, then the symbols, of the longest piece so far */
	size_t workspace_piece_bytes;
	struct tokenizer_cache *cache;
	uint32_t common_bytes, segment_count;
};

/* The bytes of the UTF-8 sequence at position, and whether it is one; an ill-formed one is its maximal subpart. */
static size_t sequence_length(const unsigned char *text, size_t position, size_t end, bool *is_valid_out)
{
    const unsigned lead = text[position];
    const size_t length = lead < TWO_BYTE_LEAD_FIRST || lead >= LEAD_END ? 1
                        : lead < THREE_BYTE_LEAD_FIRST ? TWO_BYTES
                        : lead < FOUR_BYTE_LEAD_FIRST ? THREE_BYTES : FOUR_BYTES;
    unsigned low = lead == THREE_BYTE_LEAD_FIRST ? THREE_BYTE_SECOND_FIRST
                 : lead == FOUR_BYTE_LEAD_FIRST ? FOUR_BYTE_SECOND_FIRST : CONTINUATION_FIRST;
    unsigned high = lead == SURROGATE_LEAD ? SURROGATE_LEAD_SECOND_LAST
                  : lead == LAST_LEAD ? LAST_LEAD_SECOND_LAST : CONTINUATION_LAST;
    *is_valid_out = lead < ASCII_END || length > 1;
    for (size_t index = 1; index < length; index++) {
        if (position + index >= end || text[position + index] < low || text[position + index] > high) {
            *is_valid_out = false;
            return index;
        }
        low = CONTINUATION_FIRST;
        high = CONTINUATION_LAST;
    }
    return length;
}

uint32_t tokenizer_utf8_sequence_bytes(const char *text, size_t position,
				       size_t end, bool *is_valid_out)
{
    bool is_valid = false;
    const size_t length = text == NULL || position >= end ? 0
                        : sequence_length((const unsigned char *)text, position, end, &is_valid);
    if (is_valid_out != NULL) {
        *is_valid_out = is_valid;
    }
    return is_valid_out != NULL ? (uint32_t)length : 0;
}

/* The character at position of text that is UTF-8, with its classes; nothing at the end. */
static struct character character_at(const struct text_tokenizer *tokenizer,
				     const unsigned char *text, size_t position,
				     size_t end)
{
    bool is_valid = false;
    if (position >= end) {
        return (struct character){0, 0};
    }
    if (text[position] < ASCII_END) {                                   /* the most of most text */
        return (struct character){1, tokenizer->ascii_classes[text[position]]};
    }
    const size_t length = sequence_length(text, position, end, &is_valid);
    uint32_t code_point = text[position] & lead_payload_mask[length];
    for (size_t index = 1; index < length; index++) {
        code_point = code_point << CONTINUATION_PAYLOAD_BITS | (text[position + index] & CONTINUATION_PAYLOAD);
    }
    const size_t block = tokenizer->class_block_of[code_point >> CODE_POINT_BLOCK_SHIFT];
    const size_t place = block * TOKENIZER_CLASS_BLOCK_CODE_POINTS +
			 (code_point & CODE_POINT_IN_BLOCK_MASK);
    return (struct character){(uint32_t)length, tokenizer->class_blocks[place]};
}

static bool is_line_break(unsigned char byte) { return byte == '\r' || byte == '\n'; }

/* Where the run of characters with any of the classes ends, ASCII ones only if asked. */
static size_t run_end(const struct text_tokenizer *tokenizer,
		      const unsigned char *text, size_t position, size_t end,
		      uint32_t classes, bool is_ascii_only)
{
    for (struct character character = character_at(tokenizer, text, position, end);
         (character.classes & classes) != 0 && (!is_ascii_only || text[position] < ASCII_END);
         character = character_at(tokenizer, text, position, end)) {
        position += character.byte_count;
    }
    return position;
}

/* Where the third split's match at position ends: position itself when no alternative matches there. */
static size_t match_end(const struct text_tokenizer *tokenizer,
			const unsigned char *text, size_t position, size_t end)
{
    const struct character first = character_at(tokenizer, text, position, end);
    const size_t second_position = position + first.byte_count;
    const struct character second = character_at(tokenizer, text, second_position, end);
    if (text[position] < ASCII_END && (first.classes & SYMBOL) != 0 && (second.classes & LETTER) != 0 &&
        text[second_position] < ASCII_END) {
        return run_end(tokenizer, text, second_position, end, LETTER, true);                           /* A */
    }
    if ((first.classes & LETTER_OR_MARK) != 0) {
        return run_end(tokenizer, text, position, end, LETTER_OR_MARK, false);                         /* B */
    }
    if ((first.classes & SYMBOL) == 0 && !is_line_break(text[position]) && (second.classes & LETTER_OR_MARK) != 0) {
        return run_end(tokenizer, text, second_position, end, LETTER_OR_MARK, false);                  /* B */
    }
    const size_t symbols_begin = text[position] == ' ' && (second.classes & SYMBOL) != 0 ? second_position : position;
    size_t match = run_end(tokenizer, text, symbols_begin, end, SYMBOL, false);                         /* C */
    if (match > symbols_begin) {
        while (match < end && is_line_break(text[match])) {
            match++;
        }
        return match;
    }
    size_t space_end = position, last_character = position, last_line_break_end = position;
    for (struct character character = first; (character.classes & WHITESPACE) != 0;
         character = character_at(tokenizer, text, space_end, end)) {
        last_character = space_end;
        space_end += character.byte_count;
        last_line_break_end = is_line_break(text[last_character]) ? space_end : last_line_break_end;
    }
    return last_line_break_end > position ? last_line_break_end                                         /* D */
         : space_end == end || last_character == position ? space_end : last_character;                /* E, F */
}

static size_t slot_index(const struct text_tokenizer *tokenizer, uint64_t key)
{
    size_t index = (size_t)((key * HASH_MULTIPLIER) >> tokenizer->slot_shift);
    while (tokenizer->slots[index].key != key && tokenizer->slots[index].key != EMPTY_KEY) {
        index = (index + 1) & tokenizer->slot_mask;
    }
    return index;
}

static uint64_t pair_key(uint32_t left, uint32_t right) { return (uint64_t)left << PAIR_LEFT_SHIFT | right; }

static uint64_t edge_key(uint32_t node, unsigned char byte)
{
    return EDGE_KEY_TAG | (uint64_t)node << EDGE_NODE_SHIFT | byte;
}

/* The added token that begins at position and is the longest there: its byte count, 0 when none does. */
static size_t added_token_at(const struct text_tokenizer *tokenizer,
			     const unsigned char *text, size_t position,
			     size_t end, uint32_t *token_id_out)
{
    size_t matched = 0;
    for (size_t index = position, node = 0; index < end; index++) {
        const struct slot *const edge = &tokenizer->slots[slot_index(tokenizer, edge_key((uint32_t)node, text[index]))];
        if (edge->key == EMPTY_KEY) {
            break;
        }
        node = edge->value;
        if (edge->result != NO_INDEX) {
            matched = index + 1 - position;
            *token_id_out = edge->result;
        }
    }
    return matched;
}

static MUST_CHECK result_status emit(struct encoder *encoder, uint32_t token_id)
{
    if (encoder->token_count == encoder->token_id_capacity) {
	    return STATUS_INVALID_ARGUMENT;
    }
    encoder->token_ids[encoder->token_count++] = token_id;
    return STATUS_OK;
}

static void heap_push(uint64_t *heap, size_t *size, uint64_t entry)
{
    size_t index = (*size)++;
    for (; index > 0 && heap[(index - 1) / HEAP_ARITY] > entry; index = (index - 1) / HEAP_ARITY) {
        heap[index] = heap[(index - 1) / HEAP_ARITY];
    }
    heap[index] = entry;
}

static uint64_t heap_pop(uint64_t *heap, size_t *size)
{
    const uint64_t top = heap[0];
    const uint64_t last = heap[--(*size)];
    size_t index = 0;
    for (size_t child = 1; child < *size; index = child, child = HEAP_ARITY * child + 1) {
        child += child + 1 < *size && heap[child + 1] < heap[child] ? 1 : 0;
        if (heap[child] >= last) {
            break;
        }
        heap[index] = heap[child];
    }
    heap[index] = last;
    return top;
}

/* Puts the pair that begins at symbol position on the heap, if a merge joins it. */
static void push_pair(const struct text_tokenizer *tokenizer,
		      const struct symbol *symbols, uint32_t position,
		      uint64_t *heap, size_t *heap_size)
{
    const struct slot *const merge = &tokenizer->slots[slot_index(
        tokenizer, pair_key(symbols[position].token_id, symbols[symbols[position].next].token_id))];
    if (merge->key != EMPTY_KEY) {
        heap_push(heap, heap_size, (uint64_t)merge->value << HEAP_RANK_SHIFT | position);
    }
}

/* The BPE of one piece, its tokens emitted in order. */
static MUST_CHECK result_status encode_piece(struct encoder *encoder,
					     size_t begin, size_t end)
{
	const struct text_tokenizer *const tokenizer = encoder->tokenizer;
	const size_t byte_count = end - begin;
	if (byte_count == 1) {
		return emit(encoder,
			    tokenizer->byte_token_ids[encoder->text[begin]]);
	}
    if (byte_count > encoder->workspace_piece_bytes) {
        const size_t piece_bytes = byte_count > WORKSPACE_GROWTH * encoder->workspace_piece_bytes
                                 ? byte_count : WORKSPACE_GROWTH * encoder->workspace_piece_bytes;
        free(encoder->workspace);
	encoder->workspace =
		FAULT_POINT(tokenizer_memory) ?
			NULL :
			malloc(piece_bytes *
			       (HEAP_ENTRIES_PER_BYTE * sizeof(uint64_t) +
				sizeof(struct symbol)));
	encoder->workspace_piece_bytes = encoder->workspace == NULL ? 0 : piece_bytes;
        if (encoder->workspace == NULL) {
		return STATUS_OUT_OF_MEMORY;
	}
    }
    uint64_t *const heap = encoder->workspace;
    struct symbol *const symbols = (struct symbol *)(heap + HEAP_ENTRIES_PER_BYTE * encoder->workspace_piece_bytes);
    size_t heap_size = 0;
    for (size_t index = 0; index < byte_count; index++) {
        symbols[index] = (struct symbol){tokenizer->byte_token_ids[encoder->text[begin + index]],
                                         index == 0 ? NO_INDEX : (uint32_t)index - 1,
                                         index + 1 == byte_count ? NO_INDEX : (uint32_t)index + 1};
    }
    for (uint32_t position = 0; position + 1 < byte_count; position++) {
        push_pair(tokenizer, symbols, position, heap, &heap_size);
    }
    while (heap_size > 0) {
        const uint64_t entry = heap_pop(heap, &heap_size);
        const uint32_t position = (uint32_t)entry;
        struct symbol *const left = &symbols[position];
        if (left->token_id == NO_INDEX || left->next == NO_INDEX) {
            continue;
        }
        const struct slot *const merge = &tokenizer->slots[slot_index(
            tokenizer, pair_key(left->token_id, symbols[left->next].token_id))];
        if (merge->key == EMPTY_KEY || merge->value != (uint32_t)(entry >> HEAP_RANK_SHIFT)) {
            continue;                                         /* stale: the pair at position is another now */
        }
        left->token_id = merge->result;
        symbols[left->next].token_id = NO_INDEX;
        left->next = symbols[left->next].next;
        if (left->next != NO_INDEX) {
            symbols[left->next].previous = position;
            push_pair(tokenizer, symbols, position, heap, &heap_size);
        }
        if (left->previous != NO_INDEX) {
            push_pair(tokenizer, symbols, left->previous, heap, &heap_size);
        }
    }
    result_status status = STATUS_OK;
    for (uint32_t index = 0; status == STATUS_OK && index != NO_INDEX;
	 index = symbols[index].next) {
	    status = emit(encoder, symbols[index].token_id);
    }
    return status;
}

/* The third split of a piece of the second: its matches, and the text between them, each through BPE. */
static MUST_CHECK result_status encode_third_split(struct encoder *encoder,
						   size_t position, size_t end)
{
	result_status status = STATUS_OK;
	size_t gap_begin = position;
	while (status == STATUS_OK && position < end) {
		const size_t piece_end = match_end(
			encoder->tokenizer, encoder->text, position, end);
		if (piece_end == position) {
			position += character_at(encoder->tokenizer,
						 encoder->text, position, end)
					    .byte_count;
			continue;
		}
		status = gap_begin < position ?
				 encode_piece(encoder, gap_begin, position) :
				 STATUS_OK;
		status = status == STATUS_OK ?
				 encode_piece(encoder, position, piece_end) :
				 status;
		position = gap_begin = piece_end;
	}
	return status == STATUS_OK && gap_begin < end ?
		       encode_piece(encoder, gap_begin, end) :
		       status;
}

/* Text between two added tokens: the first split's groups of numbers, the second's runs, the third inside both. */
static MUST_CHECK result_status encode_segment(struct encoder *encoder,
					       size_t position, size_t end)
{
	const struct text_tokenizer *const tokenizer = encoder->tokenizer;
	result_status status = STATUS_OK;
	while (status == STATUS_OK && position < end) {
		const struct character first =
			character_at(tokenizer, encoder->text, position, end);
		size_t piece_end = position + first.byte_count;
		struct character next =
			character_at(tokenizer, encoder->text, piece_end, end);
		if ((first.classes & NUMBER) != 0) {
			for (unsigned digits = 1; digits < DIGITS_PER_GROUP &&
						  (next.classes & NUMBER) != 0;
			     digits++) {
				piece_end += next.byte_count;
				next = character_at(tokenizer, encoder->text,
						    piece_end, end);
			}
			status = encode_piece(encoder, position, piece_end);
		} else {
			while (next.byte_count > 0 &&
			       (next.classes & NUMBER) == 0 &&
			       (next.classes & SECOND_SPLIT) ==
				       (first.classes & SECOND_SPLIT)) {
				piece_end += next.byte_count;
				next = character_at(tokenizer, encoder->text,
						    piece_end, end);
			}
			status = encode_third_split(encoder, position,
						    piece_end);
		}
		position = piece_end;
	}
    return status;
}

/* Added-token recognition scans the NEW text, choosing each boundary before this shortcut. Both endpoints agree with
 * the old segment, and every byte through its end must match. The three pre-splits depend only on this complete
 * segment (including their end/whitespace lookahead); BPE depends only on each resulting piece. Thus its ids are
 * identical. No ordinary word boundary, byte offset or guessed lookbehind is treated as a safe restart point.
 * If a changed suffix completes/lengthens an earlier added token, the new scan changes the segment endpoints
 * and this check misses. Rewriting records in order is safe: a record is read before it is replaced. */
static MUST_CHECK result_status encode_cached_segment(struct encoder *encoder,
						      size_t begin, size_t end)
{
	struct tokenizer_cache *const cache = encoder->cache;
	const uint32_t index = encoder->segment_count++;
	const uint32_t token_begin = encoder->token_count;
	result_status status = STATUS_OK;
	const bool recorded = cache != NULL && index < cache->segment_capacity;
	const struct cached_segment previous =
		recorded && index < cache->segment_count ?
			cache->segments[index] :
			(struct cached_segment){ 0 };
	if (recorded && index < cache->segment_count &&
	    end <= encoder->common_bytes && previous.begin == begin &&
	    previous.end == end) {
		const uint32_t count =
			previous.token_end - previous.token_begin;
		if (count > encoder->token_id_capacity - encoder->token_count) {
			status = STATUS_INVALID_ARGUMENT;
		} else {
			memcpy(encoder->token_ids + encoder->token_count,
			       cache->tokens + previous.token_begin,
			       (size_t)count * sizeof(uint32_t));
			encoder->token_count += count;
		}
    } else {
        status = encode_segment(encoder, begin, end);
    }
    if (recorded) {
        cache->segments[index] = (struct cached_segment){(uint32_t)begin, (uint32_t)end,
                                                        token_begin, encoder->token_count};
    }
    return status;
}

static void cache_refresh(struct encoder *encoder, size_t length)
{
	struct tokenizer_cache *const cache = encoder->cache;
	if (cache == NULL) {
		return;
	}
    if (encoder->token_count > cache->token_limit) {
        cache->text_count = cache->segment_count = 0;
        return;
    }
    const size_t text_words = (length + sizeof(uint32_t) - 1u) / sizeof(uint32_t);
    const size_t text_capacity = text_words * sizeof(uint32_t);
    const size_t bytes = text_capacity + (size_t)encoder->token_count * sizeof(uint32_t);
    if (bytes > cache->storage_capacity) {
        free(cache->storage);
	cache->storage = FAULT_POINT(tokenizer_cache_memory) ? NULL :
							       malloc(bytes);
	cache->storage_capacity = cache->storage == NULL ? 0 : bytes;
    }
    cache->text_count = cache->segment_count = 0;
    if (cache->storage != NULL) {
        cache->tokens = (uint32_t *)cache->storage + text_words;
        if (length != 0) {
            memcpy(cache->storage, encoder->text, length);
        }
        memcpy(cache->tokens, encoder->token_ids, (size_t)encoder->token_count * sizeof(uint32_t));
        cache->text_count = (uint32_t)length;
        cache->segment_count = encoder->segment_count < cache->segment_capacity
                               ? encoder->segment_count : cache->segment_capacity;
    }
}

static MUST_CHECK result_status
encode_text(const struct text_tokenizer *tokenizer, const char *text,
	    uint64_t text_byte_count, uint32_t *token_ids_out,
	    uint32_t token_id_capacity, uint32_t *token_count_out,
	    struct tokenizer_cache *cache, const struct tokenizer_span *trusted,
	    uint32_t trusted_count, bool restricts_added_tokens)
{
	if (tokenizer == NULL || (text == NULL && text_byte_count > 0) ||
	    token_ids_out == NULL || token_count_out == NULL ||
	    text_byte_count > TOKENIZER_MAXIMUM_TEXT_BYTES ||
	    text_byte_count > (uint64_t)token_id_capacity *
				      tokenizer->longest_token_bytes ||
	    (trusted == NULL && trusted_count != 0)) {
		return STATUS_INVALID_ARGUMENT;
	}
    uint32_t previous_end = 0;
    for (uint32_t index = 0; index < trusted_count; index++) {
        if (trusted[index].begin < previous_end || trusted[index].begin >= trusted[index].end ||
            trusted[index].end > text_byte_count) {
		return STATUS_INVALID_ARGUMENT;
	}
        previous_end = trusted[index].end;
    }
    struct encoder encoder = { .tokenizer = tokenizer,
			       .text = (const unsigned char *)text,
			       .token_ids = token_ids_out,
			       .token_id_capacity = token_id_capacity,
			       .cache = cache,
			       .common_bytes = tokenizer_cache_common_prefix(
				       cache, text, text_byte_count) };
    size_t length = (size_t)text_byte_count, position = 0;
    bool is_valid = true;
    while (is_valid && position < length) {
        position += encoder.text[position] < ASCII_END ? 1
                  : sequence_length(encoder.text, position, length, &is_valid);
    }
    /* Chat rendering already validates UTF-8. Keeping its byte offsets exact avoids interpreting sanitized
     * replacement bytes as template output. Raw-prompt encoding retains its established replacement behavior. */
    if (restricts_added_tokens && !is_valid) {
	    return STATUS_INVALID_ARGUMENT;
    }
    if (cache != NULL) {
        const uintptr_t input = (uintptr_t)text, output = (uintptr_t)token_ids_out;
        const bool overlaps = length != 0 && token_id_capacity != 0 &&
            (input <= output ? output - input < length
                             : input - output < (uint64_t)token_id_capacity * sizeof(uint32_t));
        /* An aliased output may change text before refresh. Preserve full encode's behavior, never cache those
         * overwritten bytes with the original text's ids. Subtraction avoids overflowing an end address. */
        if (!is_valid || length > cache->text_limit || overlaps) {
            cache->text_count = cache->segment_count = 0;
            encoder.cache = NULL;
        }
    }
    unsigned char *const sanitized =
	    is_valid ? NULL :
	    FAULT_POINT(tokenizer_memory) ?
		       NULL :
		       malloc(length * REPLACEMENT_CHARACTER_BYTES);
    if (!is_valid && sanitized == NULL) {
	    return STATUS_OUT_OF_MEMORY;
    }
    if (sanitized != NULL) {                                            /* every ill-formed subpart as U+FFFD */
        size_t written = 0;
        for (size_t source = 0, sequence = 0; source < length; source += sequence) {
            sequence = sequence_length(encoder.text, source, length, &is_valid);
            memcpy(sanitized + written, is_valid ? (const void *)(encoder.text + source) : REPLACEMENT_CHARACTER,
                   is_valid ? sequence : REPLACEMENT_CHARACTER_BYTES);
            written += is_valid ? sequence : REPLACEMENT_CHARACTER_BYTES;
        }
        encoder.text = sanitized;
        length = written;
    }
    result_status status = STATUS_OK;
    size_t segment_begin = 0;
    uint32_t trusted_index = 0;
    for (position = 0; status == STATUS_OK && position < length;) {
	    while (trusted_index < trusted_count &&
		   trusted[trusted_index].end <= position) {
		    trusted_index++;
	    }
	    const bool is_trusted = !restricts_added_tokens ||
				    (trusted_index < trusted_count &&
				     trusted[trusted_index].begin <= position);
	    const size_t trusted_end = restricts_added_tokens && is_trusted ?
					       trusted[trusted_index].end :
					       length;
	    uint32_t added_token_id = NO_INDEX;
	    const size_t matched =
		    is_trusted && tokenizer->may_start_added_token
					    [encoder.text[position]] ?
			    added_token_at(tokenizer, encoder.text, position,
					   trusted_end, &added_token_id) :
			    0;
	    if (matched == 0) {
		    position++;
		    continue;
	    }
	    status = encode_cached_segment(&encoder, segment_begin, position);
	    status = status == STATUS_OK ? emit(&encoder, added_token_id) :
					   status;
	    position += matched;
	    segment_begin = position;
    }
    status = status == STATUS_OK ?
		     encode_cached_segment(&encoder, segment_begin, length) :
		     status;
    if (status == STATUS_OK) {
	    cache_refresh(&encoder, length);
    }
    free(encoder.workspace);
    free(sanitized);
    if (status == STATUS_OK) {
	    *token_count_out = encoder.token_count;
    }
    return status;
}

result_status tokenizer_encode(const struct text_tokenizer *tokenizer,
			       const char *text, uint64_t text_byte_count,
			       uint32_t *token_ids_out,
			       uint32_t token_id_capacity,
			       uint32_t *token_count_out)
{
    return encode_text(tokenizer, text, text_byte_count, token_ids_out, token_id_capacity, token_count_out,
                       NULL, NULL, 0, false);
}

result_status tokenizer_encode_spans(const struct text_tokenizer *tokenizer,
				     const char *text, uint64_t text_byte_count,
				     const struct tokenizer_span *trusted,
				     uint32_t trusted_count,
				     uint32_t *token_ids_out,
				     uint32_t token_id_capacity,
				     uint32_t *token_count_out)
{
    return encode_text(tokenizer, text, text_byte_count, token_ids_out, token_id_capacity, token_count_out,
                       NULL, trusted, trusted_count, true);
}

result_status tokenizer_cache_open(const struct text_tokenizer *tokenizer,
				   uint32_t text_byte_limit,
				   uint32_t token_limit,
				   struct tokenizer_cache **cache_out)
{
	if (tokenizer == NULL || cache_out == NULL || text_byte_limit == 0 ||
	    token_limit == 0 ||
	    text_byte_limit > TOKENIZER_MAXIMUM_TEXT_BYTES) {
		return STATUS_INVALID_ARGUMENT;
	}
	const uint32_t segments = token_limit < TOKENIZER_CACHE_SEGMENTS ?
					  token_limit + 1u :
					  TOKENIZER_CACHE_SEGMENTS;
	struct tokenizer_cache *const cache =
		FAULT_POINT(tokenizer_cache_memory) ?
			NULL :
			calloc(1,
			       sizeof(*cache) +
				       (size_t)segments *
					       sizeof(struct cached_segment));
	if (cache == NULL) {
		return STATUS_OUT_OF_MEMORY;
	}
    cache->tokenizer = tokenizer;
    cache->text_limit = text_byte_limit;
    cache->token_limit = token_limit;
    cache->segment_capacity = segments;
    *cache_out = cache;
    return STATUS_OK;
}

void tokenizer_cache_close(struct tokenizer_cache *cache)
{
    if (cache != NULL) {
        free(cache->storage);
        free(cache);
    }
}

uint32_t tokenizer_cache_common_prefix(const struct tokenizer_cache *cache,
				       const char *text,
				       uint64_t text_byte_count)
{
    const uint32_t most = cache == NULL || text == NULL ? 0
                          : text_byte_count < cache->text_count ? (uint32_t)text_byte_count : cache->text_count;
    const char *const previous = cache == NULL ? NULL : cache->storage;
    uint32_t common = 0;
    while (most - common >= PREFIX_COMPARE_BYTES &&
           memcmp(previous + common, text + common, PREFIX_COMPARE_BYTES) == 0) {
        common += PREFIX_COMPARE_BYTES;
    }
    while (common < most && previous[common] == text[common]) {
        common++;
    }
    return common;
}

result_status
tokenizer_encode_cached(struct tokenizer_cache *cache, const char *text,
			uint64_t text_byte_count, uint32_t *token_ids_out,
			uint32_t token_id_capacity, uint32_t *token_count_out)
{
    if (cache == NULL) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const result_status status = encode_text(cache->tokenizer, text,
					     text_byte_count, token_ids_out,
					     token_id_capacity, token_count_out,
					     cache, NULL, 0, false);
    if (status != STATUS_OK) {
	    cache->text_count = cache->segment_count = 0;
    }
    return status;
}

result_status tokenizer_encode_cached_spans(
	struct tokenizer_cache *cache, const char *text,
	uint64_t text_byte_count, const struct tokenizer_span *trusted,
	uint32_t trusted_count, uint32_t *token_ids_out,
	uint32_t token_id_capacity, uint32_t *token_count_out)
{
    if (cache == NULL) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const result_status status =
	    encode_text(cache->tokenizer, text, text_byte_count, token_ids_out,
			token_id_capacity, token_count_out, cache, trusted,
			trusted_count, true);
    if (status != STATUS_OK) {
	    cache->text_count = cache->segment_count = 0;
    }
    return status;
}

result_status tokenizer_token_bytes(const struct text_tokenizer *tokenizer,
				    uint32_t token_id, const char **bytes_out,
				    uint32_t *byte_count_out)
{
    if (tokenizer == NULL || bytes_out == NULL || byte_count_out == NULL || token_id >= tokenizer->vocabulary_size) {
	    return STATUS_INVALID_ARGUMENT;
    }
    const uint32_t begin = token_id == 0 ? 0 : tokenizer->token_byte_ends[token_id - 1];
    *bytes_out = tokenizer->token_bytes + begin;
    *byte_count_out = tokenizer->token_byte_ends[token_id] - begin;
    return STATUS_OK;
}

uint32_t tokenizer_vocabulary_size(const struct text_tokenizer *tokenizer)
{
    return tokenizer != NULL ? tokenizer->vocabulary_size : 0;
}

void tokenizer_close(struct text_tokenizer *tokenizer)
{
	free(tokenizer);
} /* one allocation */

/* The bytes of a token in tables whose token_byte_ends were checked. */
static const char *bytes_of(const struct text_tokenizer *tokenizer,
			    uint32_t token_id, uint32_t *byte_count_out)
{
    const char *bytes = NULL;
    const result_status status =
	    tokenizer_token_bytes(tokenizer, token_id, &bytes, byte_count_out);
    return status == STATUS_OK ? bytes : NULL;
}

/* The checks of the tables that need the hash table, which they fill: the merges, the bytes, the added tokens and
 * the classes. NULL when every rule holds, else the rule. */
static const char *fill_and_check(struct text_tokenizer *tokenizer,
				  const struct tokenizer_tables_header *header,
				  const uint32_t *added_token_ids)
{
    uint32_t left_count = 0, right_count = 0, result_count = 0, byte_token_count = 0;
    for (uint32_t rank = 0; rank < header->merge_count; rank++) {
	    const uint32_t *const merge =
		    &tokenizer->merges[(size_t)rank * TOKENIZER_MERGE_WORDS];
	    const char *const left = bytes_of(
		    tokenizer, merge[TOKENIZER_MERGE_LEFT], &left_count);
	    const char *const right = bytes_of(
		    tokenizer, merge[TOKENIZER_MERGE_RIGHT], &right_count);
	    const char *const result = bytes_of(
		    tokenizer, merge[TOKENIZER_MERGE_RESULT], &result_count);
	    if (left == NULL || right == NULL || result == NULL ||
		result_count != left_count + right_count ||
		memcmp(result, left, left_count) != 0 ||
		memcmp(result + left_count, right, right_count) != 0) {
		    return "a merge joins two tokens into the one of their bytes";
	    }
	    const uint64_t key = pair_key(merge[TOKENIZER_MERGE_LEFT],
					  merge[TOKENIZER_MERGE_RIGHT]);
	    struct slot *const slot =
		    &tokenizer->slots[slot_index(tokenizer, key)];
	    if (slot->key != EMPTY_KEY) {
		    return "no pair is merged twice";
	    }
	    *slot = (struct slot){ key, rank, merge[TOKENIZER_MERGE_RESULT] };
    }
    for (uint32_t byte = 0; byte < TOKENIZER_BYTE_VALUES; byte++) {
	    const char *const bytes = bytes_of(tokenizer,
					       tokenizer->byte_token_ids[byte],
					       &byte_token_count);
	    if (bytes == NULL || byte_token_count != 1 ||
		(unsigned char)bytes[0] != byte) {
		    return "the token of a byte is that byte";
	    }
    }
    for (uint32_t added_index = 0, node_count = 1; added_index < header->added_token_count; added_index++) {
        uint32_t byte_count = 0, node = 0;
        const unsigned char *const bytes = (const unsigned char *)bytes_of(tokenizer, added_token_ids[added_index],
                                                                           &byte_count);
        struct slot *edge = NULL;
        for (uint32_t index = 0; bytes != NULL && index < byte_count; index++, node = edge->value) {
            edge = &tokenizer->slots[slot_index(tokenizer, edge_key(node, bytes[index]))];
            if (edge->key == EMPTY_KEY) {
                *edge = (struct slot){edge_key(node, bytes[index]), node_count++, NO_INDEX};
            }
        }
        if (edge == NULL || edge->result != NO_INDEX) {
            return "an added token is a token of the vocabulary, listed once";
        }
        edge->result = added_token_ids[added_index];
        tokenizer->may_start_added_token[bytes[0]] = true;
    }
    for (size_t index = 0; index < TOKENIZER_CLASS_BLOCK_COUNT; index++) {
	    if (tokenizer->class_block_of[index] >= header->class_block_count) {
		    return "a block of classes is one of the tables";
	    }
    }
    const size_t class_count = (size_t)header->class_block_count *
			       TOKENIZER_CLASS_BLOCK_CODE_POINTS;
    for (size_t index = 0; index < class_count; index++) {
        const unsigned classes = tokenizer->class_blocks[index];
	if ((classes & ~(unsigned)TOKENIZER_CLASS_ALL) != 0 ||
	    ((classes & NUMBER) != 0 && (classes & SECOND_SPLIT) != 0)) {
		return "a code point has known classes, never a number of the second split's range";
	}
    }
    return NULL;
}

result_status tokenizer_open(const void *tables, uint64_t table_byte_count,
			     struct text_tokenizer **tokenizer_out)
{
	if (tables == NULL || tokenizer_out == NULL ||
	    (uintptr_t)tables % TOKENIZER_TABLES_ALIGNMENT != 0) {
		return STATUS_INVALID_ARGUMENT;
	}
	const struct tokenizer_tables_header *const header = tables;
	const char *rule = table_byte_count < sizeof(*header) ?
				   "the tables hold their header" :
				   NULL;
	for (size_t index = 0;
	     rule == NULL && index < TOKENIZER_TABLES_RESERVED_WORDS; index++) {
		rule = header->reserved_zero[index] != 0 ?
			       "the reserved words are zero" :
			       NULL;
	}
	rule = rule != NULL ? rule :
	       header->format_version != TOKENIZER_TABLES_FORMAT_VERSION ||
			       memcmp(header->magic, TOKENIZER_TABLES_MAGIC,
				      TOKENIZER_TABLES_MAGIC_BYTES) != 0 ?
			      "the magic and the version are the header's" :
			      NULL;
	rule = rule != NULL ? rule :
	       header->vocabulary_size == 0 ||
			       header->vocabulary_size >
				       TOKENIZER_MAXIMUM_VOCABULARY ||
			       header->merge_count > header->vocabulary_size ||
			       header->added_token_count >
				       header->vocabulary_size ||
			       header->class_block_count == 0 ||
			       header->class_block_count >
				       TOKENIZER_CLASS_BLOCK_COUNT ?
			      "every count is within its limits" :
			      NULL;
	const uint64_t word_count =
		rule != NULL ? 0 :
			       (uint64_t)header->vocabulary_size +
				       header->added_token_count +
				       (uint64_t)header->merge_count *
					       TOKENIZER_MERGE_WORDS +
				       TOKENIZER_BYTE_VALUES;
	const uint64_t class_offset =
		sizeof(*header) + word_count * sizeof(uint32_t);
	const uint64_t bytes_offset =
		class_offset + TOKENIZER_CLASS_BLOCK_COUNT * sizeof(uint16_t) +
		(uint64_t)(rule != NULL ? 0 : header->class_block_count) *
			TOKENIZER_CLASS_BLOCK_CODE_POINTS;
	rule = rule != NULL ? rule :
	       bytes_offset + header->token_byte_count != table_byte_count ?
			      "the tables are as long as their counts say" :
			      NULL;
	if (rule != NULL) {
		LOG_ERROR("the tokenizer's tables do not open, broken rule: %s",
			  rule);
		return STATUS_CORRUPT_PACK;
	}
    const unsigned char *const base = tables;
    const void *const token_byte_ends = base + sizeof(*header);
    const void *const merges = (const uint32_t *)token_byte_ends + header->vocabulary_size;
    const void *const added_token_ids =
	    (const uint32_t *)merges +
	    (size_t)header->merge_count * TOKENIZER_MERGE_WORDS;
    const void *const byte_token_ids = (const uint32_t *)added_token_ids + header->added_token_count;
    const void *const class_block_of = base + class_offset;
    const uint32_t *const ends = token_byte_ends;
    uint64_t added_byte_count = 0;
    uint32_t longest_token_bytes = 0;
    for (uint32_t token_id = 0; rule == NULL && token_id < header->vocabulary_size; token_id++) {
        const uint32_t begin = token_id == 0 ? 0 : ends[token_id - 1];
	rule = ends[token_id] <= begin ||
			       ends[token_id] - begin >
				       TOKENIZER_MAXIMUM_TOKEN_BYTES ||
			       ends[token_id] > header->token_byte_count ?
		       "every token has one to the most bytes, in order" :
		       NULL;
	longest_token_bytes = rule == NULL && ends[token_id] - begin > longest_token_bytes ? ends[token_id] - begin
                                                                                           : longest_token_bytes;
    }
    rule = rule != NULL ? rule : ends[header->vocabulary_size - 1] != header->token_byte_count
           ? "the last token ends with the bytes" : NULL;
    for (uint32_t index = 0; rule == NULL && index < header->added_token_count; index++) {
        const uint32_t token_id = ((const uint32_t *)added_token_ids)[index];
        rule = token_id >= header->vocabulary_size ? "an added token is a token of the vocabulary, listed once" : NULL;
        added_byte_count += rule == NULL ? ends[token_id] - (token_id == 0 ? 0 : ends[token_id - 1]) : 0;
	rule = rule == NULL && added_byte_count >
				       TOKENIZER_MAXIMUM_ADDED_TOKEN_BYTES ?
		       "every count is within its limits" :
		       rule;
    }
    uint32_t slot_bits = SMALLEST_SLOT_BITS;
    while (((uint64_t)1 << slot_bits) < SLOTS_PER_KEY * (header->merge_count + added_byte_count)) {
        slot_bits++;
    }
    struct text_tokenizer *const tokenizer =
	    rule != NULL || FAULT_POINT(tokenizer_memory) ?
		    NULL :
		    malloc(sizeof(struct text_tokenizer) +
			   ((size_t)1 << slot_bits) * sizeof(struct slot));
    if (tokenizer != NULL) {
	    *tokenizer = (struct text_tokenizer){
		    .token_byte_ends = token_byte_ends,
		    .merges = merges,
		    .byte_token_ids = byte_token_ids,
		    .class_block_of = class_block_of,
		    .class_blocks =
			    (const uint8_t *)class_block_of +
			    TOKENIZER_CLASS_BLOCK_COUNT * sizeof(uint16_t),
		    .token_bytes = (const char *)base + bytes_offset,
		    .vocabulary_size = header->vocabulary_size,
		    .longest_token_bytes = longest_token_bytes,
		    .slot_shift = KEY_BITS - slot_bits,
		    .slot_mask = ((uint64_t)1 << slot_bits) - 1
	    };
	    memset(tokenizer->slots, UINT8_MAX,
		   ((size_t)1 << slot_bits) *
			   sizeof(struct slot)); /* every key empty */
	    rule = fill_and_check(tokenizer, header, added_token_ids);
    }
    if (tokenizer == NULL || rule != NULL) {
        if (rule != NULL) {
		LOG_ERROR("the tokenizer's tables do not open, broken rule: %s",
			  rule);
	}
        free(tokenizer);
	return rule != NULL ? STATUS_CORRUPT_PACK : STATUS_OUT_OF_MEMORY;
    }
    tokenizer->ascii_classes =
	    tokenizer->class_blocks + /* once every block is known to be in the tables */
	    (size_t)tokenizer->class_block_of[0] *
		    TOKENIZER_CLASS_BLOCK_CODE_POINTS;
    *tokenizer_out = tokenizer;
    return STATUS_OK;
}
