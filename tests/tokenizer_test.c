/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * tokenizer_test: source/tokenizer.c against tokenizer.h, on tables the test builds itself. The model's own tables
 * are front_end_golden_test's.
 *
 * The tables hold the 256 byte tokens, the tokens of hand_merges and of merges drawn at random over three letters,
 * a few added tokens, and classes for some blocks of code points (ASCII and Latin-1 about as Unicode has them, the
 * combining marks, a range of the second split). Every rule of open is broken by one change to valid tables. Each
 * alternative of the splits is shown by a merge that would join across the boundary if it were not there. The heap
 * of BPE is held to the plain algorithm (the pair of lowest rank, the leftmost of equals, again) on random strings.
 * Random bytes go through encode and back to their text with each ill-formed subpart as U+FFFD, written here a
 * second way. Then the long piece, the capacity, the faults and threads, and tables whose added tokens hold more
 * bytes than open allows.
 */
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "anemone/hooks.h"
#include "anemone/status.h"
#include "anemone/tokenizer.h"

#define TEST_TIME_LIMIT_SECONDS 600
#define BYTE_TOKENS TOKENIZER_BYTE_VALUES
#define MOST_TOKENS 1024
#define MOST_TOKEN_BYTES 16
#define MOST_MERGES 512
#define MOST_ADDED 8
#define NO_TOKEN UINT32_MAX
#define TABLES_ALIGNMENT TOKENIZER_TABLES_ALIGNMENT
#define BLOCK TOKENIZER_CLASS_BLOCK_CODE_POINTS
#define CODE_POINT_LIMIT 0x110000u
#define MOST_CASE_TOKENS 8
#define RANDOM_SEED 0x9E3779B97F4A7C15ull
#define DRAWN_MERGE_ATTEMPTS 600
#define DRAWN_TOKEN_MOST_BYTES 8
#define DRAWN_STRINGS 4000
#define DRAWN_STRING_MOST_BYTES 64
#define FUZZ_TEXTS 20000
#define FUZZ_TEXT_MOST_PIECES 40
#define CACHE_TEXT_BYTES 2048u
#define CACHE_TOKEN_IDS (CACHE_TEXT_BYTES * REPLACEMENT_BYTES)
#define CACHE_RANDOM_EDITS 20000u
#define CACHE_RESET_PERIOD 64u
#define CACHE_EDIT_BYTES 16u
#define CACHE_SEGMENT_OVERFLOW (TOKENIZER_CACHE_SEGMENTS + 1u)
#define REPLACEMENT_BYTES 3                       /* U+FFFD in UTF-8 */
#define PAIR 2                                    /* the two tokens a merge joins */
#define LEFT TOKENIZER_MERGE_LEFT
#define RIGHT TOKENIZER_MERGE_RIGHT
#define RESULT TOKENIZER_MERGE_RESULT
#define TOKENS_OF_ABC 2                           /* "a" and "bc" */
#define CONTINUATION_FIRST 0x80
#define CONTINUATION_LAST 0xBF
#define LONG_RUN_BYTES 1000000
#define LONG_RUN_TOKEN_BYTES 4                    /* the run of a is merged into tokens of aaaa */
#define THREAD_COUNT 4
#define FAULT_ON_FIRST_CALL 1
#define SPLITMIX_INCREMENT 0x9E3779B97F4A7C15ull
#define SPLITMIX_FIRST_MULTIPLIER 0xBF58476D1CE4E5B9ull
#define SPLITMIX_SECOND_MULTIPLIER 0x94D049BB133111EBull
#define SPLITMIX_FIRST_SHIFT 30
#define SPLITMIX_SECOND_SHIFT 27
#define SPLITMIX_THIRD_SHIFT 31
#define LETTER TOKENIZER_CLASS_LETTER
#define MARK TOKENIZER_CLASS_MARK
#define NUMBER TOKENIZER_CLASS_NUMBER
#define SYMBOL TOKENIZER_CLASS_PUNCTUATION_OR_SYMBOL
#define SPACE TOKENIZER_CLASS_WHITESPACE
#define SECOND TOKENIZER_CLASS_SECOND_SPLIT
#define CORRUPT STATUS_CORRUPT_PACK
#define INVALID STATUS_INVALID_ARGUMENT
#define OUT_OF_MEMORY STATUS_OUT_OF_MEMORY

struct class_range {
    uint32_t first;
    uint32_t last;
    uint8_t classes;
};

/* About Unicode's classes, enough for the cases below; any code point outside is of no class. */
static const struct class_range class_ranges[] = {
    {0x09, 0x0D, SPACE}, {0x20, 0x20, SPACE}, {0x21, 0x2F, SYMBOL}, {0x30, 0x39, NUMBER}, {0x3A, 0x40, SYMBOL},
    {0x41, 0x5A, LETTER}, {0x5B, 0x60, SYMBOL}, {0x61, 0x7A, LETTER}, {0x7B, 0x7E, SYMBOL}, {0x85, 0x85, SPACE},
    {0xA0, 0xA0, SPACE}, {0xA1, 0xB1, SYMBOL}, {0xB2, 0xB3, NUMBER}, {0xB4, 0xBF, SYMBOL}, {0xC0, 0xD6, LETTER},
    {0xD7, 0xD7, SYMBOL}, {0xD8, 0xF6, LETTER}, {0xF7, 0xF7, SYMBOL}, {0xF8, 0xFF, LETTER}, {0x300, 0x36F, MARK},
    {0x370, 0x3FF, LETTER}, {0x3000, 0x3000, SPACE}, {0x3041, 0x30FA, LETTER | SECOND},
    {0x30FB, 0x30FB, SYMBOL | SECOND}, {0x30FC, 0x30FF, LETTER | SECOND}, {0x4E00, 0x9FA5, LETTER | SECOND},
    {0x1F600, 0x1F64F, SYMBOL}};

/* The hand merges, by rank. Some join what must never stand in one piece: "a " and "s " (a word and a space),
 * "3" "4" (two groups of digits), "\xAC" "a" (the end of a character of the second split and a letter). */
static const char *const hand_merges[][PAIR] = {
    {"!", "a"}, {"b", "c"}, {"a", "b"}, {"a", "a"}, {"aa", "aa"}, {"1", "2"}, {"3", "4"}, {"12", "3"}, {"4", "5"},
    {" ", " "}, {" ", "b"}, {"a", " "}, {"'", "s"}, {"s", " "}, {"\n", "\n"}, {" ", "\n"}, {"\xEF", "\xBF"},
    {"\xEF\xBF", "\xBD"}, {"e", "\xCC"}, {"\xAC", "a"}, {"\x01", "\x02"}, {"\xA0", "x"}, {"!", "\r"}, {"x", "y"}};
static const char *const added_texts[] = {"<|a|>", "<|a|>b", "<|begin|>", "\xEF\xBD\x9C\x44\xEF\xBD\x9C"};
static const char drawn_letters[] = "pqr";

struct lead_range {                               /* Unicode's table 3-7, by lead byte */
    uint8_t first_lead;
    uint8_t last_lead;
    uint8_t bytes;
    uint8_t low;                                  /* of the second byte */
    uint8_t high;
};

struct utf8_case {
    const char *bytes;
    uint32_t length;
    bool is_valid;
};

static const struct lead_range lead_ranges[] = {
    {0x00, 0x7F, 1, 0, 0}, {0xC2, 0xDF, 2, 0x80, 0xBF}, {0xE0, 0xE0, 3, 0xA0, 0xBF}, {0xE1, 0xEC, 3, 0x80, 0xBF},
    {0xED, 0xED, 3, 0x80, 0x9F}, {0xEE, 0xEF, 3, 0x80, 0xBF}, {0xF0, 0xF0, 4, 0x90, 0xBF}, {0xF1, 0xF3, 4, 0x80, 0xBF},
    {0xF4, 0xF4, 4, 0x80, 0x8F}};
static const unsigned char replacement_character[REPLACEMENT_BYTES] = {0xEF, 0xBF, 0xBD};
static const struct utf8_case utf8_cases[] = {
    {"a", 1, true}, {"\xC2\x80", 2, true}, {"\xE0\xA0\x80", 3, true}, {"\xF0\x9F\x98\x80", 4, true},
    {"\xF4\x8F\xBF\xBF", 4, true}, {"\x80", 1, false}, {"\xC1\xBF", 1, false}, {"\xE0\x80\x80", 1, false},
    {"\xE0\xA0", 2, false}, {"\xED\xA0\x80", 1, false}, {"\xF0\x8F", 1, false}, {"\xF4\x90\x80\x80", 1, false},
    {"\xF5\x80", 1, false}, {"\xF0\x9F\x98", 3, false}, {"\xE1\x80\x41", 2, false}};

struct encoding_case {
    const char *what;
    const char *text;
    const char *tokens[MOST_CASE_TOKENS];         /* NULL after the last */
};

static const struct encoding_case encoding_cases[] = {
    {"nothing gives no token", "", {NULL}},
    {"the pair of lowest rank goes first", "abc", {"a", "bc", NULL}},
    {"of equal ranks the leftmost goes first", "aaa", {"aa", "a", NULL}},
    {"a merged pair merges again", "aaaa", {"aaaa", NULL}},
    {"numbers go in groups of up to three", "12345", {"123", "45", NULL}},
    {"punctuation before ASCII letters is one piece", "!ab", {"!a", "b", NULL}},
    {"a space goes with the word after it", "a b", {"a", " b", NULL}},
    {"a run of spaces gives its last one to the word", "a  b", {"a", " ", " b", NULL}},
    {"a run of spaces at the end stays whole", "a  ", {"a", "  ", NULL}},
    {"spaces up to the last line break", "s \n\nb", {"s", " ", "\n\n", "b", NULL}},
    {"the apostrophe's letters", "it's", {"i", "t", "'s", NULL}},
    {"a mark stays with its letter", "e\xCC\x81", {"e\xCC", "\x81", NULL}},
    {"line breaks stay with punctuation", "!\r\n", {"!\r", "\n", NULL}},
    {"a space of another kind goes with the word", "\xC2\xA0x", {"\xC2", "\xA0x", NULL}},
    {"a character of no class stands alone before another", "\x01\x02\x61", {"\x01", "\x02", "a", NULL}},
    {"the second split's runs are pieces of their own", "\xE6\x97\xA5\xE6\x9C\xAC\x61",
     {"\xE6", "\x97", "\xA5", "\xE6", "\x9C", "\xAC", "a", NULL}},
    {"an ill-formed byte is U+FFFD", "x\xFFy", {"x", "\xEF\xBF\xBD", "y", NULL}},
    {"an added token stands alone", "x<|a|>y", {"x", "<|a|>", "y", NULL}},
    {"the longest added token is found", "<|a|>b", {"<|a|>b", NULL}},
    {"the leftmost added token is found", "<<|a|>", {"<", "<|a|>", NULL}},
    {"half an added token is text", "<|a|", {"<", "|", "a", "|", NULL}},
    {"an added token may begin with any character", "\xEF\xBD\x9C\x44\xEF\xBD\x9C",
     {"\xEF\xBD\x9C\x44\xEF\xBD\x9C", NULL}}};

struct vocabulary {
    char tokens[MOST_TOKENS][MOST_TOKEN_BYTES];
    uint32_t lengths[MOST_TOKENS];
    uint32_t count;
    uint32_t merges[MOST_MERGES][TOKENIZER_MERGE_WORDS];
    uint32_t merge_count;
    uint32_t added[MOST_ADDED];
    uint32_t added_count;
};

struct built_tables {
    void *bytes;
    size_t byte_count;
};

struct thread_work {
	const struct text_tokenizer *tokenizer;
	const char *text;
	size_t text_bytes;
	uint32_t *ids;
	uint32_t count;
	result_status status;
};

static int failures = 0;
static uint64_t random_state = RANDOM_SEED;
static struct vocabulary vocabulary;

static void expect(bool condition, const char *what)
{
    if (!condition) {
        (void)fprintf(stderr, "FAILED: %s\n", what);
        failures++;
    }
}

static _Noreturn void give_up(const char *what)
{
    (void)fprintf(stderr, "FAILED: %s\n", what);
    exit(EXIT_FAILURE);
}

static uint64_t next_random(void)
{
    uint64_t value = (random_state += SPLITMIX_INCREMENT);
    value = (value ^ (value >> SPLITMIX_FIRST_SHIFT)) * SPLITMIX_FIRST_MULTIPLIER;
    value = (value ^ (value >> SPLITMIX_SECOND_SHIFT)) * SPLITMIX_SECOND_MULTIPLIER;
    return value ^ (value >> SPLITMIX_THIRD_SHIFT);
}

static uint32_t token_of(const char *bytes, size_t length)
{
    for (uint32_t token = 0; token < vocabulary.count; token++) {
        if (vocabulary.lengths[token] == length && memcmp(vocabulary.tokens[token], bytes, length) == 0) {
            return token;
        }
    }
    return NO_TOKEN;
}

/* The token of these bytes, made when there is none. */
static uint32_t add_token(const char *bytes, size_t length)
{
    const uint32_t found = token_of(bytes, length);
    if (found != NO_TOKEN) {
        return found;
    }
    if (vocabulary.count == MOST_TOKENS || length == 0 || length > MOST_TOKEN_BYTES) {
        give_up("the test's vocabulary has room for its tokens");
    }
    memcpy(vocabulary.tokens[vocabulary.count], bytes, length);
    vocabulary.lengths[vocabulary.count] = (uint32_t)length;
    return vocabulary.count++;
}

/* A merge of two tokens, or false when that pair is merged already or their bytes would be too many. */
static bool add_merge(uint32_t left, uint32_t right)
{
    char joined[MOST_TOKEN_BYTES] = {0};
    const size_t length = (size_t)vocabulary.lengths[left] + vocabulary.lengths[right];
    for (uint32_t rank = 0; rank < vocabulary.merge_count; rank++) {
        if (vocabulary.merges[rank][LEFT] == left && vocabulary.merges[rank][RIGHT] == right) {
            return false;
        }
    }
    if (length > DRAWN_TOKEN_MOST_BYTES || vocabulary.merge_count == MOST_MERGES) {
        return false;
    }
    memcpy(joined, vocabulary.tokens[left], vocabulary.lengths[left]);
    memcpy(joined + vocabulary.lengths[left], vocabulary.tokens[right], vocabulary.lengths[right]);
    const uint32_t result = add_token(joined, length);
    const uint32_t merge[TOKENIZER_MERGE_WORDS] = { left, right, result };
    memcpy(vocabulary.merges[vocabulary.merge_count++], merge, sizeof(merge));
    return true;
}

static uint8_t classes_of(uint32_t code_point)
{
    for (size_t index = 0; index < sizeof(class_ranges) / sizeof(class_ranges[0]); index++) {
        if (code_point >= class_ranges[index].first && code_point <= class_ranges[index].last) {
            return class_ranges[index].classes;
        }
    }
    return 0;
}

static void make_vocabulary(void)
{
    for (uint32_t byte = 0; byte < BYTE_TOKENS; byte++) {
        const char bytes[1] = {(char)byte};
        vocabulary.lengths[vocabulary.count] = 1;
        memcpy(vocabulary.tokens[vocabulary.count++], bytes, 1);
    }
    for (size_t index = 0; index < sizeof(hand_merges) / sizeof(hand_merges[0]); index++) {
        const bool is_new = add_merge(add_token(hand_merges[index][0], strlen(hand_merges[index][0])),
                                      add_token(hand_merges[index][1], strlen(hand_merges[index][1])));
        expect(is_new, "every hand merge is a pair of its own");
    }
    uint32_t drawn[MOST_TOKENS] = {0}, drawn_count = 0;
    for (const char *letter = drawn_letters; *letter != '\0'; letter++) {
        drawn[drawn_count++] = (uint32_t)(unsigned char)*letter;
    }
    for (int attempt = 0; attempt < DRAWN_MERGE_ATTEMPTS; attempt++) {     /* two different pairs may make one token */
        const uint32_t left = drawn[next_random() % drawn_count], right = drawn[next_random() % drawn_count];
        const uint32_t count_before = vocabulary.count;
        if (add_merge(left, right) && vocabulary.count > count_before) {
            drawn[drawn_count++] = vocabulary.count - 1;
        }
    }
    for (size_t index = 0; index < sizeof(added_texts) / sizeof(added_texts[0]); index++) {
        vocabulary.added[vocabulary.added_count++] = add_token(added_texts[index], strlen(added_texts[index]));
    }
}

/* The tables of the vocabulary, as tokenizer.h lays them out. */
static struct built_tables build_tables(void)
{
	static uint8_t blocks[TOKENIZER_CLASS_BLOCK_COUNT][BLOCK];
	uint16_t block_of[TOKENIZER_CLASS_BLOCK_COUNT] = { 0 };
	uint32_t block_count = 0, token_byte_count = 0;
	for (uint32_t block = 0; block < TOKENIZER_CLASS_BLOCK_COUNT; block++) {
		uint8_t classes[BLOCK] = { 0 };
		for (uint32_t offset = 0; offset < BLOCK; offset++) {
			classes[offset] = classes_of(block * BLOCK + offset);
		}
		uint32_t same = 0;
		while (same < block_count &&
		       memcmp(blocks[same], classes, BLOCK) != 0) {
			same++;
		}
		memcpy(blocks[same], classes, BLOCK);
		block_count += same == block_count ? 1 : 0;
		block_of[block] = (uint16_t)same;
	}
    for (uint32_t token = 0; token < vocabulary.count; token++) {
        token_byte_count += vocabulary.lengths[token];
    }
    struct tokenizer_tables_header header = {
	    .format_version = TOKENIZER_TABLES_FORMAT_VERSION,
	    .vocabulary_size = vocabulary.count,
	    .merge_count = vocabulary.merge_count,
	    .added_token_count = vocabulary.added_count,
	    .class_block_count = block_count,
	    .token_byte_count = token_byte_count
    };
    memcpy(header.magic, TOKENIZER_TABLES_MAGIC, TOKENIZER_TABLES_MAGIC_BYTES);
    const size_t byte_count =
	    sizeof(header) +
	    sizeof(uint32_t) *
		    ((size_t)vocabulary.count + BYTE_TOKENS +
		     (size_t)vocabulary.merge_count * TOKENIZER_MERGE_WORDS +
		     vocabulary.added_count) +
	    sizeof(block_of) + (size_t)block_count * BLOCK + token_byte_count;
    unsigned char *const bytes = aligned_alloc(TABLES_ALIGNMENT,
                                               (byte_count / TABLES_ALIGNMENT + 1) * TABLES_ALIGNMENT);
    if (bytes == NULL) {
        give_up("memory for the tables");
    }
    unsigned char *cursor = bytes;
    memcpy(cursor, &header, sizeof(header));
    cursor += sizeof(header);
    for (uint32_t token = 0, end = 0; token < vocabulary.count; token++, cursor += sizeof(end)) {
        end += vocabulary.lengths[token];
        memcpy(cursor, &end, sizeof(end));
    }
    memcpy(cursor, vocabulary.merges, (size_t)vocabulary.merge_count * sizeof(vocabulary.merges[0]));
    cursor += (size_t)vocabulary.merge_count * sizeof(vocabulary.merges[0]);
    memcpy(cursor, vocabulary.added, vocabulary.added_count * sizeof(uint32_t));
    cursor += vocabulary.added_count * sizeof(uint32_t);
    for (uint32_t byte = 0; byte < BYTE_TOKENS; byte++, cursor += sizeof(byte)) {
        memcpy(cursor, &byte, sizeof(byte));      /* the token of byte b is the token b */
    }
    memcpy(cursor, block_of, sizeof(block_of));
    cursor += sizeof(block_of);
    memcpy(cursor, blocks, (size_t)block_count * BLOCK);
    cursor += (size_t)block_count * BLOCK;
    for (uint32_t token = 0; token < vocabulary.count; token++) {
        memcpy(cursor, vocabulary.tokens[token], vocabulary.lengths[token]);
        cursor += vocabulary.lengths[token];
    }
    return (struct built_tables){bytes, byte_count};
}

/* A word of the tables: the header's, or of the arrays behind it, counted in uint32 from the first byte. */
static uint32_t *word_at(const struct built_tables *tables, size_t byte_offset)
{
    void *const word = (unsigned char *)tables->bytes + byte_offset;
    return word;
}

static size_t merges_offset(void)
{
	return sizeof(struct tokenizer_tables_header) +
	       vocabulary.count * sizeof(uint32_t);
}
static size_t added_offset(void)
{
	return merges_offset() + (size_t)vocabulary.merge_count *
					 TOKENIZER_MERGE_WORDS *
					 sizeof(uint32_t);
}
static size_t byte_tokens_offset(void) { return added_offset() + vocabulary.added_count * sizeof(uint32_t); }
static size_t class_offset(void) { return byte_tokens_offset() + BYTE_TOKENS * sizeof(uint32_t); }

/* One change to valid tables, and the status open must give for it. */
static void expect_open(const struct built_tables *tables, size_t byte_count,
			result_status expected, const char *what)
{
	struct text_tokenizer *tokenizer = NULL;
	const result_status status =
		tokenizer_open(tables->bytes, byte_count, &tokenizer);
	expect(status == expected &&
		       (status == STATUS_OK) == (tokenizer != NULL),
	       what);
	tokenizer_close(tokenizer);
}

/* The word at byte_offset set to value, open refused, the word put back. */
static void expect_refused_with(const struct built_tables *tables, size_t byte_offset, uint32_t value, const char *what)
{
    uint32_t *const word = word_at(tables, byte_offset);
    const uint32_t kept = *word;
    *word = value;
    expect_open(tables, tables->byte_count, CORRUPT, what);
    *word = kept;
}

static void test_open(const struct built_tables *tables)
{
	struct text_tokenizer *tokenizer = NULL;
	const struct tokenizer_tables_header *const header = tables->bytes;
	const size_t ends = sizeof(*header),
		     last_end =
			     ends + (vocabulary.count - 1) * sizeof(uint32_t);
	const uint32_t vocabulary_size = header->vocabulary_size;
	expect_open(tables, tables->byte_count, STATUS_OK, "valid tables open");
	expect(tokenizer_open(NULL, tables->byte_count, &tokenizer) ==
			       STATUS_INVALID_ARGUMENT &&
		       tokenizer_open(tables->bytes, tables->byte_count,
				      NULL) == STATUS_INVALID_ARGUMENT &&
		       tokenizer_open((unsigned char *)tables->bytes + 1,
				      tables->byte_count - 1,
				      &tokenizer) == STATUS_INVALID_ARGUMENT &&
		       tokenizer == NULL,
	       "a NULL argument or misaligned tables");
	expect_open(tables, sizeof(*header) - 1, CORRUPT,
		    "tables shorter than their header");
	expect_open(tables, tables->byte_count - 1, CORRUPT,
		    "tables a byte short of their counts");
	expect_open(tables, tables->byte_count + 1, CORRUPT,
		    "tables a byte longer than their counts");
	expect_refused_with(tables, 0, header->magic[0] ^ 1u, "another magic");
	expect_refused_with(
		tables,
		offsetof(struct tokenizer_tables_header, format_version),
		TOKENIZER_TABLES_FORMAT_VERSION + 1, "another version");
	expect_refused_with(tables,
			    offsetof(struct tokenizer_tables_header,
				     reserved_zero) +
				    sizeof(uint32_t),
			    1, "a reserved word that is not zero");
	expect_refused_with(tables,
			    offsetof(struct tokenizer_tables_header,
				     vocabulary_size),
			    0, "no token");
	expect_refused_with(
		tables,
		offsetof(struct tokenizer_tables_header, vocabulary_size),
		TOKENIZER_MAXIMUM_VOCABULARY + 1, "too many tokens");
	expect_refused_with(
		tables, offsetof(struct tokenizer_tables_header, merge_count),
		vocabulary_size + 1, "more merges than tokens");
	expect_refused_with(
		tables,
		offsetof(struct tokenizer_tables_header, added_token_count),
		vocabulary_size + 1, "more added tokens than tokens");
	expect_refused_with(tables,
			    offsetof(struct tokenizer_tables_header,
				     class_block_count),
			    0, "no block");
	expect_refused_with(tables,
			    offsetof(struct tokenizer_tables_header,
				     class_block_count),
			    TOKENIZER_CLASS_BLOCK_COUNT + 1, "too many blocks");
	expect_refused_with(tables, ends + sizeof(uint32_t),
			    *word_at(tables, ends), "a token of no byte");
	expect_refused_with(tables, ends, TOKENIZER_MAXIMUM_TOKEN_BYTES + 1,
			    "a token of too many bytes");
	expect_refused_with(tables, last_end, *word_at(tables, last_end) - 1,
			    "a last token short of the bytes");
	expect_refused_with(tables,
			    byte_tokens_offset() + 'x' * sizeof(uint32_t), 'y',
			    "a byte's token is another");
	expect_refused_with(tables, byte_tokens_offset(), vocabulary_size,
			    "a byte's token out of range");
	expect_refused_with(tables, merges_offset() + RESULT * sizeof(uint32_t),
			    'a', "a merge into the wrong token");
	expect_refused_with(tables, merges_offset(), vocabulary_size,
			    "a merge of a token out of range");
	const size_t second_merge =
		merges_offset() + TOKENIZER_MERGE_WORDS * sizeof(uint32_t);
	uint32_t kept_merge[TOKENIZER_MERGE_WORDS] = { 0 };
	memcpy(kept_merge, word_at(tables, second_merge), sizeof(kept_merge));
	memcpy(word_at(tables, second_merge), word_at(tables, merges_offset()),
	       sizeof(kept_merge));
	expect_open(tables, tables->byte_count, CORRUPT, "a pair merged twice");
	memcpy(word_at(tables, second_merge), kept_merge, sizeof(kept_merge));
	expect_refused_with(tables, added_offset(), vocabulary_size,
			    "an added token out of range");
	expect_refused_with(tables, added_offset() + sizeof(uint32_t),
			    vocabulary.added[0], "an added token twice");
	const uint32_t first_block_words = *word_at(tables, class_offset());
	expect_refused_with(tables, class_offset(),
			    first_block_words + header->class_block_count,
			    "a block out of range");
	unsigned char *const first_class =
		(unsigned char *)tables->bytes + class_offset() +
		TOKENIZER_CLASS_BLOCK_COUNT * sizeof(uint16_t);
	const unsigned char kept_class = *first_class;
	*first_class = TOKENIZER_CLASS_ALL + 1;
	expect_open(tables, tables->byte_count, CORRUPT,
		    "a class bit no one knows");
	*first_class = NUMBER | SECOND;
	expect_open(tables, tables->byte_count, CORRUPT,
		    "a number of the second split's range");
	*first_class = kept_class;
	expect_open(tables, tables->byte_count, STATUS_OK,
		    "every change was put back");
}

static void test_cases(const struct text_tokenizer *tokenizer)
{
    uint32_t ids[MOST_TOKEN_BYTES * MOST_CASE_TOKENS] = {0}, count = 0;
    char what[MOST_TOKENS] = {0};
    for (size_t index = 0; index < sizeof(encoding_cases) / sizeof(encoding_cases[0]); index++) {
        const struct encoding_case *const encoding_case = &encoding_cases[index];
	const result_status status = tokenizer_encode(
		tokenizer, encoding_case->text, strlen(encoding_case->text),
		ids, MOST_TOKEN_BYTES * MOST_CASE_TOKENS, &count);
	uint32_t expected = 0;
	bool is_same = status == STATUS_OK;
	for (; is_same && encoding_case->tokens[expected] != NULL; expected++) {
            is_same = expected < count && ids[expected] == token_of(encoding_case->tokens[expected],
                                                                    strlen(encoding_case->tokens[expected]));
        }
        (void)snprintf(what, sizeof(what), "the tokens of a case: %s", encoding_case->what);
        expect(is_same && count == expected, what);
    }
    const char *bytes = NULL;
    uint32_t byte_count = 0;
    expect(tokenizer_encode(tokenizer, NULL, 0, ids, 1, &count) == STATUS_OK &&
		   count == 0,
	   "no text from NULL");
    expect(tokenizer_encode(tokenizer, "abc", strlen("abc"), ids, 1, &count) ==
			   INVALID &&
		   tokenizer_encode(tokenizer, "abc", strlen("abc"), ids,
				    TOKENS_OF_ABC, &count) == STATUS_OK &&
		   count == TOKENS_OF_ABC,
	   "a capacity one token short is refused, the exact one is not");
    expect(tokenizer_encode(NULL, "a", 1, ids, 1, &count) == INVALID &&
		   tokenizer_encode(tokenizer, NULL, 1, ids, 1, &count) ==
			   INVALID &&
		   tokenizer_encode(tokenizer, "a", 1, NULL, 1, &count) ==
			   INVALID &&
		   tokenizer_encode(tokenizer, "a", 1, ids, 1, NULL) ==
			   INVALID &&
		   tokenizer_encode(tokenizer, "a",
				    TOKENIZER_MAXIMUM_TEXT_BYTES + 1, ids, 1,
				    &count) == INVALID,
	   "a NULL argument or a text too long");
    expect(tokenizer_encode(tokenizer, "a", TOKENIZER_MAXIMUM_TEXT_BYTES, ids,
			    1, &count) == INVALID,
	   "a text of more bytes than the capacity's tokens can hold is refused before a byte of it is read");
    expect(tokenizer_token_bytes(tokenizer, vocabulary.count, &bytes,
				 &byte_count) == STATUS_INVALID_ARGUMENT &&
		   tokenizer_token_bytes(NULL, 0, &bytes, &byte_count) ==
			   STATUS_INVALID_ARGUMENT &&
		   tokenizer_vocabulary_size(NULL) == 0 &&
		   tokenizer_vocabulary_size(tokenizer) == vocabulary.count,
	   "the bytes of a token out of range");
}

/* The plain BPE of a run of letters, the reference the heap is held to. */
static uint32_t plain_bpe(const char *text, size_t length, uint32_t *ids)
{
    size_t count = length;
    for (size_t index = 0; index < length; index++) {
        ids[index] = (uint32_t)(unsigned char)text[index];
    }
    for (;;) {
        uint32_t best_rank = NO_TOKEN;
        size_t best = 0;
        for (size_t index = 0; index + 1 < count; index++) {
            for (uint32_t rank = 0; rank < best_rank && rank < vocabulary.merge_count; rank++) {
                if (vocabulary.merges[rank][LEFT] == ids[index] && vocabulary.merges[rank][RIGHT] == ids[index + 1]) {
                    best_rank = rank;
                    best = index;
                }
            }
        }
        if (best_rank == NO_TOKEN) {
            return (uint32_t)count;
        }
        ids[best] = vocabulary.merges[best_rank][RESULT];
        memmove(ids + best + 1, ids + best + PAIR, (count - best - PAIR) * sizeof(ids[0]));
        count--;
    }
}

static void
test_bpe_against_the_plain_one(const struct text_tokenizer *tokenizer)
{
    char text[DRAWN_STRING_MOST_BYTES] = {0};
    uint32_t expected[DRAWN_STRING_MOST_BYTES] = {0}, ids[DRAWN_STRING_MOST_BYTES] = {0}, count = 0;
    int differing = 0;
    for (int index = 0; index < DRAWN_STRINGS; index++) {
        const size_t length = 1 + next_random() % DRAWN_STRING_MOST_BYTES;
        for (size_t place = 0; place < length; place++) {
            text[place] = drawn_letters[next_random() % (sizeof(drawn_letters) - 1)];
        }
        const uint32_t expected_count = plain_bpe(text, length, expected);
	const result_status status = tokenizer_encode(
		tokenizer, text, length, ids, DRAWN_STRING_MOST_BYTES, &count);
	differing += status != STATUS_OK || count != expected_count ||
				     memcmp(ids, expected,
					    count * sizeof(ids[0])) != 0 ?
			     1 :
			     0;
    }
    expect(differing == 0, "the heap merges as the plain BPE does");
}

/* The text with each ill-formed subpart as U+FFFD, written by Unicode's table 3-7: what the ids must spell. */
static size_t sanitized(const unsigned char *text, size_t length, unsigned char *out)
{
    const size_t lead_count = sizeof(lead_ranges) / sizeof(lead_ranges[0]);
    size_t written = 0;
    for (size_t position = 0; position < length;) {
        size_t kind = 0, good = 1;
        while (kind < lead_count && (text[position] < lead_ranges[kind].first_lead ||
                                     text[position] > lead_ranges[kind].last_lead)) {
            kind++;
        }
        const size_t bytes = kind < lead_count ? lead_ranges[kind].bytes : 0;
        while (good < bytes && position + good < length &&
               text[position + good] >= (good == 1 ? lead_ranges[kind].low : CONTINUATION_FIRST) &&
               text[position + good] <= (good == 1 ? lead_ranges[kind].high : CONTINUATION_LAST)) {
            good++;
        }
        if (bytes != 0 && good == bytes) {
            memcpy(out + written, text + position, bytes);
            written += bytes;
        } else {
            memcpy(out + written, replacement_character, REPLACEMENT_BYTES);
            written += REPLACEMENT_BYTES;
        }
        position += good;
    }
    return written;
}

/* Random pieces: bytes of any value, letters, spaces, digits, characters of every length, added tokens whole or cut. */
static size_t random_text(char *text)
{
    static const char *const pieces[] = {"a", "aa", "b c", "12345", " ", "  ", "\n\n", "\r\n", "!", "'s", "\xC2\xA0",
        "\xCC\x81", "\xE6\x97\xA5", "\xE3\x83\xBB", "\xF0\x9F\x98\x80", "\xE3\x80\x80", "<|a|>", "<|a|", "<|a|>b", "|>",
        "\xEF\xBD\x9C", "\xEF\xBD\x9C\x44\xEF\xBD\x9C", "pqrpq", "\xC3", "\xE0\xA0", "\xED\xA0\x80",
        "\xF4\x90\x80\x80"};
    size_t length = 0;
    for (uint64_t count = next_random() % FUZZ_TEXT_MOST_PIECES; count > 0; count--) {
        const uint64_t draw = next_random();
        const char *const piece = pieces[draw % (sizeof(pieces) / sizeof(pieces[0]))];
        const char byte[1] = {(char)(draw >> SPLITMIX_FIRST_SHIFT)};
        const bool is_byte = (draw >> SPLITMIX_THIRD_SHIFT) % (sizeof(pieces) / sizeof(pieces[0])) == 0;
        memcpy(text + length, is_byte ? byte : piece, is_byte ? 1 : strlen(piece));
        length += is_byte ? 1 : strlen(piece);
    }
    return length;
}

static void test_round_trip(const struct text_tokenizer *tokenizer)
{
    enum { MOST_FUZZ_BYTES = FUZZ_TEXT_MOST_PIECES * MOST_TOKEN_BYTES };
    char text[MOST_FUZZ_BYTES] = {0};
    unsigned char expected[MOST_FUZZ_BYTES * REPLACEMENT_BYTES] = {0};
    uint32_t ids[MOST_FUZZ_BYTES * REPLACEMENT_BYTES] = {0}, count = 0;
    int differing = 0;
    for (int index = 0; index < FUZZ_TEXTS; index++) {
        const size_t length = random_text(text);
        const size_t expected_bytes = sanitized((const unsigned char *)text, length, expected);
	const result_status status =
		tokenizer_encode(tokenizer, text, length, ids,
				 MOST_FUZZ_BYTES * REPLACEMENT_BYTES, &count);
	size_t decoded = 0;
	bool is_same = status == STATUS_OK;
	for (uint32_t place = 0; is_same && place < count; place++) {
            const char *bytes = NULL;
            uint32_t byte_count = 0;
	    is_same = tokenizer_token_bytes(tokenizer, ids[place], &bytes,
					    &byte_count) == STATUS_OK &&
		      byte_count <= expected_bytes - decoded &&
		      memcmp(bytes, expected + decoded, byte_count) == 0;
	    decoded += byte_count;
        }
        differing += is_same && decoded == expected_bytes ? 0 : 1;
    }
    expect(differing == 0, "random bytes come back as their text, each ill-formed subpart as U+FFFD");
}

static struct tokenizer_cache *
open_cache(const struct text_tokenizer *tokenizer, uint32_t text_limit,
	   uint32_t token_limit)
{
	struct tokenizer_cache *cache = NULL;
	if (tokenizer_cache_open(tokenizer, text_limit, token_limit, &cache) !=
	    STATUS_OK) {
		give_up("a request cache opens");
	}
    return cache;
}

/* The full encoder remains the oracle; neither the expected IDs nor the edit boundaries come from the cache. */
static bool cached_matches_full(const struct text_tokenizer *tokenizer,
				struct tokenizer_cache *cache, const char *text,
				size_t length)
{
    uint32_t expected[CACHE_TOKEN_IDS] = {0}, actual[CACHE_TOKEN_IDS] = {0};
    uint32_t expected_count = UINT32_MAX, actual_count = UINT32_MAX;
    const result_status full = tokenizer_encode(tokenizer, text, length,
						expected, CACHE_TOKEN_IDS,
						&expected_count);
    const result_status cached = tokenizer_encode_cached(
	    cache, text, length, actual, CACHE_TOKEN_IDS, &actual_count);
    return full == STATUS_OK && cached == full &&
	   actual_count == expected_count &&
	   memcmp(actual, expected,
		  (size_t)expected_count * sizeof(expected[0])) == 0;
}

static void test_cache_edits(const struct text_tokenizer *tokenizer)
{
    static const char *const turns[] = {
        "<|begin|>a b  <|a|>", "<|begin|>a b  <|a|>b", "<|begin|>a b  <|a|>bc",
        "<|begin|>a b  <|a|>", "<|begin|>a b  <|a|", "<|begin|>a b  <|a|>b<|a|>",
        "<|begin|>1234567<|a|>", "<|begin|>12345678<|a|>", "<|begin|>12305678<|a|>",
        "<|begin|>a  ", "<|begin|>a  b", "<|begin|>a \n\n", "<|begin|>a \r\nb",
        "<|begin|>aaaa", "<|begin|>aaaaa", "<|begin|>aaabaa", "<|begin|>!ab<|a|>",
        "<|begin|>!a<|a|>", "<|begin|>e\xCC\x81<|a|>", "<|begin|>e\xCC\x82<|a|>",
        "<|begin|>\xE6\x97\xA5\xE6\x9C\xAC<|a|>", "<|begin|>\xF0\x9F\x98\x80<|a|>",
        "<|begin|>\xF0\x9F\x98<|a|>", "<|begin|>\xF0\x9F\x98\x81<|a|>",
        "<|begin|>\xEF\xBD\x9C\x44\xEF\xBD\x9C", "<|begin|>\xEF\xBD\x9C\x44\xEF\xBD\x9C\x61", "", "a"};
    struct tokenizer_cache *const cache =
	    open_cache(tokenizer, CACHE_TEXT_BYTES, CACHE_TOKEN_IDS);
    for (size_t index = 0; index < sizeof(turns) / sizeof(turns[0]); index++) {
        expect(cached_matches_full(tokenizer, cache, turns[index], strlen(turns[index])),
               "a boundary, Unicode, special-token, word or digit edit matches full encoding");
        expect(cached_matches_full(tokenizer, cache, turns[index], strlen(turns[index])),
               "an exact retry matches full encoding");
    }
    /* Insert, replace, remove and truncate at arbitrary byte offsets, including inside UTF-8 and added tokens. */
    char text[CACHE_TEXT_BYTES] = {0}, replacement[CACHE_TEXT_BYTES] = {0};
    unsigned char repaired[CACHE_TOKEN_IDS] = {0};
    size_t length = 0;
    uint32_t differing = 0;
    for (uint32_t edit = 0; edit < CACHE_RANDOM_EDITS; edit++) {
        if (edit % CACHE_RESET_PERIOD == 0) {
            length = random_text(text);
            if (edit % (CACHE_RESET_PERIOD * PAIR) == 0) {
                length = sanitized((const unsigned char *)text, length, repaired);
                memcpy(text, repaired, length);
            }
            differing += cached_matches_full(tokenizer, cache, text, length) ? 0u : 1u;
        }
        const size_t begin = next_random() % (length + 1);
        const size_t available = length - begin < CACHE_EDIT_BYTES ? length - begin : CACHE_EDIT_BYTES;
        const size_t removed = next_random() % (available + 1);
        const size_t drawn = random_text(replacement);
        const size_t inserted = drawn < CACHE_EDIT_BYTES ? drawn : CACHE_EDIT_BYTES;
        if (length - removed + inserted <= sizeof(text)) {
            memmove(text + begin + inserted, text + begin + removed, length - begin - removed);
            memcpy(text + begin, replacement, inserted);
            length = length - removed + inserted;
        }
        if (edit % PAIR == 0) {
            const size_t repaired_length = sanitized((const unsigned char *)text, length, repaired);
            if (repaired_length <= sizeof(text)) {
                memcpy(text, repaired, repaired_length);
                length = repaired_length;
            }
        }
        differing += cached_matches_full(tokenizer, cache, text, length) ? 0u : 1u;
    }
    expect(differing == 0, "twenty thousand random byte edits have exactly the full encoder's IDs");
    tokenizer_cache_close(cache);
}

/* Origin boundaries change recognition only, never the ordinary pre-tokenizer/BPE segment around them. */
static void test_spans(const struct text_tokenizer *tokenizer)
{
    static const char *const ordinary[] = {"abc", "aaaa", "!ab", "a  b", "12345", "e\xCC\x81", "<|a|"};
    uint32_t expected[CACHE_TOKEN_IDS] = {0}, actual[CACHE_TOKEN_IDS] = {0};
    struct tokenizer_cache *const cache =
	    open_cache(tokenizer, CACHE_TEXT_BYTES, CACHE_TOKEN_IDS);
    for (size_t text_index = 0; text_index < sizeof(ordinary) / sizeof(ordinary[0]); text_index++) {
        const char *const text = ordinary[text_index];
        const uint32_t length = (uint32_t)strlen(text);
        for (uint32_t split = 1; split < length; split++) {
		const struct tokenizer_span trusted = { 0, split };
		uint32_t expected_count = 0, actual_count = 0;
		expect(tokenizer_encode(tokenizer, text, length, expected,
					CACHE_TOKEN_IDS,
					&expected_count) == STATUS_OK &&
			       tokenizer_encode_spans(
				       tokenizer, text, length, &trusted, 1,
				       actual, CACHE_TOKEN_IDS,
				       &actual_count) == STATUS_OK &&
			       expected_count == actual_count &&
			       memcmp(expected, actual,
				      (size_t)expected_count *
					      sizeof(*actual)) == 0,
		       "a trust boundary inside a word, split lookahead or UTF-8 leaves ordinary ids unchanged");
	}
    }
    const char *const text = "<|begin|>abc<|a|>b a  b<|a|>";
    const uint32_t length = (uint32_t)strlen(text);
    for (uint32_t end = 0; end <= length; end++) {
	    const struct tokenizer_span trusted = { 0, end };
	    const uint32_t spans = end != 0 ? 1u : 0u;
	    uint32_t expected_count = 0, actual_count = 0;
	    expect(tokenizer_encode_spans(tokenizer, text, length, &trusted,
					  spans, expected, CACHE_TOKEN_IDS,
					  &expected_count) == STATUS_OK &&
			   tokenizer_encode_cached_spans(
				   cache, text, length, &trusted, spans, actual,
				   CACHE_TOKEN_IDS,
				   &actual_count) == STATUS_OK &&
			   expected_count == actual_count &&
			   memcmp(expected, actual,
				  (size_t)expected_count * sizeof(*actual)) ==
				   0,
		   "changed trust and longest-added-token endpoints cannot reuse cached control ids");
    }
    tokenizer_cache_close(cache);
}

static void test_cache_contract(const struct text_tokenizer *tokenizer)
{
	struct tokenizer_cache *cache = NULL;
	uint32_t ids[CACHE_TOKEN_IDS] = { 0 }, count = UINT32_MAX;
	expect(tokenizer_cache_open(NULL, 1, 1, &cache) == INVALID &&
		       tokenizer_cache_open(tokenizer, 0, 1, &cache) ==
			       INVALID &&
		       tokenizer_cache_open(tokenizer, 1, 0, &cache) ==
			       INVALID &&
		       tokenizer_cache_open(
			       tokenizer,
			       (uint32_t)TOKENIZER_MAXIMUM_TEXT_BYTES + 1, 1,
			       &cache) == INVALID &&
		       tokenizer_cache_open(tokenizer, 1, 1, NULL) == INVALID &&
		       cache == NULL,
	       "cache open refuses bad arguments without writing its output");
	cache = open_cache(tokenizer, CACHE_TEXT_BYTES, CACHE_TOKEN_IDS);
	struct tokenizer_cache *const kept_cache = cache;
	expect(tokenizer_cache_open(NULL, 1, 1, &cache) == INVALID &&
		       cache == kept_cache,
	       "a failed open preserves a nonempty output pointer");
	expect(tokenizer_cache_common_prefix(NULL, "a", 1) == 0 &&
		       tokenizer_cache_common_prefix(cache, NULL, 1) == 0 &&
		       tokenizer_cache_common_prefix(cache, "a", 1) == 0,
	       "an empty cache or absent argument has no common prefix");
	expect(cached_matches_full(tokenizer, cache, "abc", strlen("abc")) &&
		       tokenizer_cache_common_prefix(
			       cache, "abc", strlen("abc")) == strlen("abc") &&
		       tokenizer_cache_common_prefix(
			       cache, "abx", strlen("abx")) == strlen("ab"),
	       "common prefix counts bytes through the first difference");
	expect(tokenizer_encode_cached(cache, "abc", strlen("abc"), ids, 1,
				       &count) == INVALID &&
		       count == UINT32_MAX &&
		       tokenizer_cache_common_prefix(cache, "abc",
						     strlen("abc")) == 0,
	       "an insufficient output capacity invalidates cached data and preserves the count");
	expect(cached_matches_full(tokenizer, cache, "abc", strlen("abc")),
	       "a failed encode recovers");
	count = UINT32_MAX;
	expect(tokenizer_encode_cached(NULL, "a", 1, ids, 1, &count) ==
			       INVALID &&
		       tokenizer_encode_cached(cache, NULL, 1, ids, 1,
					       &count) == INVALID &&
		       tokenizer_encode_cached(cache, "a", 1, NULL, 1,
					       &count) == INVALID &&
		       tokenizer_encode_cached(cache, "a", 1, ids, 1, NULL) ==
			       INVALID &&
		       tokenizer_encode_cached(cache, "a",
					       TOKENIZER_MAXIMUM_TEXT_BYTES + 1,
					       ids, 1, &count) == INVALID &&
		       count == UINT32_MAX,
	       "cached encode has full encode's invalid-argument refusals");
	expect(cached_matches_full(tokenizer, cache, NULL, 0),
	       "an empty text may be NULL through the cache");
	expect(cached_matches_full(tokenizer, cache, "abc\xFF",
				   strlen("abc\xFF")) &&
		       tokenizer_cache_common_prefix(cache, "abc",
						     strlen("abc")) == 0,
	       "malformed UTF-8 uses the full encoder and evicts cached data");
	tokenizer_cache_close(cache);
	cache = open_cache(tokenizer, 1, 1);
	expect(cached_matches_full(tokenizer, cache, "abc<|a|>",
				   strlen("abc<|a|>")) &&
		       tokenizer_cache_common_prefix(cache, "abc",
						     strlen("abc")) == 0 &&
		       cached_matches_full(tokenizer, cache, "a", 1) &&
		       tokenizer_cache_common_prefix(cache, "a", 1) == 1,
	       "cache limits cause a fallback, never a request refusal, and a smaller next request recovers");
	tokenizer_cache_close(cache);
	cache = open_cache(tokenizer, CACHE_TEXT_BYTES, 1);
	expect(cached_matches_full(tokenizer, cache, "abc", strlen("abc")) &&
		       tokenizer_cache_common_prefix(cache, "abc",
						     strlen("abc")) == 0,
	       "a token-only cache limit also falls back");
	tokenizer_cache_close(cache);
	cache = open_cache(tokenizer, CACHE_TEXT_BYTES, CACHE_TOKEN_IDS);
	struct tokenizer_cache *const peer =
		open_cache(tokenizer, CACHE_TEXT_BYTES, CACHE_TOKEN_IDS);
	expect(cached_matches_full(tokenizer, cache, "abc<|a|>",
				   strlen("abc<|a|>")) &&
		       cached_matches_full(tokenizer, peer, "12345<|a|>",
					   strlen("12345<|a|>")) &&
		       tokenizer_cache_common_prefix(
			       cache, "abc", strlen("abc")) == strlen("abc") &&
		       tokenizer_cache_common_prefix(peer, "abc",
						     strlen("abc")) == 0 &&
		       cached_matches_full(tokenizer, cache, "abc<|a|>b",
					   strlen("abc<|a|>b")) &&
		       cached_matches_full(tokenizer, peer, "12345<|a|>b",
					   strlen("12345<|a|>b")),
	       "interleaved caches of one tokenizer own independent histories");
	tokenizer_cache_close(peer);
	tokenizer_cache_close(cache);
	tokenizer_cache_close(NULL);
}

static void test_cache_alias(const struct text_tokenizer *tokenizer)
{
	struct tokenizer_cache *const cache =
		open_cache(tokenizer, CACHE_TEXT_BYTES, CACHE_TOKEN_IDS);
	uint32_t actual[CACHE_TOKEN_IDS] = { 0 },
		 expected[CACHE_TOKEN_IDS] = { 0 };
	uint32_t actual_count = UINT32_MAX, expected_count = UINT32_MAX;
	const size_t length = strlen("abc");
	expect(cached_matches_full(tokenizer, cache, "abc", length),
	       "the cache is warm before an aliased encode");
	memcpy(actual, "abc", length);
	memcpy(expected, "abc", length);
	const result_status full =
		tokenizer_encode(tokenizer, (const char *)expected, length,
				 expected, CACHE_TOKEN_IDS, &expected_count);
	const result_status cached =
		tokenizer_encode_cached(cache, (const char *)actual, length,
					actual, CACHE_TOKEN_IDS, &actual_count);
	expect(full == STATUS_OK && cached == full &&
		       actual_count == expected_count &&
		       memcmp(actual, expected,
			      (size_t)expected_count * sizeof(expected[0])) ==
			       0,
	       "aliased input and output retain the full encoder's behavior");
	expect(tokenizer_cache_common_prefix(cache, (const char *)actual,
					     length) == 0,
	       "aliased output bytes are never published as cached input text");
	expect(cached_matches_full(tokenizer, cache, (const char *)actual,
				   length),
	       "the overwritten binary bytes encode normally after an aliased call");
	tokenizer_cache_close(cache);
}

static void test_cache_limits_and_faults(const struct text_tokenizer *tokenizer)
{
    const size_t added_bytes = strlen(added_texts[0]);
    const size_t length = CACHE_SEGMENT_OVERFLOW * added_bytes;
    char *const text = malloc(length);
    uint32_t *const ids = malloc(CACHE_SEGMENT_OVERFLOW * sizeof(uint32_t));
    struct tokenizer_cache *cache =
	    open_cache(tokenizer, (uint32_t)length, CACHE_SEGMENT_OVERFLOW);
    if (text == NULL || ids == NULL) {
        give_up("memory for too many cached segments");
    }
    for (uint32_t index = 0; index < CACHE_SEGMENT_OVERFLOW; index++) {
        memcpy(text + index * added_bytes, added_texts[0], added_bytes);
    }
    uint32_t count = UINT32_MAX;
    bool is_right = tokenizer_encode_cached(cache, text, length, ids,
					    CACHE_SEGMENT_OVERFLOW,
					    &count) == STATUS_OK &&
		    count == CACHE_SEGMENT_OVERFLOW;
    const uint32_t added_id = token_of(added_texts[0], added_bytes);
    for (uint32_t index = 0; is_right && index < count; index++) {
        is_right = ids[index] == added_id;
    }
    expect(is_right, "more special segments than the cache holds still encode exactly");
    expect(cached_matches_full(tokenizer, cache, "abc<|a|>", strlen("abc<|a|>")),
           "a small request after segment overflow recovers");
    tokenizer_cache_close(cache);
    free(text);
    free(ids);
#if CHECKING_HOOKS_ENABLED
    cache = NULL;
    expect(fault_arm("tokenizer_cache_memory", FAULT_ON_FIRST_CALL),
	   "cache allocation failure is armed");
    expect(tokenizer_cache_open(tokenizer, CACHE_TEXT_BYTES, CACHE_TOKEN_IDS,
				&cache) == OUT_OF_MEMORY &&
		   cache == NULL && fault_disarm() == 1,
	   "cache open reports allocation failure");
    cache = open_cache(tokenizer, CACHE_TEXT_BYTES, CACHE_TOKEN_IDS);
    expect(fault_arm("tokenizer_cache_memory", FAULT_ON_FIRST_CALL),
	   "cache refresh failure is armed");
    expect(cached_matches_full(tokenizer, cache, "abc<|a|>",
			       strlen("abc<|a|>")) &&
		   fault_disarm() >= 1 &&
		   tokenizer_cache_common_prefix(cache, "abc", strlen("abc")) ==
			   0,
	   "a cache refresh allocation failure preserves the successful encoding and evicts the cache");
    expect(cached_matches_full(tokenizer, cache, "abc<|a|>", strlen("abc<|a|>")),
           "cache refresh recovers from memory failure");
    uint32_t output[CACHE_TOKEN_IDS] = {0};
    count = UINT32_MAX;
    expect(fault_arm("tokenizer_memory", FAULT_ON_FIRST_CALL),
	   "suffix BPE memory failure is armed");
    expect(tokenizer_encode_cached(cache, "abc<|a|>aaaa",
				   strlen("abc<|a|>aaaa"), output,
				   CACHE_TOKEN_IDS, &count) == OUT_OF_MEMORY &&
		   count == UINT32_MAX && fault_disarm() >= 1 &&
		   tokenizer_cache_common_prefix(cache, "abc", strlen("abc")) ==
			   0,
	   "suffix allocation failure preserves the count and invalidates partial cache state");
    expect(cached_matches_full(tokenizer, cache, "abc<|a|>", strlen("abc<|a|>")),
           "cached encoding recovers after suffix allocation failure");
    tokenizer_cache_close(cache);
#endif
}

static void test_utf8_sequences(void)
{
    for (size_t index = 0; index < sizeof(utf8_cases) / sizeof(utf8_cases[0]); index++) {
        bool is_valid = !utf8_cases[index].is_valid;
	const uint32_t length = tokenizer_utf8_sequence_bytes(
		utf8_cases[index].bytes, 0, strlen(utf8_cases[index].bytes),
		&is_valid);
	expect(length == utf8_cases[index].length && is_valid == utf8_cases[index].is_valid,
               "a UTF-8 sequence, or the maximal subpart of one that is not");
    }
    bool is_valid = true;
    expect(tokenizer_utf8_sequence_bytes("a", 1, 1, &is_valid) == 0 &&
		   !is_valid &&
		   tokenizer_utf8_sequence_bytes(NULL, 0, 1, &is_valid) == 0 &&
		   !is_valid &&
		   tokenizer_utf8_sequence_bytes("a", 0, 1, NULL) == 0,
	   "no sequence at the end or from NULL");
}

static void *encode_in_a_thread(void *argument)
{
    struct thread_work *const work = argument;
    work->status = tokenizer_encode(work->tokenizer, work->text,
				    work->text_bytes, work->ids, LONG_RUN_BYTES,
				    &work->count);
    return NULL;
}

/* A million letters in one piece; the same text from several threads at once. */
static void test_long_piece_and_threads(const struct text_tokenizer *tokenizer)
{
    char *const text = malloc(LONG_RUN_BYTES);
    struct thread_work works[THREAD_COUNT] = {{0}};
    pthread_t threads[THREAD_COUNT];
    if (text == NULL) {
        give_up("memory for the long piece");
    }
    memset(text, 'a', LONG_RUN_BYTES);
    for (int index = 0; index < THREAD_COUNT; index++) {
	    works[index] = (struct thread_work){ tokenizer,
						 text,
						 LONG_RUN_BYTES,
						 malloc(LONG_RUN_BYTES *
							sizeof(uint32_t)),
						 0,
						 STATUS_INVALID_ARGUMENT };
	    if (works[index].ids == NULL ||
		pthread_create(&threads[index], NULL, encode_in_a_thread,
			       &works[index]) != 0) {
		    give_up("a thread to encode with");
	    }
    }
    const uint32_t four_a = token_of("aaaa", LONG_RUN_TOKEN_BYTES);
    for (int index = 0; index < THREAD_COUNT; index++) {
        if (pthread_join(threads[index], NULL) != 0) {
            give_up("a thread that ends");
        }
	bool is_right = works[index].status == STATUS_OK &&
			works[index].count ==
				LONG_RUN_BYTES / LONG_RUN_TOKEN_BYTES;
	for (uint32_t place = 0; is_right && place < works[index].count; place++) {
            is_right = works[index].ids[place] == four_a;
        }
        expect(is_right, "a million letters, in one piece, from several threads at once");
        free(works[index].ids);
    }
    free(text);
}

static void test_faults(const struct built_tables *tables,
			const struct text_tokenizer *tokenizer)
{
#if CHECKING_HOOKS_ENABLED
	struct text_tokenizer *opened = NULL;
	uint32_t ids[MOST_CASE_TOKENS] = { 0 }, count = 0;
	expect(fault_arm("tokenizer_memory", FAULT_ON_FIRST_CALL),
	       "a lack of memory is armed");
	expect(tokenizer_open(tables->bytes, tables->byte_count, &opened) ==
			       STATUS_OUT_OF_MEMORY &&
		       opened == NULL && fault_disarm() == 1,
	       "an open that lacks memory says so");
	expect(fault_arm("tokenizer_memory", FAULT_ON_FIRST_CALL),
	       "a lack of memory is armed");
	expect(tokenizer_encode(tokenizer, "ab", strlen("ab"), ids,
				MOST_CASE_TOKENS, &count) == OUT_OF_MEMORY &&
		       fault_disarm() == 1,
	       "an encode that lacks memory for its pieces says so");
	expect(fault_arm("tokenizer_memory", FAULT_ON_FIRST_CALL),
	       "a lack of memory is armed");
	expect(tokenizer_encode(tokenizer, "\xFF", 1, ids, MOST_CASE_TOKENS,
				&count) == OUT_OF_MEMORY &&
		       fault_disarm() == 1,
	       "an encode that lacks memory for U+FFFD says so");
#else
    (void)tables;
    (void)tokenizer;
#endif
}

/* Tables of the byte tokens, a token of the most bytes and one-byte tokens, whose added tokens are that token again
 * and again: as many bytes of them as open allows, then one token more. Every count is within its own limit; the
 * second must be refused before the hash table is sized, so an armed lack of memory is never reached. */
static void test_added_token_bytes(void)
{
	enum {
		MOST = TOKENIZER_MAXIMUM_ADDED_TOKEN_BYTES /
			TOKENIZER_MAXIMUM_TOKEN_BYTES,
		TOKENS = BYTE_TOKENS + MOST + 1,
		LONG_TOKEN = BYTE_TOKENS
	};
	const size_t token_byte_count =
		(size_t)TOKENS - 1 + TOKENIZER_MAXIMUM_TOKEN_BYTES;
	for (uint32_t added_count = MOST; added_count <= MOST + 1;
	     added_count++) {
		struct tokenizer_tables_header header = {
			.format_version = TOKENIZER_TABLES_FORMAT_VERSION,
			.vocabulary_size = TOKENS,
			.added_token_count = added_count,
			.class_block_count = 1,
			.token_byte_count = (uint32_t)token_byte_count
		};
		const size_t byte_count =
			sizeof(header) +
			sizeof(uint32_t) *
				((size_t)TOKENS + added_count + BYTE_TOKENS) +
			TOKENIZER_CLASS_BLOCK_COUNT * sizeof(uint16_t) + BLOCK +
			token_byte_count;
		unsigned char *const bytes =
			calloc(1, byte_count); /* the blocks of classes are 0 */
		struct text_tokenizer *tokenizer = NULL;
		if (bytes == NULL) {
			give_up("memory for the tables of many added tokens");
		}
		unsigned char *cursor = bytes + sizeof(header);
		memcpy(header.magic, TOKENIZER_TABLES_MAGIC,
		       TOKENIZER_TABLES_MAGIC_BYTES);
		memcpy(bytes, &header, sizeof(header));
		for (uint32_t token = 0, end = 0; token < TOKENS;
		     token++, cursor += sizeof(end)) {
			end += token == LONG_TOKEN ?
				       TOKENIZER_MAXIMUM_TOKEN_BYTES :
				       1;
			memcpy(cursor, &end, sizeof(end));
		}
		for (uint32_t index = 0, long_token = LONG_TOKEN;
		     index < added_count;
		     index++, cursor += sizeof(long_token)) {
			memcpy(cursor, &long_token, sizeof(long_token));
		}
		for (uint32_t byte = 0; byte < BYTE_TOKENS;
		     byte++, cursor += sizeof(byte)) {
			memcpy(cursor, &byte, sizeof(byte));
			bytes[byte_count - token_byte_count + byte] =
				(unsigned char)byte;
		}
#if CHECKING_HOOKS_ENABLED
		expect(fault_arm("tokenizer_memory", FAULT_ON_FIRST_CALL),
		       "a lack of memory is armed");
		const result_status status =
			tokenizer_open(bytes, byte_count, &tokenizer);
		expect(status == (added_count == MOST ? OUT_OF_MEMORY :
							CORRUPT) &&
			       fault_disarm() ==
				       (added_count == MOST ? 1u : 0u),
		       "added tokens of the most bytes are sized, of more are refused before");
#else
		expect(tokenizer_open(bytes, byte_count, &tokenizer) == CORRUPT,
		       "an added token listed twice");
#endif
		tokenizer_close(tokenizer);
		free(bytes);
    }
}

int main(void)
{
	struct text_tokenizer *tokenizer = NULL;
	(void)alarm(TEST_TIME_LIMIT_SECONDS);
	make_vocabulary();
	struct built_tables tables = build_tables();
	test_open(&tables);
	if (tokenizer_open(tables.bytes, tables.byte_count, &tokenizer) !=
	    STATUS_OK) {
		give_up("the test's tables open");
	}
    test_cases(tokenizer);
    test_bpe_against_the_plain_one(tokenizer);
    test_round_trip(tokenizer);
    test_cache_edits(tokenizer);
    test_spans(tokenizer);
    test_cache_contract(tokenizer);
    test_cache_alias(tokenizer);
    test_cache_limits_and_faults(tokenizer);
    test_utf8_sequences();
    test_long_piece_and_threads(tokenizer);
    test_faults(&tables, tokenizer);
    test_added_token_bytes();
    tokenizer_close(tokenizer);
    tokenizer_close(NULL);
    free(tables.bytes);
    if (failures != 0) {
        return EXIT_FAILURE;
    }
    (void)printf(
	    "tokenizer_test passed (%u tokens, %u merges, checking hooks %s)\n",
	    vocabulary.count, vocabulary.merge_count,
	    CHECKING_HOOKS_ENABLED ? "enabled" : "compiled out");
    return EXIT_SUCCESS;
}
