/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * linear_card_fixture.h: what the card tests of the linear share: the lines a test fails by, one open card with the
 * seven kernels of linear_records.h prepared and an arena of which the test knows every byte, the ranges of a case,
 * and the synthetic numbers a case is made of. Test support: no part of the engine.
 *
 * Contract:
 *   - Failures. A test fails by lines "FAILED: <context>: <what>" on the standard error stream;
 *     test_linear_failures counts them. test_linear_context says what the lines that follow are about. The two
 *     expect functions return whether the expectation held. All of this is for the one thread of a test.
 *   - A card (struct test_linear_card) is one card of backend_cuda_value, open with an arena of work_capacity plus
 *     weights_capacity bytes. The work region, bytes 0 to work_capacity, holds what launches read and write; the
 *     weights region behind it holds matrices. image is page-locked memory of the arena's size: what a test lays out
 *     in it is what test_linear_card_load uploads, at the same offsets. received is page-locked memory of the work
 *     region's size: test_linear_card_fetch downloads the work region set aside so far into it, at the same offsets.
 *     A card that has no room for the arena, does not open or lacks a kernel FAILS the test:
 *     test_linear_card_open then answers false with nothing left open. work_capacity is a multiple of
 *     BACKEND_STAGING_ALIGNMENT_BYTES, so that the weights region of the image begins on one.
 *   - Ranges. test_linear_card_clear begins a case with no range set aside. test_linear_card_work and
 *     test_linear_card_weights set one aside: it begins on a multiple of TEST_LINEAR_RANGE_ALIGNMENT_BYTES, and its
 *     bytes in the image are zero. Behind a range of the work region lie at least TEST_LINEAR_GUARD_BYTES bytes of
 *     TEST_LINEAR_GUARD_BYTE that no launch may write: every fetch looks at every guard. A range that does not fit
 *     is one failure line and a range of no bytes; test_linear_card_has_room then answers false until the next
 *     clear, and a case that sees it runs nothing.
 *   - test_linear_card_put copies into the image and test_linear_card_get out of received, both inside one range:
 *     bytes beyond the range are a failure line and not copied. test_linear_card_stored is the image at a range,
 *     for the functions that read a stored matrix, or NULL for a range that is not inside the arena;
 *     test_linear_card_writable is the same place for a reader that fills it, of a range of the weights region
 *     only: the weights region of the image begins on a multiple of BACKEND_STAGING_ALIGNMENT_BYTES.
 *     test_linear_card_put_codes puts the codes of one vector, given in the order of its values, into the image in
 *     the lane order of linear_records.h; test_linear_card_expect_codes fails unless received holds them so.
 *     test_linear_card_put_operands and test_linear_card_expect_operands do the same with the operands the
 *     reference makes of the vector's codes and exponents (test_linear_operands_lay_out).
 *   - load, upload (a part of the image again, after a test changed it there), run (launches queued and waited for)
 *     and fetch each wait for what they began, at most TEST_LINEAR_WAIT_LIMIT_NANOSECONDS, and answer false behind
 *     a failure line when the backend refuses or fails. fetch adds the pack digest of the downloaded work region
 *     to the card's digest: two cards that computed the same bits in every case have the same one. It also adds
 *     the digest of the first prefix_byte_count bytes of the region (whole digest words; a test sets the count, 0
 *     by default) to prefix_digest: a test that keeps there the ranges of an earlier version of itself, in their
 *     order, reproduces the digest that version printed.
 *   - What came back. test_linear_card_differing counts the bytes in which two stretches of received differ;
 *     test_linear_card_expect_same fails unless they are the same, and test_linear_card_expect_bytes unless a
 *     stretch of received is the bytes a test hands it: one failure line each, with the count and the first byte
 *     that differs. test_linear_card_expect_untouched fails unless a range of the work region came back as the
 *     image holds it; test_linear_card_expect_weights_untouched fails unless the card's own digest of the weights
 *     region set aside is the host's digest of the image there: no launch wrote a matrix.
 *   - test_linear_expect_products holds outputs of a card against the reference's products with
 *     test_linear_compare: one failure line if any is outside the tolerance, one line on the standard output that
 *     says what was measured, and the counts added to a total. test_linear_share_holds says whether as many of a
 *     total are bit-equal as the gate of the linear asks: TEST_LINEAR_SHARE_REQUIRED_PER_THOUSAND of a thousand.
 *   - test_linear_card_set_weight changes ONE weight of a matrix in the image, and nothing on the card: it finds
 *     the stored byte that holds the code of the weight of an output and an input by changing a bit of each byte of
 *     that weight's 512 and asking weight_format_reference.h which one the weight follows, writes the new code
 *     there (of a 4-bit code, the one half of the byte) and gives the byte's place in the matrix, for the upload.
 *   - Numbers. test_linear_hash is a fixed mixer of a stream and a position: the same number on every machine.
 *     test_linear_values fills a vector of bfloat16 values, none an infinity or a not-a-number:
 *       PLAIN     magnitudes from 2^-4 to 4, both signs;
 *       SMALL     for rows that are not quantized: blocks of plain values, of zeros of both signs, and of magnitudes
 *                 from 2^-12 down to 2^-28, where a 16-bit float keeps fewer bits or none;
 *       EDGES     blocks of 32 values of six classes, chosen by the hash of the block: plain; zeros of both signs; a
 *                 block whose largest magnitude is below the floor of 1e-4; a block whose first value is 448 * 2^k,
 *                 the largest E4M3 number at its scale, and whose other values lie on ties between two neighbouring
 *                 E4M3 numbers, subnormal ones included; exponents from 2^-40 to 2^19; one value near 8 among values
 *                 down to 2^-17 of it, which become subnormal codes and zeros of both signs;
 *       EXTREMES  the six, and: a block that holds the largest finite bfloat16; a block of subnormal bfloat16.
 *     test_linear_card_matrix lays a matrix out in the weights region with weight_format_reference.h: the code of
 *     every weight from the recipe's function, the scale codes scale_code_base plus the hash modulo
 *     scale_code_spread + 1. The two code functions here give every E2M1 code alike, and E4M3 bytes whose exponents
 *     thin out by half from the largest binade down (one byte in 64 is any number), never a byte that is no number.
 */
#ifndef TEST_LINEAR_CARD_FIXTURE_H
#define TEST_LINEAR_CARD_FIXTURE_H

#include <stdbool.h>
#include <stdint.h>

#include "anemone/backend_seam.h"
#include "anemone/linear.h"
#include "anemone/status.h"
#include "anemone/weight_format_seam.h"
#include "linear_reference.h"

#define TEST_LINEAR_WAIT_LIMIT_NANOSECONDS 120000000000ull  /* two minutes: far above any copy or kernel of a test */
#define TEST_LINEAR_RANGE_ALIGNMENT_BYTES 64u   /* a multiple of the alignment of an offset, and a matrix's own */
#define TEST_LINEAR_GUARD_BYTES 64u
#define TEST_LINEAR_GUARD_BYTE 0xC3
#define TEST_LINEAR_MAXIMUM_GUARDS 4096u        /* ranges of the work region in one case */
#define TEST_LINEAR_PRINTF_FORMAT_INDEX 1
#define TEST_LINEAR_PRINTF_VALUES_INDEX 2
#define TEST_LINEAR_NOTHING_DIFFERS UINT64_MAX  /* in place of the index of a byte */
#define TEST_LINEAR_SHARE_REQUIRED_PER_THOUSAND 999u    /* 99.9% bit-equal */

struct test_linear_card {
	struct device_card *card; /* NULL while no card is open */
	struct linear_kernels_record kernels;
	unsigned char *image;
	unsigned char *received;
	uint64_t arena_byte_count;
	uint64_t work_capacity;
	uint64_t work_byte_count; /* set aside in the work region, guards included */
	uint64_t weights_byte_count; /* set aside in the weights region */
	uint64_t digest;
	uint64_t prefix_byte_count;
	uint64_t prefix_digest;
	uint32_t card_index;
	uint32_t arithmetic_class;
	uint32_t guard_count;
	bool is_out_of_room;
	uint64_t guard_offsets[TEST_LINEAR_MAXIMUM_GUARDS];
};

enum test_linear_values_kind {
    TEST_LINEAR_VALUES_PLAIN,
    TEST_LINEAR_VALUES_SMALL,
    TEST_LINEAR_VALUES_EDGES,
    TEST_LINEAR_VALUES_EXTREMES
};

struct test_linear_matrix_recipe {
	enum weight_format_code format;
	struct matrix_shape_record shape;
	uint64_t stream;
	uint8_t (*code)(uint64_t stream, uint32_t output_index,
			uint32_t input_index);
	uint32_t scale_code_base;
	uint32_t scale_code_spread;
};

void test_linear_context(const char *format, ...)
    __attribute__((format(printf, TEST_LINEAR_PRINTF_FORMAT_INDEX, TEST_LINEAR_PRINTF_VALUES_INDEX)));
void test_linear_fail(const char *format, ...)
    __attribute__((format(printf, TEST_LINEAR_PRINTF_FORMAT_INDEX, TEST_LINEAR_PRINTF_VALUES_INDEX)));
bool test_linear_expect(bool condition, const char *what);
bool test_linear_expect_status(result_status actual, result_status expected,
			       const char *what);
int test_linear_failures(void);

bool test_linear_card_open(struct test_linear_card *self, uint32_t card_index, uint64_t work_capacity,
                           uint64_t weights_capacity);
void test_linear_card_close(struct test_linear_card *self);     /* also of a card that did not open */

void test_linear_card_clear(struct test_linear_card *self);
struct linear_range test_linear_card_work(struct test_linear_card *self,
					  uint64_t byte_count);
struct linear_range test_linear_card_weights(struct test_linear_card *self,
					     uint64_t byte_count);
bool test_linear_card_has_room(const struct test_linear_card *self);
/* The slices a card runs of a launch stamped for slice_count: in the exact profile class 120 takes at most
 * LINEAR_CLASS_120_EXACT_MOST_SLICES (gate with up: _MOST_GATE_UP_SLICES), in the same order at any count. */
uint32_t test_linear_card_slices(const struct test_linear_card *self, uint32_t slice_count, bool is_gate_up);
/* The weights region set aside so far, as one range: what holds every matrix of a case. */
struct linear_range
test_linear_card_all_weights(const struct test_linear_card *self);

bool test_linear_card_put(struct test_linear_card *self,
			  struct linear_range range, uint64_t byte_offset,
			  const void *bytes, uint64_t byte_count);
bool test_linear_card_get(const struct test_linear_card *self,
			  struct linear_range range, uint64_t byte_offset,
			  void *bytes_out, uint64_t byte_count);
const uint8_t *test_linear_card_stored(const struct test_linear_card *self,
				       struct linear_range range);
unsigned char *test_linear_card_writable(struct test_linear_card *self,
					 struct linear_range range);
/* codes [width] in the order of their values; the vector lies at byte_offset of the range. */
bool test_linear_card_put_codes(struct test_linear_card *self,
				struct linear_range range, uint64_t byte_offset,
				const uint8_t *codes, uint32_t width);
bool test_linear_card_expect_codes(const struct test_linear_card *self,
				   struct linear_range range,
				   uint64_t byte_offset, const uint8_t *codes,
				   uint32_t width, const char *what);
/* codes [width] in the order of their values, exponents [width / 32]; the operands lie at byte_offset. */
bool test_linear_card_put_operands(struct test_linear_card *self,
				   struct linear_range range,
				   uint64_t byte_offset, const uint8_t *codes,
				   const uint8_t *exponents, uint32_t width);
bool test_linear_card_expect_operands(const struct test_linear_card *self,
				      struct linear_range range,
				      uint64_t byte_offset,
				      const uint8_t *codes,
				      const uint8_t *exponents, uint32_t width,
				      const char *what);

bool test_linear_card_load(struct test_linear_card *self);
bool test_linear_card_upload(struct test_linear_card *self,
			     struct linear_range range, uint64_t byte_offset,
			     uint64_t byte_count);
bool test_linear_card_run(struct test_linear_card *self,
			  const struct kernel_launch *launches,
			  uint32_t launch_count);
bool test_linear_card_fetch(struct test_linear_card *self);

/* Bytes byte_offset .. of two ranges of received, compared: how many differ, and the first that does (or
 * TEST_LINEAR_NOTHING_DIFFERS). Bytes beyond either range are a failure line and count as differing. */
uint64_t test_linear_card_differing(const struct test_linear_card *self,
				    struct linear_range first,
				    uint64_t first_byte_offset,
				    struct linear_range second,
				    uint64_t second_byte_offset,
				    uint64_t byte_count,
				    uint64_t *first_differing_out);
bool test_linear_card_expect_same(const struct test_linear_card *self,
				  struct linear_range first,
				  uint64_t first_byte_offset,
				  struct linear_range second,
				  uint64_t second_byte_offset,
				  uint64_t byte_count, const char *what);
bool test_linear_card_expect_bytes(const struct test_linear_card *self,
				   struct linear_range range,
				   uint64_t byte_offset, const void *expected,
				   uint64_t byte_count, const char *what);
bool test_linear_card_expect_untouched(const struct test_linear_card *self,
				       struct linear_range range,
				       const char *what);
bool test_linear_card_expect_weights_untouched(struct test_linear_card *self);

void test_linear_expect_products(const char *what, const uint16_t *outputs, const struct test_linear_product *products,
                                 uint64_t output_count, uint32_t input_size, uint32_t slice_count,
                                 struct test_linear_comparison *total);
bool test_linear_share_holds(uint64_t equal_count, uint64_t count);
double test_linear_percent(uint64_t part, uint64_t whole);     /* 0 of nothing */

uint64_t test_linear_hash(uint64_t stream, uint64_t position);
void test_linear_values(uint16_t *values_out, uint32_t width, uint64_t stream, enum test_linear_values_kind kind);
uint8_t test_linear_e2m1_code(uint64_t stream, uint32_t output_index, uint32_t input_index);
uint8_t test_linear_e4m3_code(uint64_t stream, uint32_t output_index, uint32_t input_index);
bool test_linear_card_matrix(struct test_linear_card *self,
			     const struct test_linear_matrix_recipe *recipe,
			     struct linear_range *stored_out);
bool test_linear_card_set_weight(struct test_linear_card *self,
				 enum weight_format_code format,
				 const struct matrix_shape_record *shape,
				 struct linear_range stored,
				 uint32_t output_index, uint32_t input_index,
				 uint32_t new_code, uint64_t *byte_offset_out);

#endif
