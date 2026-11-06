/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * linear_card_fixture.c: the fixture of linear_card_fixture.h. Test support: no part of the engine.
 *
 * Contract: linear_card_fixture.h. What it leaves open is decided here:
 *   - A range is set aside by a counter that only rises: the work region from byte 0, the weights region from
 *     work_capacity. A range of the work region is followed by guard bytes up to the next multiple of the alignment
 *     and TEST_LINEAR_GUARD_BYTES more; the weights region has no guards, its digest on the card is its check.
 *   - Every copy is begun and then waited for: a test is a sequence, and its uploads never share the image with a
 *     change the test makes to it.
 *   - The hash is the pack digest's term of a position in a stream: a bijection of the position for each stream.
 *   - A bfloat16 of a chosen exponent is made by its three fields; one of a chosen value by the reference's rounding
 *     of that value, which is exact for the values asked for here.
 */
#include "linear_card_fixture.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "anemone/pack_digest.h"
#include "linear_reference.h"
#include "weight_format_reference.h"

#define CONTEXT_CAPACITY 200
#define DIGEST_WORD_BYTES ((uint64_t)PACK_DIGEST_WORD_BYTES)
#define UNTOUCHED_TICKET 0ull                   /* zero is never a ticket */
#define THOUSAND 1000u
#define PERCENT 100.0

#define SCALE_STREAM 0x5CA1E5ull                /* added to a recipe's stream: the hashes of its scale codes */
#define CLASS_STREAM 0xC1A55ull                 /* added to a vector's stream: the hashes of its blocks' classes */
#define E8M0_BIAS 127u                          /* an E8M0 byte c stands for 2^(c - 127) */
#define E8M0_LARGEST_CODE 254u
#define VALUES_PER_BLOCK TEST_LINEAR_VALUES_PER_BLOCK
#define OPERAND_BYTES LINEAR_BFLOAT16_BYTES /* of one operand */

/* The fields of a bfloat16, and the bits of a hash that fill them. */
#define BFLOAT16_SIGN_SHIFT 15u
#define BFLOAT16_FRACTION_SHIFT 7u
#define BFLOAT16_FRACTION_MASK 0x7Fu
#define BFLOAT16_EXPONENT_OF_ONE 127u           /* the exponent field of 1.0 */
#define BFLOAT16_LARGEST_FINITE 0x7F7Fu
#define BFLOAT16_LARGEST_EXPONENT_FIELD 254u
#define HASH_SIGN_SHIFT 7u                      /* bit 7: the sign; bits 0 to 6: the fraction */
#define HASH_CHOICE_SHIFT 8u                    /* bits from 8 on: the exponent, or which of two kinds */
#define HASH_WHOLE_SHIFT 16u
#define HASH_SECOND_CHOICE_SHIFT 24u
#define HASH_MANTISSA_SHIFT 40u
#define HASH_CODE_SIGN_SHIFT 48u

/* The values of test_linear_values: an exponent field is its first, less the hash modulo its spread. */
#define PLAIN_FIRST_EXPONENT (BFLOAT16_EXPONENT_OF_ONE + 1u)    /* magnitudes below 4 */
#define PLAIN_EXPONENT_SPREAD 6u                                /* down to 2^-4 */
#define BELOW_FLOOR_FIRST_EXPONENT (BFLOAT16_EXPONENT_OF_ONE - 14u)     /* 2^-14 is below 1e-4 */
#define BELOW_FLOOR_EXPONENT_SPREAD 30u
#define WIDE_FIRST_EXPONENT (BFLOAT16_EXPONENT_OF_ONE + 19u)
#define WIDE_EXPONENT_SPREAD 60u
#define ONE_LARGE_EXPONENT (BFLOAT16_EXPONENT_OF_ONE + 3u)      /* the block's first value: 8 to 16 */
#define ONE_LARGE_REST_FIRST_EXPONENT (BFLOAT16_EXPONENT_OF_ONE - 7u)   /* the others: from 2^-10 of it */
#define ONE_LARGE_REST_EXPONENT_SPREAD 9u
#define NEAR_LARGEST_EXPONENT_SPREAD 40u
#define SUBNORMAL_EXPONENT_SPREAD 2u                            /* exponent fields 0 and 1 */
#define TINY_FIRST_EXPONENT (BFLOAT16_EXPONENT_OF_ONE - 12u)
#define TINY_EXPONENT_SPREAD 17u
#define E4M3_LARGEST 448.0
#define TIES_SCALE_SPREAD 5u                    /* the block's scale: 2^-2 to 2^2 */
#define TIES_SCALE_OFFSET 2u
#define TIES_KINDS 2u
#define TIES_SUBNORMAL_PLACE 10u                /* a multiple of 2^-10 of the scale: a tie of subnormal codes if odd */
#define TIES_SUBNORMAL_WHOLE_COUNT 256u
#define TIES_MIDDLE_FIRST 17u                   /* (17 + 2 m) 2^q: halfway between the mantissas 8 + m and 9 + m */
#define TIES_MIDDLE_STEP 2u
#define TIES_MANTISSA_COUNT 8u
#define TIES_MIDDLE_PLACE 10u                   /* q from -10 */
#define TIES_MIDDLE_PLACE_SPREAD 14u            /* to 3: 31 * 2^3 stays below 448 */

/* The codes of the two code functions. */
#define E2M1_CODE_MASK 0xFu
#define E4M3_ANY_BYTE_ONE_IN 64u
#define E4M3_BYTE_MASK 0xFFu
#define E4M3_NOT_A_NUMBER_MAGNITUDE 0x7Fu
#define E4M3_SIGN_SHIFT 7u
#define E4M3_EXPONENT_SHIFT 3u
#define E4M3_LARGEST_EXPONENT_FIELD 15u
#define E4M3_MANTISSA_MASK 7u
#define E4M3_LARGEST_MANTISSA_OF_LARGEST_EXPONENT 6u    /* 0x7E is 448; 0x7F is no number */
#define E4M3_THINNING_STOP (1u << 14u)          /* the exponent field falls by one per trailing zero bit, to 1 */
#define OUTPUT_INDEX_SHIFT 32u

/* Where a weight lies, for test_linear_card_set_weight: weight_format_seam.h. */
#define OUTPUTS_PER_TILE 16u
#define E2M1_INPUTS_PER_LOAD 64u
#define BYTES_PER_TILE_STRETCH 512u             /* 16 outputs by 64 inputs of 4 bits, or by 32 inputs of 8 bits */
#define E2M1_CODE_BITS 4u
#define BITS_PER_BYTE 8u

/* What a block of 32 values holds: the first EDGE_CLASS_COUNT are the classes of TEST_LINEAR_VALUES_EDGES, the first
 * EXTREME_CLASS_COUNT those of TEST_LINEAR_VALUES_EXTREMES; the classes of TEST_LINEAR_VALUES_SMALL are a table. */
enum values_class {
    CLASS_PLAIN,
    CLASS_ZEROS,
    CLASS_BELOW_FLOOR,
    CLASS_TIES,
    CLASS_WIDE,
    CLASS_ONE_LARGE,
    CLASS_NEAR_LARGEST,
    CLASS_SUBNORMAL,
    CLASS_TINY
};
#define EDGE_CLASS_COUNT 6u
#define EXTREME_CLASS_COUNT 8u

static const enum values_class small_classes[] = {CLASS_PLAIN, CLASS_ZEROS, CLASS_TINY};
#define SMALL_CLASS_COUNT (sizeof(small_classes) / sizeof(small_classes[0]))

_Static_assert(CLASS_NEAR_LARGEST == EDGE_CLASS_COUNT && CLASS_TINY == EXTREME_CLASS_COUNT,
               "the classes of the edges come first, then those of the extremes");

static const struct backend_operations *const backend = &backend_cuda_value;
static int failure_count = 0;
static char context_text[CONTEXT_CAPACITY] = "the linear on the cards";

void test_linear_context(const char *format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(context_text, sizeof(context_text), format, arguments);
    va_end(arguments);
}

void test_linear_fail(const char *format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    (void)fprintf(stderr, "FAILED: %s: ", context_text);
    (void)vfprintf(stderr, format, arguments);
    (void)fprintf(stderr, "\n");
    va_end(arguments);
    failure_count++;
}

bool test_linear_expect(bool condition, const char *what)
{
    if (!condition) {
        test_linear_fail("%s", what);
    }
    return condition;
}

bool test_linear_expect_status(result_status actual, result_status expected,
			       const char *what)
{
    if (actual != expected) {
	    test_linear_fail("%s: the status was \"%s\", not \"%s\"", what,
			     status_text(actual), status_text(expected));
    }
    return actual == expected;
}

int test_linear_failures(void)
{
    return failure_count;
}

bool test_linear_card_open(struct test_linear_card *self, uint32_t card_index, uint64_t work_capacity,
                           uint64_t weights_capacity)
{
	struct card_description_record description;
	void *image = NULL;
	void *received = NULL;
	memset(&description, 0, sizeof(description));
	memset(self, 0, sizeof(*self));
	self->card_index = card_index;
	self->work_capacity = work_capacity;
	self->arena_byte_count = work_capacity + weights_capacity;
	test_linear_context("card %" PRIu32, card_index);
	if (!test_linear_expect(
		    work_capacity != 0 &&
			    work_capacity % BACKEND_STAGING_ALIGNMENT_BYTES ==
				    0 &&
			    weights_capacity <= UINT64_MAX - work_capacity,
		    "the work region is whole staging pages and the arena a size 64 bits hold") ||
	    !test_linear_expect_status(backend->card_describe(card_index,
							      &description),
				       STATUS_OK, "the card is described")) {
		return false;
	}
    (void)printf("card %" PRIu32 ": %s at %s, class %" PRIu32 "\n", card_index, description.name,
                 description.bus_address, description.arithmetic_class);
    self->arithmetic_class = description.arithmetic_class;
    if (description.memory_free_bytes < description.memory_reserved_bytes ||
        description.memory_free_bytes - description.memory_reserved_bytes < self->arena_byte_count) {
        test_linear_fail("the cards are not free: this card has no room for the arena of the test");
        return false;
    }
    if (test_linear_expect_status(backend->card_open(card_index,
						     self->arena_byte_count,
						     &self->card),
				  STATUS_OK, "the card opens") &&
	test_linear_expect_status(
		backend->staging_allocate(self->arena_byte_count, &image),
		STATUS_OK, "the image of the arena is page-locked") &&
	test_linear_expect_status(
		backend->staging_allocate(work_capacity, &received), STATUS_OK,
		"the block of the downloads is page-locked") &&
	test_linear_expect_status(
		linear_kernels_prepare(backend, self->card,
				       self->arithmetic_class, &self->kernels),
		STATUS_OK, "the kernels of the linear are prepared")) {
	    self->image = image;
	    self->received = received;
	    test_linear_card_clear(self);
	    return true;
    }
    backend->card_close(self->card, TEST_LINEAR_WAIT_LIMIT_NANOSECONDS);
    backend->staging_free(received);
    backend->staging_free(image);
    self->card = NULL;
    return false;
}

void test_linear_card_close(struct test_linear_card *self)
{
    backend->card_close(self->card, TEST_LINEAR_WAIT_LIMIT_NANOSECONDS);
    backend->staging_free(self->received);
    backend->staging_free(self->image);
    self->card = NULL;
    self->image = NULL;
    self->received = NULL;
}

void test_linear_card_clear(struct test_linear_card *self)
{
    self->work_byte_count = 0;
    self->weights_byte_count = 0;
    self->guard_count = 0;
    self->is_out_of_room = false;
}

/* The bytes from a range's end to where the next range may begin, less the guard a work range has behind it. */
static uint64_t fill_to_alignment(uint64_t byte_count)
{
    const uint64_t beyond = byte_count % TEST_LINEAR_RANGE_ALIGNMENT_BYTES;
    return beyond == 0 ? 0 : TEST_LINEAR_RANGE_ALIGNMENT_BYTES - beyond;
}

/* Sets byte_count bytes aside at *used_byte_count of a region of capacity bytes that begins at region_offset, with
 * fill_byte in guard_byte_count bytes behind them and up to the alignment. A range of no bytes if it does not fit. */
static struct linear_range set_aside(struct test_linear_card *self,
				     uint64_t region_offset, uint64_t capacity,
				     uint64_t *used_byte_count,
				     uint64_t byte_count,
				     uint64_t guard_byte_count, int fill_byte)
{
	const struct linear_range range = { region_offset + *used_byte_count,
					    byte_count };
	const uint64_t room = capacity - *used_byte_count;
	if (self->image == NULL || byte_count == 0 || byte_count > room ||
	    fill_to_alignment(byte_count) + guard_byte_count >
		    room - byte_count) {
		test_linear_fail("a range of %" PRIu64
				 " bytes does not fit: %" PRIu64 " of %" PRIu64
				 " bytes of its region are left",
				 byte_count, room, capacity);
		self->is_out_of_room = true;
		return (struct linear_range){ 0, 0 };
	}
    const uint64_t behind_byte_count = fill_to_alignment(byte_count) + guard_byte_count;
    memset(self->image + range.byte_offset, 0, byte_count);
    memset(self->image + range.byte_offset + byte_count, fill_byte, behind_byte_count);
    *used_byte_count += byte_count + behind_byte_count;
    return range;
}

struct linear_range test_linear_card_work(struct test_linear_card *self,
					  uint64_t byte_count)
{
    if (self->guard_count >= TEST_LINEAR_MAXIMUM_GUARDS) {
        test_linear_fail("a case with more ranges than the fixture holds guards for");
        self->is_out_of_room = true;
	return (struct linear_range){ 0, 0 };
    }
    const struct linear_range range = set_aside(
	    self, 0, self->work_capacity, &self->work_byte_count, byte_count,
	    TEST_LINEAR_GUARD_BYTES, TEST_LINEAR_GUARD_BYTE);
    if (range.byte_count != 0) {
        self->guard_offsets[self->guard_count] = range.byte_offset + range.byte_count;
        self->guard_count++;
    }
    return range;
}

struct linear_range test_linear_card_weights(struct test_linear_card *self,
					     uint64_t byte_count)
{
    return set_aside(self, self->work_capacity, self->arena_byte_count - self->work_capacity,
                     &self->weights_byte_count, byte_count, 0, 0);
}

bool test_linear_card_has_room(const struct test_linear_card *self)
{
    return !self->is_out_of_room;
}

uint32_t test_linear_card_slices(const struct test_linear_card *self, uint32_t slice_count, bool is_gate_up)
{
#if defined(LINEAR_CLASS_FORMS)
	(void)self;
	(void)is_gate_up;
	return slice_count;
#else
	const uint32_t most =
		is_gate_up ? LINEAR_CLASS_120_EXACT_MOST_GATE_UP_SLICES :
			     LINEAR_CLASS_120_EXACT_MOST_SLICES;
	return self->arithmetic_class == ARITHMETIC_CLASS_SM_120 &&
			       slice_count > most ?
		       most :
		       slice_count;
#endif
}

struct linear_range
test_linear_card_all_weights(const struct test_linear_card *self)
{
	return (struct linear_range){ self->work_capacity,
				      self->weights_byte_count };
}

/* Whether bytes byte_offset .. byte_offset + byte_count - 1 of a range lie in it, and the range in limit bytes. */
static bool is_inside(struct linear_range range, uint64_t byte_offset,
		      uint64_t byte_count, uint64_t limit)
{
    return range.byte_offset <= limit && range.byte_count <= limit - range.byte_offset &&
           byte_offset <= range.byte_count && byte_count <= range.byte_count - byte_offset;
}

bool test_linear_card_put(struct test_linear_card *self,
			  struct linear_range range, uint64_t byte_offset,
			  const void *bytes, uint64_t byte_count)
{
    if (self->image == NULL || bytes == NULL || !is_inside(range, byte_offset, byte_count, self->arena_byte_count)) {
        test_linear_fail("%" PRIu64 " bytes put at byte %" PRIu64 " of a range of %" PRIu64 " bytes", byte_count,
                         byte_offset, range.byte_count);
        return false;
    }
    memcpy(self->image + range.byte_offset + byte_offset, bytes, byte_count);
    return true;
}

bool test_linear_card_get(const struct test_linear_card *self,
			  struct linear_range range, uint64_t byte_offset,
			  void *bytes_out, uint64_t byte_count)
{
    if (self->received == NULL || bytes_out == NULL ||
        !is_inside(range, byte_offset, byte_count, self->work_byte_count)) {
        test_linear_fail("%" PRIu64 " bytes got from byte %" PRIu64 " of a range of %" PRIu64 " bytes", byte_count,
                         byte_offset, range.byte_count);
        return false;
    }
    memcpy(bytes_out, self->received + range.byte_offset + byte_offset, byte_count);
    return true;
}

const uint8_t *test_linear_card_stored(const struct test_linear_card *self,
				       struct linear_range range)
{
    return self->image != NULL && is_inside(range, 0, range.byte_count, self->arena_byte_count)
               ? self->image + range.byte_offset : NULL;
}

unsigned char *test_linear_card_writable(struct test_linear_card *self,
					 struct linear_range range)
{
    return self->image != NULL && range.byte_offset >= self->work_capacity &&
                   is_inside(range, 0, range.byte_count, self->work_capacity + self->weights_byte_count)
               ? self->image + range.byte_offset : NULL;
}

bool test_linear_card_put_codes(struct test_linear_card *self,
				struct linear_range range, uint64_t byte_offset,
				const uint8_t *codes, uint32_t width)
{
	if (self->image == NULL || codes == NULL ||
	    !is_inside(range, byte_offset, width, self->arena_byte_count) ||
	    test_linear_codes_lay_out(codes, width,
				      self->image + range.byte_offset +
					      byte_offset) != STATUS_OK) {
		test_linear_fail("%" PRIu32 " codes put at byte %" PRIu64
				 " of a range of %" PRIu64 " bytes",
				 width, byte_offset, range.byte_count);
		return false;
	}
    return true;
}

bool test_linear_card_expect_codes(const struct test_linear_card *self,
				   struct linear_range range,
				   uint64_t byte_offset, const uint8_t *codes,
				   uint32_t width, const char *what)
{
    uint8_t *const stored = malloc(width);
    const bool is_same =
	    test_linear_expect(
		    stored != NULL && codes != NULL &&
			    test_linear_codes_lay_out(codes, width, stored) ==
				    STATUS_OK,
		    "the reference lays the codes of a vector out") &&
	    test_linear_card_expect_bytes(self, range, byte_offset, stored,
					  width, what);
    free(stored);
    return is_same;
}

bool test_linear_card_put_operands(struct test_linear_card *self,
				   struct linear_range range,
				   uint64_t byte_offset, const uint8_t *codes,
				   const uint8_t *exponents, uint32_t width)
{
    const uint64_t byte_count = (uint64_t)width * OPERAND_BYTES;
    uint16_t *const operands = malloc((size_t)byte_count);
    const bool is_put = operands != NULL &&
			test_linear_operands_lay_out(codes, exponents, width,
						     operands) == STATUS_OK &&
			test_linear_card_put(self, range, byte_offset, operands,
					     byte_count);
    free(operands);
    if (!is_put) {
        test_linear_fail("%" PRIu32 " operands are not put at byte %" PRIu64 " of a range of %" PRIu64 " bytes",
                         width, byte_offset, range.byte_count);
    }
    return is_put;
}

bool test_linear_card_expect_operands(const struct test_linear_card *self,
				      struct linear_range range,
				      uint64_t byte_offset,
				      const uint8_t *codes,
				      const uint8_t *exponents, uint32_t width,
				      const char *what)
{
    uint16_t *const operands = malloc((size_t)width * OPERAND_BYTES);
    const bool is_same =
	    test_linear_expect(
		    operands != NULL && test_linear_operands_lay_out(
						codes, exponents, width,
						operands) == STATUS_OK,
		    "the reference lays the operands of a vector out") &&
	    test_linear_card_expect_bytes(self, range, byte_offset, operands,
					  (uint64_t)width * OPERAND_BYTES,
					  what);
    free(operands);
    return is_same;
}

/* One copy of the image to the arena, begun and waited for. */
static bool upload(struct test_linear_card *self, uint64_t byte_offset, uint64_t byte_count)
{
    uint64_t ticket = UNTOUCHED_TICKET;
    return test_linear_expect_status(
		   backend->upload_begin(self->card, byte_offset,
					 self->image + byte_offset, byte_count,
					 &ticket),
		   STATUS_OK, "an upload begins") &&
	   test_linear_expect_status(
		   backend->upload_wait(self->card, ticket,
					TEST_LINEAR_WAIT_LIMIT_NANOSECONDS),
		   STATUS_OK, "an upload finishes within the limit");
}

bool test_linear_card_load(struct test_linear_card *self)
{
    return test_linear_expect(self->card != NULL && self->work_byte_count != 0 && !self->is_out_of_room,
                              "a case that is loaded has its ranges") &&
           upload(self, 0, self->work_byte_count) &&
           (self->weights_byte_count == 0 || upload(self, self->work_capacity, self->weights_byte_count));
}

bool test_linear_card_upload(struct test_linear_card *self,
			     struct linear_range range, uint64_t byte_offset,
			     uint64_t byte_count)
{
    return test_linear_expect(self->card != NULL && byte_count != 0 &&
                                  is_inside(range, byte_offset, byte_count, self->arena_byte_count),
                              "what is uploaded again lies inside its range") &&
           upload(self, range.byte_offset + byte_offset, byte_count);
}

bool test_linear_card_run(struct test_linear_card *self,
			  const struct kernel_launch *launches,
			  uint32_t launch_count)
{
	return test_linear_expect_status(
		       backend->launch_run(self->card, launches, launch_count),
		       STATUS_OK, "the launches are queued") &&
	       test_linear_expect_status(
		       backend->launch_wait(self->card,
					    TEST_LINEAR_WAIT_LIMIT_NANOSECONDS),
		       STATUS_OK, "the launches finish within the limit");
}

bool test_linear_card_fetch(struct test_linear_card *self)
{
    const uint64_t digested_byte_count = self->work_byte_count - self->work_byte_count % DIGEST_WORD_BYTES;
    const uint64_t prefix_byte_count = self->prefix_byte_count < digested_byte_count
                                           ? self->prefix_byte_count - self->prefix_byte_count % DIGEST_WORD_BYTES
                                           : digested_byte_count;
    uint64_t ticket = UNTOUCHED_TICKET;
    uint64_t digest = 0;
    uint64_t prefix_digest = 0;
    uint32_t broken_guard_count = 0;
    if (!test_linear_expect(self->card != NULL && self->work_byte_count != 0,
			    "a case that is fetched has ranges") ||
	!test_linear_expect_status(
		backend->download_begin(self->card, 0, self->received,
					self->work_byte_count, &ticket),
		STATUS_OK, "the download of the work region begins") ||
	!test_linear_expect_status(
		backend->upload_wait(self->card, ticket,
				     TEST_LINEAR_WAIT_LIMIT_NANOSECONDS),
		STATUS_OK,
		"the download of the work region finishes within the limit")) {
	    return false;
    }
    for (uint32_t guard_index = 0; guard_index < self->guard_count; guard_index++) {
        const uint64_t guard_offset = self->guard_offsets[guard_index];
        const uint64_t guard_byte_count = fill_to_alignment(guard_offset) + TEST_LINEAR_GUARD_BYTES;
        uint64_t byte_index = 0;
        while (byte_index < guard_byte_count && self->received[guard_offset + byte_index] == TEST_LINEAR_GUARD_BYTE) {
            byte_index++;
        }
        if (byte_index < guard_byte_count && broken_guard_count == 0) {
            test_linear_fail("byte %" PRIu64 " of the arena, behind the range that ends at byte %" PRIu64
                             ", was written: no range of the case holds it", guard_offset + byte_index, guard_offset);
        }
        broken_guard_count += byte_index < guard_byte_count ? 1u : 0u;
    }
    if (test_linear_expect_status(pack_digest_of_host_bytes(self->received,
							    digested_byte_count,
							    &digest),
				  STATUS_OK, "the work region is digested") &&
	test_linear_expect_status(pack_digest_of_host_bytes(self->received,
							    prefix_byte_count,
							    &prefix_digest),
				  STATUS_OK, "its first bytes are digested")) {
	    self->digest += digest;
	    self->prefix_digest += prefix_digest;
    }
    return broken_guard_count == 0;
}

uint64_t test_linear_card_differing(const struct test_linear_card *self,
				    struct linear_range first,
				    uint64_t first_byte_offset,
				    struct linear_range second,
				    uint64_t second_byte_offset,
				    uint64_t byte_count,
				    uint64_t *first_differing_out)
{
    uint64_t differing_count = 0;
    uint64_t first_differing = TEST_LINEAR_NOTHING_DIFFERS;
    if (self->received == NULL || !is_inside(first, first_byte_offset, byte_count, self->work_byte_count) ||
        !is_inside(second, second_byte_offset, byte_count, self->work_byte_count)) {
        test_linear_fail("%" PRIu64 " bytes compared beyond a range", byte_count);
        differing_count = byte_count;
        first_differing = 0;
    } else {
        const unsigned char *const first_bytes = self->received + first.byte_offset + first_byte_offset;
        const unsigned char *const second_bytes = self->received + second.byte_offset + second_byte_offset;
        for (uint64_t byte_index = 0; byte_index < byte_count; byte_index++) {
            if (first_bytes[byte_index] != second_bytes[byte_index]) {
                first_differing = differing_count == 0 ? byte_index : first_differing;
                differing_count++;
            }
        }
    }
    if (first_differing_out != NULL) {
        *first_differing_out = first_differing;
    }
    return differing_count;
}

bool test_linear_card_expect_same(const struct test_linear_card *self,
				  struct linear_range first,
				  uint64_t first_byte_offset,
				  struct linear_range second,
				  uint64_t second_byte_offset,
				  uint64_t byte_count, const char *what)
{
    uint64_t first_differing = TEST_LINEAR_NOTHING_DIFFERS;
    const uint64_t differing_count = test_linear_card_differing(self, first, first_byte_offset, second,
                                                                second_byte_offset, byte_count, &first_differing);
    if (differing_count != 0) {
        test_linear_fail("%s: %" PRIu64 " of %" PRIu64 " bytes differ, the first of them byte %" PRIu64, what,
                         differing_count, byte_count, first_differing);
    }
    return differing_count == 0;
}

bool test_linear_card_expect_bytes(const struct test_linear_card *self,
				   struct linear_range range,
				   uint64_t byte_offset, const void *expected,
				   uint64_t byte_count, const char *what)
{
    const unsigned char *const expected_bytes = expected;
    uint64_t differing_count = 0;
    uint64_t first_differing = 0;
    if (self->received == NULL || expected == NULL ||
        !is_inside(range, byte_offset, byte_count, self->work_byte_count)) {
        test_linear_fail("%s: %" PRIu64 " bytes compared beyond a range", what, byte_count);
        return false;
    }
    for (uint64_t byte_index = 0; byte_index < byte_count; byte_index++) {
        if (self->received[range.byte_offset + byte_offset + byte_index] != expected_bytes[byte_index]) {
            first_differing = differing_count == 0 ? byte_index : first_differing;
            differing_count++;
        }
    }
    if (differing_count != 0) {
        test_linear_fail("%s: %" PRIu64 " of %" PRIu64 " bytes are not the expected ones, the first of them byte %"
                         PRIu64 ": 0x%02X, expected 0x%02X", what, differing_count, byte_count, first_differing,
                         (unsigned int)self->received[range.byte_offset + byte_offset + first_differing],
                         (unsigned int)expected_bytes[first_differing]);
    }
    return differing_count == 0;
}

bool test_linear_card_expect_untouched(const struct test_linear_card *self,
				       struct linear_range range,
				       const char *what)
{
    const bool is_untouched =
        self->image != NULL && self->received != NULL && is_inside(range, 0, 0, self->work_byte_count) &&
        memcmp(self->image + range.byte_offset, self->received + range.byte_offset, range.byte_count) == 0;
    if (!is_untouched) {
        test_linear_fail("%s: what the launches only read came back with other bytes", what);
    }
    return is_untouched;
}

bool test_linear_card_expect_weights_untouched(struct test_linear_card *self)
{
    uint64_t card_digest = 0;
    uint64_t host_digest = 1;                   /* another value than the card's, until both are delivered */
    /* the weights region set aside is whole ranges and their fill: a multiple of the alignment, and so of a word */
    return self->weights_byte_count == 0 ||
	   (test_linear_expect_status(
		    backend->digest(self->card, self->work_capacity,
				    self->weights_byte_count,
				    TEST_LINEAR_WAIT_LIMIT_NANOSECONDS,
				    &card_digest),
		    STATUS_OK, "the card digests its matrices") &&
	    test_linear_expect_status(
		    pack_digest_of_host_bytes(self->image + self->work_capacity,
					      self->weights_byte_count,
					      &host_digest),
		    STATUS_OK, "the host digests the matrices of the image") &&
	    test_linear_expect(card_digest == host_digest,
			       "no launch changed a byte of a matrix"));
}

void test_linear_expect_products(const char *what, const uint16_t *outputs, const struct test_linear_product *products,
                                 uint64_t output_count, uint32_t input_size, uint32_t slice_count,
                                 struct test_linear_comparison *total)
{
    struct test_linear_comparison comparison = {0};
    const uint64_t first_outside =
        test_linear_compare(outputs, products, output_count, input_size, slice_count, &comparison);
    if (first_outside != TEST_LINEAR_NO_OUTPUT) {
        test_linear_fail("%s: %" PRIu64 " of %" PRIu64 " outputs are outside the tolerance, the first of them "
                         "output %" PRIu64 ": 0x%04X, where the reference has %.9g (its magnitudes add up to %.9g)",
                         what, comparison.outside_count, output_count, first_outside,
                         (unsigned int)outputs[first_outside], products[first_outside].sum,
                         products[first_outside].magnitude_sum);
    }
    (void)printf("  %s: %" PRIu64 " outputs, %.4f%% bit-equal to the double result rounded once, the largest "
                 "difference %" PRIu32 " last places, %" PRIu64 " outside the tolerance\n",
                 what, comparison.output_count, test_linear_percent(comparison.equal_count, comparison.output_count),
                 comparison.largest_distance, comparison.outside_count);
    if (total != NULL) {
        total->output_count += comparison.output_count;
        total->equal_count += comparison.equal_count;
        total->outside_count += comparison.outside_count;
        total->largest_distance = comparison.largest_distance > total->largest_distance
                                      ? comparison.largest_distance : total->largest_distance;
    }
}

bool test_linear_share_holds(uint64_t equal_count, uint64_t count)
{
    return equal_count * THOUSAND >= count * TEST_LINEAR_SHARE_REQUIRED_PER_THOUSAND;
}

double test_linear_percent(uint64_t part, uint64_t whole)
{
    return whole == 0 ? 0.0 : (double)part * PERCENT / (double)whole;
}

uint64_t test_linear_hash(uint64_t stream, uint64_t position)
{
	return pack_digest_term(position, stream);
}

/* A bfloat16 of an exponent field, its sign and its fraction from a hash. */
static uint16_t bfloat16_of_exponent(uint64_t hash, uint32_t exponent_field)
{
    return (uint16_t)((((uint32_t)(hash >> HASH_SIGN_SHIFT) & 1u) << BFLOAT16_SIGN_SHIFT) |
                      (exponent_field << BFLOAT16_FRACTION_SHIFT) | ((uint32_t)hash & BFLOAT16_FRACTION_MASK));
}

/* The bfloat16 of whole * 2^exponent, of the sign of a hash: whole is below 256, so the value is one exactly. */
static uint16_t bfloat16_of_value(uint64_t hash, double whole, uint32_t exponent_code)
{
    const double magnitude = whole * test_weight_format_e8m0_value(exponent_code);
    return test_linear_bfloat16_nearest(((hash >> HASH_SIGN_SHIFT) & 1u) != 0 ? -magnitude : magnitude);
}

/* One value of a block of the class CLASS_TIES whose scale is 2^(scale_code - 127): see the header. */
static uint16_t tie_value(uint64_t hash, uint32_t value_in_block, uint32_t scale_code)
{
    const uint32_t whole = (uint32_t)(hash >> HASH_WHOLE_SHIFT);
    if (value_in_block == 0) {
        return bfloat16_of_value(hash, E4M3_LARGEST, scale_code);
    }
    if ((hash >> HASH_CHOICE_SHIFT) % TIES_KINDS == 0) {
        return bfloat16_of_value(hash, (double)(whole % TIES_SUBNORMAL_WHOLE_COUNT), scale_code - TIES_SUBNORMAL_PLACE);
    }
    return bfloat16_of_value(hash, (double)(TIES_MIDDLE_FIRST + TIES_MIDDLE_STEP * (whole % TIES_MANTISSA_COUNT)),
                             scale_code - TIES_MIDDLE_PLACE +
                                 (uint32_t)(hash >> HASH_SECOND_CHOICE_SHIFT) % TIES_MIDDLE_PLACE_SPREAD);
}

/* The class of the block a value lies in. */
static enum values_class class_of_block(uint64_t block_hash, enum test_linear_values_kind kind)
{
    switch (kind) {
    case TEST_LINEAR_VALUES_SMALL:
        return small_classes[block_hash % SMALL_CLASS_COUNT];
    case TEST_LINEAR_VALUES_EDGES:
        return (enum values_class)(block_hash % EDGE_CLASS_COUNT);
    case TEST_LINEAR_VALUES_EXTREMES:
        return (enum values_class)(block_hash % EXTREME_CLASS_COUNT);
    case TEST_LINEAR_VALUES_PLAIN:
    default:
        return CLASS_PLAIN;
    }
}

void test_linear_values(uint16_t *values_out, uint32_t width, uint64_t stream, enum test_linear_values_kind kind)
{
    for (uint32_t value_index = 0; value_index < width; value_index++) {
        const uint64_t hash = test_linear_hash(stream, value_index);
        const uint64_t block_hash = test_linear_hash(stream + CLASS_STREAM, value_index / VALUES_PER_BLOCK);
        const uint32_t choice = (uint32_t)(hash >> HASH_CHOICE_SHIFT);
        const bool is_first_of_block = value_index % VALUES_PER_BLOCK == 0;
        uint16_t bits = 0;
        switch (class_of_block(block_hash, kind)) {
        case CLASS_ZEROS:
            bits = bfloat16_of_exponent(hash & ~(uint64_t)BFLOAT16_FRACTION_MASK, 0);
            break;
        case CLASS_BELOW_FLOOR:
            bits = bfloat16_of_exponent(hash, BELOW_FLOOR_FIRST_EXPONENT - choice % BELOW_FLOOR_EXPONENT_SPREAD);
            break;
        case CLASS_TIES:
            bits = tie_value(hash, value_index % VALUES_PER_BLOCK,
                             E8M0_BIAS - TIES_SCALE_OFFSET +
                                 (uint32_t)(block_hash >> HASH_CHOICE_SHIFT) % TIES_SCALE_SPREAD);
            break;
        case CLASS_WIDE:
            bits = bfloat16_of_exponent(hash, WIDE_FIRST_EXPONENT - choice % WIDE_EXPONENT_SPREAD);
            break;
        case CLASS_ONE_LARGE:
            bits = bfloat16_of_exponent(
                hash, is_first_of_block ? ONE_LARGE_EXPONENT
                                        : ONE_LARGE_REST_FIRST_EXPONENT - choice % ONE_LARGE_REST_EXPONENT_SPREAD);
            break;
        case CLASS_NEAR_LARGEST:
            bits = is_first_of_block ? BFLOAT16_LARGEST_FINITE
                                     : bfloat16_of_exponent(hash, BFLOAT16_LARGEST_EXPONENT_FIELD -
                                                                      choice % NEAR_LARGEST_EXPONENT_SPREAD);
            break;
        case CLASS_SUBNORMAL:
            bits = bfloat16_of_exponent(hash, choice % SUBNORMAL_EXPONENT_SPREAD);
            break;
        case CLASS_TINY:
            bits = bfloat16_of_exponent(hash, TINY_FIRST_EXPONENT - choice % TINY_EXPONENT_SPREAD);
            break;
        case CLASS_PLAIN:
        default:
            bits = bfloat16_of_exponent(hash, PLAIN_FIRST_EXPONENT - choice % PLAIN_EXPONENT_SPREAD);
            break;
        }
        values_out[value_index] = bits;
    }
}

uint8_t test_linear_e2m1_code(uint64_t stream, uint32_t output_index, uint32_t input_index)
{
    return (uint8_t)(test_linear_hash(stream, ((uint64_t)output_index << OUTPUT_INDEX_SHIFT) | input_index) &
                     E2M1_CODE_MASK);
}

uint8_t test_linear_e4m3_code(uint64_t stream, uint32_t output_index, uint32_t input_index)
{
    const uint64_t hash = test_linear_hash(stream, ((uint64_t)output_index << OUTPUT_INDEX_SHIFT) | input_index);
    const uint32_t sign = (uint32_t)(hash >> HASH_CODE_SIGN_SHIFT) & 1u;
    uint32_t thinning = (uint32_t)(hash >> HASH_WHOLE_SHIFT) | E4M3_THINNING_STOP;
    uint32_t exponent_field = E4M3_LARGEST_EXPONENT_FIELD;
    uint32_t mantissa = (uint32_t)(hash >> HASH_MANTISSA_SHIFT) & E4M3_MANTISSA_MASK;
    if ((hash >> HASH_CHOICE_SHIFT) % E4M3_ANY_BYTE_ONE_IN == 0) {
        const uint32_t any_byte = (uint32_t)(hash >> HASH_WHOLE_SHIFT) & E4M3_BYTE_MASK;
        const bool is_no_number = (any_byte & E4M3_NOT_A_NUMBER_MAGNITUDE) == E4M3_NOT_A_NUMBER_MAGNITUDE;
        return (uint8_t)(is_no_number ? any_byte ^ 1u : any_byte);
    }
    while ((thinning & 1u) == 0) {
        thinning >>= 1u;
        exponent_field--;
    }
    if (exponent_field == E4M3_LARGEST_EXPONENT_FIELD && mantissa > E4M3_LARGEST_MANTISSA_OF_LARGEST_EXPONENT) {
        mantissa = E4M3_LARGEST_MANTISSA_OF_LARGEST_EXPONENT;
    }
    return (uint8_t)((sign << E4M3_SIGN_SHIFT) | (exponent_field << E4M3_EXPONENT_SHIFT) | mantissa);
}

bool test_linear_card_matrix(struct test_linear_card *self,
			     const struct test_linear_matrix_recipe *recipe,
			     struct linear_range *stored_out)
{
    struct test_weight_format_sizes sizes = {0};
    struct linear_range stored = { 0, 0 };
    bool is_laid_out = false;
    if (!test_linear_expect(
		recipe->code != NULL &&
			recipe->scale_code_base + recipe->scale_code_spread <=
				E8M0_LARGEST_CODE,
		"a recipe has a code function and scale codes that are numbers") ||
	!test_linear_expect_status(
		test_weight_format_sizes_of(recipe->format, &recipe->shape,
					    &sizes),
		STATUS_OK,
		"the reference knows the sizes of a matrix of the test")) {
	    return false;
    }
    uint8_t *const codes = malloc(sizes.weight_count);
    uint8_t *const scale_codes = malloc(sizes.scale_count);
    (void)test_linear_expect(codes != NULL && scale_codes != NULL, "the codes of a matrix have memory");
    if (codes != NULL && scale_codes != NULL) {
        for (uint32_t output_index = 0; output_index < recipe->shape.output_size; output_index++) {
            uint8_t *const row_codes = codes + (uint64_t)output_index * recipe->shape.input_size;
            for (uint32_t input_index = 0; input_index < recipe->shape.input_size; input_index++) {
                row_codes[input_index] = recipe->code(recipe->stream, output_index, input_index);
            }
        }
        for (uint64_t scale_index = 0; scale_index < sizes.scale_count; scale_index++) {
            scale_codes[scale_index] =
                (uint8_t)(recipe->scale_code_base + test_linear_hash(recipe->stream + SCALE_STREAM, scale_index) %
                                                        ((uint64_t)recipe->scale_code_spread + 1));
        }
        stored = test_linear_card_weights(self, sizes.stored_byte_count);
	is_laid_out = stored.byte_count != 0 &&
		      test_linear_expect_status(
			      test_weight_format_lay_out(
				      recipe->format, &recipe->shape, codes,
				      sizes.weight_count, scale_codes,
				      sizes.scale_count,
				      self->image + stored.byte_offset,
				      sizes.stored_byte_count),
			      STATUS_OK,
			      "the reference lays a matrix of the test out");
    }
    free(scale_codes);
    free(codes);
    *stored_out = stored;
    return is_laid_out;
}

bool test_linear_card_set_weight(struct test_linear_card *self,
				 enum weight_format_code format,
				 const struct matrix_shape_record *shape,
				 struct linear_range stored,
				 uint32_t output_index, uint32_t input_index,
				 uint32_t new_code, uint64_t *byte_offset_out)
{
	const bool is_e2m1 = format == WEIGHT_FORMAT_E2M1_SCALED;
	const uint64_t inputs_per_stretch = is_e2m1 ? E2M1_INPUTS_PER_LOAD :
						      VALUES_PER_BLOCK;
	const uint64_t stretches_per_tile =
		shape->input_size / inputs_per_stretch;
	/* both formats keep the weights of 16 outputs by one such stretch of inputs in 512 consecutive bytes */
	const uint64_t first_byte =
		((uint64_t)(output_index / OUTPUTS_PER_TILE) *
			 stretches_per_tile +
		 input_index / inputs_per_stretch) *
		BYTES_PER_TILE_STRETCH;
	const uint8_t *const stored_bytes =
		test_linear_card_stored(self, stored);
	struct test_weight_format_weight before = { 0 };
	if (stored_bytes == NULL ||
	    test_weight_format_read_weight(format, shape, stored_bytes,
					   stored.byte_count, output_index,
					   input_index, &before) != STATUS_OK ||
	    new_code > (is_e2m1 ? E2M1_CODE_MASK : E4M3_BYTE_MASK)) {
		test_linear_fail(
			"a weight to change that the matrix does not hold, or a code that is none");
		return false;
	}
    for (uint64_t byte_offset = first_byte; byte_offset < first_byte + BYTES_PER_TILE_STRETCH; byte_offset++) {
        for (uint32_t shift = 0; shift < (is_e2m1 ? BITS_PER_BYTE : E2M1_CODE_BITS); shift += E2M1_CODE_BITS) {
            struct test_weight_format_weight after = {0};
            const uint32_t old_bits = stored_bytes[byte_offset];
            const uint32_t code_mask = is_e2m1 ? E2M1_CODE_MASK << shift : E4M3_BYTE_MASK;
            const uint8_t old_byte = (uint8_t)old_bits;
            const uint8_t probe_byte = (uint8_t)(old_bits ^ (1u << shift));
            const uint8_t new_byte = (uint8_t)((old_bits & ~code_mask) | ((new_code << shift) & code_mask));
	    if (!test_linear_card_put(self, stored, byte_offset, &probe_byte,
				      1) ||
		test_weight_format_read_weight(
			format, shape, stored_bytes, stored.byte_count,
			output_index, input_index, &after) != STATUS_OK) {
		    return false;
	    }
	    /* the byte the weight follows gets the new code; any other gets its own byte back */
            if (!test_linear_card_put(self, stored, byte_offset, after.code != before.code ? &new_byte : &old_byte,
                                      1)) {
                return false;
            }
            if (after.code != before.code) {
                *byte_offset_out = byte_offset;
                return true;
            }
        }
    }
    test_linear_fail("no stored byte holds the weight of output %" PRIu32 " and input %" PRIu32, output_index,
                     input_index);
    return false;
}
