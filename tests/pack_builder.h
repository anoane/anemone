/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * pack_builder.h: writes a small valid pack into memory, for the tests of the pack reader, the placement and the
 * loader, and walks an open pack as its users will. Test support: no part of the engine.
 *
 * Contract:
 *   - test_pack_build writes a pack that obeys every rule of pack.h: the header, the tables and the extents, with
 *     the digests pack_digest_of_host_bytes gives. Units and extents lie in the order of the plan. The
 *     bytes of an extent are a pattern drawn from content_seed, the extent's index and the position, so two
 *     extents never hold the same bytes and a misplaced block is seen.
 *   - Every extent holds tensors_per_extent tensors named "unit<u>.extent<e>.tensor<t>" with the numbers in
 *     decimal, zero-filled to TEST_PACK_NAME_DIGITS digits; they share the extent's bytes in equal parts of whole
 *     alignment units (what is left over belongs to no tensor), one dimension each, element type INT8. Every
 *     tensor is plain: weight_format WEIGHT_FORMAT_NONE, bits_per_weight 0. A test that wants a matrix of
 *     a weight format changes the record of a tensor and seals the pack again.
 *   - The pack is one allocation aligned to PACK_BLOCK_BYTES; the caller frees it with free(). A plan
 *     that breaks a rule of pack.h (no unit, a unit without extents or blocks, a kind out of range, an on-demand
 *     unit first, a model name that does not fit, tensors that do not fit the extent) is
 *     STATUS_INVALID_ARGUMENT: a test that wants a broken pack builds a valid one and breaks its bytes.
 *   - test_pack_reseal recomputes both digests of a pack whose header or tables a test changed on purpose, so
 *     that the change reaches the rule it was made for and is not stopped at the digest.
 *   - test_pack_walk follows every index, offset and count an open pack gives, as its users will, and holds the
 *     pack against what pack.h says of an open pack, whatever the bytes it was opened from were: the header, the
 *     units derived from the order of the extents, the extents one after another up to the end of the file, and
 *     every tensor: inside its extent, aligned, found by its name, and holding what it says (a plain tensor no
 *     bits per weight; a matrix of a weight format two dimensions of UINT8 in exactly the bytes that
 *     tests/weight_format_reference.h counts for its format, its shape and its bits per weight). It reads every
 *     byte of every extent and of every tensor, so that a sanitizer sees what a rule failed to bound; a name is
 *     read by the search for it.
 */
#ifndef TEST_PACK_BUILDER_H
#define TEST_PACK_BUILDER_H

#include <stdint.h>

#include "anemone/pack.h"
#include "anemone/status.h"

#define TEST_PACK_NAME_DIGITS 5

struct test_pack_unit_plan {
	uint32_t unit_kind; /* enum placement_unit_kind */
	uint32_t layer_index;
	uint32_t extent_count;
	uint32_t blocks_per_extent;
};

struct test_pack_plan {
    const char *model_name;
    const struct test_pack_unit_plan *units;
    uint32_t unit_count;
    uint32_t tensors_per_extent;                     /* may be 0 */
    uint64_t content_seed;
};

MUST_CHECK result_status test_pack_build(const struct test_pack_plan *plan,
					 void **pack_bytes_out,
					 uint64_t *pack_byte_count_out);

/* pack_bytes is a pack test_pack_build wrote, with a header that still gives its true tables_byte_count. */
void test_pack_reseal(void *pack_bytes);

/* The byte test_pack_build wrote at byte_index of the extent extent_index: what a test compares an arena with. */
unsigned char test_pack_content_byte(uint64_t content_seed, uint32_t extent_index, uint64_t byte_index);

/* pack is open and pack_bytes are the pack_byte_count bytes it was opened from. Returns what is not as pack.h says,
 * in words for a test to print, or NULL; a missing argument is such a thing. Every byte it reads is added to
 * *byte_sum, which the test prints, so that no read is optimised away. */
const char *test_pack_walk(const struct pack_record *pack,
			   const unsigned char *pack_bytes,
			   uint64_t pack_byte_count, uint64_t *byte_sum);

#endif
