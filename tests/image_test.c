/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Intrinsic hostile-input cases always run. An optional official fixture stream
 * compares decoder RGB, resize planning and every BF16 patch word independently. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "anemone/image.h"

#define TEST_PAIR 2u
#define TEST_TRIPLE 3u
#define TEST_QUAD 4u
#define TEST_BYTE_BITS 8u
#define TEST_WORD_BITS 16u
#define TEST_TOP_SHIFT 24u
#define TEST_BYTE_MAXIMUM 255u
#define TEST_BUFFER_BYTES 1024u
#define TEST_PNG_SIGNATURE_BYTES 8u
#define TEST_CRC_POLYNOMIAL UINT32_C(0xedb88320)
#define TEST_ADLER_MODULUS 65521u
#define TEST_PNG_HEADER_BYTES 13u
#define TEST_PNG_RGB 2u
#define TEST_PNG_IHDR_OFFSET 16u
#define TEST_PNG_CRC_OFFSET 29u
#define TEST_PNG_IDAT_DATA 41u
#define TEST_PNG_ZLIB_BYTES 15u
#define TEST_DEFLATE_STORED_LENGTH 3u
#define TEST_DEFLATE_STORED_COMPLEMENT 5u
#define TEST_PNG_DEPTH_OFFSET 24u
#define TEST_PNG_COLOR_OFFSET 25u
#define TEST_PNG_16_DEPTH 16u
#define TEST_PNG_RGBA_COLOR 6u
#define TEST_DECOMPRESSED_WIDTH 4096u
#define TEST_DECOMPRESSED_HEIGHT 4096u
#define TEST_ZLIB_HEADER_BYTES 2u
#define TEST_BAD_DYNAMIC_HEADER UINT32_C(0x04920005)
#define TEST_DEFLATE_TOO_LONG 5u
#define TEST_DEFLATE_TOO_LONG_COMPLEMENT 250u
#define TEST_JPEG_DHT_COUNT_OFFSET 89u
#define TEST_JPEG_BAD_HUFFMAN_COUNT 3u
#define TEST_JPEG_ENTROPY_FROM_END 3u
#define TEST_JPEG_BLOCK 8u
#define TEST_JPEG_COEFFICIENTS 64u
#define TEST_JPEG_DQT_BYTES 67u
#define TEST_JPEG_DHT_BYTES 38u
#define TEST_JPEG_GRAY 128u
#define TEST_GOLDEN_NAME_BYTES 64u
#define TEST_GOLDEN_FIELDS 8u
#define TEST_GOLDEN_SIGNATURE_BYTES 8u
#define TEST_GOLDEN_MAX_BYTES (512u * 1024u * 1024u)
#define TEST_GOLDEN_MAX_CASES 256u
#define TEST_GOLDEN_ENCODED_FIELD 0u
#define TEST_GOLDEN_WIDTH_FIELD 1u
#define TEST_GOLDEN_HEIGHT_FIELD 2u
#define TEST_GOLDEN_VIT_HEIGHT_FIELD 3u
#define TEST_GOLDEN_VIT_WIDTH_FIELD 4u
#define TEST_GOLDEN_LLM_HEIGHT_FIELD 5u
#define TEST_GOLDEN_LLM_WIDTH_FIELD 6u
#define TEST_GOLDEN_ELEMENTS_FIELD 7u
#define TEST_BAD_WIDTH 32769u
#define TEST_CORRUPT_MASK 128u
#define TEST_FUZZ_CASES 1000u
#define TEST_FUZZ_STEP 1103515245u
#define TEST_FUZZ_SEED 12345u
#define TEST_FUZZ_SHIFT 16u
#define TEST_EXPECTED_RED 17u
#define TEST_EXPECTED_GREEN 131u
#define TEST_EXPECTED_BLUE 249u
#define TEST_MIN_GRID 39u
#define TEST_MIN_TOKENS 184u

static unsigned failures;
static void image_test_require(bool condition, const char *case_name)
{
	if (!condition) {
		(void)fprintf(stderr, "FAILED: image %s\n", case_name);
		++failures;
	}
}
static void image_test_u32be(uint8_t *bytes, uint32_t value)
{
	bytes[0] = (uint8_t)(value >> TEST_TOP_SHIFT);
	bytes[1] = (uint8_t)(value >> TEST_WORD_BITS);
	bytes[TEST_PAIR] = (uint8_t)(value >> TEST_BYTE_BITS);
	bytes[TEST_TRIPLE] = (uint8_t)value;
}
static uint32_t image_test_crc(const uint8_t *bytes, size_t count)
{
	uint32_t crc = UINT32_MAX;
	for (size_t offset = 0; offset < count; ++offset) {
		crc ^= bytes[offset];
		for (unsigned bit = 0; bit < TEST_BYTE_BITS; ++bit)
			crc = (crc >> 1u) ^
			      ((crc & 1u) != 0 ? TEST_CRC_POLYNOMIAL : 0);
	}
	return ~crc;
}
static size_t image_test_chunk(uint8_t *output, const char *kind,
			       const uint8_t *bytes, size_t count)
{
	image_test_u32be(output, (uint32_t)count);
	memcpy(output + TEST_QUAD, kind, TEST_QUAD);
	if (count != 0)
		memcpy(output + (size_t)TEST_PAIR * TEST_QUAD, bytes, count);
	image_test_u32be(output + (size_t)TEST_PAIR * TEST_QUAD + count,
			 image_test_crc(output + TEST_QUAD, count + TEST_QUAD));
	return count + (size_t)TEST_TRIPLE * TEST_QUAD;
}
static size_t image_test_png(uint8_t *output)
{
	static const uint8_t signature[] = { 137, 80, 78, 71, 13, 10, 26, 10 };
	static const uint8_t header[] = {
		0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0
	};
	static const uint8_t zlib_template[] = { 0x78,
						 0x01,
						 0x01,
						 0x04,
						 0x00,
						 0xfb,
						 0xff,
						 0x00,
						 TEST_EXPECTED_RED,
						 TEST_EXPECTED_GREEN,
						 TEST_EXPECTED_BLUE,
						 0,
						 0,
						 0,
						 0 };
	uint8_t zlib[sizeof(zlib_template)];
	memcpy(zlib, zlib_template, sizeof(zlib));
	uint32_t first = 1, second = 0;
	for (unsigned offset = 0; offset < TEST_QUAD; ++offset) {
		first = (first + zlib[sizeof(zlib) -
				      (size_t)TEST_PAIR * TEST_QUAD + offset]) %
			TEST_ADLER_MODULUS;
		second = (second + first) % TEST_ADLER_MODULUS;
	}
	image_test_u32be(zlib + sizeof(zlib) - TEST_QUAD,
			 (second << TEST_WORD_BITS) | first);
	memcpy(output, signature, sizeof(signature));
	size_t length = sizeof(signature);
	length += image_test_chunk(output + length, "IHDR", header,
				   sizeof(header));
	length += image_test_chunk(output + length, "IDAT", zlib, sizeof(zlib));
	length += image_test_chunk(output + length, "IEND", NULL, 0);
	return length;
}
static size_t image_test_jpeg(uint8_t *output)
{
	static const uint8_t frame[] = { 0xff, 0xc0, 0, 11, 8,	  0, 1,
					 0,    1,    1, 1,  0x11, 0 };
	static const uint8_t scan[] = { 0xff, 0xda, 0, 8,    1,	   1,	0,
					0,    63,   0, 0x3f, 0xff, 0xd9 };
	static const uint8_t start[] = { 0xff, 0xd8, 0xff,
					 0xdb, 0,    TEST_JPEG_DQT_BYTES,
					 0 };
	static const uint8_t huffman_start[] = { 0xff, 0xc4, 0,
						 TEST_JPEG_DHT_BYTES };
	memcpy(output, start, sizeof(start));
	size_t position = sizeof(start);
	memset(output + position, 1, TEST_JPEG_COEFFICIENTS);
	position += TEST_JPEG_COEFFICIENTS;
	memcpy(output + position, frame, sizeof(frame));
	position += sizeof(frame);
	memcpy(output + position, huffman_start, sizeof(huffman_start));
	position += sizeof(huffman_start);
	for (unsigned table = 0; table < TEST_PAIR; ++table) {
		output[position++] = (uint8_t)(table << TEST_QUAD);
		output[position++] = 1;
		memset(output + position, 0, TEST_WORD_BITS - 1u);
		position += TEST_WORD_BITS - 1u;
		output[position++] = 0;
	}
	memcpy(output + position, scan, sizeof(scan));
	return position + sizeof(scan);
}
static void image_test_truncations(const uint8_t *bytes, size_t length)
{
	for (size_t count = 0; count < length; ++count) {
		struct image_rgb sentinel = { .pixels = NULL,
					      .width = TEST_BAD_WIDTH,
					      .height = 1 };
		result_status status = image_decode(bytes, count, &sentinel);
		image_test_require(status != STATUS_OK, "truncation refused");
		if (status == STATUS_OK)
			image_rgb_destroy(&sentinel);
		else
			image_test_require(sentinel.pixels == NULL &&
						   sentinel.width ==
							   TEST_BAD_WIDTH &&
						   sentinel.height == 1,
					   "decode leaves outputs untouched");
	}
}
static void image_test_mutations(const uint8_t *source, size_t length)
{
	uint8_t bytes[TEST_BUFFER_BYTES];
	uint32_t random = TEST_FUZZ_SEED;
	for (unsigned attempt = 0; attempt < TEST_FUZZ_CASES; ++attempt) {
		memcpy(bytes, source, length);
		random = random * TEST_FUZZ_STEP + TEST_FUZZ_SEED;
		size_t offset = (random >> TEST_FUZZ_SHIFT) % length;
		random = random * TEST_FUZZ_STEP + TEST_FUZZ_SEED;
		bytes[offset] ^= (uint8_t)(1u << ((random >> TEST_FUZZ_SHIFT) %
						  TEST_BYTE_BITS));
		struct image_rgb image = { 0 };
		result_status status = image_decode(bytes, length, &image);
		if (status == STATUS_OK) {
			image_test_require(
				image.width != 0 && image.height != 0 &&
					(uint64_t)image.width * image.height <=
						IMAGE_MAX_PIXELS,
				"mutation success dimensions");
			image_rgb_destroy(&image);
		} else {
			image_test_require(
				image.pixels == NULL && image.width == 0 &&
					image.height == 0,
				"mutation failure leaves output untouched");
		}
	}
}
static void image_test_intrinsic(void)
{
	uint8_t bytes[TEST_BUFFER_BYTES];
	size_t length = image_test_png(bytes);
	struct image_rgb image = { 0 };
	result_status status = image_decode(bytes, length, &image);
	image_test_require(status == STATUS_OK && image.width == 1 &&
				   image.height == 1,
			   "stored PNG decode");
	if (status == STATUS_OK) {
		image_test_require(
			image.pixels[0] == TEST_EXPECTED_RED &&
				image.pixels[1] == TEST_EXPECTED_GREEN &&
				image.pixels[TEST_PAIR] == TEST_EXPECTED_BLUE,
			"PNG RGB values");
		image_rgb_destroy(&image);
	}
	image_test_truncations(bytes, length);
	image_test_mutations(bytes, length);
	bytes[TEST_PNG_CRC_OFFSET] ^= TEST_CORRUPT_MASK;
	status = image_decode(bytes, length, &image);
	image_test_require(status == STATUS_INVALID_ARGUMENT,
			   "bad PNG checksum");
	length = image_test_png(bytes);
	image_test_u32be(bytes + TEST_PNG_IHDR_OFFSET, TEST_BAD_WIDTH);
	image_test_u32be(
		bytes + TEST_PNG_CRC_OFFSET,
		image_test_crc(bytes + TEST_PNG_IHDR_OFFSET - TEST_QUAD,
			       TEST_PNG_HEADER_BYTES + TEST_QUAD));
	status = image_decode(bytes, length, &image);
	image_test_require(status == STATUS_INVALID_ARGUMENT,
			   "dimension bound before inflate");
	length = image_test_png(bytes);
	bytes[TEST_PNG_IDAT_DATA + TEST_PNG_ZLIB_BYTES - 1u] ^= 1u;
	image_test_u32be(bytes + TEST_PNG_IDAT_DATA + TEST_PNG_ZLIB_BYTES,
			 image_test_crc(bytes + TEST_PNG_IDAT_DATA - TEST_QUAD,
					TEST_PNG_ZLIB_BYTES + TEST_QUAD));
	status = image_decode(bytes, length, &image);
	image_test_require(status == STATUS_INVALID_ARGUMENT,
			   "bad zlib Adler with valid PNG CRC");
	length = image_test_png(bytes);
	bytes[TEST_PNG_IDAT_DATA + TEST_DEFLATE_STORED_LENGTH] =
		TEST_DEFLATE_TOO_LONG;
	bytes[TEST_PNG_IDAT_DATA + TEST_DEFLATE_STORED_COMPLEMENT] =
		TEST_DEFLATE_TOO_LONG_COMPLEMENT;
	image_test_u32be(bytes + TEST_PNG_IDAT_DATA + TEST_PNG_ZLIB_BYTES,
			 image_test_crc(bytes + TEST_PNG_IDAT_DATA - TEST_QUAD,
					TEST_PNG_ZLIB_BYTES + TEST_QUAD));
	status = image_decode(bytes, length, &image);
	image_test_require(status == STATUS_INVALID_ARGUMENT,
			   "inflate output bound with valid PNG CRC");
	length = image_test_png(bytes);
	uint32_t bad_dynamic = TEST_BAD_DYNAMIC_HEADER;
	for (unsigned offset = 0; offset < TEST_QUAD; ++offset)
		bytes[TEST_PNG_IDAT_DATA + TEST_ZLIB_HEADER_BYTES + offset] =
			(uint8_t)(bad_dynamic >> (offset * TEST_BYTE_BITS));
	image_test_u32be(bytes + TEST_PNG_IDAT_DATA + TEST_PNG_ZLIB_BYTES,
			 image_test_crc(bytes + TEST_PNG_IDAT_DATA - TEST_QUAD,
					TEST_PNG_ZLIB_BYTES + TEST_QUAD));
	status = image_decode(bytes, length, &image);
	image_test_require(status == STATUS_INVALID_ARGUMENT,
			   "oversubscribed deflate Huffman tree");
	length = image_test_png(bytes);
	image_test_u32be(bytes + TEST_PNG_IHDR_OFFSET, TEST_DECOMPRESSED_WIDTH);
	image_test_u32be(bytes + TEST_PNG_IHDR_OFFSET + TEST_QUAD,
			 TEST_DECOMPRESSED_HEIGHT);
	bytes[TEST_PNG_DEPTH_OFFSET] = TEST_PNG_16_DEPTH;
	bytes[TEST_PNG_COLOR_OFFSET] = TEST_PNG_RGBA_COLOR;
	image_test_u32be(
		bytes + TEST_PNG_CRC_OFFSET,
		image_test_crc(bytes + TEST_PNG_IHDR_OFFSET - TEST_QUAD,
			       TEST_PNG_HEADER_BYTES + TEST_QUAD));
	status = image_decode(bytes, length, &image);
	image_test_require(status == STATUS_INVALID_ARGUMENT,
			   "decompressed-byte limit before inflate");
	length = image_test_jpeg(bytes);
	status = image_decode(bytes, length, &image);
	image_test_require(status == STATUS_OK && image.width == 1 &&
				   image.height == 1,
			   "baseline JPEG decode");
	if (status == STATUS_OK) {
		image_test_require(image.pixels[0] == TEST_JPEG_GRAY &&
					   image.pixels[1] == TEST_JPEG_GRAY &&
					   image.pixels[TEST_PAIR] ==
						   TEST_JPEG_GRAY,
				   "JPEG DC and grayscale");
		image_rgb_destroy(&image);
	}
	image_test_truncations(bytes, length);
	image_test_mutations(bytes, length);
	length = image_test_jpeg(bytes);
	bytes[TEST_JPEG_DHT_COUNT_OFFSET] = TEST_JPEG_BAD_HUFFMAN_COUNT;
	status = image_decode(bytes, length, &image);
	image_test_require(status == STATUS_INVALID_ARGUMENT,
			   "oversubscribed JPEG Huffman table");
	length = image_test_jpeg(bytes);
	bytes[length - TEST_JPEG_ENTROPY_FROM_END] = TEST_BYTE_MAXIMUM;
	status = image_decode(bytes, length, &image);
	image_test_require(status == STATUS_INVALID_ARGUMENT,
			   "marker inside JPEG entropy");
	status = image_decode(bytes, IMAGE_MAX_BYTES + 1u, &image);
	image_test_require(status == STATUS_INVALID_ARGUMENT,
			   "byte bound before read");
	status = image_decode("GIF89a", sizeof("GIF89a") - 1u, &image);
	image_test_require(status == STATUS_UNSUPPORTED, "unknown format");
	status = image_decode(NULL, 1, &image);
	image_test_require(status == STATUS_INVALID_ARGUMENT, "null bytes");
	struct image_plan plan;
	status = image_grid(1, 1, &plan);
	image_test_require(status == STATUS_OK &&
				   plan.vit_height == TEST_MIN_GRID &&
				   plan.vit_width == TEST_MIN_GRID &&
				   plan.token_count == TEST_MIN_TOKENS,
			   "minimum grid");
	status = image_grid(TEST_BAD_WIDTH, 1, &plan);
	image_test_require(status == STATUS_INVALID_ARGUMENT,
			   "plan dimension bound");
}
static bool image_test_read(FILE *file, void *data, size_t bytes)
{
	if (bytes == 0)
		return true;
	if (ferror(file) || feof(file))
		return false;
	return fread(data, 1, bytes, file) == bytes;
}
static uint32_t image_test_le32(const uint8_t *bytes)
{
	return bytes[0] | ((uint32_t)bytes[1] << TEST_BYTE_BITS) |
	       ((uint32_t)bytes[TEST_PAIR] << TEST_WORD_BITS) |
	       ((uint32_t)bytes[TEST_TRIPLE] << TEST_TOP_SHIFT);
}
static bool image_test_golden_case(FILE *file)
{
	char name[TEST_GOLDEN_NAME_BYTES + 1u] = { 0 };
	uint8_t fields[TEST_GOLDEN_FIELDS * TEST_QUAD];
	if (!image_test_read(file, name, TEST_GOLDEN_NAME_BYTES) ||
	    !image_test_read(file, fields, sizeof(fields)))
		return false;
	uint32_t values[TEST_GOLDEN_FIELDS];
	for (unsigned index = 0; index < TEST_GOLDEN_FIELDS; ++index)
		values[index] =
			image_test_le32(fields + (size_t)index * TEST_QUAD);
	size_t encoded_bytes = values[TEST_GOLDEN_ENCODED_FIELD];
	uint32_t width = values[TEST_GOLDEN_WIDTH_FIELD],
		 height = values[TEST_GOLDEN_HEIGHT_FIELD];
	size_t patch_elements = values[TEST_GOLDEN_ELEMENTS_FIELD];
	if (encoded_bytes > IMAGE_MAX_BYTES || width == 0 || height == 0 ||
	    width > IMAGE_MAX_DIMENSION || height > IMAGE_MAX_DIMENSION ||
	    (uint64_t)width * height > IMAGE_MAX_PIXELS ||
	    patch_elements > (size_t)IMAGE_MAX_PATCHES * IMAGE_PATCH_ELEMENTS)
		return false;
	size_t rgb_bytes = (size_t)width * height * IMAGE_CHANNELS;
	uint8_t *encoded = malloc(encoded_bytes), *rgb = malloc(rgb_bytes);
	uint16_t *golden = malloc(patch_elements * sizeof(*golden));
	uint16_t *patches = malloc(patch_elements * sizeof(*patches));
	struct image_rgb decoded = { 0 };
	bool success = false;
	if (encoded == NULL || rgb == NULL || golden == NULL || patches == NULL)
		goto cleanup;
	if (!image_test_read(file, encoded, encoded_bytes) ||
	    !image_test_read(file, rgb, rgb_bytes) ||
	    !image_test_read(file, golden, patch_elements * sizeof(*golden)))
		goto cleanup;
	struct image_plan plan;
	result_status status = image_grid(width, height, &plan);
	if (status != STATUS_OK ||
	    plan.vit_height != values[TEST_GOLDEN_VIT_HEIGHT_FIELD] ||
	    plan.vit_width != values[TEST_GOLDEN_VIT_WIDTH_FIELD] ||
	    plan.llm_height != values[TEST_GOLDEN_LLM_HEIGHT_FIELD] ||
	    plan.llm_width != values[TEST_GOLDEN_LLM_WIDTH_FIELD] ||
	    plan.patch_elements != patch_elements)
		goto cleanup;
	status = image_decode(encoded, encoded_bytes, &decoded);
	if (status != STATUS_OK || decoded.width != width ||
	    decoded.height != height)
		goto cleanup;
	size_t rgb_differences = 0, maximum_difference = 0;
	for (size_t offset = 0; offset < rgb_bytes; ++offset) {
		int difference = abs((int)decoded.pixels[offset] - rgb[offset]);
		if (difference != 0)
			++rgb_differences;
		if ((size_t)difference > maximum_difference)
			maximum_difference = (size_t)difference;
	}
	struct image_rgb reference_rgb = { .pixels = rgb,
					   .width = width,
					   .height = height };
	status = image_preprocess(&reference_rgb, &plan, patches,
				  patch_elements);
	if (status != STATUS_OK)
		goto cleanup;
	size_t preprocess_differences = 0;
	for (size_t offset = 0; offset < patch_elements; ++offset)
		if (patches[offset] != golden[offset])
			++preprocess_differences;
	status = image_preprocess(&decoded, &plan, patches, patch_elements);
	if (status != STATUS_OK)
		goto cleanup;
	size_t end_to_end_differences = 0;
	for (size_t offset = 0; offset < patch_elements; ++offset)
		if (patches[offset] != golden[offset])
			++end_to_end_differences;
	(void)printf(
		"image fixture %s RGB %zu max %zu preprocess %zu end_to_end %zu\n",
		name, rgb_differences, maximum_difference,
		preprocess_differences, end_to_end_differences);
	success = rgb_differences == 0 && preprocess_differences == 0 &&
		  end_to_end_differences == 0;
cleanup:
	if (!success)
		(void)fprintf(stderr, "FAILED: image fixture %s\n", name);
	image_rgb_destroy(&decoded);
	free(encoded);
	free(rgb);
	free(golden);
	free(patches);
	return success;
}
static void image_test_golden(const char *path)
{
	FILE *file = fopen(path, "rb");
	if (file == NULL) {
		image_test_require(false, "open golden");
		return;
	}
	uint8_t header[TEST_GOLDEN_SIGNATURE_BYTES + TEST_QUAD];
	bool valid = image_test_read(file, header, sizeof(header)) &&
		     memcmp(header, "ANIMG001", TEST_GOLDEN_SIGNATURE_BYTES) ==
			     0;
	uint32_t cases =
		valid ? image_test_le32(header + TEST_GOLDEN_SIGNATURE_BYTES) :
			0;
	if (!valid || cases == 0 || cases > TEST_GOLDEN_MAX_CASES)
		image_test_require(false, "golden header");
	else {
		bool compared = true;
		for (uint32_t index = 0; index < cases; ++index) {
			if (!image_test_golden_case(file)) {
				image_test_require(false, "golden case");
				compared = false;
				break;
			}
		}
		if (compared)
			image_test_require(fgetc(file) == EOF && !ferror(file),
					   "golden consumed exactly");
	}
	image_test_require(fclose(file) == 0, "close golden");
}
int main(int argument_count, char **arguments)
{
	image_test_intrinsic();
	if (argument_count == (int)TEST_PAIR)
		image_test_golden(arguments[1]);
	else
		image_test_require(argument_count == 1,
				   "usage: image_test [golden stream]");
	(void)printf("image_test: %u failures\n", failures);
	return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
