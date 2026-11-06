/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* The entropy readers never read beyond their declared byte slice. PNG inflate is
 * bounded by the exact sum of scanline lengths, not by attacker-supplied zlib sizes. */
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "anemone/image.h"

#define JPEG_SIGNATURE_SECOND 0xd8u
#define BYTE_BITS 8u
#define BYTE_MAXIMUM 255u
#define HALF_BYTE_BITS 4u
#define WORD_BITS 16u
#define WORD_LOW_MASK 65535u
#define PNG_SIGNATURE_BYTES 8u
#define PNG_CHUNK_OVERHEAD 12u
#define PNG_IHDR_BYTES 13u
#define PNG_CRC_POLYNOMIAL UINT32_C(0xedb88320)
#define PNG_ADLER_MODULUS 65521u
#define PNG_PALETTE_ENTRIES 256u
#define PNG_GREY 0u
#define PNG_RGB 2u
#define PNG_INDEXED 3u
#define PNG_GREY_ALPHA 4u
#define PNG_RGBA 6u
#define PNG_MAX_CHANNELS 4u
#define PNG_FILTER_NONE 0u
#define PNG_FILTER_SUB 1u
#define PNG_FILTER_UP 2u
#define PNG_FILTER_AVERAGE 3u
#define PNG_FILTER_PAETH 4u
#define PNG_ADAM_PASSES 7u
#define PNG_16_DEPTH 16u
#define PNG_8_DEPTH 8u
#define PNG_4_DEPTH 4u
#define PNG_2_DEPTH 2u
#define PNG_1_DEPTH 1u
#define ZLIB_HEADER_BYTES 2u
#define ZLIB_CHECKSUM_BYTES 4u
#define ZLIB_METHOD_DEFLATE 8u
#define ZLIB_HEADER_MODULUS 31u
#define ZLIB_DICTIONARY_FLAG 32u
#define DEFLATE_MAX_BITS 15u
#define DEFLATE_MAX_SYMBOLS 288u
#define DEFLATE_MAX_DISTANCES 32u
#define DEFLATE_END 256u
#define DEFLATE_FIRST_LENGTH 257u
#define DEFLATE_LAST_LENGTH 285u
#define DEFLATE_CODE_LENGTHS 19u
#define DEFLATE_LENGTH_BASE 257u
#define DEFLATE_DISTANCE_BASE 1u
#define DEFLATE_CODE_BASE 4u
#define DEFLATE_LENGTH_REPEAT 16u
#define DEFLATE_ZERO_SHORT 17u
#define DEFLATE_ZERO_LONG 18u
#define DEFLATE_REPEAT_MINIMUM 3u
#define DEFLATE_ZERO_LONG_MINIMUM 11u
#define DEFLATE_HEADER_BITS 3u
#define DEFLATE_HLIT_BITS 5u
#define DEFLATE_HDIST_BITS 5u
#define DEFLATE_HCLEN_BITS 4u
#define DEFLATE_REPEAT_BITS 2u
#define DEFLATE_SHORT_BITS 3u
#define DEFLATE_LONG_BITS 7u
#define DEFLATE_STORED 0u
#define DEFLATE_FIXED 1u
#define DEFLATE_DYNAMIC 2u
#define DEFLATE_STORED_HEADER_BYTES 4u
#define DEFLATE_FIXED_BOUNDARY_A 144u
#define DEFLATE_FIXED_BOUNDARY_B 256u
#define DEFLATE_FIXED_BOUNDARY_C 280u
#define DEFLATE_FIXED_A_BITS 8u
#define DEFLATE_FIXED_B_BITS 9u
#define DEFLATE_FIXED_C_BITS 7u
#define DEFLATE_FIXED_D_BITS 8u
#define DEFLATE_FIXED_DISTANCE_BITS 5u
#define DEFLATE_REAL_DISTANCES 30u
#define PAIR 2u
#define TRIPLE 3u
#define QUAD 4u
#define CRC_BYTE_MASK 255u
#define PNG_CHUNK_IHDR UINT32_C(0x49484452)
#define PNG_CHUNK_PLTE UINT32_C(0x504c5445)
#define PNG_CHUNK_IDAT UINT32_C(0x49444154)
#define PNG_CHUNK_IEND UINT32_C(0x49454e44)
#define PNG_CRITICAL_MASK UINT32_C(0x20000000)
#define PNG_RESERVED_MASK UINT32_C(0x00002000)
#define PNG_CHUNK_TYPE_BYTES 4u
#define BIG_ENDIAN_TOP_SHIFT 24u
#define BIG_ENDIAN_MIDDLE_SHIFT 16u
#define CORRUPT STATUS_INVALID_ARGUMENT

struct image_bits {
	const uint8_t *data;
	size_t length;
	size_t position;
	uint32_t bits;
	unsigned available;
	bool failed;
};
struct image_huffman {
	uint16_t counts[DEFLATE_MAX_BITS + 1u];
	uint16_t symbols[DEFLATE_MAX_SYMBOLS];
};
static uint32_t image_u32(const uint8_t *data)
{
	return ((uint32_t)data[0] << BIG_ENDIAN_TOP_SHIFT) |
	       ((uint32_t)data[1] << BIG_ENDIAN_MIDDLE_SHIFT) |
	       ((uint32_t)data[PAIR] << BYTE_BITS) | data[TRIPLE];
}
static uint32_t image_u16(const uint8_t *data)
{
	return ((uint32_t)data[0] << BYTE_BITS) | data[1];
}
static bool image_dimensions(uint32_t width, uint32_t height)
{
	return width != 0 && height != 0 && width <= IMAGE_MAX_DIMENSION &&
	       height <= IMAGE_MAX_DIMENSION &&
	       (uint64_t)width * height <= IMAGE_MAX_PIXELS;
}
static uint32_t image_take(struct image_bits *reader, unsigned count)
{
	if (count == 0)
		return 0;
	while (reader->available < count) {
		if (reader->position == reader->length) {
			reader->failed = true;
			return 0;
		}
		reader->bits |= (uint32_t)reader->data[reader->position++]
				<< reader->available;
		reader->available += BYTE_BITS;
	}
	uint32_t result = reader->bits & ((1u << count) - 1u);
	reader->bits >>= count;
	reader->available -= count;
	return result;
}
static bool image_huffman_build(struct image_huffman *table,
				const uint8_t *lengths, unsigned count)
{
	memset(table, 0, sizeof(*table));
	for (unsigned symbol = 0; symbol < count; ++symbol) {
		if (lengths[symbol] > DEFLATE_MAX_BITS)
			return false;
		++table->counts[lengths[symbol]];
	}
	int remaining = 1;
	uint16_t offsets[DEFLATE_MAX_BITS + 1u] = { 0 };
	for (unsigned length = 1; length <= DEFLATE_MAX_BITS; ++length) {
		remaining = remaining * (int)PAIR - table->counts[length];
		if (remaining < 0)
			return false;
		if (length < DEFLATE_MAX_BITS)
			offsets[length + 1u] =
				(uint16_t)(offsets[length] +
					   table->counts[length]);
	}
	for (unsigned symbol = 0; symbol < count; ++symbol)
		if (lengths[symbol] != 0)
			table->symbols[offsets[lengths[symbol]]++] =
				(uint16_t)symbol;
	return true;
}
static unsigned image_huffman_take(struct image_bits *reader,
				   const struct image_huffman *table)
{
	unsigned code = 0, first = 0, offset = 0;
	for (unsigned length = 1; length <= DEFLATE_MAX_BITS; ++length) {
		code |= image_take(reader, 1);
		unsigned count = table->counts[length];
		if (code >= first && code - first < count)
			return table->symbols[offset + code - first];
		offset += count;
		first = (first + count) << 1u;
		code <<= 1u;
	}
	reader->failed = true;
	return 0;
}
static bool image_inflate_tables(struct image_bits *reader, unsigned type,
				 struct image_huffman *literals,
				 struct image_huffman *distances)
{
	uint8_t lengths[DEFLATE_MAX_SYMBOLS + DEFLATE_MAX_DISTANCES] = { 0 };
	if (type == DEFLATE_FIXED) {
		for (unsigned symbol = 0; symbol < DEFLATE_MAX_SYMBOLS;
		     ++symbol)
			lengths[symbol] = symbol < DEFLATE_FIXED_BOUNDARY_A ?
						  DEFLATE_FIXED_A_BITS :
					  symbol < DEFLATE_FIXED_BOUNDARY_B ?
						  DEFLATE_FIXED_B_BITS :
					  symbol < DEFLATE_FIXED_BOUNDARY_C ?
						  DEFLATE_FIXED_C_BITS :
						  DEFLATE_FIXED_D_BITS;
		for (unsigned symbol = 0; symbol < DEFLATE_MAX_DISTANCES;
		     ++symbol)
			lengths[DEFLATE_MAX_SYMBOLS + symbol] =
				DEFLATE_FIXED_DISTANCE_BITS;
		return image_huffman_build(literals, lengths,
					   DEFLATE_MAX_SYMBOLS) &&
		       image_huffman_build(distances,
					   lengths + DEFLATE_MAX_SYMBOLS,
					   DEFLATE_MAX_DISTANCES);
	}
	unsigned literal_count =
		image_take(reader, DEFLATE_HLIT_BITS) + DEFLATE_LENGTH_BASE;
	unsigned distance_count =
		image_take(reader, DEFLATE_HDIST_BITS) + DEFLATE_DISTANCE_BASE;
	unsigned code_count =
		image_take(reader, DEFLATE_HCLEN_BITS) + DEFLATE_CODE_BASE;
	static const uint8_t order[DEFLATE_CODE_LENGTHS] = {
		16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15
	};
	uint8_t code_lengths[DEFLATE_CODE_LENGTHS] = { 0 };
	struct image_huffman code_table;
	for (unsigned index = 0; index < code_count; ++index)
		code_lengths[order[index]] =
			(uint8_t)image_take(reader, DEFLATE_SHORT_BITS);
	if (literal_count > DEFLATE_LAST_LENGTH + 1u ||
	    !image_huffman_build(&code_table, code_lengths,
				 DEFLATE_CODE_LENGTHS))
		return false;
	unsigned total = literal_count + distance_count;
	for (unsigned index = 0; index < total;) {
		unsigned symbol = image_huffman_take(reader, &code_table);
		if (reader->failed)
			return false;
		if (symbol < DEFLATE_LENGTH_REPEAT) {
			lengths[index++] = (uint8_t)symbol;
			continue;
		}
		unsigned count;
		uint8_t value = 0;
		if (symbol == DEFLATE_LENGTH_REPEAT) {
			if (index == 0)
				return false;
			value = lengths[index - 1u];
			count = image_take(reader, DEFLATE_REPEAT_BITS) +
				DEFLATE_REPEAT_MINIMUM;
		} else if (symbol == DEFLATE_ZERO_SHORT) {
			count = image_take(reader, DEFLATE_SHORT_BITS) +
				DEFLATE_REPEAT_MINIMUM;
		} else {
			count = image_take(reader, DEFLATE_LONG_BITS) +
				DEFLATE_ZERO_LONG_MINIMUM;
		}
		if (count > total - index)
			return false;
		memset(lengths + index, value, count);
		index += count;
	}
	return !reader->failed && lengths[DEFLATE_END] != 0 &&
	       image_huffman_build(literals, lengths, literal_count) &&
	       image_huffman_build(distances, lengths + literal_count,
				   distance_count);
}
static bool image_inflate(const uint8_t *input, size_t input_bytes,
			  uint8_t *output, size_t output_bytes)
{
	static const uint16_t length_base[] = {
		3,  4,	5,  6,	7,  8,	9,  10, 11,  13,  15,  17,  19,	 23, 27,
		31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258
	};
	static const uint8_t length_bits[] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
					       1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
					       4, 4, 4, 4, 5, 5, 5, 5, 0 };
	static const uint16_t distance_base[] = {
		1,    2,    3,	  4,	5,    7,    9,	  13,	 17,	25,
		33,   49,   65,	  97,	129,  193,  257,  385,	 513,	769,
		1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577
	};
	static const uint8_t distance_bits[] = { 0,  0,	 0,  0,	 1,  1, 2,  2,
						 3,  3,	 4,  4,	 5,  5, 6,  6,
						 7,  7,	 8,  8,	 9,  9, 10, 10,
						 11, 11, 12, 12, 13, 13 };
	if (input_bytes < ZLIB_HEADER_BYTES + ZLIB_CHECKSUM_BYTES ||
	    (input[0] & ((1u << HALF_BYTE_BITS) - 1u)) != ZLIB_METHOD_DEFLATE ||
	    (input[0] >> HALF_BYTE_BITS) > BYTE_BITS - 1u ||
	    image_u16(input) % ZLIB_HEADER_MODULUS != 0 ||
	    (input[1] & ZLIB_DICTIONARY_FLAG) != 0)
		return false;
	struct image_bits reader = { .data = input + ZLIB_HEADER_BYTES,
				     .length = input_bytes - ZLIB_HEADER_BYTES -
					       ZLIB_CHECKSUM_BYTES };
	size_t written = 0;
	bool final;
	do {
		final = image_take(&reader, 1) != 0;
		unsigned type = image_take(&reader, DEFLATE_REPEAT_BITS);
		if (reader.failed)
			return false;
		if (type == DEFLATE_STORED) {
			reader.available = 0;
			reader.bits = 0;
			if (reader.length - reader.position <
			    DEFLATE_STORED_HEADER_BYTES)
				return false;
			const uint8_t *header = reader.data + reader.position;
			unsigned count = header[0] |
					 ((unsigned)header[1] << BYTE_BITS);
			unsigned inverse =
				header[PAIR] |
				((unsigned)header[TRIPLE] << BYTE_BITS);
			reader.position += DEFLATE_STORED_HEADER_BYTES;
			if ((count ^ inverse) != WORD_LOW_MASK ||
			    count > output_bytes - written ||
			    count > reader.length - reader.position)
				return false;
			memcpy(output + written, reader.data + reader.position,
			       count);
			reader.position += count;
			written += count;
			continue;
		}
		if (type != DEFLATE_FIXED && type != DEFLATE_DYNAMIC)
			return false;
		struct image_huffman literals, distances;
		if (!image_inflate_tables(&reader, type, &literals, &distances))
			return false;
		for (;;) {
			unsigned symbol =
				image_huffman_take(&reader, &literals);
			if (reader.failed)
				return false;
			if (symbol == DEFLATE_END)
				break;
			if (symbol < DEFLATE_END) {
				if (written == output_bytes)
					return false;
				output[written++] = (uint8_t)symbol;
				continue;
			}
			if (symbol > DEFLATE_LAST_LENGTH)
				return false;
			unsigned index = symbol - DEFLATE_FIRST_LENGTH;
			unsigned count =
				length_base[index] +
				image_take(&reader, length_bits[index]);
			unsigned distance =
				image_huffman_take(&reader, &distances);
			if (distance >= DEFLATE_REAL_DISTANCES)
				return false;
			distance = distance_base[distance] +
				   image_take(&reader, distance_bits[distance]);
			if (reader.failed || distance > written ||
			    distance > (1u << ((input[0] >> HALF_BYTE_BITS) +
					       BYTE_BITS)) ||
			    count > output_bytes - written)
				return false;
			for (unsigned offset = 0; offset < count; ++offset) {
				output[written] = output[written - distance];
				++written;
			}
		}
	} while (!final);
	if (written != output_bytes || reader.position != reader.length)
		return false;
	uint32_t first = 1, second = 0;
	for (size_t offset = 0; offset < written; ++offset) {
		first = (first + output[offset]) % PNG_ADLER_MODULUS;
		second = (second + first) % PNG_ADLER_MODULUS;
	}
	return image_u32(input + input_bytes - ZLIB_CHECKSUM_BYTES) ==
	       ((second << WORD_BITS) | first);
}
static uint32_t image_crc(const uint8_t *data, size_t count)
{
	uint32_t crc = UINT32_MAX;
	for (size_t offset = 0; offset < count; ++offset) {
		crc ^= data[offset];
		for (unsigned bit = 0; bit < BYTE_BITS; ++bit)
			crc = (crc >> 1u) ^
			      ((crc & 1u) != 0 ? PNG_CRC_POLYNOMIAL : 0);
	}
	return ~crc;
}
static unsigned image_paeth(unsigned left, unsigned above, unsigned upper_left)
{
	int prediction = (int)left + (int)above - (int)upper_left;
	int left_distance = abs(prediction - (int)left);
	int above_distance = abs(prediction - (int)above);
	int corner_distance = abs(prediction - (int)upper_left);
	return left_distance <= above_distance &&
			       left_distance <= corner_distance ?
		       left :
	       above_distance <= corner_distance ? above :
						   upper_left;
}
static unsigned image_png_sample(const uint8_t *row, uint32_t index,
				 unsigned depth)
{
	if (depth == PNG_16_DEPTH)
		return image_u16(row + (size_t)index * PAIR);
	if (depth == PNG_8_DEPTH)
		return row[index];
	unsigned bit = index * depth;
	return ((uint32_t)row[bit / BYTE_BITS] >>
		(BYTE_BITS - depth - bit % BYTE_BITS)) &
	       ((1u << depth) - 1u);
}
static MUST_CHECK result_status image_png(const uint8_t *data, size_t count,
					  struct image_rgb *image_out)
{
	if (count < PNG_SIGNATURE_BYTES + PNG_CHUNK_OVERHEAD + PNG_IHDR_BYTES ||
	    image_u32(data + PNG_SIGNATURE_BYTES) != PNG_IHDR_BYTES ||
	    image_u32(data + PNG_SIGNATURE_BYTES + PNG_CHUNK_TYPE_BYTES) !=
		    PNG_CHUNK_IHDR)
		return CORRUPT;
	const uint8_t *header = data + PNG_SIGNATURE_BYTES +
				(size_t)PAIR * PNG_CHUNK_TYPE_BYTES;
	uint32_t width = image_u32(header),
		 height = image_u32(header + PNG_CHUNK_TYPE_BYTES);
	unsigned depth = header[(size_t)PAIR * PNG_CHUNK_TYPE_BYTES];
	unsigned color = header[(size_t)PAIR * PNG_CHUNK_TYPE_BYTES + 1u];
	unsigned channels = color == PNG_GREY || color == PNG_INDEXED ? 1u :
			    color == PNG_RGB			      ? TRIPLE :
			    color == PNG_GREY_ALPHA ? PAIR :
						      PNG_MAX_CHANNELS;
	unsigned interlace = header[PNG_IHDR_BYTES - 1u];
	bool valid_depth = depth == PNG_8_DEPTH || depth == PNG_16_DEPTH ||
			   ((color == PNG_GREY || color == PNG_INDEXED) &&
			    (depth == PNG_1_DEPTH || depth == PNG_2_DEPTH ||
			     depth == PNG_4_DEPTH));
	if (!image_dimensions(width, height) || !valid_depth ||
	    (color != PNG_GREY && color != PNG_RGB && color != PNG_INDEXED &&
	     color != PNG_GREY_ALPHA && color != PNG_RGBA) ||
	    (color == PNG_INDEXED && depth == PNG_16_DEPTH) ||
	    header[PNG_IHDR_BYTES - TRIPLE] != 0 ||
	    header[PNG_IHDR_BYTES - PAIR] != 0 || interlace > 1)
		return CORRUPT;
	uint8_t palette[PNG_PALETTE_ENTRIES * TRIPLE] = { 0 };
	unsigned palette_entries = 0;
	uint8_t *compressed = malloc(count);
	if (compressed == NULL)
		return STATUS_OUT_OF_MEMORY;
	size_t compressed_count = 0;
	bool seen_header = false, seen_end = false, ended_data = false;
	result_status status = CORRUPT;
	for (size_t position = PNG_SIGNATURE_BYTES; position < count;) {
		if (count - position < PNG_CHUNK_OVERHEAD)
			goto cleanup;
		uint32_t length = image_u32(data + position);
		uint32_t type =
			image_u32(data + position + PNG_CHUNK_TYPE_BYTES);
		if (length > count - position - PNG_CHUNK_OVERHEAD ||
		    (type & PNG_RESERVED_MASK) != 0 ||
		    image_crc(data + position + PNG_CHUNK_TYPE_BYTES,
			      (size_t)length + PNG_CHUNK_TYPE_BYTES) !=
			    image_u32(data + position +
				      (size_t)PAIR * PNG_CHUNK_TYPE_BYTES +
				      length))
			goto cleanup;
		const uint8_t *chunk =
			data + position + (size_t)PAIR * PNG_CHUNK_TYPE_BYTES;
		if (type == PNG_CHUNK_IHDR) {
			if (seen_header || position != PNG_SIGNATURE_BYTES)
				goto cleanup;
			seen_header = true;
		} else if (type == PNG_CHUNK_PLTE) {
			if (palette_entries != 0 || compressed_count != 0 ||
			    length == 0 || length % TRIPLE != 0 ||
			    length > sizeof(palette))
				goto cleanup;
			memcpy(palette, chunk, length);
			palette_entries = length / TRIPLE;
		} else if (type == PNG_CHUNK_IDAT) {
			if (ended_data ||
			    (color == PNG_INDEXED && palette_entries == 0))
				goto cleanup;
			memcpy(compressed + compressed_count, chunk, length);
			compressed_count += length;
		} else if (type == PNG_CHUNK_IEND) {
			if (length != 0 || compressed_count == 0 ||
			    position + PNG_CHUNK_OVERHEAD != count)
				goto cleanup;
			seen_end = true;
		} else {
			if ((type & PNG_CRITICAL_MASK) == 0)
				goto cleanup;
			if (compressed_count != 0)
				ended_data = true;
		}
		position += (size_t)length + PNG_CHUNK_OVERHEAD;
	}
	if (!seen_end)
		goto cleanup;
	static const uint8_t pass_x[] = { 0, 4, 0, 2, 0, 1, 0 };
	static const uint8_t pass_y[] = { 0, 0, 4, 0, 2, 0, 1 };
	static const uint8_t pass_dx[] = { 8, 8, 4, 4, 2, 2, 1 };
	static const uint8_t pass_dy[] = { 8, 8, 8, 4, 4, 2, 2 };
	unsigned passes = interlace != 0 ? PNG_ADAM_PASSES : 1u;
	size_t raw_bytes = 0;
	for (unsigned pass = 0; pass < passes; ++pass) {
		unsigned start_x = interlace != 0 ? pass_x[pass] : 0;
		unsigned start_y = interlace != 0 ? pass_y[pass] : 0;
		unsigned step_x = interlace != 0 ? pass_dx[pass] : 1u;
		unsigned step_y = interlace != 0 ? pass_dy[pass] : 1u;
		uint32_t pass_width =
			width > start_x ?
				(width - start_x + step_x - 1u) / step_x :
				0;
		uint32_t pass_height =
			height > start_y ?
				(height - start_y + step_y - 1u) / step_y :
				0;
		if (pass_width != 0)
			raw_bytes += (((size_t)pass_width * channels * depth +
				       BYTE_BITS - 1u) /
					      BYTE_BITS +
				      1u) *
				     pass_height;
	}
	if (raw_bytes > IMAGE_MAX_DECOMPRESSED_BYTES)
		goto cleanup;
	uint8_t *raw = malloc(raw_bytes);
	uint8_t *pixels = malloc((size_t)width * height * TRIPLE);
	if (raw == NULL || pixels == NULL) {
		free(raw);
		free(pixels);
		status = STATUS_OUT_OF_MEMORY;
		goto cleanup;
	}
	if (!image_inflate(compressed, compressed_count, raw, raw_bytes)) {
		free(raw);
		free(pixels);
		goto cleanup;
	}
	size_t position = 0;
	for (unsigned pass = 0; pass < passes; ++pass) {
		unsigned start_x = interlace != 0 ? pass_x[pass] : 0;
		unsigned start_y = interlace != 0 ? pass_y[pass] : 0;
		unsigned step_x = interlace != 0 ? pass_dx[pass] : 1u;
		unsigned step_y = interlace != 0 ? pass_dy[pass] : 1u;
		uint32_t pass_width =
			width > start_x ?
				(width - start_x + step_x - 1u) / step_x :
				0;
		uint32_t pass_height =
			height > start_y ?
				(height - start_y + step_y - 1u) / step_y :
				0;
		size_t row_bytes = ((size_t)pass_width * channels * depth +
				    BYTE_BITS - 1u) /
				   BYTE_BITS;
		unsigned pixel_bytes =
			(channels * depth + BYTE_BITS - 1u) / BYTE_BITS;
		uint8_t *previous = NULL;
		if (pass_width == 0)
			continue;
		for (uint32_t row = 0; row < pass_height; ++row) {
			unsigned filter = raw[position++];
			uint8_t *current = raw + position;
			if (filter > PNG_FILTER_PAETH) {
				free(raw);
				free(pixels);
				goto cleanup;
			}
			for (size_t offset = 0; offset < row_bytes; ++offset) {
				unsigned left =
					offset >= pixel_bytes ?
						current[offset - pixel_bytes] :
						0;
				unsigned above =
					previous != NULL ? previous[offset] : 0;
				unsigned corner =
					previous != NULL &&
							offset >= pixel_bytes ?
						previous[offset - pixel_bytes] :
						0;
				unsigned predictor =
					filter == PNG_FILTER_SUB ?
						left :
					filter == PNG_FILTER_UP ?
						above :
					filter == PNG_FILTER_AVERAGE ?
						(left + above) / PAIR :
					filter == PNG_FILTER_PAETH ?
						image_paeth(left, above,
							    corner) :
						0;
				current[offset] =
					(uint8_t)(current[offset] + predictor);
			}
			for (uint32_t column = 0; column < pass_width;
			     ++column) {
				uint8_t *pixel =
					pixels +
					((size_t)(start_y + row * step_y) *
						 width +
					 start_x + (size_t)column * step_x) *
						TRIPLE;
				unsigned first = image_png_sample(
					current, column * channels, depth);
				if (color == PNG_INDEXED) {
					if (first >= palette_entries) {
						free(raw);
						free(pixels);
						goto cleanup;
					}
					memcpy(pixel,
					       palette + (size_t)first * TRIPLE,
					       TRIPLE);
				} else if (color == PNG_GREY ||
					   color == PNG_GREY_ALPHA) {
					/* Pillow opens 16-bit grayscale as integer mode; convert RGB clamps. */
					unsigned gray;
					if (depth == PNG_16_DEPTH)
						gray = color == PNG_GREY ?
							       (first > BYTE_MAXIMUM ?
									BYTE_MAXIMUM :
									first) :
							       first >>
								       BYTE_BITS;
					else
						gray = first * BYTE_MAXIMUM /
						       ((1u << depth) - 1u);
					pixel[0] = pixel[1] = pixel[PAIR] =
						(uint8_t)gray;
				} else {
					for (unsigned channel = 0;
					     channel < TRIPLE; ++channel) {
						unsigned value = image_png_sample(
							current,
							column * channels +
								channel,
							depth);
						pixel[channel] =
							(uint8_t)(depth == PNG_16_DEPTH ?
									  value >>
										  BYTE_BITS :
									  value);
					}
				}
			}
			previous = current;
			position += row_bytes;
		}
	}
	free(raw);
	*image_out = (struct image_rgb){ .pixels = pixels,
					 .width = width,
					 .height = height };
	status = STATUS_OK;
cleanup:
	free(compressed);
	return status;
}

#include "image_decode.h"
MUST_CHECK
result_status image_decode(const void *bytes, size_t byte_count,
			   struct image_rgb *image_out)
{
	static const uint8_t png_signature[PNG_SIGNATURE_BYTES] = { 137, 80, 78,
								    71,	 13, 10,
								    26,	 10 };
	const uint8_t *data = bytes;
	if (data == NULL || image_out == NULL || byte_count == 0 ||
	    byte_count > IMAGE_MAX_BYTES)
		return STATUS_INVALID_ARGUMENT;
	if (byte_count >= PNG_SIGNATURE_BYTES &&
	    memcmp(data, png_signature, PNG_SIGNATURE_BYTES) == 0)
		return image_png(data, byte_count, image_out);
	if (byte_count >= PAIR && data[0] == BYTE_MAXIMUM &&
	    data[1] == JPEG_SIGNATURE_SECOND)
		return image_jpeg_decode(data, byte_count, image_out);
	return STATUS_UNSUPPORTED;
}
void image_rgb_destroy(struct image_rgb *image)
{
	if (image != NULL) {
		free(image->pixels);
		*image = (struct image_rgb){ 0 };
	}
}
