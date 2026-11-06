/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Baseline sequential Huffman JPEG, including component scans and restart markers.
 * Planes are bounded before allocation. Entropy, coefficient and table bounds are
 * independent of decoded dimensions. The integer IDCT fixes every rounding point. */
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "image_decode.h"

#define JPEG_MARKER_PREFIX 255u
#define JPEG_SOI 0xd8u
#define JPEG_EOI 0xd9u
#define JPEG_SOF0 0xc0u
#define JPEG_SOF_FIRST 0xc0u
#define JPEG_SOF_LAST 0xcfu
#define JPEG_DHT 0xc4u
#define JPEG_DQT 0xdbu
#define JPEG_DRI 0xddu
#define JPEG_SOS 0xdau
#define JPEG_APP14 0xeeu
#define JPEG_RESTART_FIRST 0xd0u
#define JPEG_RESTART_LAST 0xd7u
#define JPEG_BLOCK 8u
#define JPEG_COEFFICIENTS 64u
#define JPEG_COMPONENTS 4u
#define JPEG_TABLES 4u
#define JPEG_HUFFMAN_BITS 16u
#define JPEG_HUFFMAN_SYMBOLS 256u
#define JPEG_MAX_SAMPLING 4u
#define JPEG_MAX_MCU_BLOCKS 10u
#define JPEG_DC_MAX_BITS 11u
#define JPEG_AC_MAX_BITS 10u
#define JPEG_DC_BOUND 2048
#define JPEG_SAMPLE_MAXIMUM 255
#define JPEG_SAMPLE_CENTER 128
#define JPEG_SPECTRAL_END 63u
#define JPEG_AC_ZERO_RUN 15u
#define JPEG_AC_RUN_MASK 15u
#define JPEG_HALF_BYTE 4u
#define JPEG_SEGMENT_LENGTH_BYTES 2u
#define JPEG_FRAME_FIXED_BYTES 6u
#define JPEG_FRAME_COMPONENT_BYTES 3u
#define JPEG_SCAN_FIXED_BYTES 4u
#define JPEG_SCAN_COMPONENT_BYTES 2u
#define JPEG_ADOBE_BYTES 12u
#define JPEG_ADOBE_TRANSFORM_INDEX 11u
#define JPEG_ADOBE_YCCK 2
#define JPEG_COLOR_CHANNELS 3u
#define JPEG_PAIR 2u
#define JPEG_WORD_BITS 16u
#define JPEG_FIXED_HALF 32768
#define JPEG_RED_CR 91881
#define JPEG_GREEN_CB 22554
#define JPEG_GREEN_CR 46802
#define JPEG_BLUE_CB 116130
#define JPEG_BYTE_BITS 8u
#define JPEG_IDCT_CONSTANT_BITS 13u
#define JPEG_IDCT_PASS_BITS 2u
#define JPEG_IDCT_ROW_SHIFT (JPEG_IDCT_CONSTANT_BITS + JPEG_IDCT_PASS_BITS + 3u)
#define JPEG_IDCT_COLUMN_SHIFT (JPEG_IDCT_CONSTANT_BITS - JPEG_IDCT_PASS_BITS)
#define JPEG_FIX_0_298631336 2446
#define JPEG_FIX_0_390180644 3196
#define JPEG_FIX_0_541196100 4433
#define JPEG_FIX_0_765366865 6270
#define JPEG_FIX_0_899976223 7373
#define JPEG_FIX_1_175875602 9633
#define JPEG_FIX_1_501321110 12299
#define JPEG_FIX_1_847759065 15137
#define JPEG_FIX_1_961570560 16069
#define JPEG_FIX_2_053119869 16819
#define JPEG_FIX_2_562915447 20995
#define JPEG_FIX_3_072711026 25172
#define JPEG_IDCT_FREQUENCY_2 2u
#define JPEG_IDCT_FREQUENCY_3 3u
#define JPEG_IDCT_FREQUENCY_4 4u
#define JPEG_IDCT_FREQUENCY_5 5u
#define JPEG_IDCT_FREQUENCY_6 6u
#define JPEG_IDCT_FREQUENCY_7 7u
#define JPEG_UPSAMPLE_H2V1_EVEN 1u
#define JPEG_UPSAMPLE_H2V1_ODD 2u
#define JPEG_UPSAMPLE_H2V2_EVEN 8u
#define JPEG_UPSAMPLE_H2V2_ODD 7u
#define JPEG_UPSAMPLE_NEAR_WEIGHT 3u
#define JPEG_UPSAMPLE_HORIZONTAL_SCALE 4u
#define JPEG_UPSAMPLE_COMBINED_SCALE 16u
#define JPEG_COLOR_UNSPECIFIED (-1)
#define JPEG_COLOR_RGB 0
#define JPEG_CORRUPT STATUS_INVALID_ARGUMENT

struct jpeg_huffman {
	uint16_t count[JPEG_HUFFMAN_BITS + 1u];
	uint8_t symbol[JPEG_HUFFMAN_SYMBOLS];
	bool present;
};
struct jpeg_component {
	uint8_t *pixels;
	uint32_t width;
	uint32_t height;
	uint32_t stride;
	uint32_t rows;
	unsigned identifier;
	unsigned horizontal;
	unsigned vertical;
	unsigned quantizer;
	unsigned dc_table;
	unsigned ac_table;
	int dc;
	bool decoded;
};
struct jpeg_decoder {
	const uint8_t *data;
	size_t bytes;
	size_t position;
	unsigned bits;
	unsigned available;
	bool failed;
	bool allocation_failed;
	uint32_t width;
	uint32_t height;
	unsigned component_count;
	unsigned max_horizontal;
	unsigned max_vertical;
	unsigned restart_interval;
	int adobe_transform;
	bool quantizer_present[JPEG_TABLES];
	uint16_t quantizer[JPEG_TABLES][JPEG_COEFFICIENTS];
	struct jpeg_huffman huffman[JPEG_PAIR][JPEG_TABLES];
	struct jpeg_component components[JPEG_COMPONENTS];
};
static const uint8_t jpeg_zigzag[JPEG_COEFFICIENTS] = {
	0,  1,	8,  16, 9,  2,	3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
	12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,	7,  14, 21, 28,
	35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
	58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
};
static unsigned jpeg_u16(const uint8_t *bytes)
{
	return ((unsigned)bytes[0] << JPEG_BYTE_BITS) | bytes[1];
}
static unsigned jpeg_take(struct jpeg_decoder *decoder, unsigned count)
{
	unsigned result = 0;
	for (unsigned index = 0; index < count; ++index) {
		if (decoder->available == 0) {
			if (decoder->position >= decoder->bytes) {
				decoder->failed = true;
				return 0;
			}
			decoder->bits = decoder->data[decoder->position++];
			if (decoder->bits == JPEG_MARKER_PREFIX) {
				if (decoder->position >= decoder->bytes ||
				    decoder->data[decoder->position] != 0) {
					decoder->failed = true;
					return 0;
				}
				++decoder->position;
			}
			decoder->available = JPEG_BYTE_BITS;
		}
		--decoder->available;
		result = (result << 1u) |
			 ((decoder->bits >> decoder->available) & 1u);
	}
	return result;
}
static unsigned jpeg_symbol(struct jpeg_decoder *decoder,
			    const struct jpeg_huffman *table)
{
	unsigned code = 0, first = 0, offset = 0;
	if (!table->present) {
		decoder->failed = true;
		return 0;
	}
	for (unsigned bits = 1; bits <= JPEG_HUFFMAN_BITS; ++bits) {
		code = (code << 1u) | jpeg_take(decoder, 1);
		unsigned count = table->count[bits];
		if (code >= first && code - first < count)
			return table->symbol[offset + code - first];
		offset += count;
		first = (first + count) << 1u;
	}
	decoder->failed = true;
	return 0;
}
static int jpeg_signed(struct jpeg_decoder *decoder, unsigned count)
{
	if (count == 0)
		return 0;
	unsigned value = jpeg_take(decoder, count);
	return value < (1u << (count - 1u)) ?
		       (int)value - (int)((1u << count) - 1u) :
		       (int)value;
}
static uint8_t jpeg_clamp(int value)
{
	return (uint8_t)(value < 0		     ? 0 :
			 value > JPEG_SAMPLE_MAXIMUM ? JPEG_SAMPLE_MAXIMUM :
						       value);
}
/* Factor the eight-point inverse transform into even/odd butterflies.
 * Multipliers are rounded Q13 cosines. The column pass retains two extra bits;
 * the row pass rounds those bits and the transform's factor of eight together.
 * int64 intermediates preserve the result and defined behavior on corrupt input. */
static int64_t jpeg_descale(int64_t value, unsigned shift)
{
	int64_t divisor = INT64_C(1) << shift;
	value += divisor / JPEG_PAIR;
	return value >= 0 ? value / divisor :
			    -((-value + divisor - 1) / divisor);
}
static void jpeg_idct_line(const int64_t *input, size_t stride, int64_t *output,
			   unsigned shift)
{
	int64_t frequency_2 = input[JPEG_IDCT_FREQUENCY_2 * stride];
	int64_t frequency_6 = input[JPEG_IDCT_FREQUENCY_6 * stride];
	int64_t rotation = (frequency_2 + frequency_6) * JPEG_FIX_0_541196100;
	int64_t even_2 = rotation - frequency_6 * JPEG_FIX_1_847759065;
	int64_t even_3 = rotation + frequency_2 * JPEG_FIX_0_765366865;
	int64_t even_0 = (input[0] + input[JPEG_IDCT_FREQUENCY_4 * stride]) *
			 (INT64_C(1) << JPEG_IDCT_CONSTANT_BITS);
	int64_t even_1 = (input[0] - input[JPEG_IDCT_FREQUENCY_4 * stride]) *
			 (INT64_C(1) << JPEG_IDCT_CONSTANT_BITS);
	int64_t sum_0 = even_0 + even_3, sum_3 = even_0 - even_3;
	int64_t sum_1 = even_1 + even_2, sum_2 = even_1 - even_2;
	int64_t odd_0 = input[JPEG_IDCT_FREQUENCY_7 * stride];
	int64_t odd_1 = input[JPEG_IDCT_FREQUENCY_5 * stride];
	int64_t odd_2 = input[JPEG_IDCT_FREQUENCY_3 * stride];
	int64_t odd_3 = input[stride];
	int64_t rotate_1 = odd_0 + odd_3, rotate_2 = odd_1 + odd_2;
	int64_t rotate_3 = odd_0 + odd_2, rotate_4 = odd_1 + odd_3;
	int64_t rotate_5 = (rotate_3 + rotate_4) * JPEG_FIX_1_175875602;
	rotate_1 *= -JPEG_FIX_0_899976223;
	rotate_2 *= -JPEG_FIX_2_562915447;
	rotate_3 = rotate_3 * -JPEG_FIX_1_961570560 + rotate_5;
	rotate_4 = rotate_4 * -JPEG_FIX_0_390180644 + rotate_5;
	odd_0 = odd_0 * JPEG_FIX_0_298631336 + rotate_1 + rotate_3;
	odd_1 = odd_1 * JPEG_FIX_2_053119869 + rotate_2 + rotate_4;
	odd_2 = odd_2 * JPEG_FIX_3_072711026 + rotate_2 + rotate_3;
	odd_3 = odd_3 * JPEG_FIX_1_501321110 + rotate_1 + rotate_4;
	output[0] = jpeg_descale(sum_0 + odd_3, shift);
	output[JPEG_IDCT_FREQUENCY_7] = jpeg_descale(sum_0 - odd_3, shift);
	output[1] = jpeg_descale(sum_1 + odd_2, shift);
	output[JPEG_IDCT_FREQUENCY_6] = jpeg_descale(sum_1 - odd_2, shift);
	output[JPEG_IDCT_FREQUENCY_2] = jpeg_descale(sum_2 + odd_1, shift);
	output[JPEG_IDCT_FREQUENCY_5] = jpeg_descale(sum_2 - odd_1, shift);
	output[JPEG_IDCT_FREQUENCY_3] = jpeg_descale(sum_3 + odd_0, shift);
	output[JPEG_IDCT_FREQUENCY_4] = jpeg_descale(sum_3 - odd_0, shift);
}
static bool jpeg_block(struct jpeg_decoder *decoder,
		       struct jpeg_component *component, uint32_t block_x,
		       uint32_t block_y)
{
	int64_t coefficients[JPEG_COEFFICIENTS] = { 0 };
	unsigned size =
		jpeg_symbol(decoder, &decoder->huffman[0][component->dc_table]);
	if (size > JPEG_DC_MAX_BITS)
		return false;
	component->dc += jpeg_signed(decoder, size);
	if (component->dc < -JPEG_DC_BOUND || component->dc >= JPEG_DC_BOUND ||
	    decoder->failed)
		return false;
	coefficients[0] = (int64_t)component->dc *
			  (int32_t)decoder->quantizer[component->quantizer][0];

	for (unsigned offset = 1; offset < JPEG_COEFFICIENTS;) {
		unsigned symbol = jpeg_symbol(
			decoder, &decoder->huffman[1][component->ac_table]);
		unsigned run = symbol >> JPEG_HALF_BYTE;
		size = symbol & JPEG_AC_RUN_MASK;
		if (decoder->failed)
			return false;
		if (size == 0) {
			if (run == 0)
				break;
			if (run != JPEG_AC_ZERO_RUN)
				return false;
			offset += JPEG_HUFFMAN_BITS;
			if (offset > JPEG_COEFFICIENTS)
				return false;
			continue;
		}
		offset += run;
		if (size > JPEG_AC_MAX_BITS || offset >= JPEG_COEFFICIENTS)
			return false;
		unsigned index = jpeg_zigzag[offset];
		coefficients[index] =
			(int64_t)jpeg_signed(decoder, size) *
			(int32_t)decoder->quantizer[component->quantizer][index];
		++offset;
	}
	if (decoder->failed)
		return false;
	uint8_t *output = component->pixels +
			  (size_t)block_y * JPEG_BLOCK * component->stride +
			  (size_t)block_x * JPEG_BLOCK;
	int64_t workspace[JPEG_COEFFICIENTS], transformed[JPEG_BLOCK];
	for (unsigned column = 0; column < JPEG_BLOCK; ++column) {
		jpeg_idct_line(coefficients + column, JPEG_BLOCK, transformed,
			       JPEG_IDCT_COLUMN_SHIFT);
		for (unsigned row = 0; row < JPEG_BLOCK; ++row)
			workspace[(size_t)row * JPEG_BLOCK + column] =
				transformed[row];
	}
	for (unsigned row = 0; row < JPEG_BLOCK; ++row) {
		jpeg_idct_line(workspace + (size_t)row * JPEG_BLOCK, 1,
			       transformed, JPEG_IDCT_ROW_SHIFT);
		for (unsigned column = 0; column < JPEG_BLOCK; ++column) {
			int64_t value =
				transformed[column] + JPEG_SAMPLE_CENTER;
			output[(size_t)row * component->stride + column] =
				(uint8_t)(value < 0 ? 0 :
					  value > JPEG_SAMPLE_MAXIMUM ?
						      JPEG_SAMPLE_MAXIMUM :
						      value);
		}
	}
	return true;
}
static bool jpeg_tables(struct jpeg_decoder *decoder, unsigned marker,
			const uint8_t *segment, size_t length)
{
	for (size_t offset = 0; offset < length;) {
		unsigned identifier = segment[offset++];
		unsigned table_index = identifier & JPEG_AC_RUN_MASK;
		unsigned table_class = identifier >> JPEG_HALF_BYTE;
		if (table_index >= JPEG_TABLES)
			return false;
		if (marker == JPEG_DQT) {
			if (table_class != 0 ||
			    length - offset < JPEG_COEFFICIENTS)
				return false;
			for (unsigned index = 0; index < JPEG_COEFFICIENTS;
			     ++index) {
				if (segment[offset + index] == 0)
					return false;
				decoder->quantizer[table_index]
						  [jpeg_zigzag[index]] =
					segment[offset + index];
			}
			decoder->quantizer_present[table_index] = true;
			offset += JPEG_COEFFICIENTS;
		} else {
			if (table_class >= JPEG_PAIR ||
			    length - offset < JPEG_HUFFMAN_BITS)
				return false;
			struct jpeg_huffman *table =
				&decoder->huffman[table_class][table_index];
			memset(table, 0, sizeof(*table));
			unsigned total = 0, remaining = 1;
			for (unsigned bits = 1; bits <= JPEG_HUFFMAN_BITS;
			     ++bits) {
				unsigned count = segment[offset++];
				remaining *= JPEG_PAIR;
				if (count > remaining)
					return false;
				remaining -= count;
				table->count[bits] = (uint16_t)count;
				total += count;
			}
			if (total == 0 || total > JPEG_HUFFMAN_SYMBOLS ||
			    total > length - offset || remaining == 0)
				return false;
			memcpy(table->symbol, segment + offset, total);
			table->present = true;
			offset += total;
		}
	}
	return true;
}
static bool jpeg_frame(struct jpeg_decoder *decoder, const uint8_t *segment,
		       size_t length)
{
	if (decoder->component_count != 0 || length < JPEG_FRAME_FIXED_BYTES ||
	    segment[0] != JPEG_BYTE_BITS)
		return false;
	uint32_t height = jpeg_u16(segment + 1u),
		 width = jpeg_u16(segment + JPEG_COLOR_CHANNELS);
	unsigned count = segment[JPEG_FRAME_FIXED_BYTES - 1u];
	if (width == 0 || height == 0 || width > IMAGE_MAX_DIMENSION ||
	    height > IMAGE_MAX_DIMENSION ||
	    (uint64_t)width * height > IMAGE_MAX_PIXELS ||
	    (count != 1 && count != JPEG_COLOR_CHANNELS &&
	     count != JPEG_COMPONENTS) ||
	    length !=
		    JPEG_FRAME_FIXED_BYTES + count * JPEG_FRAME_COMPONENT_BYTES)
		return false;
	decoder->width = width;
	decoder->height = height;
	decoder->component_count = count;
	unsigned mcu_blocks = 0;
	for (unsigned index = 0; index < count; ++index) {
		const uint8_t *record =
			segment + JPEG_FRAME_FIXED_BYTES +
			(size_t)index * JPEG_FRAME_COMPONENT_BYTES;
		struct jpeg_component *component = &decoder->components[index];
		component->identifier = record[0];
		component->horizontal = record[1] >> JPEG_HALF_BYTE;
		component->vertical = record[1] & JPEG_AC_RUN_MASK;
		component->quantizer = record[JPEG_PAIR];
		if (component->horizontal == 0 ||
		    component->horizontal > JPEG_MAX_SAMPLING ||
		    component->vertical == 0 ||
		    component->vertical > JPEG_MAX_SAMPLING ||
		    component->quantizer >= JPEG_TABLES)
			return false;
		for (unsigned previous = 0; previous < index; ++previous)
			if (decoder->components[previous].identifier ==
			    component->identifier)
				return false;
		if (component->horizontal > decoder->max_horizontal)
			decoder->max_horizontal = component->horizontal;
		if (component->vertical > decoder->max_vertical)
			decoder->max_vertical = component->vertical;
		mcu_blocks += component->horizontal * component->vertical;
	}
	if (mcu_blocks > JPEG_MAX_MCU_BLOCKS)
		return false;
	uint32_t mcu_columns =
		(width + decoder->max_horizontal * JPEG_BLOCK - 1u) /
		(decoder->max_horizontal * JPEG_BLOCK);
	uint32_t mcu_rows = (height + decoder->max_vertical * JPEG_BLOCK - 1u) /
			    (decoder->max_vertical * JPEG_BLOCK);
	size_t allocated = 0;
	for (unsigned index = 0; index < count; ++index) {
		struct jpeg_component *component = &decoder->components[index];
		component->width = (width * component->horizontal +
				    decoder->max_horizontal - 1u) /
				   decoder->max_horizontal;
		component->height = (height * component->vertical +
				     decoder->max_vertical - 1u) /
				    decoder->max_vertical;
		component->stride =
			mcu_columns * component->horizontal * JPEG_BLOCK;
		component->rows = mcu_rows * component->vertical * JPEG_BLOCK;
		size_t bytes = (size_t)component->stride * component->rows;
		if (bytes > IMAGE_MAX_DECOMPRESSED_BYTES - allocated)
			return false;
		allocated += bytes;
		component->pixels = calloc(bytes, 1);
		if (component->pixels == NULL) {
			decoder->allocation_failed = true;
			return false;
		}
	}
	return true;
}
static bool jpeg_padding_valid(const struct jpeg_decoder *decoder)
{
	unsigned mask = (1u << decoder->available) - 1u;
	return (decoder->bits & mask) == mask;
}
static bool jpeg_scan(struct jpeg_decoder *decoder, const uint8_t *segment,
		      size_t length)
{
	if (length < JPEG_SCAN_FIXED_BYTES || decoder->component_count == 0)
		return false;
	unsigned count = segment[0];
	if (count == 0 || count > decoder->component_count ||
	    length !=
		    JPEG_SCAN_FIXED_BYTES + count * JPEG_SCAN_COMPONENT_BYTES ||
	    segment[length - JPEG_COLOR_CHANNELS] != 0 ||
	    segment[length - JPEG_PAIR] != JPEG_SPECTRAL_END ||
	    segment[length - 1u] != 0)
		return false;
	unsigned indices[JPEG_COMPONENTS] = { 0 };
	for (unsigned index = 0; index < count; ++index) {
		unsigned identifier = segment[1u + index * JPEG_PAIR];
		unsigned found = decoder->component_count;
		for (unsigned search = 0; search < decoder->component_count;
		     ++search)
			if (decoder->components[search].identifier ==
			    identifier)
				found = search;
		if (found == decoder->component_count ||
		    decoder->components[found].decoded)
			return false;
		for (unsigned previous = 0; previous < index; ++previous)
			if (indices[previous] == found)
				return false;
		indices[index] = found;
		struct jpeg_component *component = &decoder->components[found];
		unsigned tables = segment[JPEG_PAIR + index * JPEG_PAIR];
		component->dc_table = tables >> JPEG_HALF_BYTE;
		component->ac_table = tables & JPEG_AC_RUN_MASK;
		component->dc = 0;
		if (component->dc_table >= JPEG_TABLES ||
		    component->ac_table >= JPEG_TABLES ||
		    !decoder->huffman[0][component->dc_table].present ||
		    !decoder->huffman[1][component->ac_table].present ||
		    !decoder->quantizer_present[component->quantizer])
			return false;
	}
	uint32_t columns, rows;
	if (count == 1) {
		struct jpeg_component *component =
			&decoder->components[indices[0]];
		columns = (component->width + JPEG_BLOCK - 1u) / JPEG_BLOCK;
		rows = (component->height + JPEG_BLOCK - 1u) / JPEG_BLOCK;
	} else {
		columns = (decoder->width +
			   decoder->max_horizontal * JPEG_BLOCK - 1u) /
			  (decoder->max_horizontal * JPEG_BLOCK);
		rows = (decoder->height + decoder->max_vertical * JPEG_BLOCK -
			1u) /
		       (decoder->max_vertical * JPEG_BLOCK);
	}
	decoder->available = 0;
	unsigned restart = 0, since_restart = 0;
	for (uint32_t row = 0; row < rows; ++row)
		for (uint32_t column = 0; column < columns; ++column) {
			if (decoder->restart_interval != 0 &&
			    since_restart == decoder->restart_interval) {
				if (!jpeg_padding_valid(decoder))
					return false;
				decoder->available = 0;
				if (decoder->position == decoder->bytes ||
				    decoder->data[decoder->position] !=
					    JPEG_MARKER_PREFIX)
					return false;
				++decoder->position;
				while (decoder->position < decoder->bytes &&
				       decoder->data[decoder->position] ==
					       JPEG_MARKER_PREFIX)
					++decoder->position;
				if (decoder->position == decoder->bytes ||
				    decoder->data[decoder->position] !=
					    JPEG_RESTART_FIRST + restart)
					return false;
				++decoder->position;
				restart = (restart + 1u) % JPEG_BLOCK;
				since_restart = 0;
				for (unsigned index = 0; index < count; ++index)
					decoder->components[indices[index]].dc =
						0;
			}
			for (unsigned index = 0; index < count; ++index) {
				struct jpeg_component *component =
					&decoder->components[indices[index]];
				unsigned horizontal =
					count == 1 ? 1u : component->horizontal;
				unsigned vertical =
					count == 1 ? 1u : component->vertical;
				for (unsigned block_y = 0; block_y < vertical;
				     ++block_y)
					for (unsigned block_x = 0;
					     block_x < horizontal; ++block_x)
						if (!jpeg_block(
							    decoder, component,
							    column * horizontal +
								    block_x,
							    row * vertical +
								    block_y))
							return false;
			}
			++since_restart;
		}
	if (!jpeg_padding_valid(decoder))
		return false;
	decoder->available = 0;
	for (unsigned index = 0; index < count; ++index)
		decoder->components[indices[index]].decoded = true;
	return true;
}
static unsigned jpeg_sample(const struct jpeg_decoder *decoder, unsigned index,
			    uint32_t column, uint32_t row)
{
	const struct jpeg_component *component = &decoder->components[index];
	if (decoder->max_horizontal == component->horizontal * JPEG_PAIR &&
	    (decoder->max_vertical == component->vertical ||
	     decoder->max_vertical == component->vertical * JPEG_PAIR)) {
		unsigned source_column = column / JPEG_PAIR;
		unsigned neighbor_column =
			(column & 1u) != 0 ?
				(source_column + 1u < component->width ?
					 source_column + 1u :
					 source_column) :
				(source_column != 0 ? source_column - 1u : 0);
		if (decoder->max_vertical == component->vertical) {
			unsigned near_value =
				component->pixels[(size_t)row *
							  component->stride +
						  source_column];
			unsigned far_value =
				component->pixels[(size_t)row *
							  component->stride +
						  neighbor_column];
			unsigned bias = (column & 1u) != 0 ?
						JPEG_UPSAMPLE_H2V1_ODD :
						JPEG_UPSAMPLE_H2V1_EVEN;
			return (JPEG_UPSAMPLE_NEAR_WEIGHT * near_value +
				far_value + bias) /
			       JPEG_UPSAMPLE_HORIZONTAL_SCALE;
		}
		unsigned source_row = row / JPEG_PAIR;
		unsigned neighbor_row =
			(row & 1u) != 0 ?
				(source_row + 1u < component->height ?
					 source_row + 1u :
					 source_row) :
				(source_row != 0 ? source_row - 1u : 0);
		unsigned near_value =
			JPEG_UPSAMPLE_NEAR_WEIGHT *
				component->pixels[(size_t)source_row *
							  component->stride +
						  source_column] +
			component->pixels[(size_t)neighbor_row *
						  component->stride +
					  source_column];
		unsigned far_value =
			JPEG_UPSAMPLE_NEAR_WEIGHT *
				component->pixels[(size_t)source_row *
							  component->stride +
						  neighbor_column] +
			component->pixels[(size_t)neighbor_row *
						  component->stride +
					  neighbor_column];
		unsigned bias = (column & 1u) != 0 ? JPEG_UPSAMPLE_H2V2_ODD :
						     JPEG_UPSAMPLE_H2V2_EVEN;
		return (JPEG_UPSAMPLE_NEAR_WEIGHT * near_value + far_value +
			bias) /
		       JPEG_UPSAMPLE_COMBINED_SCALE;
	}
	/* Sample centers define the standard bilinear fancy upsampling. The two-axis
	 * interpolation rounds once; integral and edge coordinates reduce exactly. */
	int horizontal_denominator = (int)(JPEG_PAIR * decoder->max_horizontal);
	int vertical_denominator = (int)(JPEG_PAIR * decoder->max_vertical);
	int horizontal_position =
		(int)((JPEG_PAIR * column + 1u) * component->horizontal) -
		(int)decoder->max_horizontal;
	int vertical_position =
		(int)((JPEG_PAIR * row + 1u) * component->vertical) -
		(int)decoder->max_vertical;
	if (horizontal_position < 0)
		horizontal_position = 0;
	if (vertical_position < 0)
		vertical_position = 0;
	unsigned left =
		(unsigned)(horizontal_position / horizontal_denominator);
	unsigned top = (unsigned)(vertical_position / vertical_denominator);
	unsigned right = left + 1u < component->width ? left + 1u : left;
	unsigned bottom = top + 1u < component->height ? top + 1u : top;
	unsigned horizontal_fraction =
		(unsigned)(horizontal_position % horizontal_denominator);
	unsigned vertical_fraction =
		(unsigned)(vertical_position % vertical_denominator);
	unsigned horizontal_scale = (unsigned)horizontal_denominator,
		 vertical_scale = (unsigned)vertical_denominator;
	unsigned top_value =
		component->pixels[(size_t)top * component->stride + left] *
			(horizontal_scale - horizontal_fraction) +
		component->pixels[(size_t)top * component->stride + right] *
			horizontal_fraction;
	unsigned bottom_value =
		component->pixels[(size_t)bottom * component->stride + left] *
			(horizontal_scale - horizontal_fraction) +
		component->pixels[(size_t)bottom * component->stride + right] *
			horizontal_fraction;
	unsigned denominator = horizontal_scale * vertical_scale;
	return (top_value * (vertical_scale - vertical_fraction) +
		bottom_value * vertical_fraction + denominator / JPEG_PAIR) /
	       denominator;
}
static int jpeg_fixed_round(int value)
{
	value += JPEG_FIXED_HALF;
	return value >= 0 ? value / (1 << JPEG_WORD_BITS) :
			    -((-value + (1 << JPEG_WORD_BITS) - 1) /
			      (1 << JPEG_WORD_BITS));
}
static MUST_CHECK result_status jpeg_rgb(struct jpeg_decoder *decoder,
					 struct image_rgb *image_out)
{
	for (unsigned index = 0; index < decoder->component_count; ++index)
		if (!decoder->components[index].decoded)
			return JPEG_CORRUPT;
	uint8_t *pixels = malloc((size_t)decoder->width * decoder->height *
				 JPEG_COLOR_CHANNELS);
	if (pixels == NULL)
		return STATUS_OUT_OF_MEMORY;
	bool rgb = decoder->adobe_transform == JPEG_COLOR_RGB ||
		   (decoder->adobe_transform == JPEG_COLOR_UNSPECIFIED &&
		    decoder->component_count == JPEG_COLOR_CHANNELS &&
		    decoder->components[0].identifier == 'R' &&
		    decoder->components[1].identifier == 'G' &&
		    decoder->components[JPEG_PAIR].identifier == 'B');
	for (uint32_t row = 0; row < decoder->height; ++row)
		for (uint32_t column = 0; column < decoder->width; ++column) {
			uint8_t *output =
				pixels +
				((size_t)row * decoder->width + column) *
					JPEG_COLOR_CHANNELS;
			int first = (int)jpeg_sample(decoder, 0, column, row);
			if (decoder->component_count == 1) {
				output[0] = output[1] = output[JPEG_PAIR] =
					(uint8_t)first;
				continue;
			}
			int second = (int)jpeg_sample(decoder, 1, column, row);
			int third = (int)jpeg_sample(decoder, JPEG_PAIR, column,
						     row);
			if (decoder->component_count == JPEG_COMPONENTS &&
			    decoder->adobe_transform != JPEG_ADOBE_YCCK) {
				int black = (int)jpeg_sample(
					decoder, JPEG_COLOR_CHANNELS, column,
					row);
				if (decoder->adobe_transform ==
				    JPEG_COLOR_UNSPECIFIED) {
					first = JPEG_SAMPLE_MAXIMUM - first;
					second = JPEG_SAMPLE_MAXIMUM - second;
					third = JPEG_SAMPLE_MAXIMUM - third;
					black = JPEG_SAMPLE_MAXIMUM - black;
				}
				output[0] = (uint8_t)((first * black +
						       JPEG_SAMPLE_CENTER) /
						      JPEG_SAMPLE_MAXIMUM);
				output[1] = (uint8_t)((second * black +
						       JPEG_SAMPLE_CENTER) /
						      JPEG_SAMPLE_MAXIMUM);
				output[JPEG_PAIR] =
					(uint8_t)((third * black +
						   JPEG_SAMPLE_CENTER) /
						  JPEG_SAMPLE_MAXIMUM);
				continue;
			}
			if (!rgb) {
				second -= JPEG_SAMPLE_CENTER;
				third -= JPEG_SAMPLE_CENTER;
				output[0] = jpeg_clamp(
					first +
					jpeg_fixed_round(JPEG_RED_CR * third));
				output[1] = jpeg_clamp(
					first +
					jpeg_fixed_round(
						-JPEG_GREEN_CB * second -
						JPEG_GREEN_CR * third));
				output[JPEG_PAIR] = jpeg_clamp(
					first + jpeg_fixed_round(JPEG_BLUE_CB *
								 second));
			} else {
				output[0] = (uint8_t)first;
				output[1] = (uint8_t)second;
				output[JPEG_PAIR] = (uint8_t)third;
			}
			if (decoder->component_count == JPEG_COMPONENTS) {
				unsigned black = jpeg_sample(
					decoder, JPEG_COLOR_CHANNELS, column,
					row);
				for (unsigned channel = 0;
				     channel < JPEG_COLOR_CHANNELS; ++channel)
					output[channel] =
						(uint8_t)(((unsigned)(JPEG_SAMPLE_MAXIMUM -
								      output[channel]) *
								   black +
							   JPEG_SAMPLE_CENTER) /
							  JPEG_SAMPLE_MAXIMUM);
			}
		}
	*image_out = (struct image_rgb){ .pixels = pixels,
					 .width = decoder->width,
					 .height = decoder->height };
	return STATUS_OK;
}
MUST_CHECK result_status image_jpeg_decode(const uint8_t *bytes,
					   size_t byte_count,
					   struct image_rgb *image_out)
{
	if (bytes == NULL || image_out == NULL || byte_count < JPEG_PAIR ||
	    byte_count > IMAGE_MAX_BYTES || bytes[0] != JPEG_MARKER_PREFIX ||
	    bytes[1] != JPEG_SOI)
		return JPEG_CORRUPT;
	struct jpeg_decoder decoder = { .data = bytes,
					.bytes = byte_count,
					.position = JPEG_PAIR,
					.adobe_transform =
						JPEG_COLOR_UNSPECIFIED };
	result_status status = JPEG_CORRUPT;
	while (decoder.position < byte_count) {
		if (bytes[decoder.position++] != JPEG_MARKER_PREFIX)
			break;
		while (decoder.position < byte_count &&
		       bytes[decoder.position] == JPEG_MARKER_PREFIX)
			++decoder.position;
		if (decoder.position == byte_count)
			break;
		unsigned marker = bytes[decoder.position++];
		if (marker == JPEG_EOI) {
			if (decoder.component_count != 0 &&
			    decoder.position == byte_count)
				status = jpeg_rgb(&decoder, image_out);
			break;
		}
		if (marker == 0 || marker == JPEG_SOI ||
		    (marker >= JPEG_RESTART_FIRST &&
		     marker <= JPEG_RESTART_LAST) ||
		    byte_count - decoder.position < JPEG_SEGMENT_LENGTH_BYTES)
			break;
		unsigned length = jpeg_u16(bytes + decoder.position);
		if (length < JPEG_SEGMENT_LENGTH_BYTES ||
		    length > byte_count - decoder.position)
			break;
		const uint8_t *segment =
			bytes + decoder.position + JPEG_SEGMENT_LENGTH_BYTES;
		size_t payload = length - JPEG_SEGMENT_LENGTH_BYTES;
		decoder.position += length;
		if (marker == JPEG_DQT || marker == JPEG_DHT) {
			if (!jpeg_tables(&decoder, marker, segment, payload))
				break;
		} else if (marker == JPEG_SOF0) {
			if (!jpeg_frame(&decoder, segment, payload))
				break;
		} else if (marker >= JPEG_SOF_FIRST &&
			   marker <= JPEG_SOF_LAST) {
			status = STATUS_UNSUPPORTED;
			break;
		} else if (marker == JPEG_DRI) {
			if (payload != JPEG_PAIR)
				break;
			decoder.restart_interval = jpeg_u16(segment);
		} else if (marker == JPEG_APP14) {
			if (payload >= JPEG_ADOBE_BYTES &&
			    memcmp(segment, "Adobe", sizeof("Adobe") - 1u) == 0)
				decoder.adobe_transform =
					segment[JPEG_ADOBE_TRANSFORM_INDEX];
		} else if (marker == JPEG_SOS) {
			if (!jpeg_scan(&decoder, segment, payload))
				break;
		}
	}
	for (unsigned index = 0; index < JPEG_COMPONENTS; ++index)
		free(decoder.components[index].pixels);
	return decoder.allocation_failed ? STATUS_OUT_OF_MEMORY : status;
}
