/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* vision_reference.c: scalar Flash-rounding oracle plus double mathematical control; BF16 library boundaries. */
#include "vision_reference.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define REFERENCE_HALF_SHIFT 16u
#define REFERENCE_ROUND_BIAS 0x7FFFu
#define REFERENCE_NAN 0x7FC0u
#define REFERENCE_EXPONENT 0x7F800000u
#define REFERENCE_MANTISSA 0x7FFFFFu
#define REFERENCE_HALF 0.5
#define REFERENCE_SQRT_TWO 1.414213562373095048801688724209698079
#define REFERENCE_LOG2_E 1.442695040888963407359924681001892137

void *test_vision_at(unsigned char *arena, uint64_t byte_offset)
{
	return arena + byte_offset;
}

const void *test_vision_const_at(const unsigned char *arena,
				 uint64_t byte_offset)
{
	return arena + byte_offset;
}

float test_vision_float(uint16_t bits)
{
	const uint32_t word = (uint32_t)bits << REFERENCE_HALF_SHIFT;
	float value;
	memcpy(&value, &word, sizeof(value));
	return value;
}

uint16_t test_vision_bfloat(float value)
{
	uint32_t word;
	memcpy(&word, &value, sizeof(word));
	if ((word & REFERENCE_EXPONENT) == REFERENCE_EXPONENT &&
	    (word & REFERENCE_MANTISSA) != 0) {
		return REFERENCE_NAN;
	}
	word += REFERENCE_ROUND_BIAS + ((word >> REFERENCE_HALF_SHIFT) & 1u);
	return (uint16_t)(word >> REFERENCE_HALF_SHIFT);
}

static void reference_linear(struct vision_arguments record,
			     unsigned char *arena, bool mathematical)
{
	const uint16_t *const source =
		test_vision_const_at(arena, record.source_offset);
	const uint16_t *const weights =
		test_vision_const_at(arena, record.weights_offset);
	const uint16_t *const bias =
		test_vision_const_at(arena, record.bias_offset);
	uint16_t *const output =
		test_vision_at(arena, record.destination_offset);
	const uint32_t requested_partition =
		(record.flags & VISION_LINEAR_K_PARTITION_MASK) >>
		VISION_LINEAR_K_PARTITION_SHIFT;
	/* The independent mathematical oracle keeps one whole-K FP64 product. */
	const bool use_partitions = !mathematical && requested_partition != 0;
	const uint32_t partition_width = use_partitions ? requested_partition :
							  record.input_width;
	for (uint32_t row = 0; row < record.rows; row++) {
		for (uint32_t channel = 0; channel < record.output_width;
		     channel++) {
			uint16_t carried = 0;
			for (uint32_t first_inner = 0;
			     first_inner < record.input_width;
			     first_inner += partition_width) {
				const uint32_t remaining =
					record.input_width - first_inner;
				const uint32_t end =
					first_inner +
					(partition_width < remaining ?
						 partition_width :
						 remaining);
				double wide = 0.0;
				float narrow = 0.0f;
				/* Scalar K order and BF16 boundaries stay explicit.
				 * Device goldens qualify MMA's internal tree. */
				for (uint32_t inner = first_inner; inner < end;
				     inner++) {
					const float input = test_vision_float(
						source[(size_t)row *
							       record.input_width +
						       inner]);
					const float weight = test_vision_float(
						weights[(size_t)channel *
								record.input_width +
							inner]);
					if (mathematical) {
						wide += (double)input *
							(double)weight;
					} else {
						narrow = fmaf(input, weight,
							      narrow);
					}
				}
				float value = mathematical ? (float)wide :
							     narrow;
				const bool bias_last =
					(record.flags &
					 VISION_LINEAR_BIAS_LAST) != 0;
				const bool bias_partition =
					bias_last ? end == record.input_width :
						    first_inner == 0;
				if ((record.flags & VISION_LINEAR_BIAS) != 0 &&
				    bias_partition) {
					const float bias_value =
						test_vision_float(
							bias[channel]);
					value = mathematical ?
							(float)(wide +
								(double)bias_value) :
							value + bias_value;
				}
				if (first_inner != 0) {
					value += test_vision_float(carried);
				}
				carried = test_vision_bfloat(value);
			}
			output[(size_t)row * record.output_width + channel] =
				carried;
		}
	}
}

static void reference_rms(struct vision_arguments record, unsigned char *arena)
{
	const uint16_t *const source =
		test_vision_const_at(arena, record.source_offset);
	const float *const weights =
		test_vision_const_at(arena, record.weights_offset);
	const uint16_t *const bfloat_weights =
		test_vision_const_at(arena, record.weights_offset);
	uint16_t *const output =
		test_vision_at(arena, record.destination_offset);
	for (uint32_t row = 0; row < record.rows; row++) {
		double squares = 0.0;
		for (uint32_t channel = 0; channel < record.input_width;
		     channel++) {
			const double value = (double)test_vision_float(
				source[(size_t)row * record.input_width +
				       channel]);
			squares += value * value;
		}
		const double inverse = 1.0 / sqrt(squares / record.input_width +
						  (double)record.epsilon);
		for (uint32_t channel = 0; channel < record.input_width;
		     channel++) {
			const size_t index =
				(size_t)row * record.input_width + channel;
			const double weight =
				(record.flags & VISION_NORM_BFLOAT16_WEIGHT) !=
						0 ?
					(double)test_vision_float(
						bfloat_weights[channel]) :
					(double)weights[channel];
			output[index] = test_vision_bfloat(
				(float)((double)test_vision_float(
						source[index]) *
					inverse * weight));
		}
	}
}

static void reference_rope(struct vision_arguments record, unsigned char *arena)
{
	if (record.heads == 0 ||
	    record.input_width / record.heads < VISION_ROPE_PARTS) {
		return;
	}
	const uint32_t head_width = record.input_width / record.heads;
	const uint32_t half = head_width / VISION_HALF_DIVISOR;
	const uint32_t axis = head_width / VISION_ROPE_PARTS;
	if (record.grid_width == 0 || axis == 0) {
		return;
	}
	uint16_t *const source = test_vision_at(arena, record.source_offset);
	for (uint32_t row = 0; row < record.rows; row++) {
		for (uint32_t component = 0; component < VISION_GATE_COMPONENTS;
		     component++) {
			for (uint32_t head = 0; head < record.heads; head++) {
				for (uint32_t channel = 0; channel < half;
				     channel++) {
					const uint32_t position =
						channel < axis ?
							row / record.grid_width :
							row % record.grid_width;
					const double angle =
						position /
						pow(record.theta,
						    (double)(VISION_HALF_DIVISOR *
							     (channel % axis)) /
							    half);
					const size_t index =
						(size_t)row *
							VISION_QKV_COMPONENTS *
							record.input_width +
						(size_t)component *
							record.input_width +
						(size_t)(size_t)head *
							head_width +
						channel;
					const double first =
						(double)test_vision_float(
							source[index]);
					const double second =
						(double)test_vision_float(
							source[index + half]);
					source[index] = test_vision_bfloat(
						(float)(first * cos(angle) -
							second * sin(angle)));
					source[index + half] = test_vision_bfloat(
						(float)(second * cos(angle) +
							first * sin(angle)));
				}
			}
		}
	}
}

static void reference_attention(struct vision_arguments record,
				unsigned char *arena)
{
	const uint16_t *const source =
		test_vision_const_at(arena, record.source_offset);
	uint16_t *const output =
		test_vision_at(arena, record.destination_offset);
	if (record.heads == 0 ||
	    record.input_width / record.heads < VISION_ROPE_PARTS) {
		return;
	}
	const uint32_t head_width = record.input_width / record.heads;
	const uint32_t stride = VISION_QKV_COMPONENTS * record.input_width;
	double scores[VISION_MAXIMUM_PATCHES];
	for (uint32_t row = 0; row < record.rows; row++) {
		for (uint32_t head = 0; head < record.heads; head++) {
			double maximum = -INFINITY;
			for (uint32_t key = 0; key < record.rows; key++) {
				double value = 0.0;
				for (uint32_t channel = 0; channel < head_width;
				     channel++) {
					value +=
						(double)test_vision_float(
							source[(size_t)row *
								       stride +
							       (size_t)head *
								       head_width +
							       channel]) *
						(double)test_vision_float(
							source[(size_t)key *
								       stride +
							       record.input_width +
							       (size_t)head *
								       head_width +
							       channel]);
				}
				scores[key] = value / sqrt((double)head_width);
				maximum = fmax(maximum, scores[key]);
			}
			double total = 0.0;
			for (uint32_t key = 0; key < record.rows; key++) {
				scores[key] = exp(scores[key] - maximum);
				total += scores[key];
			}
			if (total <= 0.0 || !isfinite(total)) {
				return;
			}
			for (uint32_t channel = 0; channel < head_width;
			     channel++) {
				double value = 0.0;
				for (uint32_t key = 0; key < record.rows;
				     key++) {
					value +=
						scores[key] *
						(double)test_vision_float(
							source[(size_t)key *
								       stride +
							       (size_t)VISION_GATE_COMPONENTS *
								       record.input_width +
							       (size_t)head *
								       head_width +
							       channel]);
				}
				output[(size_t)row * record.input_width +
				       (size_t)head * head_width + channel] =
					test_vision_bfloat(
						(float)(value / total));
			}
		}
	}
}

/* Flash's FP32 online state and BF16 probability boundary are distinct from the mathematical control above. */
static void reference_flash_attention(struct vision_arguments record,
				      unsigned char *arena)
{
	if (record.heads == 0) {
		return;
	}
	const uint32_t head_width = record.input_width / record.heads;
	if (head_width == 0 || head_width > VISION_THREADS) {
		return;
	}
	const uint16_t *const source =
		test_vision_const_at(arena, record.source_offset);
	uint16_t *const output =
		test_vision_at(arena, record.destination_offset);
	const uint32_t stride = VISION_QKV_COMPONENTS * record.input_width;
	const float scale = (float)(1.0 / sqrt((double)head_width));
	const float scale_log2 = (float)((double)scale * REFERENCE_LOG2_E);
	for (uint32_t row = 0; row < record.rows; row++) {
		for (uint32_t head = 0; head < record.heads; head++) {
			float accumulated[VISION_THREADS] = { 0 };
			float denominator[VISION_ATTENTION_SUM_LANES] = { 0 };
			float maximum = -INFINITY;
			for (uint32_t tile = (record.rows +
					      VISION_ATTENTION_KEY_TILE - 1u) /
					     VISION_ATTENTION_KEY_TILE;
			     tile != 0; tile--) {
				const uint32_t first_key =
					(tile - 1u) * VISION_ATTENTION_KEY_TILE;
				const uint32_t remaining =
					record.rows - first_key;
				const uint32_t columns =
					remaining < VISION_ATTENTION_KEY_TILE ?
						remaining :
						VISION_ATTENTION_KEY_TILE;
				float scores[VISION_ATTENTION_KEY_TILE];
				const float previous_maximum = maximum;
				for (uint32_t column = 0; column < columns;
				     column++) {
					float score = 0.0f;
					for (uint32_t channel = 0;
					     channel < head_width; channel++) {
						const size_t query_index =
							(size_t)row * stride +
							(size_t)head *
								head_width +
							channel;
						const size_t key_index =
							(size_t)(first_key +
								 column) *
								stride +
							record.input_width +
							(size_t)head *
								head_width +
							channel;
						score = fmaf(
							test_vision_float(
								source[query_index]),
							test_vision_float(
								source[key_index]),
							score);
					}
					scores[column] = score;
					maximum = fmaxf(maximum, score);
				}
				const float rescale =
					previous_maximum == -INFINITY ?
						0.0f :
						exp2f((previous_maximum -
						       maximum) *
						      scale_log2);
				for (uint32_t channel = 0; channel < head_width;
				     channel++) {
					accumulated[channel] *= rescale;
				}
				for (uint32_t column = 0; column < columns;
				     column++) {
					scores[column] = exp2f(
						scores[column] * scale_log2 -
						maximum * scale_log2);
				}
				for (uint32_t lane = 0;
				     lane < VISION_ATTENTION_SUM_LANES;
				     lane++) {
					for (uint32_t column =
						     lane *
						     VISION_ATTENTION_COLUMNS_PER_LANE;
					     column < columns;
					     column +=
					     VISION_ATTENTION_SUM_LANES *
					     VISION_ATTENTION_COLUMNS_PER_LANE) {
						/* Flash fuses this lane's carry rescale with its first probability. */
						if (column ==
						    lane * VISION_ATTENTION_COLUMNS_PER_LANE) {
							denominator[lane] = fmaf(
								denominator[lane],
								rescale,
								scores[column]);
						} else {
							denominator[lane] +=
								scores[column];
						}
						if (column + 1u < columns) {
							denominator[lane] +=
								scores[column +
								       1u];
						}
					}
				}
				for (uint32_t channel = 0; channel < head_width;
				     channel++) {
					for (uint32_t column = 0;
					     column < columns; column++) {
						const size_t value_index =
							(size_t)(first_key +
								 column) *
								stride +
							(size_t)VISION_GATE_COMPONENTS *
								record.input_width +
							(size_t)head *
								head_width +
							channel;
						accumulated[channel] = fmaf(
							test_vision_float(test_vision_bfloat(
								scores[column])),
							test_vision_float(
								source[value_index]),
							accumulated[channel]);
					}
				}
			}
			const float total =
				(denominator[0] +
				 denominator[VISION_HALF_DIVISOR]) +
				(denominator[1] +
				 denominator[VISION_ATTENTION_SUM_LANES - 1u]);
			for (uint32_t channel = 0; channel < head_width;
			     channel++) {
				output[(size_t)row * record.input_width +
				       (size_t)head * head_width + channel] =
					test_vision_bfloat(
						accumulated[channel] *
						(1.0f / total));
			}
		}
	}
}

static void reference_element(enum vision_operation operation,
			      struct vision_arguments record,
			      unsigned char *arena)
{
	const uint16_t *const source =
		test_vision_const_at(arena, record.source_offset);
	uint16_t *const output =
		test_vision_at(arena, record.destination_offset);
	for (uint32_t index = 0; index < record.rows * record.output_width;
	     index++) {
		const uint32_t row = index / record.output_width;
		const uint32_t column = index % record.output_width;
		double value = 0.0;
		if (operation == VISION_ADD) {
			const uint16_t *const auxiliary = test_vision_const_at(
				arena, record.auxiliary_offset);
			value = (double)test_vision_float(source[index]) +
				(double)test_vision_float(auxiliary[index]);
		} else if (operation == VISION_SILU) {
			const double gate = (double)test_vision_float(
				source[(size_t)row * record.input_width +
				       column]);
			const double activated =
				(double)test_vision_float(test_vision_bfloat(
					(float)(gate / (1.0 + exp(-gate)))));
			value = activated *
				(double)test_vision_float(
					source[(size_t)row * record.input_width +
					       (size_t)record.output_width +
					       column]);
		} else if (operation == VISION_GELU) {
			value = (double)test_vision_float(source[index]);
			value = (REFERENCE_HALF * value) *
				(1.0 + erf(value / REFERENCE_SQRT_TWO));
		} else if (operation == VISION_UNFOLD) {
			const uint32_t ratio = record.flags;
			if (ratio == 0 || record.grid_width == 0) {
				return;
			}
			const uint32_t cell_width =
				(record.grid_width + ratio - 1u) / ratio;
			const uint32_t channel = column / (ratio * ratio);
			const uint32_t delta = column % (ratio * ratio);
			const uint32_t patch_y =
				(row / cell_width) * ratio + delta / ratio;
			const uint32_t patch_x =
				(row % cell_width) * ratio + delta % ratio;
			if (patch_y < record.grid_height &&
			    patch_x < record.grid_width) {
				value = (double)test_vision_float(
					source[((size_t)patch_y *
							record.grid_width +
						patch_x) *
						       record.input_width +
					       channel]);
			}
		} else if (operation == VISION_MERGE) {
			const uint32_t channel = index % record.input_width;
			const uint32_t in_row =
				row == 0 ?
					0 :
					(row - 1u) % (record.grid_width + 1u);
			uint64_t offset = record.source_offset;
			uint32_t cell = 0;
			if (row == 0) {
				offset = record.weights_offset;
			} else if (row + 1u == record.rows) {
				offset = record.auxiliary_offset;
			} else if (in_row == record.grid_width) {
				offset = record.bias_offset;
			} else {
				cell = ((row - 1u) / (record.grid_width + 1u)) *
					       record.grid_width +
				       in_row;
			}
			const uint16_t *const selected =
				test_vision_const_at(arena, offset);
			value = (double)test_vision_float(
				selected[(size_t)cell * record.input_width +
					 channel]);
		}
		output[index] = test_vision_bfloat((float)value);
	}
}

static bool reference_run(const struct kernel_launch *launches, uint32_t count,
			  unsigned char *arena, bool round_probabilities)
{
	if (launches == NULL || arena == NULL) {
		return false;
	}
	for (uint32_t index = 0; index < count; index++) {
		struct vision_arguments record;
		if (launches[index].kernel >= VISION_OPERATIONS ||
		    launches[index].argument_byte_count != sizeof(record)) {
			return false;
		}
		memcpy(&record, launches[index].argument, sizeof(record));
		if (record.rows == 0 || record.rows > VISION_MAXIMUM_PATCHES ||
		    record.input_width == 0 || record.output_width == 0) {
			return false;
		}
		const enum vision_operation operation =
			(enum vision_operation)launches[index].kernel;
		switch (operation) {
		case VISION_LINEAR:
			reference_linear(record, arena, !round_probabilities);
			break;
		case VISION_RMS:
			reference_rms(record, arena);
			break;
		case VISION_ROPE:
			reference_rope(record, arena);
			break;
		case VISION_ATTENTION:
			if (round_probabilities) {
				reference_flash_attention(record, arena);
			} else {
				reference_attention(record, arena);
			}
			break;
		case VISION_ADD:
		case VISION_SILU:
		case VISION_UNFOLD:
		case VISION_GELU:
		case VISION_MERGE:
			reference_element(operation, record, arena);
			break;
		case VISION_STEP_MERGE:
		case VISION_OPERATIONS:
			return false;
		}
	}
	return true;
}

bool test_vision_reference(const struct kernel_launch *launches, uint32_t count,
			   unsigned char *arena)
{
	return reference_run(launches, count, arena, true);
}

bool test_vision_math_reference(const struct kernel_launch *launches,
				uint32_t count, unsigned char *arena)
{
	return reference_run(launches, count, arena, false);
}
