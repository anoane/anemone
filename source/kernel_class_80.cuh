/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_class_80.cuh: the exact profile on class 120, class 80's tensor-core sums without its tensor cores. Device
 * code only: kernel_linear_common.cuh includes it after its constants (the lane layout, the operand registers, the
 * bfloat16 helpers) and nothing else does. One arithmetic, two forms:
 *   - class_80_instruction: one m16n8k16 instruction emulated lane by lane in the instruction's operand layout (the
 *     attention core's fixed order, the products' chains of the instruction's layout).
 *   - class_80_thread_group: one group of 8 products formed by one thread from inputs it decodes itself (the
 *     products and the attention core's reference order, LINEAR_THREAD_SUMS).
 * Contract: kernel_linear_common.cuh (the instruction and its operands).
 */
#ifndef KERNEL_CLASS_80_CUH
#define KERNEL_CLASS_80_CUH

#if !defined(KERNEL_LINEAR_COMMON_CUH)
#error "kernel_linear_common.cuh includes kernel_class_80.cuh after its constants"
#endif

/* ---- the exact profile on class 120: one instruction as class 80's tensor cores sum it ----
 * Measured (tools/measure/cross_class: every case of 24,000 random instructions, subnormals and zeros among them): a
 * card of class 80 adds an output's 16 products in two groups of 8, k 0 to 7 and then 8 to 15, each group with the
 * running sum (the accumulator first). Every term of a group is aligned to the group's anchor, the largest of its
 * nonzero products' exponent sums ea + eb (the exponent of an input, a subnormal's taken as the smallest normal
 * one's, its significand unnormalized) and of the running sum's own exponent; truncated toward zero to 24 bits below
 * the anchor; the terms are added exactly and the sum truncated toward zero to float32. A card of class 120 adds all
 * 16 in one group, 25 bits below the anchor, so the two classes differ wherever a sum is not exact: in the exact
 * profile class 120 computes class 80's sums with the instructions below, and a layer's bits do not depend on its
 * card. A not-a-number or an infinity among the inputs gives a not-a-number. */
#define CLASS_80_GROUP_PRODUCTS 8u
#define CLASS_80_GROUPS 2u
#define CLASS_80_KEPT_BITS 24                   /* below the anchor */
#define CLASS_80_NO_ANCHOR INT32_MIN
#define CLASS_80_HIGH_ROW_OFFSET 1u             /* weights[1] and weights[3]: the lane group's output + 8 */
#define CLASS_80_HIGH_K_OFFSET 2u   /* weights[2] and weights[3]: inputs + 8 */
#define CLASS_80_EVERY_LANE 0xFFFFFFFFu
#define CLASS_80_FRACTION_MASK 0x007FFFFFu      /* of a float32 */
#define CLASS_80_HIDDEN_BIT 0x00800000u
#define CLASS_80_SIGN_MASK 0x80000000u
#define CLASS_80_FIELD_MASK 0xFFu
#define CLASS_80_FRACTION_BITS 23
#define CLASS_80_EXPONENT_BIAS 127
#define CLASS_80_SMALLEST_EXPONENT (-126)
#define CLASS_80_LARGEST_FINITE_BITS 0x7F7FFFFFu
#define CLASS_80_NOT_A_NUMBER_BITS 0x7FC00000u
#define CLASS_80_BFLOAT16_FIELD_SHIFT 7u
#define CLASS_80_BFLOAT16_FIELD_MASK 0xFFu
#define CLASS_80_BFLOAT16_BIAS 127
#define CLASS_80_HALF_FIELD_SHIFT 10u
#define CLASS_80_HALF_FIELD_MASK 0x1Fu
#define CLASS_80_HALF_BIAS 15
#define CLASS_80_HALF_BITS 0xFFFFu
#define CLASS_80_HALF_SIGN_BIT 0x8000u
#define CLASS_80_WORD_BITS 32

/* An input as a signed significand (its hidden bit set but for a subnormal) and an exponent: its value is the
 * significand times 2^(exponent - fraction bits), a subnormal's exponent that of the smallest normal number. An
 * infinity or a not-a-number clears is_finite. */
__device__ static __forceinline__ void class_80_decode(uint32_t bits, bool is_bfloat16, int32_t *significand_out,
                                                       int32_t *exponent_out, bool *is_finite)
{
    const uint32_t fraction_bits = is_bfloat16 ? CLASS_80_BFLOAT16_FIELD_SHIFT : CLASS_80_HALF_FIELD_SHIFT;
    const uint32_t largest_field = is_bfloat16 ? CLASS_80_BFLOAT16_FIELD_MASK : CLASS_80_HALF_FIELD_MASK;
    const uint32_t field = (bits >> fraction_bits) & largest_field;
    const uint32_t magnitude = (bits & ((1u << fraction_bits) - 1u)) | (field != 0 ? 1u << fraction_bits : 0u);
    *significand_out = (bits & CLASS_80_HALF_SIGN_BIT) != 0 ? -(int32_t)magnitude : (int32_t)magnitude;
    *exponent_out = (int32_t)(field != 0 ? field : 1u) - (is_bfloat16 ? CLASS_80_BFLOAT16_BIAS : CLASS_80_HALF_BIAS);
    *is_finite = *is_finite && field != largest_field;
}

/* A float32's exponent for the anchor (a subnormal's the smallest normal one's). */
__device__ static __forceinline__ int32_t class_80_float_exponent(float value)
{
    const int32_t field = (int32_t)((__float_as_uint(value) >> CLASS_80_FRACTION_BITS) & CLASS_80_FIELD_MASK);
    return (field == 0 ? 1 : field) - CLASS_80_EXPONENT_BIAS;
}

/* A float32 in units of 2^unit, truncated toward zero: below 2^26 in magnitude for every term of a group. */
__device__ static __forceinline__ int32_t class_80_units(float value, int32_t unit)
{
    const uint32_t bits = __float_as_uint(value);
    const int32_t field = (int32_t)((bits >> CLASS_80_FRACTION_BITS) & CLASS_80_FIELD_MASK);
    const uint32_t significand = (bits & CLASS_80_FRACTION_MASK) | (field != 0 ? CLASS_80_HIDDEN_BIT : 0u);
    const int32_t shift = (field == 0 ? 1 : field) - CLASS_80_EXPONENT_BIAS - CLASS_80_FRACTION_BITS - unit;
    const uint32_t magnitude = shift >= 0                     ? significand << shift
                               : -shift >= CLASS_80_WORD_BITS ? 0u
                                                              : significand >> -shift;
    return (bits & CLASS_80_SIGN_MASK) != 0 ? -(int32_t)magnitude : (int32_t)magnitude;
}

/* units times 2^unit, truncated toward zero to float32. */
__device__ static __forceinline__ float class_80_float(int32_t units, int32_t unit)
{
    const uint32_t magnitude = units < 0 ? (uint32_t)-units : (uint32_t)units;
    const uint32_t sign = units < 0 ? CLASS_80_SIGN_MASK : 0u;
    if (magnitude == 0) {
        return 0.0f;
    }
    const int32_t lead = CLASS_80_WORD_BITS - 1 - __clz(magnitude);
    const int32_t exponent = unit + lead;
    uint32_t significand = lead > CLASS_80_FRACTION_BITS ? magnitude >> (lead - CLASS_80_FRACTION_BITS)
                                                         : magnitude << (CLASS_80_FRACTION_BITS - lead);
    if (exponent > CLASS_80_EXPONENT_BIAS) {
        return __uint_as_float(sign | CLASS_80_LARGEST_FINITE_BITS);
    }
    if (exponent < CLASS_80_SMALLEST_EXPONENT) {
        const int32_t extra = CLASS_80_SMALLEST_EXPONENT - exponent;
        significand = extra >= CLASS_80_WORD_BITS ? 0u : significand >> extra;
        return __uint_as_float(sign | significand);
    }
    return __uint_as_float(sign | (uint32_t)(exponent + CLASS_80_EXPONENT_BIAS) << CLASS_80_FRACTION_BITS |
                           (significand & CLASS_80_FRACTION_MASK));
}

/* One group of an output: the running sum and 8 products of decoded inputs, row k by column k. A product is an exact
 * integer of units: the significands multiplied, scaled by 2^(exponent sum - anchor + product_shift), product_shift
 * being 24 less the two inputs' fraction bits. */
__device__ static __forceinline__ float class_80_group(float running,
                                                       const int32_t row_significands[CLASS_80_GROUP_PRODUCTS],
                                                       const int32_t row_exponents[CLASS_80_GROUP_PRODUCTS],
                                                       const int32_t column_significands[CLASS_80_GROUP_PRODUCTS],
                                                       const int32_t column_exponents[CLASS_80_GROUP_PRODUCTS],
                                                       int32_t product_shift)
{
    int32_t anchor = running != 0.0f ? class_80_float_exponent(running) : CLASS_80_NO_ANCHOR;
#pragma unroll
    for (uint32_t k = 0; k < CLASS_80_GROUP_PRODUCTS; k++) {
        const int32_t exponent = row_exponents[k] + column_exponents[k];
        anchor = row_significands[k] != 0 && column_significands[k] != 0 && exponent > anchor ? exponent : anchor;
    }
    if (anchor == CLASS_80_NO_ANCHOR) {
        return 0.0f;
    }
    int32_t units = class_80_units(running, anchor - CLASS_80_KEPT_BITS);
#pragma unroll
    for (uint32_t k = 0; k < CLASS_80_GROUP_PRODUCTS; k++) {
        const int32_t product = row_significands[k] * column_significands[k];
        const int32_t shift = row_exponents[k] + column_exponents[k] - anchor + product_shift;
        const uint32_t magnitude = (uint32_t)(product < 0 ? -product : product);
        const uint32_t kept = shift >= 0 ? magnitude << shift : -shift >= CLASS_80_WORD_BITS ? 0u : magnitude >> -shift;
        units += product < 0 ? -(int32_t)kept : (int32_t)kept;
    }
    return class_80_float(units, anchor - CLASS_80_KEPT_BITS);
}

/* The words of one group a lane needs from its row lanes and column lanes, decoded: the rows g and g + 8 of A, the
 * columns of B the lane computes (columns 2 t and 2 t + 1; column 0 for both where one column is in use). */
struct class_80_operands {
    int32_t row_significands[OUTPUT_HALVES][CLASS_80_GROUP_PRODUCTS];
    int32_t row_exponents[OUTPUT_HALVES][CLASS_80_GROUP_PRODUCTS];
    int32_t column_significands[COLUMNS_PER_LANE][CLASS_80_GROUP_PRODUCTS];
    int32_t column_exponents[COLUMNS_PER_LANE][CLASS_80_GROUP_PRODUCTS];
};

__device__ static __forceinline__ void class_80_gather(struct class_80_operands *operands,
                                                       const uint32_t weights[REGISTERS_PER_OPERAND_A],
                                                       uint32_t fed_inputs, uint32_t group, bool is_one_column,
                                                       bool is_bfloat16, bool *is_finite)
{
    const uint32_t lane = threadIdx.x % THREADS_PER_WARP;
    const uint32_t lane_group = lane / LANES_PER_LANE_GROUP, lane_place = lane % LANES_PER_LANE_GROUP;
#pragma unroll
    for (uint32_t source = 0; source < LANES_PER_LANE_GROUP; source++) {
        const int row_lane = (int)(lane_group * LANES_PER_LANE_GROUP + source);
        const uint32_t k = COLUMNS_PER_LANE * source; /* the even k of the source's two */
#pragma unroll
        for (uint32_t half = 0; half < OUTPUT_HALVES; half++) {
            const uint32_t row_word = weights[half * CLASS_80_HIGH_ROW_OFFSET + group * CLASS_80_HIGH_K_OFFSET];
            const uint32_t word = __shfl_sync(CLASS_80_EVERY_LANE, row_word, row_lane);
            class_80_decode(word & CLASS_80_HALF_BITS, is_bfloat16, &operands->row_significands[half][k],
                            &operands->row_exponents[half][k], is_finite);
            class_80_decode(word >> BITS_PER_HALF, is_bfloat16, &operands->row_significands[half][k + 1u],
                            &operands->row_exponents[half][k + 1u], is_finite);
        }
#pragma unroll
        for (uint32_t column = 0; column < COLUMNS_PER_LANE; column++) {
            const uint32_t fed_column = is_one_column ? 0u : COLUMNS_PER_LANE * lane_place + column;
            const uint32_t word =
                __shfl_sync(CLASS_80_EVERY_LANE, fed_inputs, (int)(fed_column * LANES_PER_LANE_GROUP + source));
            class_80_decode(word & CLASS_80_HALF_BITS, is_bfloat16, &operands->column_significands[column][k],
                            &operands->column_exponents[column][k], is_finite);
            class_80_decode(word >> BITS_PER_HALF, is_bfloat16, &operands->column_significands[column][k + 1u],
                            &operands->column_exponents[column][k + 1u], is_finite);
        }
    }
}

/* The words of one group for one output of column 0: row g (lane place 0) or g + 8 (lane place 1) of A, chosen by
 * value after the shuffles (every lane sends both), and column 0 of B. */
struct class_80_one_operands {
    int32_t row_significands[CLASS_80_GROUP_PRODUCTS], row_exponents[CLASS_80_GROUP_PRODUCTS];
    int32_t column_significands[CLASS_80_GROUP_PRODUCTS], column_exponents[CLASS_80_GROUP_PRODUCTS];
};

__device__ static __forceinline__ void class_80_gather_one(struct class_80_one_operands *operands,
                                                           const uint32_t weights[REGISTERS_PER_OPERAND_A],
                                                           uint32_t fed_inputs, uint32_t group, bool is_bfloat16,
                                                           bool *is_finite)
{
    const uint32_t lane = threadIdx.x % THREADS_PER_WARP;
    const uint32_t lane_group = lane / LANES_PER_LANE_GROUP, lane_place = lane % LANES_PER_LANE_GROUP;
#pragma unroll
    for (uint32_t source = 0; source < LANES_PER_LANE_GROUP; source++) {
        const int row_lane = (int)(lane_group * LANES_PER_LANE_GROUP + source);
        const uint32_t k = COLUMNS_PER_LANE * source;
        const uint32_t low_row = __shfl_sync(CLASS_80_EVERY_LANE, weights[group * CLASS_80_HIGH_K_OFFSET], row_lane);
        const uint32_t high_row = __shfl_sync(
            CLASS_80_EVERY_LANE, weights[CLASS_80_HIGH_ROW_OFFSET + group * CLASS_80_HIGH_K_OFFSET], row_lane);
        const uint32_t row_word = lane_place == 1u ? high_row : low_row;
        const uint32_t column_word = __shfl_sync(CLASS_80_EVERY_LANE, fed_inputs, (int)source); /* column 0 */
        class_80_decode(row_word & CLASS_80_HALF_BITS, is_bfloat16, &operands->row_significands[k],
                        &operands->row_exponents[k], is_finite);
        class_80_decode(row_word >> BITS_PER_HALF, is_bfloat16, &operands->row_significands[k + 1u],
                        &operands->row_exponents[k + 1u], is_finite);
        class_80_decode(column_word & CLASS_80_HALF_BITS, is_bfloat16, &operands->column_significands[k],
                        &operands->column_exponents[k], is_finite);
        class_80_decode(column_word >> BITS_PER_HALF, is_bfloat16, &operands->column_significands[k + 1u],
                        &operands->column_exponents[k + 1u], is_finite);
    }
}

/* accumulator = A B + accumulator of one m16n8k16 instruction (operand layouts of the PTX ISA), as class 80 sums it,
 * for the columns below column_count (the others are left as they are). The lane's outputs are rows g and g + 8 of
 * A by columns 2 t and 2 t + 1 of B: lane group g holds those rows, lane groups 2 t and 2 t + 1 those columns, each
 * of its 4 lanes two k of each half (the even k low). With one column in use, lane (g, 1) computes the output of
 * row g + 8 for lane (g, 0), which keeps row g's: half the work of a lane, on twice the lanes. */
__device__ static __forceinline__ void class_80_instruction(float accumulator[ELEMENTS_PER_LANE],
                                                            const uint32_t weights[REGISTERS_PER_OPERAND_A],
                                                            uint32_t fed_low_inputs, uint32_t fed_high_inputs,
                                                            bool is_bfloat16, uint32_t column_count)
{
    const uint32_t lane = threadIdx.x % THREADS_PER_WARP;
    const uint32_t lane_group = lane / LANES_PER_LANE_GROUP, lane_place = lane % LANES_PER_LANE_GROUP;
    const int32_t fraction_bits = (int32_t)(is_bfloat16 ? CLASS_80_BFLOAT16_FIELD_SHIFT : CLASS_80_HALF_FIELD_SHIFT);
    const int32_t product_shift = CLASS_80_KEPT_BITS - COLUMNS_PER_LANE * fraction_bits;
    const bool is_one_column = column_count == 1u;
    const int owner_lane = (int)(lane_group * LANES_PER_LANE_GROUP);
    /* one column: lane (g, 1) takes row g + 8's running sum from lane (g, 0) */
    const float high_running = __shfl_sync(CLASS_80_EVERY_LANE, accumulator[HIGH_OUTPUT_EVEN_COLUMN], owner_lane);
    float one_running = lane_place == 0 ? accumulator[LOW_OUTPUT_EVEN_COLUMN] : high_running;
    bool is_finite = true;
#pragma unroll
    for (uint32_t group = 0; group < CLASS_80_GROUPS; group++) {
        if (is_one_column) {
            struct class_80_one_operands one;
            class_80_gather_one(&one, weights, group == 0 ? fed_low_inputs : fed_high_inputs, group, is_bfloat16,
                                &is_finite);
            if (lane_place < OUTPUT_HALVES) {
                one_running = class_80_group(one_running, one.row_significands, one.row_exponents,
                                             one.column_significands, one.column_exponents, product_shift);
            }
            continue;
        }
        struct class_80_operands operands;
        class_80_gather(&operands, weights, group == 0 ? fed_low_inputs : fed_high_inputs, group, false, is_bfloat16,
                        &is_finite);
#pragma unroll
        for (uint32_t element = 0; element < ELEMENTS_PER_LANE; element++) {
            const uint32_t half = element / COLUMNS_PER_LANE, column = element % COLUMNS_PER_LANE;
            if (COLUMNS_PER_LANE * lane_place + column < column_count) {
                accumulator[element] = class_80_group(
                    accumulator[element], operands.row_significands[half], operands.row_exponents[half],
                    operands.column_significands[column], operands.column_exponents[column], product_shift);
            }
        }
    }
    const float high_result = __shfl_sync(CLASS_80_EVERY_LANE, one_running, owner_lane + 1);
    if (is_one_column && lane_place == 0) {
        accumulator[LOW_OUTPUT_EVEN_COLUMN] = one_running;
        accumulator[HIGH_OUTPUT_EVEN_COLUMN] = high_result;
    }
#pragma unroll
    for (uint32_t element = 0; element < ELEMENTS_PER_LANE; element++) {
        accumulator[element] = is_finite ? accumulator[element] : __uint_as_float(CLASS_80_NOT_A_NUMBER_BITS);
    }
}

#if LINEAR_THREAD_SUMS
/* ---- the exact profile on class 120: class 80's groups formed by one thread each ----
 * Emulating an instruction lane by lane (class_80_instruction) costs every lane shuffles and a decode of all the
 * inputs it gathers. Here one thread forms a whole group of 8 products of one output and one column from inputs it
 * decodes itself: a magnitude (the hidden bit set but for a subnormal) and an exponent biased by the format's bias
 * (a subnormal's that of the smallest normal number, a zero's CLASS_80_ZERO_EXPONENT, an infinity's or a
 * not-a-number's CLASS_80_NOT_FINITE_EXPONENT). The column magnitudes come shifted left by 24 less the two inputs'
 * fraction bits: a product of magnitudes is then in units of 2^(anchor - 24) when its exponent sum is the anchor.
 * An input or a running sum that is not finite gives what IEEE 754 gives (class_80_special). */
#define CLASS_80_ZERO_EXPONENT (-1024)          /* below every anchor, and every shift of its products past 31 */
#define CLASS_80_NOT_FINITE_EXPONENT 4096
#define CLASS_80_NOT_FINITE_ANCHOR 2048         /* from here up an input was not finite, even times a zero */
#define CLASS_80_NO_RUNNING_ANCHOR (-8192)      /* a running sum of zero */
#define CLASS_80_SCALED_SMALLEST_UNIT (-126)    /* units 2^unit whose float32 scales 2^-unit and 2^unit are exact */
#define CLASS_80_SCALED_LARGEST_UNIT 96         /* and a sum below 2^31 units stays below the largest float32 */
#define CLASS_80_HALF_PAIR_BIAS 30              /* two 16-bit floats' exponent biases */
#define CLASS_80_BFLOAT16_PAIR_BIAS 254
#define CLASS_80_HALF_COLUMN_SHIFT 4u           /* 24 less two inputs' 10 fraction bits */
#define CLASS_80_BFLOAT16_COLUMN_SHIFT 10u      /* 24 less two inputs' 7 fraction bits */

/* A group of class_80_thread_group with an input that is not finite, as IEEE 754 has it: a not-a-number among the
 * inputs or the running sum, an infinity times a zero, or infinities of both signs give a not-a-number; else the
 * infinity of the infinite terms' sign. An infinity is told from a not-a-number by its magnitude: the hidden bit
 * alone. */
__device__ static __forceinline__ float class_80_special(float running,
                                                        const uint32_t row_magnitudes[CLASS_80_GROUP_PRODUCTS],
                                                        const int32_t row_exponents[CLASS_80_GROUP_PRODUCTS],
                                                        const uint32_t column_magnitudes[CLASS_80_GROUP_PRODUCTS],
                                                        const int32_t column_exponents[CLASS_80_GROUP_PRODUCTS],
                                                        const int32_t product_signs[CLASS_80_GROUP_PRODUCTS],
                                                        int32_t pair_bias)
{
    const bool is_bfloat16 = pair_bias == CLASS_80_BFLOAT16_PAIR_BIAS;
    const uint32_t row_hidden = 1u << (is_bfloat16 ? CLASS_80_BFLOAT16_FIELD_SHIFT : CLASS_80_HALF_FIELD_SHIFT);
    const uint32_t column_hidden =
        row_hidden << (is_bfloat16 ? CLASS_80_BFLOAT16_COLUMN_SHIFT : CLASS_80_HALF_COLUMN_SHIFT);
    bool is_number = !isnan(running);
    bool has_positive = isinf(running) && running > 0.0f, has_negative = isinf(running) && running < 0.0f;
    for (uint32_t k = 0; k < CLASS_80_GROUP_PRODUCTS; k++) {
        const bool is_row_special = row_exponents[k] == CLASS_80_NOT_FINITE_EXPONENT;
        const bool is_column_special = column_exponents[k] == CLASS_80_NOT_FINITE_EXPONENT;
        is_number = is_number && !(is_row_special && row_magnitudes[k] != row_hidden) &&
                    !(is_column_special && column_magnitudes[k] != column_hidden) &&
                    !(is_row_special && column_exponents[k] == CLASS_80_ZERO_EXPONENT) &&
                    !(is_column_special && row_exponents[k] == CLASS_80_ZERO_EXPONENT);
        has_positive = has_positive || ((is_row_special || is_column_special) && product_signs[k] > 0);
        has_negative = has_negative || ((is_row_special || is_column_special) && product_signs[k] < 0);
    }
    return !is_number || (has_positive && has_negative) ? __uint_as_float(CLASS_80_NOT_A_NUMBER_BITS)
           : has_positive                               ? INFINITY
                                                        : -INFINITY;
}

/* One group of class 80's sum: running and the 8 products of row k by column k, each product's sign as 1 or -1,
 * the exponents biased by pair_bias together. A product is an exact integer of units: its magnitudes multiplied,
 * shifted right by the anchor less its exponent sum (truncated toward zero); the running sum's units are truncated
 * toward zero the same way; the units are added exactly and their sum truncated toward zero to float32 (by the
 * float32 scales where they are exact, else class_80_units and class_80_float). An input or a running sum that is
 * not finite: class_80_special. */
__device__ static __forceinline__ float class_80_thread_group(float running,
                                                              const uint32_t row_magnitudes[CLASS_80_GROUP_PRODUCTS],
                                                              const int32_t row_exponents[CLASS_80_GROUP_PRODUCTS],
                                                              const uint32_t column_magnitudes[CLASS_80_GROUP_PRODUCTS],
                                                              const int32_t column_exponents[CLASS_80_GROUP_PRODUCTS],
                                                              const int32_t product_signs[CLASS_80_GROUP_PRODUCTS],
                                                              int32_t pair_bias)
{
    const uint32_t running_bits = __float_as_uint(running);
    const int32_t running_field = (int32_t)((running_bits >> CLASS_80_FRACTION_BITS) & CLASS_80_FIELD_MASK);
    int32_t anchor = (running_bits & FLOAT32_MAGNITUDE_MASK) == 0 ? CLASS_80_NO_RUNNING_ANCHOR
                     : running_field == (int32_t)CLASS_80_FIELD_MASK
                         ? CLASS_80_NOT_FINITE_EXPONENT
                         : (running_field != 0 ? running_field : 1) - CLASS_80_EXPONENT_BIAS + pair_bias;
#pragma unroll
    for (uint32_t k = 0; k < CLASS_80_GROUP_PRODUCTS; k++) {
        anchor = max(anchor, row_exponents[k] + column_exponents[k]);
    }
    const int32_t unit = anchor - pair_bias - CLASS_80_KEPT_BITS;
    const bool is_scaled = unit >= CLASS_80_SCALED_SMALLEST_UNIT && unit <= CLASS_80_SCALED_LARGEST_UNIT;
    int32_t units = is_scaled ? __float2int_rz(running * __uint_as_float((uint32_t)(CLASS_80_EXPONENT_BIAS - unit)
                                                                         << CLASS_80_FRACTION_BITS))
                              : class_80_units(running, unit);
#pragma unroll
    for (uint32_t k = 0; k < CLASS_80_GROUP_PRODUCTS; k++) {
        const uint32_t product = row_magnitudes[k] * column_magnitudes[k];
        const uint32_t kept =
            __funnelshift_rc(product, 0u, (uint32_t)(anchor - row_exponents[k] - column_exponents[k]));
        units += (int32_t)kept * product_signs[k];
    }
    const float sum = is_scaled ? __int2float_rz(units) * __uint_as_float((uint32_t)(CLASS_80_EXPONENT_BIAS + unit)
                                                                          << CLASS_80_FRACTION_BITS)
                                : class_80_float(units, unit);
    if (anchor >= CLASS_80_NOT_FINITE_ANCHOR) {
        return class_80_special(running, row_magnitudes, row_exponents, column_magnitudes, column_exponents,
                                product_signs, pair_bias);
    }
    return units == 0 ? 0.0f : sum;
}

/* ---- the routed experts on class 120: a group of E2M1 weights by E4M3 activations in the carried form ----
 * products: the group's 8 products W A. With w a weight's value and x an activation's (its block's exponent c
 * outside it), W = 2 w and A = 512 x are whole numbers (|W| at most 12, |A| at most 480 * 512) and W A is w x in
 * units of 2^-10. In those units no product's
 * anchor can make the 24-bit window coarser than 1 (the largest product, 12 * 245,760, is below 2^22), and before a
 * block's fourth group its running sum is below 2^26 (24 products of at most 6 * 480): so class_80_thread_group's
 * group of the placed operands is the exact sum of the 8 products W A, each first cut toward zero to an even number
 * where the running sum is 2^25 or more (the window's unit is then 2; the running sum, of 24 significant bits, is a
 * multiple of 4 and keeps its value), added to the running sum, the total cut toward zero to 24 significant bits
 * (agent D's proof: tmp/rnd/20261025/codex_linear_v1/REPORT_D_class120_exact.md; tools/measure/expert_group_check.cu
 * holds it to class_80_thread_group). The placed block sum is the result times 2^(c - 163). */
#define CLASS_80_E2M1_EVEN_FROM (1 << 25)   /* a running sum from here up cuts each product to an even number */
#define CLASS_80_E2M1_KEPT_FROM (1 << 24)   /* a total from here up keeps 24 significant bits */
#define CLASS_80_E2M1_KEPT_SHIFT 8          /* the bits a total keeps below its leading one: 31 less 23 */

__device__ static __forceinline__ int32_t class_80_e2m1_group(int32_t running,
                                                             const int32_t products[CLASS_80_GROUP_PRODUCTS])
{
    int32_t sum = 0;
#pragma unroll
    for (uint32_t k = 0; k < CLASS_80_GROUP_PRODUCTS; k++) {
        sum += products[k];
    }
    if (abs(running) >= CLASS_80_E2M1_EVEN_FROM) {
#pragma unroll
        for (uint32_t k = 0; k < CLASS_80_GROUP_PRODUCTS; k++) {     /* an odd product toward zero */
            sum -= (products[k] & 1) * ((products[k] >> (CLASS_80_WORD_BITS - 1)) | 1);
        }
    }
    const int32_t total = running + sum;
    const uint32_t magnitude = (uint32_t)abs(total);
    if (magnitude < CLASS_80_E2M1_KEPT_FROM) {
        return total;
    }
    const uint32_t cut = (uint32_t)(CLASS_80_E2M1_KEPT_SHIFT - __clz(magnitude));
    const int32_t kept = (int32_t)((magnitude >> cut) << cut);
    return total < 0 ? -kept : kept;
}

/* A 16-bit float or a bfloat16 decoded (see above): its magnitude and its biased exponent; sign_out -1 or 0. */
template <bool is_bfloat16>
__device__ __forceinline__ void class_80_thread_decode(uint32_t bits, uint32_t *magnitude_out, int32_t *exponent_out,
                                                       int32_t *sign_out)
{
    const uint32_t fraction_bits = is_bfloat16 ? CLASS_80_BFLOAT16_FIELD_SHIFT : CLASS_80_HALF_FIELD_SHIFT;
    const uint32_t largest_field = is_bfloat16 ? CLASS_80_BFLOAT16_FIELD_MASK : CLASS_80_HALF_FIELD_MASK;
    const uint32_t field = (bits >> fraction_bits) & largest_field;
    *magnitude_out = (bits & ((1u << fraction_bits) - 1u)) | (field != 0 ? 1u << fraction_bits : 0u);
    *exponent_out = field == largest_field                      ? CLASS_80_NOT_FINITE_EXPONENT
                    : (bits & (CLASS_80_HALF_BITS >> 1)) == 0 ? CLASS_80_ZERO_EXPONENT
                                                                : (int32_t)(field != 0 ? field : 1u);
    *sign_out = (int32_t)(bits << BITS_PER_HALF) >> (CLASS_80_WORD_BITS - 1);
}
#endif

#endif
