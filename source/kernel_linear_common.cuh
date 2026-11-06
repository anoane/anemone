/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * kernel_linear_common.cuh: what the kernels of the linear share: how stored codes become operands of the
 * tensor-core instruction, the instruction, the power of two that undoes the placements and applies the scales, the
 * chain of one slice of the input width, the order in which slices are added, the rounding to bfloat16 and the
 * hidden step of an expert. Device code only; kernel_expert_linear.cu, kernel_trunk_linear.cu and
 * kernel_activation_quantize.cu include it.
 *
 * Contract: include/anemone/linear_records.h (the layouts, the order of additions, the scale codes), and:
 *   - The instruction is m16n8k16 with float32 accumulation. Lane 4 g + t of a warp holds, of the 16-by-16
 *     operand A (the weights), outputs g and g + 8 at inputs 2 t, 2 t + 1, 2 t + 8 and 2 t + 9; of the 16-by-8
 *     operand B column g (one vector) at the same four inputs; of the result, outputs g and g + 8 for columns 2 t
 *     and 2 t + 1. Both weight formats are stored so that a lane's bytes are its part of A, and a quantized
 *     vector so that the bytes of a lane place are its part of B.
 *   - Placement moves bits and computes nothing. The three magnitude bits of an E2M1 code go to bits 9 to 11 of a
 *     16-bit float and its sign to bit 15: the code's value times 2^-14 (0.5 becomes a subnormal); to bits 6 to 8
 *     of a bfloat16: times 2^-126. The seven magnitude bits of an E4M3 byte go to bits 7 to 13 of a 16-bit float
 *     and its sign to bit 15: the byte's value times 2^-8, subnormal bytes included; to bits 4 to 10 of a
 *     bfloat16: times 2^-120. The two bytes that are no number become +480 and -480 times those powers.
 *   - The power of two of a block of 32 inputs is built in the exponent field of a float32: the weights' scale
 *     code, plus what their placement took; in the codes form plus the vector's exponent less 127 and the 8 of
 *     its placement; in the carried form less the 100 of the operands. A field of 1 to 254 is that power exactly;
 *     a field below 1 becomes +0 and one above 254 becomes 2^127, so no code can reach the sign bit or make an
 *     infinity of a scale. (The carried field is at least 20: only its upper end is held.)
 *   - The block-scaled form (class 120, E2M1 weights) is one instruction m16n8k32 of 4-bit by 8-bit floats per
 *     block of 32 inputs, both scale codes inside it, on a zero accumulator; its sum is added to the chain. Of a
 *     stored word of eight weight codes the even nibbles are output g, the odd ones output g + 8, at inputs 2 t,
 *     2 t + 8, 2 t + 1 and 2 t + 9 of their step: the inputs of bytes 0 to 3 of the vector's code word of that step.
 *     (word << 2) & 0x3C3C3C3C and (word >> 2) & 0x3C3C3C3C are then operand A with every spare bit zero (the
 *     instruction reads them), and the code words operand B as stored: both sides meet in one order, which the sum
 *     does not show. The scale of outputs g and g + 8 is byte b of lane places 0 and 1 of lane group g, that of
 *     column g byte b of lane place 0: the weights' scale word (the odd lane's a byte down) with b = 2 q, and the
 *     fed vector's two exponents with b = q (the selector map of the instruction measured by batch i03a).
 *   - The forms each class of card takes (linear_records.h): LINEAR_TRUNK_FORM and LINEAR_EXPERT_FORM.
 *   - A slice asks for the weights of its next load before it works on the ones it holds, and never for a load
 *     beyond its own last one. In the trunk's carried form and with bfloat16 rows it asks so for the rest of its
 *     next load too (the scale codes of its weights, what its column feeds), which the compiler otherwise asks for
 *     between the instructions of a load (batch i01_r2). The codes form keeps round 1's loop, which i01 and i04
 *     timed; an expert asks for those where it uses them (at the launch bound batch i02 measured, its instance of
 *     4 slices has no registers for more: it spills).
 *   - Every function here is pure arithmetic on what it is handed or reads through the pointers of its lane: the
 *     kernel that calls it has turned the offsets of a checked record into those pointers.
 *   - A product kernel is a template of its slice count S, stamped once for each count of
 *     LINEAR_EVERY_SLICE_COUNT: its block, the cuts and the sum of its slices are constants.
 *   - LINEAR_DEVICE_CHECK states a promise of the host that a kernel relies on. In the checked modules (the build
 *     sets CHECKING_HOOKS_ENABLED to 1: the debug and thread builds) a broken promise stops the card with
 *     a trap, which the driver reports as a fault of the launch; in the release modules the check is compiled out
 *     and they hold no trap.
 */
#ifndef KERNEL_LINEAR_COMMON_CUH
#define KERNEL_LINEAR_COMMON_CUH

#include <cuda_fp16.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "anemone/linear_records.h"

#if !defined(CHECKING_HOOKS_ENABLED)
#error "the build sets ANEMONE_CHECKING_HOOKS_ENABLED: 1 for the checked kernel modules, 0 for the release modules"
#endif
#if CHECKING_HOOKS_ENABLED
#define LINEAR_DEVICE_CHECK(condition) ((condition) ? (void)0 : __trap())
#else
#define LINEAR_DEVICE_CHECK(condition) ((void)0)
#endif

#define THREADS_PER_WARP LINEAR_THREADS_PER_WARP
#define LANES_PER_LANE_GROUP 4u                 /* lane = 4 g + t: g the lane group, t the lane place */
#define LANE_GROUPS_PER_WARP 8u
#define OUTPUTS_PER_TILE LINEAR_OUTPUTS_PER_TILE
#define OUTPUTS_PER_HALF_TILE 8u                /* a lane holds an output of each half of its tile */
#define OUTPUT_HALVES 2u
#define TILES_PER_SCALE_BLOCK 2u                /* an E4M3 scale is shared by 32 outputs: two tiles */
#define COLUMNS_PER_LANE 2u                     /* a lane holds the results of columns 2 t and 2 t + 1 */
#define ELEMENTS_PER_LANE 4u                    /* its two outputs by those two columns */
#define VALUES_PER_LOAD LINEAR_VALUES_PER_LOAD
#define VALUES_PER_BLOCK LINEAR_VALUES_PER_BLOCK
#define BLOCKS_PER_LOAD 2u
#define STEPS_PER_BLOCK 2u                      /* instructions: each takes 16 inputs */
#define STEPS_PER_LOAD 4u
#define WORDS_PER_LOAD 8u                       /* of E4M3 weights in a lane: two to an instruction */
#define WORDS_PER_STEP 2u
#define REGISTERS_PER_OPERAND_A 4u
#define PAIRS_PER_LOAD 32u                      /* of bfloat16 values: a pair is one register of operand B */
#define PAIRS_PER_STEP 8u
#define PAIRS_PER_HALF_STEP 4u
#define BITS_PER_BYTE 8u
#define BYTE_MASK 0xFFu
#define BITS_PER_HALF 16u
#define UPPER_HALF_MASK 0xFFFF0000u
#define NIBBLE_BITS 4u
#define E2M1_WORD_BYTES 4u                      /* a stored word of eight codes, and a word of four scale codes */
#define E2M1_WEIGHTS_PER_BYTE 2u
#define E2M1_MAGNITUDES 0x00070007u             /* the magnitude bits of a code in both halves of a register */
#define E2M1_PAIR_MASK 0x000F000Fu              /* a code in both halves of a register */
#define E2M1_UPPER_SHIFT 8u                     /* codes 2, 3, 6 and 7 of a word: registers 2 and 3 */
#define E2M1_HALF_BIT 9u                        /* where the lowest magnitude bit of a code goes: 16-bit float */
#define E2M1_BFLOAT16_BIT 6u                    /* bfloat16 */
#define E2M1_SIGN_SHIFT 12u                     /* moves bit 3 of a code to bit 15 */
#define HALF_SIGN_MASK 0x80008000u              /* bit 15 of both halves of a register */
#define E2M1_PLACEMENT_EXPONENT 14
#define E2M1_CARRIED_FIELD 26                   /* 126 of the bfloat16 placement less the 100 of the operands */
#define E4M3_EVEN_SIGN_MASK 0x00800080u         /* bytes 0 and 2 of a word: the halves of one register */
#define E4M3_EVEN_SIGN_SHIFT 8u
#define E4M3_EVEN_MAGNITUDE_MASK 0x007F007Fu
#define E4M3_HALF_BIT 7u                        /* where the lowest magnitude bit of a byte goes: 16-bit float */
#define E4M3_BFLOAT16_BIT 4u                    /* bfloat16 */
#define E4M3_ODD_MAGNITUDE_MASK 0x7F007F00u     /* bytes 1 and 3: the halves of the other, their signs in place */
#define E4M3_PLACEMENT_EXPONENT 8
#define E4M3_CARRIED_FIELD 20                   /* 120 of the bfloat16 placement less the 100 of the operands */
#define OPERAND_LOADS 2u                        /* 16-byte loads of a lane place's operands of one load of inputs */
#define E2M1_BYTE_SHIFT 2u                      /* a code in bits 2 to 5 of a byte: the block-scaled operand A */
#define E2M1_BYTE_MASK 0x3C3C3C3Cu
#define FED_WORDS 8u                            /* the most words a lane is fed for one load of inputs */
#define WORDS_PER_PART 4u                       /* of a 16-byte load */
#define E4M3_CODE_IN_BOTH_HALVES 0x0101u        /* one scale code for both outputs of a lane */
#define E8M0_BIAS 127
#define FLOAT32_EXPONENT_SHIFT 23u
#define FLOAT32_LARGEST_EXPONENT_FIELD 254
#define FLOAT32_MAGNITUDE_MASK 0x7FFFFFFFu
#define FLOAT32_INFINITY_BITS 0x7F800000u
#define BFLOAT16_ROUNDING_BIAS 0x7FFFu          /* plus the lowest kept bit: to nearest, a tie to even */
#define BFLOAT16_NOT_A_NUMBER 0x7FC0u

/* What a template is stamped for: the weights, and the form of linear_records.h (with the rows that are not
 * quantized, which are fed as 16-bit floats). */
enum linear_weight_format { LINEAR_WEIGHTS_E2M1 = 0, LINEAR_WEIGHTS_E4M3 = 1 };
enum linear_form { LINEAR_FORM_CODES, LINEAR_FORM_CARRIED, LINEAR_FORM_BLOCK_SCALED, LINEAR_FORM_BFLOAT16_ROWS };
/* The exact profile (the default) runs class 80's carried form on every class, so that a product's bits do not depend
 * on the card it runs on. LINEAR_CLASS_FORMS selects each class's own forms (codes and block-scaled on class
 * 120), a profile of its own whose bits differ between the classes. */
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) && \
	defined(LINEAR_CLASS_FORMS) /* class 120, its own forms */
#define LINEAR_TRUNK_FORM LINEAR_FORM_CODES
#define LINEAR_EXPERT_FORM LINEAR_FORM_BLOCK_SCALED
#else                                           /* every class: class 80's */
#define LINEAR_TRUNK_FORM LINEAR_FORM_CARRIED
#define LINEAR_EXPERT_FORM LINEAR_FORM_CARRIED
#endif

/* The four registers of operand A in a lane, and the four results it holds. */
enum linear_weight_register {
    LOW_OUTPUT_LOW_INPUTS = 0,
    HIGH_OUTPUT_LOW_INPUTS = 1,
    LOW_OUTPUT_HIGH_INPUTS = 2,
    HIGH_OUTPUT_HIGH_INPUTS = 3
};
enum linear_element {
    LOW_OUTPUT_EVEN_COLUMN = 0,
    LOW_OUTPUT_ODD_COLUMN = 1,
    HIGH_OUTPUT_EVEN_COLUMN = 2,
    HIGH_OUTPUT_ODD_COLUMN = 3
};

/* What one lane of a product reads. The kernel sets it from a record whose ranges the host has checked. */
struct linear_lane {
    const uint4 *weights;                       /* its 16 bytes of the first load of weights of its tile */
    const unsigned char *weight_scales;         /* E2M1: the word of its lane group at that load; E4M3: the byte
                                                   of the first block of 32 inputs of its tile's block of outputs */
    const unsigned char *fed;                   /* the vector its column feeds, at the first load: its lane place's
                                                   16 bytes of codes or 32 of operands; bfloat16 rows: the first
                                                   value */
    const unsigned char *exponents[COLUMNS_PER_LANE];   /* quantized: of the vectors of columns 2 t and 2 t + 1 */
    uint32_t column_count;                      /* the columns in use: the exact profile's emulation computes these */
};

/* What a lane reads for one load beside its weights. */
struct linear_fed {
    uint32_t weight_codes;                      /* byte 2 q + r: block q of the load, output half r */
    uint32_t words[FED_WORDS];                  /* codes: the lane place's words of the four instructions; carried:
                                                   its eight operand registers; bfloat16 rows: its eight pairs */
    uint32_t fed_codes[COLUMNS_PER_LANE];       /* codes: byte q is block q of the load (block-scaled: column 0
                                                   only, of the vector the lane feeds) */
};

/* The value of a stored bfloat16. */
__device__ static __forceinline__ float linear_bfloat16_value(uint32_t stored_bits)
{
    return __uint_as_float(stored_bits << BITS_PER_HALF);
}

/* A float32 rounded to bfloat16, its upper half: to nearest, a tie to even; a not-a-number becomes the quiet one
 * (the plain sum would make minus zero of a not-a-number whose magnitude bits are all set, the one a card makes). */
__device__ static __forceinline__ uint32_t linear_bfloat16_bits(float value)
{
    const uint32_t bits = __float_as_uint(value);
    const uint32_t rounded = (bits + BFLOAT16_ROUNDING_BIAS + ((bits >> BITS_PER_HALF) & 1u)) >> BITS_PER_HALF;
    return (bits & FLOAT32_MAGNITUDE_MASK) > FLOAT32_INFINITY_BITS ? BFLOAT16_NOT_A_NUMBER : rounded;
}

/* The hidden value of an expert as bfloat16 bits, in the order linear_records.h states. The comparisons keep a
 * not-a-number, where a minimum or maximum of the card would drop it. */
__device__ static __forceinline__ uint32_t linear_hidden_bits(float gate, float up, float limit, float route_weight)
{
    const float bounded_gate = gate > limit ? limit : gate;
    const float bounded_up = up > limit ? limit : (up < -limit ? -limit : up);
    const float gated = bounded_gate / (1.0f + expf(-bounded_gate));
    const float hidden = gated * bounded_up;
    return linear_bfloat16_bits(hidden * route_weight);
}

/* accumulator = A B + accumulator for the lane's part of each: see the contract. 16-bit floats, or bfloat16 for
 * the carried form. */
#define LINEAR_MMA(kind)                                                                                          \
    "mma.sync.aligned.m16n8k16.row.col.f32." kind "." kind ".f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "         \
    "{%10,%11,%12,%13};\n"
#define LINEAR_MMA_OPERANDS                                                                                       \
    "=f"(accumulator[LOW_OUTPUT_EVEN_COLUMN]), "=f"(accumulator[LOW_OUTPUT_ODD_COLUMN]),                          \
        "=f"(accumulator[HIGH_OUTPUT_EVEN_COLUMN]), "=f"(accumulator[HIGH_OUTPUT_ODD_COLUMN])                     \
        : "r"(weights[LOW_OUTPUT_LOW_INPUTS]), "r"(weights[HIGH_OUTPUT_LOW_INPUTS]),                              \
          "r"(weights[LOW_OUTPUT_HIGH_INPUTS]), "r"(weights[HIGH_OUTPUT_HIGH_INPUTS]), "r"(fed_low_inputs),       \
          "r"(fed_high_inputs), "f"(accumulator[LOW_OUTPUT_EVEN_COLUMN]), "f"(accumulator[LOW_OUTPUT_ODD_COLUMN]), \
          "f"(accumulator[HIGH_OUTPUT_EVEN_COLUMN]), "f"(accumulator[HIGH_OUTPUT_ODD_COLUMN])
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) && !defined(LINEAR_CLASS_FORMS)
#define LINEAR_THREAD_SUMS 1                    /* the exact profile on class 120: groups formed by threads */
#else
#define LINEAR_THREAD_SUMS 0
#endif
#include "kernel_class_80.cuh"
#define LINEAR_ORDERED_TERM_FLOATS        \
	(LINEAR_SHARED_BYTES_PER_THREAD / \
	 4u) /* a thread's share of the slots */
#define LINEAR_ORDERED_SLOTS 2u                 /* one filled while the other is added from */
#define LINEAR_ORDERED_CHAIN_SLOTS (ELEMENTS_PER_LANE * THREADS_PER_WARP)  /* a tile's chains: 16 outputs, 8 columns */
#define LINEAR_ORDERED_NO_CHAIN UINT32_MAX
/* The most loads of a slot (16 slices, one lane place), and E4M3's powers of two (one a block for the whole tile):
 * a pair a load of each slot of each of two groups of slices (gate with up). */
#define LINEAR_ORDERED_SLOT_LOADS_MOST                       \
	(LINEAR_MAXIMUM_SLICES * THREADS_PER_WARP *          \
	 LINEAR_ORDERED_TERM_FLOATS / LINEAR_ORDERED_SLOTS / \
	 (BLOCKS_PER_LOAD * ELEMENTS_PER_LANE * LANE_GROUPS_PER_WARP))
#define LINEAR_ORDERED_LOAD_POWERS                        \
	(EXPERT_GATE_UP_MATRICES * LINEAR_ORDERED_SLOTS * \
	 LINEAR_ORDERED_SLOT_LOADS_MOST)

/* A slice's chains: in the exact profile linear_ordered_chains (the reference order whatever the slices), else
 * linear_slice_chains over the slice's run of loads. The caller's names: lane, lane_place, slice_count, thread (its
 * index in the block: threadIdx.x, or a copy the compiler cannot see through, kernel_trunk_linear.cu). */
#if defined(LINEAR_CLASS_FORMS)
#define LINEAR_CHAINS(weight_format, form, slice, load_count, chains_out)                                              \
    linear_slice_chains<weight_format, form>(&lane, lane_place, linear_slice_start(slice, slice_count, load_count),    \
                                             linear_slice_start((slice) + 1u, slice_count, load_count), chains_out)
#else
#define LINEAR_CHAINS(weight_format, form, slice, load_count, chains_out)                                              \
    if (slice_count == 1u) {                    /* one slice: its own chain is the reference order */                  \
        linear_slice_chains<weight_format, form>(&lane, lane_place, 0u, load_count, chains_out);                       \
    } else {                                                                                                           \
        linear_ordered_chains<weight_format, form>(&lane, lane_place, slice, slice_count, load_count, thread,          \
                                                   chains_out);                                                        \
    }
#endif

/* The exact profile on class 120 runs products of at most `most` slices (linear_records.h): an instance of more is a
 * stub, its block too large for the registers of the emulated instruction; the host promises never to launch it. */
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) && !defined(LINEAR_CLASS_FORMS)
#define LINEAR_SLICED_BODY(slices, most, call)                                                                         \
    if ((slices) <= (most)) {                                                                                          \
        call;                                                                                                          \
    } else {                                                                                                           \
        LINEAR_DEVICE_CHECK(false);                                                                                    \
    }
#else
#define LINEAR_SLICED_BODY(slices, most, call) call;
#endif

/* One instruction of which only the columns below column_count are used (the products' vectors in use). */
template <int form>
__device__ __forceinline__ void
linear_tensor_core_of_columns(float accumulator[ELEMENTS_PER_LANE], const uint32_t weights[REGISTERS_PER_OPERAND_A],
                              uint32_t fed_low_inputs, uint32_t fed_high_inputs, uint32_t column_count)
{
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) && \
	!defined(LINEAR_CLASS_FORMS) /* the exact profile */
	class_80_instruction(accumulator, weights, fed_low_inputs,
			     fed_high_inputs, form == LINEAR_FORM_CARRIED,
			     column_count);
#else
    (void)column_count;
    if (form == LINEAR_FORM_CARRIED) {
        asm volatile(LINEAR_MMA("bf16") : LINEAR_MMA_OPERANDS);
    } else {
        asm volatile(LINEAR_MMA("f16") : LINEAR_MMA_OPERANDS);
    }
#endif
}

/* One instruction, every column in use. */
template <int form>
__device__ __forceinline__ void linear_tensor_core(float accumulator[ELEMENTS_PER_LANE],
                                                   const uint32_t weights[REGISTERS_PER_OPERAND_A],
                                                   uint32_t fed_low_inputs, uint32_t fed_high_inputs)
{
	linear_tensor_core_of_columns<form>(accumulator, weights,
					    fed_low_inputs, fed_high_inputs,
					    LINEAR_MAXIMUM_ROWS);
}

/* accumulator = A B times the scales weight_scales and fed_scales: the block-scaled instruction, the selectors of
 * the contract for the block of a load (block 0: bytes 0 and 0; block 1: bytes 2 and 1). */
#define LINEAR_BLOCK_SCALED(selectors)                                                                            \
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 " \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, " selectors ";\n"               \
                 : LINEAR_MMA_OPERANDS, "r"(weight_scales), "r"(fed_scales))

/* The eight E2M1 codes of one stored word as the four registers of operand A: the code at bit 4 r + 16 h of the
 * word is half h of register r; its magnitude bits go to magnitude_bit on. Where a code shifted to magnitude_bit and
 * to its sign's place cannot overlap (the bfloat16 placement), one multiply makes both shifts of a pair of codes
 * isolated in place, those of registers 1 and 3 from 4 bits higher (all 2^32 words: tmp/rnd/20261025/e2m1_place). */
__device__ static __forceinline__ void linear_place_e2m1(uint32_t word, uint32_t magnitude_bit,
                                                         uint32_t weights[REGISTERS_PER_OPERAND_A])
{
    const uint32_t mask = (E2M1_MAGNITUDES << magnitude_bit) | HALF_SIGN_MASK;
    if (magnitude_bit + NIBBLE_BITS <= E2M1_SIGN_SHIFT) {
        const uint32_t shifts = (1u << magnitude_bit) | (1u << E2M1_SIGN_SHIFT);
        const uint32_t upper = word >> E2M1_UPPER_SHIFT;
        weights[LOW_OUTPUT_LOW_INPUTS] = ((word & E2M1_PAIR_MASK) * shifts) & mask;
        weights[HIGH_OUTPUT_LOW_INPUTS] = ((word & (E2M1_PAIR_MASK << NIBBLE_BITS)) * (shifts >> NIBBLE_BITS)) & mask;
        weights[LOW_OUTPUT_HIGH_INPUTS] = ((upper & E2M1_PAIR_MASK) * shifts) & mask;
        weights[HIGH_OUTPUT_HIGH_INPUTS] = ((upper & (E2M1_PAIR_MASK << NIBBLE_BITS)) * (shifts >> NIBBLE_BITS)) & mask;
        return;
    }
    for (uint32_t operand_register = 0; operand_register < REGISTERS_PER_OPERAND_A; operand_register++) {
        const uint32_t codes = word >> (NIBBLE_BITS * operand_register);
        weights[operand_register] = ((codes << magnitude_bit) & (E2M1_MAGNITUDES << magnitude_bit)) |
                                    ((codes << E2M1_SIGN_SHIFT) & HALF_SIGN_MASK);
    }
}

/* Four E4M3 bytes of one word as two registers: bytes 0 and 2 are the low and the high half of the first, bytes 1
 * and 3 those of the second; the magnitude bits go to magnitude_bit on. Weights: the registers of the two outputs
 * of a lane. A quantized vector: the registers of the low and of the high inputs of an instruction. */
__device__ static __forceinline__ void linear_place_e4m3(uint32_t word, uint32_t magnitude_bit,
                                                         uint32_t *first_register, uint32_t *second_register)
{
    *first_register = ((word & E4M3_EVEN_SIGN_MASK) << E4M3_EVEN_SIGN_SHIFT) |
                      ((word & E4M3_EVEN_MAGNITUDE_MASK) << magnitude_bit);
    *second_register =
        (word & HALF_SIGN_MASK) | ((word & E4M3_ODD_MAGNITUDE_MASK) >> (E4M3_EVEN_SIGN_SHIFT - magnitude_bit));
}

/* Two bfloat16 values of one word as one register: each converted to a 16-bit float as IEEE 754 converts a
 * float32, to nearest, a tie to even. linear_records.h says what that keeps. */
__device__ static __forceinline__ uint32_t linear_place_bfloat16(uint32_t pair)
{
    const uint32_t low_half = __half_as_ushort(__float2half(__uint_as_float(pair << BITS_PER_HALF)));
    const uint32_t high_half = __half_as_ushort(__float2half(__uint_as_float(pair & UPPER_HALF_MASK)));
    return low_half | (high_half << BITS_PER_HALF);
}

/* 2^(exponent_field - 127) for a field of 1 to 254; +0 below, 2^127 above. */
__device__ static __forceinline__ float linear_power_of_two(int32_t exponent_field)
{
    const int32_t above_zero = exponent_field < 0 ? 0 : exponent_field;
    const int32_t bounded = above_zero > FLOAT32_LARGEST_EXPONENT_FIELD ? FLOAT32_LARGEST_EXPONENT_FIELD : above_zero;
    return __uint_as_float((uint32_t)bounded << FLOAT32_EXPONENT_SHIFT);
}

/* The lane's part of the weights of one load of 64 inputs: 16 bytes of E2M1 codes, or the 16 bytes of each of its
 * two blocks of 32 inputs of E4M3 bytes. */
template <int weight_format>
__device__ __forceinline__ void linear_fetch_weights(const struct linear_lane *lane, uint32_t load, uint4 *first,
                                                     uint4 *second)
{
    if (weight_format == LINEAR_WEIGHTS_E2M1) {
        *first = lane->weights[(size_t)load * THREADS_PER_WARP];
        *second = *first;
    } else {
        *first = lane->weights[(size_t)load * BLOCKS_PER_LOAD * THREADS_PER_WARP];
        *second = lane->weights[((size_t)load * BLOCKS_PER_LOAD + 1u) * THREADS_PER_WARP];
    }
}

/* The scale codes of the weights of load `load` (struct linear_fed's weight_codes); weight_scales is a lane's. */
template <int weight_format>
__device__ __forceinline__ uint32_t linear_weight_codes(const unsigned char *weight_scales, uint32_t load)
{
    if (weight_format == LINEAR_WEIGHTS_E2M1) {
        return ((const uint32_t *)weight_scales)[(size_t)load * LANE_GROUPS_PER_WARP];
    }
    const unsigned char *block_codes = weight_scales + (size_t)load * BLOCKS_PER_LOAD;
    return ((uint32_t)block_codes[0] | ((uint32_t)block_codes[1] << BITS_PER_HALF)) * E4M3_CODE_IN_BOTH_HALVES;
}

/* The power of two of block `block` of a load for output half `output_half`: the weights' scale code (of the load's
 * weight_codes) plus what their placement took plus fed_field (the codes form: the vector's exponent less 127 and
 * the 8 of its placement; else 0). A carried field is above 0: only its upper end is held. */
template <int weight_format, int form>
__device__ __forceinline__ float linear_block_power(uint32_t weight_codes, uint32_t block, uint32_t output_half,
                                                    int32_t fed_field)
{
    const bool is_e2m1 = weight_format == LINEAR_WEIGHTS_E2M1;
    const bool is_carried = form == LINEAR_FORM_CARRIED;
    const int32_t weight_field = is_carried ? (is_e2m1 ? E2M1_CARRIED_FIELD : E4M3_CARRIED_FIELD)
                                            : (is_e2m1 ? E2M1_PLACEMENT_EXPONENT : E4M3_PLACEMENT_EXPONENT);
    const uint32_t weight_code = (weight_codes >> ((block * OUTPUT_HALVES + output_half) * BITS_PER_BYTE)) & BYTE_MASK;
    const int32_t exponent_field = (int32_t)weight_code + weight_field + fed_field;
    const int32_t bounded =
        exponent_field > FLOAT32_LARGEST_EXPONENT_FIELD ? FLOAT32_LARGEST_EXPONENT_FIELD : exponent_field;
    return is_carried ? __uint_as_float((uint32_t)bounded << FLOAT32_EXPONENT_SHIFT)
                      : linear_power_of_two(exponent_field);
}

/* What the lane reads for load `load` beside its weights, in a form: see struct linear_fed. */
template <int weight_format, int form>
__device__ __forceinline__ struct linear_fed linear_fetch_fed(const struct linear_lane *lane, uint32_t lane_place,
                                                              uint32_t load)
{
    struct linear_fed fed = {};
    fed.weight_codes = linear_weight_codes<weight_format>(lane->weight_scales, load);
    if (form == LINEAR_FORM_BFLOAT16_ROWS) {
        const uint32_t *pairs = (const uint32_t *)lane->fed + (size_t)load * PAIRS_PER_LOAD + lane_place;
        for (uint32_t step = 0; step < STEPS_PER_LOAD; step++) {
            fed.words[step * WORDS_PER_STEP] = pairs[step * PAIRS_PER_STEP];
            fed.words[step * WORDS_PER_STEP + 1u] = pairs[step * PAIRS_PER_STEP + PAIRS_PER_HALF_STEP];
        }
        return fed;
    }
    const uint32_t part_count = form == LINEAR_FORM_CARRIED ? OPERAND_LOADS : 1u;
    const uint4 *parts = (const uint4 *)lane->fed + (size_t)load * LANES_PER_LANE_GROUP * part_count;
    for (uint32_t part = 0; part < part_count; part++) {
        const uint4 loaded = parts[part];
        const uint32_t words[WORDS_PER_PART] = {loaded.x, loaded.y, loaded.z, loaded.w};
        for (uint32_t word = 0; word < WORDS_PER_PART; word++) {
            fed.words[part * WORDS_PER_PART + word] = words[word];
        }
    }
    const uint32_t column_count =
        form == LINEAR_FORM_CODES ? COLUMNS_PER_LANE : (form == LINEAR_FORM_BLOCK_SCALED ? 1u : 0u);
    for (uint32_t column = 0; column < column_count; column++) {
        fed.fed_codes[column] = ((const uint16_t *)lane->exponents[column])[load];
    }
    return fed;
}

/* The sums of the two blocks of 32 inputs of one load for a lane's four results, and the powers of two that scale
 * them: a chain adds each as fma(sum, power, chain), block 0 first (block-scaled: the sum with its scales inside it,
 * power 1, which is the chain's plain addition). weight_words and fed are the load's. */
template <int weight_format, int form>
__device__ __forceinline__ void
linear_load_sums(const struct linear_lane *lane, const uint32_t weight_words[WORDS_PER_LOAD],
                 const struct linear_fed *fed, float sums[BLOCKS_PER_LOAD][ELEMENTS_PER_LANE],
                 float powers[BLOCKS_PER_LOAD][ELEMENTS_PER_LANE])
{
    const bool is_e2m1 = weight_format == LINEAR_WEIGHTS_E2M1;
    const bool is_carried = form == LINEAR_FORM_CARRIED;
    const uint32_t weight_bit =
        is_carried ? (is_e2m1 ? E2M1_BFLOAT16_BIT : E4M3_BFLOAT16_BIT) : (is_e2m1 ? E2M1_HALF_BIT : E4M3_HALF_BIT);
#pragma unroll
    for (uint32_t block = 0; form == LINEAR_FORM_BLOCK_SCALED && block < BLOCKS_PER_LOAD; block++) {
        float accumulator[ELEMENTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
        const uint32_t first_word = weight_words[block * STEPS_PER_BLOCK];
        const uint32_t second_word = weight_words[block * STEPS_PER_BLOCK + 1u];
        const uint32_t weights[REGISTERS_PER_OPERAND_A] = {
            (first_word << E2M1_BYTE_SHIFT) & E2M1_BYTE_MASK, (first_word >> E2M1_BYTE_SHIFT) & E2M1_BYTE_MASK,
            (second_word << E2M1_BYTE_SHIFT) & E2M1_BYTE_MASK, (second_word >> E2M1_BYTE_SHIFT) & E2M1_BYTE_MASK};
        const uint32_t fed_low_inputs = fed->words[block * STEPS_PER_BLOCK];
        const uint32_t fed_high_inputs = fed->words[block * STEPS_PER_BLOCK + 1u];
        /* the scale word of the weights, the odd lane's a byte down */
        const uint32_t weight_scales = fed->weight_codes >> ((threadIdx.x & 1u) * BITS_PER_BYTE);
        const uint32_t fed_scales = fed->fed_codes[0];
        if (block == 0) {
            LINEAR_BLOCK_SCALED("{0, 0}, %15, {0, 0}");
        } else {
            LINEAR_BLOCK_SCALED("{2, 0}, %15, {1, 0}");
        }
#endif
        for (uint32_t element = 0; element < ELEMENTS_PER_LANE; element++) {
            sums[block][element] = accumulator[element];
            powers[block][element] = 1.0f;
        }
    }
#pragma unroll
    for (uint32_t block = 0; form != LINEAR_FORM_BLOCK_SCALED && block < BLOCKS_PER_LOAD; block++) {
        float accumulator[ELEMENTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (uint32_t step = block * STEPS_PER_BLOCK; step < (block + 1u) * STEPS_PER_BLOCK; step++) {
            uint32_t weights[REGISTERS_PER_OPERAND_A];
            uint32_t fed_low_inputs, fed_high_inputs;
            if (is_e2m1) {
                linear_place_e2m1(weight_words[step], weight_bit, weights);
            } else {
                linear_place_e4m3(weight_words[step * WORDS_PER_STEP], weight_bit, &weights[LOW_OUTPUT_LOW_INPUTS],
                                  &weights[HIGH_OUTPUT_LOW_INPUTS]);
                linear_place_e4m3(weight_words[step * WORDS_PER_STEP + 1u], weight_bit,
                                  &weights[LOW_OUTPUT_HIGH_INPUTS], &weights[HIGH_OUTPUT_HIGH_INPUTS]);
            }
            if (form == LINEAR_FORM_CODES) {
                linear_place_e4m3(fed->words[step], E4M3_HALF_BIT, &fed_low_inputs, &fed_high_inputs);
            } else if (is_carried) {
                fed_low_inputs = fed->words[step * WORDS_PER_STEP];
                fed_high_inputs = fed->words[step * WORDS_PER_STEP + 1u];
            } else {
                fed_low_inputs = linear_place_bfloat16(fed->words[step * WORDS_PER_STEP]);
                fed_high_inputs = linear_place_bfloat16(fed->words[step * WORDS_PER_STEP + 1u]);
            }
            linear_tensor_core_of_columns<form>(accumulator, weights, fed_low_inputs, fed_high_inputs,
                                                lane->column_count);
        }
#pragma unroll
        for (uint32_t element = 0; element < ELEMENTS_PER_LANE; element++) {
            const uint32_t output_half = element / COLUMNS_PER_LANE, column = element % COLUMNS_PER_LANE;
            const int32_t fed_field =
                form == LINEAR_FORM_CODES
                    ? (int32_t)((fed->fed_codes[column] >> (block * BITS_PER_BYTE)) & BYTE_MASK) +
                          (E4M3_PLACEMENT_EXPONENT - E8M0_BIAS)
                    : 0;
            sums[block][element] = accumulator[element];
            powers[block][element] =
                linear_block_power<weight_format, form>(fed->weight_codes, block, output_half, fed_field);
        }
    }
}

/* One slice of one tile for one lane: the chains of its four results over loads first_load .. end_load - 1, each
 * from +0 with one fused multiply-add per block of 32 inputs in ascending order (block-scaled: the addition of the
 * block's sum, its scales inside it). lane_place is t. The chains are the lane's own until the slice ends:
 * chains_out, the lane's place in the block's shared memory, is written once. */
template <int weight_format, int form>
__device__ __forceinline__ void linear_slice_chains(const struct linear_lane *lane, uint32_t lane_place,
                                                    uint32_t first_load, uint32_t end_load,
                                                    float chains_out[ELEMENTS_PER_LANE])
{
    const bool is_ahead = weight_format != LINEAR_WEIGHTS_E2M1 && form != LINEAR_FORM_CODES;   /* see the top */
    float chains[ELEMENTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint4 ahead_first, ahead_second;
    struct linear_fed ahead_fed = {};
    linear_fetch_weights<weight_format>(lane, first_load, &ahead_first, &ahead_second);
    if (is_ahead) {
        ahead_fed = linear_fetch_fed<weight_format, form>(lane, lane_place, first_load);
    }
    for (uint32_t load = first_load; load < end_load; load++) {
        const uint4 first = ahead_first, second = ahead_second;
        const uint32_t weight_words[WORDS_PER_LOAD] = {first.x,  first.y,  first.z,  first.w,
                                                       second.x, second.y, second.z, second.w};
        struct linear_fed fed = ahead_fed;
        if (load + 1u < end_load) {
            linear_fetch_weights<weight_format>(lane, load + 1u, &ahead_first, &ahead_second);
            if (is_ahead) {
                ahead_fed = linear_fetch_fed<weight_format, form>(lane, lane_place, load + 1u);
            }
        }
        if (!is_ahead) {
            fed = linear_fetch_fed<weight_format, form>(lane, lane_place, load);
        }
        float sums[BLOCKS_PER_LOAD][ELEMENTS_PER_LANE], powers[BLOCKS_PER_LOAD][ELEMENTS_PER_LANE];
        linear_load_sums<weight_format, form>(lane, weight_words, &fed, sums, powers);
#pragma unroll
        for (uint32_t block = 0; block < BLOCKS_PER_LOAD; block++) {
#pragma unroll
            for (uint32_t element = 0; element < ELEMENTS_PER_LANE; element++) {
                chains[element] = fmaf(sums[block][element], powers[block][element], chains[element]);
            }
        }
    }
    for (uint32_t element = 0; element < ELEMENTS_PER_LANE; element++) {
        chains_out[element] = chains[element];
    }
}

/* Chain i of a tile in the exact profile's slices (linear_ordered_chains), of a layout of `places` lane places: its
 * element e, lane 4 g + t (i = (e 8 + g) places + t) and column; it is in use when the column is. */
struct linear_chain_slot {
    uint32_t element, lane_index, column;
};
__device__ static __forceinline__ struct linear_chain_slot linear_chain_slot_of(uint32_t chain, uint32_t places)
{
    const uint32_t lane_slot = chain / places, place = chain - lane_slot * places;   /* lane_slot: e 8 + g */
    const uint32_t element = lane_slot / LANE_GROUPS_PER_WARP;
    const uint32_t lane_index = lane_slot % LANE_GROUPS_PER_WARP * LANES_PER_LANE_GROUP + place;
    return (struct linear_chain_slot){element, lane_index, place * COLUMNS_PER_LANE + element % COLUMNS_PER_LANE};
}

/* The exact profile's slices of one tile (the reference order whatever S). Slice s of S takes loads s, s + S, ..,
 * asking for the next one before it works on the one it holds, and leaves the sums of its blocks in one of two slots
 * of its group's shared memory (LINEAR_ORDERED_TERM_FLOATS a thread between them), for the P lane places whose
 * columns are in use: a load's at [e][g][t], element e of lane 4 g + t, t < P, the sums of its two blocks side by
 * side (one 8-byte store and one 8-byte load each, no two lanes of a store in a bank); a slot holds the rounds of S
 * loads it has room for; with E4M3 weights its lane 0 leaves the two blocks' powers of two beside them. After the
 * barrier that closes a slot, thread i of the group (and i + 32 S, ..) adds that slot's blocks to chain i of the
 * layout in ascending order, as one slice adds them over every load, each with its power of two (E2M1: from the
 * scale code of the chain's output), while the slices fill the other slot: a slot is filled again only after the
 * next barrier, which a thread passes once it has added from it. After the last slot a barrier frees the memory: the
 * first slice's slots get every chain, the other slices' +0 (their sum is then the first one's). Every warp of the
 * block passes the same barriers. thread is the thread's index in the block (LINEAR_CHAINS). */
template <int weight_format, int form>
__device__ __forceinline__ void linear_ordered_chains(const struct linear_lane *lane, uint32_t lane_place,
                                                      uint32_t slice, uint32_t slice_count, uint32_t load_count,
                                                      uint32_t thread, float chains_out[ELEMENTS_PER_LANE])
{
    extern __shared__ float linear_ordered_terms[]; /* the block's: the slice chains' memory */
    __shared__ float2 linear_ordered_powers[weight_format == LINEAR_WEIGHTS_E2M1 ? 1u : LINEAR_ORDERED_LOAD_POWERS];
    const bool is_e2m1 = weight_format == LINEAR_WEIGHTS_E2M1, is_ahead = !is_e2m1;   /* the experts: no registers */
    const uint32_t lane_index = thread % THREADS_PER_WARP, lane_group = lane_index / LANES_PER_LANE_GROUP;
    const uint32_t group_threads = slice_count * THREADS_PER_WARP;
    const uint32_t group_thread = slice * THREADS_PER_WARP + lane_index;
    float *const slots = linear_ordered_terms + (size_t)(thread - group_thread) * LINEAR_ORDERED_TERM_FLOATS;
    const uint32_t group = (thread - group_thread) / group_threads;   /* of gate with up: 0 or 1 */
    const uint32_t slot_floats = group_threads * LINEAR_ORDERED_TERM_FLOATS / LINEAR_ORDERED_SLOTS;
    const uint32_t places = lane->column_count < COLUMNS_PER_LANE ? 1u : (lane->column_count + 1u) / COLUMNS_PER_LANE;
    const uint32_t element_floats = LANE_GROUPS_PER_WARP * places, block_floats = ELEMENTS_PER_LANE * element_floats;
    const uint32_t slot_loads = slot_floats / (BLOCKS_PER_LOAD * block_floats * slice_count) * slice_count;
    /* the chains this thread adds to: their place in a block's sums (LINEAR_ORDERED_NO_CHAIN: not in use) */
    float chains[ELEMENTS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t chain_sums[ELEMENTS_PER_LANE];
#pragma unroll
    for (uint32_t taken = 0; taken * group_threads < LINEAR_ORDERED_CHAIN_SLOTS; taken++) {
        const uint32_t chain = group_thread + taken * group_threads;
        const struct linear_chain_slot of = linear_chain_slot_of(chain, places);
        chain_sums[taken] = chain < block_floats && of.column < lane->column_count ? chain : LINEAR_ORDERED_NO_CHAIN;
    }
    uint4 ahead_first = {}, ahead_second = {};
    struct linear_fed ahead_fed = {};
    if (is_ahead && slice < load_count) {
        linear_fetch_weights<weight_format>(lane, slice, &ahead_first, &ahead_second);
        ahead_fed = linear_fetch_fed<weight_format, form>(lane, lane_place, slice);
    }
    for (uint32_t slot_first = 0, slot = 0; slot_first < load_count; slot_first += slot_loads, slot ^= 1u) {
        const uint32_t slot_end = slot_loads < load_count - slot_first ? slot_first + slot_loads : load_count;
        float *const slot_sums = slots + (size_t)slot * slot_floats;
        const uint32_t slot_index = group * LINEAR_ORDERED_SLOTS + slot;   /* of the powers' slots */
        float2 *const slot_powers =
            linear_ordered_powers + (is_e2m1 ? 0u : slot_index * LINEAR_ORDERED_SLOT_LOADS_MOST);
        for (uint32_t load = slot_first + slice; load < slot_end; load += slice_count) {
            uint4 first = ahead_first, second = ahead_second;
            struct linear_fed fed = ahead_fed;
            if (!is_ahead) {
                linear_fetch_weights<weight_format>(lane, load, &first, &second);
                fed = linear_fetch_fed<weight_format, form>(lane, lane_place, load);
            } else if (load + slice_count < load_count) {
                linear_fetch_weights<weight_format>(lane, load + slice_count, &ahead_first, &ahead_second);
                ahead_fed = linear_fetch_fed<weight_format, form>(lane, lane_place, load + slice_count);
            }
            const uint32_t weight_words[WORDS_PER_LOAD] = {first.x,  first.y,  first.z,  first.w,
                                                           second.x, second.y, second.z, second.w};
            float sums[BLOCKS_PER_LOAD][ELEMENTS_PER_LANE], powers[BLOCKS_PER_LOAD][ELEMENTS_PER_LANE];
            linear_load_sums<weight_format, form>(lane, weight_words, &fed, sums, powers);
            float2 *const load_sums = (float2 *)(slot_sums + (size_t)(load - slot_first) * BLOCKS_PER_LOAD *
                                                                 block_floats) +
                                      lane_group * places + lane_place;
#pragma unroll
            for (uint32_t element = 0; lane_place < places && element < ELEMENTS_PER_LANE; element++) {
                load_sums[element * element_floats] = make_float2(sums[0][element], sums[1][element]);
            }
            if (!is_e2m1 && lane_index == 0) {
                slot_powers[load - slot_first] = make_float2(powers[0][0], powers[1][0]);
            }
        }
        __syncthreads();
#pragma unroll
        for (uint32_t taken = 0; taken * group_threads < LINEAR_ORDERED_CHAIN_SLOTS; taken++) {
            if (chain_sums[taken] == LINEAR_ORDERED_NO_CHAIN) {
                continue;
            }
            /* E2M1: the scale word of the chain's lane group, and its output half */
            const struct linear_chain_slot of =
                is_e2m1 ? linear_chain_slot_of(chain_sums[taken], places) : (struct linear_chain_slot){};
            const unsigned char *const group_scales =
                lane->weight_scales +
                ((int32_t)(of.lane_index / LANES_PER_LANE_GROUP) - (int32_t)lane_group) * (int32_t)E2M1_WORD_BYTES;
            const float2 *load_sums = (const float2 *)slot_sums + chain_sums[taken], *load_powers = slot_powers;
            for (uint32_t load = slot_first; load < slot_end; load++, load_sums += block_floats, load_powers++) {
                const uint32_t codes = is_e2m1 ? linear_weight_codes<weight_format>(group_scales, load) : 0u;
                const uint32_t half = of.element / COLUMNS_PER_LANE;
                const float2 sum = *load_sums;
                const float2 power = is_e2m1 ? make_float2(linear_block_power<weight_format, form>(codes, 0, half, 0),
                                                           linear_block_power<weight_format, form>(codes, 1, half, 0))
                                             : *load_powers;
                chains[taken] = fmaf(sum.x, power.x, chains[taken]);   /* block 0, then block 1 */
                chains[taken] = fmaf(sum.y, power.y, chains[taken]);
            }
        }
    }
    __syncthreads();
    float *const first_slice_out = chains_out - (size_t)group_thread * ELEMENTS_PER_LANE;   /* its lane 0 */
    for (uint32_t element = 0; slice != 0 && element < ELEMENTS_PER_LANE; element++) {
        chains_out[element] = 0.0f;
    }
#pragma unroll
    for (uint32_t taken = 0; taken * group_threads < LINEAR_ORDERED_CHAIN_SLOTS; taken++) {
        const uint32_t chain = group_thread + taken * group_threads;
        const struct linear_chain_slot of = linear_chain_slot_of(chain, places);
        if (chain < block_floats && of.column < lane->column_count) {
            first_slice_out[of.lane_index * ELEMENTS_PER_LANE + of.element] = chains[taken];
        }
    }
}

/* The lane of a thread for one matrix and the tile of its block: where its weights and their scales lie. matrix is
 * the first stored byte of a matrix of output_size by input_size. */
template <int weight_format>
__device__ __forceinline__ void linear_lane_weights(struct linear_lane *lane, const unsigned char *matrix,
                                                    uint32_t output_size, uint32_t input_size, uint32_t tile,
                                                    uint32_t lane_index)
{
    const size_t weight_count = (size_t)output_size * input_size;
    if (weight_format == LINEAR_WEIGHTS_E2M1) {
        const size_t tile_load = (size_t)tile * (input_size / VALUES_PER_LOAD);
        lane->weights = (const uint4 *)matrix + tile_load * THREADS_PER_WARP + lane_index;
        lane->weight_scales = matrix + weight_count / E2M1_WEIGHTS_PER_BYTE +
                              (tile_load * LANE_GROUPS_PER_WARP + lane_index / LANES_PER_LANE_GROUP) * E2M1_WORD_BYTES;
    } else {
        const size_t block_count = input_size / VALUES_PER_BLOCK;
        lane->weights = (const uint4 *)matrix + (size_t)tile * block_count * THREADS_PER_WARP + lane_index;
        lane->weight_scales = matrix + weight_count + (size_t)(tile / TILES_PER_SCALE_BLOCK) * block_count;
    }
}

/* The lane of a thread for quantized vectors of a width in a form: the vector its column feeds and the vectors of
 * the two columns it holds results of. codes, exponents and operands are the arrays of linear_records.h; the
 * carried form reads the operands alone. */
template <int form>
__device__ __forceinline__ void linear_lane_quantized(struct linear_lane *lane, const unsigned char *codes,
                                                      const unsigned char *exponents, const unsigned char *operands,
                                                      uint32_t width, uint32_t lane_place, uint32_t fed_vector,
                                                      uint32_t even_column_vector, uint32_t odd_column_vector)
{
    const size_t exponents_per_vector = width / VALUES_PER_BLOCK;
    const size_t lane_place_operands = (size_t)lane_place * OPERAND_LOADS * sizeof(uint4);   /* bytes */
    lane->fed =
	    form == LINEAR_FORM_CARRIED ?
		    operands +
			    (size_t)fed_vector * width * LINEAR_BFLOAT16_BYTES +
			    lane_place_operands :
		    codes + (size_t)fed_vector * width +
			    (size_t)lane_place * sizeof(uint4);
    lane->exponents[0] = exponents + (size_t)(form == LINEAR_FORM_BLOCK_SCALED ? fed_vector : even_column_vector) *
                                         exponents_per_vector;
    lane->exponents[1] = exponents + (size_t)odd_column_vector * exponents_per_vector;
}

/* The first load a slice holds, of load_count loads cut into slice_count slices: slice slice_count is the end.
 * A slice is at most the 32 warps of the largest block and a width holds fewer than 2^26 loads: no overflow. */
__device__ static __forceinline__ uint32_t linear_slice_start(uint32_t slice, uint32_t slice_count,
                                                              uint32_t load_count)
{
    return slice * load_count / slice_count;
}

/* One result of one matrix: the chains of its slices, which warps first_warp .. first_warp + slice_count - 1 left
 * in the block's shared memory as [warp][lane][element], added from +0 in ascending slice order. */
template <uint32_t slice_count>
__device__ __forceinline__ float linear_sum_of_slices(const float *slice_chains, uint32_t first_warp,
                                                      uint32_t lane_index, uint32_t element)
{
    const float *lane_chains = slice_chains + ((size_t)first_warp * THREADS_PER_WARP + lane_index) * ELEMENTS_PER_LANE;
    float sum = 0.0f;
#pragma unroll
    for (uint32_t slice = 0; slice < slice_count; slice++) {
        sum += lane_chains[(size_t)slice * THREADS_PER_WARP * ELEMENTS_PER_LANE + element];
    }
    return sum;
}

#if !LINEAR_THREAD_SUMS
#define LINEAR_THREAD_CACHE(name, threads) uint32_t *const name = NULL
#endif
#if LINEAR_THREAD_SUMS
/* ---- the exact profile on class 120: a product's chains from groups formed by threads (kernel_class_80.cuh) ---- */
#define LINEAR_THREAD_ROWS OUTPUTS_PER_TILE     /* a tile's outputs: a thread takes one at a block of 32 inputs */
#define LINEAR_THREAD_GROUPS_PER_BLOCK (VALUES_PER_BLOCK / CLASS_80_GROUP_PRODUCTS)
#define LINEAR_THREAD_SUMS_STRIDE \
	(LINEAR_MAXIMUM_ROWS + 1u) /* floats: the threads apart in the banks */
#define LINEAR_THREAD_SCRATCH_FLOATS (LINEAR_THREAD_SUMS_STRIDE + 1u)  /* of a thread: its sums, a power */
#define LINEAR_THREAD_CHAINS_MOST 4u            /* chains a thread folds: 16 outputs by 8 columns over 32 threads */
/* The vectors the columns read: eight 16-bit indices, columns 0 to 3 in a first word and 4 to 7 in a second (a wide
 * step's jobs read vectors up to 24,575); the trunk's column c reads vector c. */
#define LINEAR_THREAD_COLUMNS_A_WORD 4u
#define LINEAR_THREAD_COLUMN_INDEX_BITS 16u
#define LINEAR_THREAD_COLUMN_INDEX_MASK 0xFFFFu
#define LINEAR_THREAD_COLUMN_VECTORS_LOW 0x0003000200010000ull
#define LINEAR_THREAD_COLUMN_VECTORS_HIGH 0x0007000600050004ull
#define LINEAR_THREAD_E4M3_ZERO_MASK 0x7Fu
#define LINEAR_THREAD_E4M3_SIGN_BIT 0x80u
#define LINEAR_THREAD_E4M3_FIELD_SHIFT 3u
#define LINEAR_THREAD_E4M3_FIELD_MASK 0xFu
#define LINEAR_THREAD_E4M3_MANTISSA_MASK 0x7u
#define LINEAR_THREAD_E2M1_ZERO_MASK 0x7u
#define LINEAR_THREAD_E2M1_SIGN_BIT 0x8u
#define LINEAR_THREAD_E2M1_FIELD_MASK 0x3u
#define LINEAR_THREAD_E2M1_MANTISSA_MASK 0x1u
#define LINEAR_THREAD_NIBBLE_MASK 0xFu
#define LINEAR_THREAD_E2M1_WHOLE_ONE 2u         /* 2 w of a normal E2M1 code: (2 + mantissa) times 2^(field - 1) */
#define LINEAR_THREAD_WHOLE_FIELD_OFFSET 98     /* an operand's field less c + 98: the shift of 128 + its fraction */
#define LINEAR_THREAD_WHOLE_POWER_OFFSET 163    /* a block's placed sum: its whole numbers' sum times 2^(c - 163) */
#define LINEAR_THREAD_FED_LOAD_BYTES \
	(VALUES_PER_LOAD * LINEAR_BFLOAT16_BYTES) /* of a vector, either form */
#define LINEAR_THREAD_FED_PLACE_BYTES (OPERAND_LOADS * sizeof(uint4))                   /* carried: a lane place's */
#define LINEAR_THREAD_BLOCK_WORDS (WORDS_PER_LOAD / BLOCKS_PER_LOAD)                     /* a lane's of a block */
/* A round's fed values decoded once (linear_thread_chains): a word each, its column magnitude, its exponent biased by
 * the zero exponent's magnitude and its sign; a thread's share of a round of 8 columns. */
#define LINEAR_THREAD_CACHE_WORDS \
	(LINEAR_MAXIMUM_ROWS * VALUES_PER_BLOCK / LINEAR_THREAD_ROWS)
#define LINEAR_THREAD_PACKED_MAGNITUDE_MASK 0x3FFFFu
#define LINEAR_THREAD_PACKED_EXPONENT_SHIFT 18u
#define LINEAR_THREAD_PACKED_EXPONENT_MASK 0x1FFFu
#define LINEAR_THREAD_PACKED_EXPONENT_BIAS (-CLASS_80_ZERO_EXPONENT)
#define LINEAR_THREAD_PACKED_SIGN_BIT 0x80000000u
#define LINEAR_THREAD_CACHE_VECTORS(threads) \
    ((threads) * LINEAR_THREAD_CACHE_WORDS / (sizeof(uint4) / sizeof(uint32_t)))
/* A kernel's round caches (static shared memory) for threads threads, as name; NULL outside the exact profile on
 * class 120. */
#define LINEAR_THREAD_CACHE(name, threads)                                                                            \
    __shared__ uint4 name##_vectors[LINEAR_THREAD_CACHE_VECTORS(threads)];                                         \
    uint32_t *const name = (uint32_t *)name##_vectors
static_assert((((1u << (CLASS_80_BFLOAT16_FIELD_SHIFT + 1u)) - 1u) << CLASS_80_BFLOAT16_COLUMN_SHIFT) <=
                      LINEAR_THREAD_PACKED_MAGNITUDE_MASK &&
                  CLASS_80_NOT_FINITE_EXPONENT + LINEAR_THREAD_PACKED_EXPONENT_BIAS <=
                      (int32_t)LINEAR_THREAD_PACKED_EXPONENT_MASK,
              "a packed word holds a column magnitude and every exponent");

/* A fed value decoded (class_80_thread_decode) into its packed word. */
template <bool is_bfloat16>
__device__ __forceinline__ uint32_t linear_thread_packed(uint32_t bits, uint32_t column_shift)
{
    uint32_t magnitude = 0;
    int32_t exponent = 0, sign = 0;
    class_80_thread_decode<is_bfloat16>(bits, &magnitude, &exponent, &sign);
    return (magnitude << column_shift) |
           ((uint32_t)(exponent + LINEAR_THREAD_PACKED_EXPONENT_BIAS) << LINEAR_THREAD_PACKED_EXPONENT_SHIFT) |
           ((uint32_t)sign & LINEAR_THREAD_PACKED_SIGN_BIT);
}

/* A weight as the form places it, decoded: an E2M1 code (the bfloat16 placement: carried) or an E4M3 byte (the
 * bfloat16 placement carried, the 16-bit float placement for bfloat16 rows). */
template <int weight_format, int form>
__device__ __forceinline__ void linear_thread_weight(uint32_t code, uint32_t *magnitude_out, int32_t *exponent_out,
                                                     int32_t *sign_out)
{
    const bool is_e2m1 = weight_format == LINEAR_WEIGHTS_E2M1;
    const uint32_t field = is_e2m1 ? (code >> 1) & LINEAR_THREAD_E2M1_FIELD_MASK
                                   : (code >> LINEAR_THREAD_E4M3_FIELD_SHIFT) & LINEAR_THREAD_E4M3_FIELD_MASK;
    const uint32_t mantissa =
        is_e2m1 ? code & LINEAR_THREAD_E2M1_MANTISSA_MASK : code & LINEAR_THREAD_E4M3_MANTISSA_MASK;
    const uint32_t fraction_bits =
        form == LINEAR_FORM_CARRIED ? CLASS_80_BFLOAT16_FIELD_SHIFT : CLASS_80_HALF_FIELD_SHIFT;
    const uint32_t mantissa_bit = fraction_bits - (is_e2m1 ? 1u : LINEAR_THREAD_E4M3_FIELD_SHIFT);
    const bool is_zero = (code & (is_e2m1 ? LINEAR_THREAD_E2M1_ZERO_MASK : LINEAR_THREAD_E4M3_ZERO_MASK)) == 0;
    *magnitude_out = (mantissa << mantissa_bit) | (field != 0 ? 1u << fraction_bits : 0u);
    *exponent_out = is_zero ? CLASS_80_ZERO_EXPONENT : (int32_t)(field != 0 ? field : 1u);
    *sign_out = (code & (is_e2m1 ? LINEAR_THREAD_E2M1_SIGN_BIT : LINEAR_THREAD_E4M3_SIGN_BIT)) != 0 ? -1 : 0;
}

/* The weight code of input 2 t + input_half of group `group` of a block (its step group / 2, the lower or upper 8
 * inputs of that step) for output half `half` of a lane group: words are its four lanes' words of the block. */
template <int weight_format>
__device__ __forceinline__ uint32_t linear_thread_code(const uint32_t words[LANES_PER_LANE_GROUP]
                                                                           [LINEAR_THREAD_BLOCK_WORDS],
                                                       uint32_t group, uint32_t half, uint32_t lane_place,
                                                       uint32_t input_half)
{
    const uint32_t step = group / OUTPUT_HALVES, upper = group % OUTPUT_HALVES;
    if (weight_format == LINEAR_WEIGHTS_E2M1) {    /* a word a step: register half + 2 upper, half input_half */
        return (words[lane_place][step] >>
                (NIBBLE_BITS * (half + OUTPUT_HALVES * upper) + BITS_PER_HALF * input_half)) &
               LINEAR_THREAD_NIBBLE_MASK;
    }
    /* two words a step, the lower and the upper inputs: output g in bytes 0 and 2, output g + 8 in bytes 1 and 3 */
    return (words[lane_place][step * WORDS_PER_STEP + upper] >> (BITS_PER_BYTE * (half + OUTPUT_HALVES * input_half))) &
           BYTE_MASK;
}

/* An E2M1 code as the whole number W = 2 w of class_80_e2m1_group: 0, 1, 2, 3, 4, 6, 8 or 12, signed. */
__device__ __forceinline__ int32_t linear_thread_whole_weight(uint32_t code)
{
    const uint32_t field = (code >> 1) & LINEAR_THREAD_E2M1_FIELD_MASK;
    const uint32_t mantissa = code & LINEAR_THREAD_E2M1_MANTISSA_MASK;
    const int32_t value =
        field == 0 ? (int32_t)mantissa : (int32_t)((LINEAR_THREAD_E2M1_WHOLE_ONE + mantissa) << (field - 1u));
    return (code & LINEAR_THREAD_E2M1_SIGN_BIT) != 0 ? -value : value;
}

/* A carried operand (bfloat16 of an E4M3 value x times 2^(c - 27)) of a block of exponent c as the whole number
 * A = 512 x of class_80_e2m1_group, signed: (128 + its 7 fraction bits) moved by its field less c + 98 (exact: x
 * has 4 significant bits; a zero's field is 0). */
__device__ __forceinline__ int32_t linear_thread_whole_activation(uint32_t operand, uint32_t exponent)
{
    const uint32_t field = (operand >> CLASS_80_BFLOAT16_FIELD_SHIFT) & CLASS_80_BFLOAT16_FIELD_MASK;
    const uint32_t significand =
        (1u << CLASS_80_BFLOAT16_FIELD_SHIFT) | (operand & ((1u << CLASS_80_BFLOAT16_FIELD_SHIFT) - 1u));
    const int32_t shift = (int32_t)field - (int32_t)exponent - LINEAR_THREAD_WHOLE_FIELD_OFFSET;
    const int32_t magnitude = field == 0 ? 0
                              : shift >= 0 ? (int32_t)(significand << shift)
                                           : (int32_t)(significand >> -shift);
    return (operand & CLASS_80_HALF_SIGN_BIT) != 0 ? -magnitude : magnitude;
}

/* The vector column `column` reads: index `column` of eight 16-bit ones, four a word. */
__device__ static __forceinline__ size_t linear_thread_column_vector(uint64_t low_columns, uint64_t high_columns,
                                                                     uint32_t column)
{
    const uint64_t word = column < LINEAR_THREAD_COLUMNS_A_WORD ? low_columns : high_columns;
    return (size_t)(word >> (column % LINEAR_THREAD_COLUMNS_A_WORD * LINEAR_THREAD_COLUMN_INDEX_BITS)) &
           LINEAR_THREAD_COLUMN_INDEX_MASK;
}

/* The exact profile's chains of one tile of one matrix on class 120 in the reference order (linear_ordered_chains):
 * for every output and column, from +0, chain = fma(p, 2^e, chain) over the blocks of 32 inputs in ascending order,
 * p a block's sum as class 80's two instructions form it. The matrix's group of threads (group_thread of
 * group_threads, whole warps: its slices) runs in rounds of group_threads / 16 blocks: thread i forms the sums of
 * output i % 16 at block i / 16 of the round for every column in use and leaves them, with the block's power, in
 * the group's scratch (LINEAR_THREAD_SCRATCH_FLOATS a thread, over no slot); after a barrier the chains of 16
 * outputs by column_count columns, taken in turn by the threads, add the round's terms in ascending blocks; another
 * barrier frees the scratch. Every thread of the block passes the same barriers (two matrices of one block have one
 * input width). The first slice's slots of the group's chains (warp 0: lane 4 (o % 8) + c / 2, element 2 (o / 8) +
 * c % 2) get the chains of the columns in use, the other slices' slots +0. Column c reads the vector of index c of
 * fed_vectors_low and fed_vectors_high (linear_thread_column_vector), fed_base + index * fed_stride (carried: its
 * operands; bfloat16 rows: its values of the tile's segment). E2M1 weights in the carried form (the routed experts)
 * take each block whose exponent (fed_exponents, exponents_stride bytes a vector) is at most
 * ACTIVATION_CARRIED_EXPONENT_LARGEST as whole numbers: the round cache holds each operand's A, a block's 4
 * groups are class_80_e2m1_group's and its sum is theirs times 2^(c - 163), the same bits; a block of infinite
 * operands (an exponent above) keeps class_80_thread_group's groups of packed words. */
template <int weight_format, int form>
__device__ __forceinline__ void linear_thread_chains(const unsigned char *matrix, uint32_t output_size,
                                                     uint32_t input_size, uint32_t tile, const unsigned char *fed_base,
                                                     size_t fed_stride, const unsigned char *fed_exponents,
                                                     uint32_t exponents_stride, uint64_t fed_vectors_low,
                                                     uint64_t fed_vectors_high, uint32_t column_count,
                                                     uint32_t group_thread,
                                                     uint32_t group_threads, float *scratch, uint32_t *fed_cache,
                                                     float *chains_slots)
{
    const bool is_whole = weight_format == LINEAR_WEIGHTS_E2M1 && form == LINEAR_FORM_CARRIED;
    const bool is_e2m1 = weight_format == LINEAR_WEIGHTS_E2M1;
    const bool is_carried = form == LINEAR_FORM_CARRIED;
    const int32_t pair_bias = is_carried ? CLASS_80_BFLOAT16_PAIR_BIAS : CLASS_80_HALF_PAIR_BIAS;
    const uint32_t column_shift = is_carried ? CLASS_80_BFLOAT16_COLUMN_SHIFT : CLASS_80_HALF_COLUMN_SHIFT;
    const int32_t weight_field =
        is_carried ? (is_e2m1 ? E2M1_CARRIED_FIELD : E4M3_CARRIED_FIELD) : E4M3_PLACEMENT_EXPONENT;
    const uint32_t load_count = input_size / VALUES_PER_LOAD;
    const uint32_t block_count = load_count * BLOCKS_PER_LOAD;
    const uint32_t round_blocks = group_threads / LINEAR_THREAD_ROWS;
    const uint32_t output = group_thread % LINEAR_THREAD_ROWS, round_block = group_thread / LINEAR_THREAD_ROWS;
    const uint32_t lane_group = output % OUTPUTS_PER_HALF_TILE, half = output / OUTPUTS_PER_HALF_TILE;
    float *const sums = scratch;                /* [round block][output][column], LINEAR_THREAD_SUMS_STRIDE apart */
    float *const powers = scratch + (size_t)group_threads * LINEAR_THREAD_SUMS_STRIDE;   /* [round block][output] */
    float *const own_sums = sums + (size_t)group_thread * LINEAR_THREAD_SUMS_STRIDE;
    const size_t weight_count = (size_t)output_size * input_size;
    const size_t tile_load = (size_t)tile * load_count;
    const uint4 *const tile_weights =
        (const uint4 *)matrix + (is_e2m1 ? tile_load : tile_load * BLOCKS_PER_LOAD) * THREADS_PER_WARP;
    const unsigned char *const scales =
        is_e2m1 ? matrix + weight_count / E2M1_WEIGHTS_PER_BYTE + tile_load * LANE_GROUPS_PER_WARP * E2M1_WORD_BYTES
                : matrix + weight_count + (size_t)(tile / TILES_PER_SCALE_BLOCK) * block_count;
    float chains[LINEAR_THREAD_CHAINS_MOST] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t round_first = 0; round_first < block_count; round_first += round_blocks) {
        /* the round's fed values decoded once, two of a pair at a time: block b of the round, column c, group g,
         * product k at ((b column_count + c) groups + g) 8 + k */
        for (uint32_t value = group_thread * COLUMNS_PER_LANE; value < round_blocks * column_count * VALUES_PER_BLOCK;
             value += group_threads * COLUMNS_PER_LANE) {
            const uint32_t k = value % CLASS_80_GROUP_PRODUCTS;
            const uint32_t group = value / CLASS_80_GROUP_PRODUCTS % LINEAR_THREAD_GROUPS_PER_BLOCK;
            const uint32_t column = value / VALUES_PER_BLOCK % column_count;
            const uint32_t block = round_first + value / VALUES_PER_BLOCK / column_count;
            const uint32_t load = block / BLOCKS_PER_LOAD, lane_place = k / COLUMNS_PER_LANE;
            const uint32_t step = block % BLOCKS_PER_LOAD * STEPS_PER_BLOCK + group / OUTPUT_HALVES;
            const uint32_t upper = group % OUTPUT_HALVES;
            const unsigned char *const vector =
                fed_base + linear_thread_column_vector(fed_vectors_low, fed_vectors_high, column) * fed_stride +
                (size_t)load * LINEAR_THREAD_FED_LOAD_BYTES;
            const uint32_t pair =
                block >= block_count ? 0u
                : is_carried         ? ((const uint32_t *)(vector + lane_place * LINEAR_THREAD_FED_PLACE_BYTES))
                                           [step * WORDS_PER_STEP + upper]
                                     : ((const uint32_t *)vector)[step * PAIRS_PER_STEP +
                                                                  upper * PAIRS_PER_HALF_STEP + lane_place];
            const uint32_t operands = is_carried ? pair : linear_place_bfloat16(pair);
	    const uint32_t exponent =
		    is_whole && block < block_count ?
			    fed_exponents[linear_thread_column_vector(
						  fed_vectors_low,
						  fed_vectors_high, column) *
						  exponents_stride +
					  block] :
			    ACTIVATION_EXPONENT_SMALLEST_VALUE;
	    if (is_whole) {                         /* an infinite operand (exponent above the bound): 1, signed */
		    const bool is_finite = exponent <=
					   ACTIVATION_CARRIED_EXPONENT_LARGEST;
		    const uint32_t low = operands & CLASS_80_HALF_BITS,
				   high = operands >> BITS_PER_HALF;
		    fed_cache[value] =
			    (uint32_t)(is_finite ?
					       linear_thread_whole_activation(
						       low, exponent) :
					       ((low &
						 CLASS_80_HALF_SIGN_BIT) != 0 ?
							-1 :
							1));
		    fed_cache[value + 1u] =
			    (uint32_t)(is_finite ?
					       linear_thread_whole_activation(
						       high, exponent) :
					       ((high &
						 CLASS_80_HALF_SIGN_BIT) != 0 ?
							-1 :
							1));
            } else {
                fed_cache[value] = linear_thread_packed<is_carried>(operands & CLASS_80_HALF_BITS, column_shift);
                fed_cache[value + 1u] = linear_thread_packed<is_carried>(operands >> BITS_PER_HALF, column_shift);
            }
        }
        __syncthreads();
        const uint32_t block = round_first + round_block;
        if (block < block_count) {
            const uint32_t load = block / BLOCKS_PER_LOAD, load_block = block % BLOCKS_PER_LOAD;
            uint32_t words[LANES_PER_LANE_GROUP][LINEAR_THREAD_BLOCK_WORDS];
#pragma unroll
            for (uint32_t lane_place = 0; lane_place < LANES_PER_LANE_GROUP; lane_place++) {
                const uint32_t lane_index = lane_group * LANES_PER_LANE_GROUP + lane_place;
                /* E2M1: the load's four steps, two a block; E4M3: 16 bytes a block, two words a step */
                const uint4 chunk = is_e2m1 ? tile_weights[(size_t)load * THREADS_PER_WARP + lane_index]
                                            : tile_weights[((size_t)load * BLOCKS_PER_LOAD + load_block) *
                                                               THREADS_PER_WARP + lane_index];
                const bool is_upper = is_e2m1 && load_block != 0;
                const uint32_t chunk_words[LINEAR_THREAD_BLOCK_WORDS] = {
                    is_upper ? chunk.z : chunk.x, is_upper ? chunk.w : chunk.y, chunk.z, chunk.w};
#pragma unroll
                for (uint32_t word = 0; word < LINEAR_THREAD_BLOCK_WORDS; word++) {
                    words[lane_place][word] = chunk_words[word];
                }
            }
            const uint32_t scale_code =
                is_e2m1 ? (((const uint32_t *)scales)[(size_t)load * LANE_GROUPS_PER_WARP + lane_group] >>
                           ((load_block * OUTPUT_HALVES + half) * BITS_PER_BYTE)) & BYTE_MASK
                        : scales[block];
            const int32_t exponent_field = (int32_t)scale_code + weight_field;
            powers[group_thread] =
                is_carried ? __uint_as_float((uint32_t)min(exponent_field, FLOAT32_LARGEST_EXPONENT_FIELD)
                                             << FLOAT32_EXPONENT_SHIFT)
                           : linear_power_of_two(exponent_field);
            for (uint32_t column = 0; column < column_count; column++) {
                own_sums[column] = 0.0f;
            }
            for (uint32_t column = 0; is_whole && column < column_count; column++) {
                const uint32_t exponent =
                    fed_exponents[linear_thread_column_vector(fed_vectors_low, fed_vectors_high, column) *
                                      exponents_stride +
                                  block];
		const bool is_finite = exponent <=
				       ACTIVATION_CARRIED_EXPONENT_LARGEST;
		int32_t running = 0;
                bool has_zero = false, has_positive = false, has_negative = false;   /* of infinite operands */
#pragma unroll
                for (uint32_t group = 0; group < LINEAR_THREAD_GROUPS_PER_BLOCK; group++) {
                    const int4 *const whole =
                        (const int4 *)(fed_cache + (((size_t)round_block * column_count + column) *
                                                        LINEAR_THREAD_GROUPS_PER_BLOCK + group) *
                                                       CLASS_80_GROUP_PRODUCTS);
                    const int4 halves[COLUMNS_PER_LANE] = {whole[0], whole[1]};
                    const int32_t activations[CLASS_80_GROUP_PRODUCTS] = {halves[0].x, halves[0].y, halves[0].z,
                                                                         halves[0].w, halves[1].x, halves[1].y,
                                                                         halves[1].z, halves[1].w};
                    int32_t products[CLASS_80_GROUP_PRODUCTS];
#pragma unroll
                    for (uint32_t k = 0; k < CLASS_80_GROUP_PRODUCTS; k++) {
                        products[k] = linear_thread_whole_weight(linear_thread_code<weight_format>(
                                          words, group, half, k / COLUMNS_PER_LANE, k % COLUMNS_PER_LANE)) *
                                      activations[k];
                        has_zero = has_zero || products[k] == 0;
                        has_positive = has_positive || products[k] > 0;
                        has_negative = has_negative || products[k] < 0;
                    }
                    running = class_80_e2m1_group(running, products);
                }
                /* infinite operands as class_80_special has them: a weight of 0 or products of both signs give a
                 * not-a-number, else the infinity of the products' sign (an operand's sign times its weight's) */
                own_sums[column] =
                    is_finite ? (float)running *
                                    __uint_as_float((uint32_t)((int32_t)exponent - LINEAR_THREAD_WHOLE_POWER_OFFSET +
                                                               CLASS_80_EXPONENT_BIAS)
                                                    << FLOAT32_EXPONENT_SHIFT)
                    : has_zero || (has_positive && has_negative) ? __uint_as_float(CLASS_80_NOT_A_NUMBER_BITS)
                    : has_negative                                ? -INFINITY
                                                                  : INFINITY;
            }
#pragma unroll
            for (uint32_t group = 0; !is_whole && group < LINEAR_THREAD_GROUPS_PER_BLOCK; group++) {
                uint32_t row_magnitudes[CLASS_80_GROUP_PRODUCTS];
                int32_t row_exponents[CLASS_80_GROUP_PRODUCTS], row_signs[CLASS_80_GROUP_PRODUCTS];
#pragma unroll
                for (uint32_t k = 0; k < CLASS_80_GROUP_PRODUCTS; k++) {
                    linear_thread_weight<weight_format, form>(
                        linear_thread_code<weight_format>(words, group, half, k / COLUMNS_PER_LANE,
                                                          k % COLUMNS_PER_LANE),
                        &row_magnitudes[k], &row_exponents[k], &row_signs[k]);
                }
                for (uint32_t column = 0; column < column_count; column++) {
                    const uint4 *const packed =
                        (const uint4 *)(fed_cache + (((size_t)round_block * column_count + column) *
                                                         LINEAR_THREAD_GROUPS_PER_BLOCK + group) *
                                                        CLASS_80_GROUP_PRODUCTS);
                    const uint4 halves[COLUMNS_PER_LANE] = {packed[0], packed[1]};
                    const uint32_t words[CLASS_80_GROUP_PRODUCTS] = {halves[0].x, halves[0].y, halves[0].z,
                                                                     halves[0].w, halves[1].x, halves[1].y,
                                                                     halves[1].z, halves[1].w};
                    uint32_t column_magnitudes[CLASS_80_GROUP_PRODUCTS];
                    int32_t column_exponents[CLASS_80_GROUP_PRODUCTS], product_signs[CLASS_80_GROUP_PRODUCTS];
#pragma unroll
                    for (uint32_t k = 0; k < CLASS_80_GROUP_PRODUCTS; k++) {
                        column_magnitudes[k] = words[k] & LINEAR_THREAD_PACKED_MAGNITUDE_MASK;
                        column_exponents[k] =
                            (int32_t)((words[k] >> LINEAR_THREAD_PACKED_EXPONENT_SHIFT) &
                                      LINEAR_THREAD_PACKED_EXPONENT_MASK) - LINEAR_THREAD_PACKED_EXPONENT_BIAS;
                        product_signs[k] = (row_signs[k] ^ ((int32_t)words[k] >> (CLASS_80_WORD_BITS - 1u))) | 1;
                    }
                    own_sums[column] = class_80_thread_group(own_sums[column], row_magnitudes, row_exponents,
                                                             column_magnitudes, column_exponents, product_signs,
                                                             pair_bias);
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (uint32_t chain = 0; chain < LINEAR_THREAD_CHAINS_MOST; chain++) {
            const uint32_t taken = group_thread + chain * group_threads;   /* output taken % 16, column taken / 16 */
            const uint32_t taken_output = taken % LINEAR_THREAD_ROWS, taken_column = taken / LINEAR_THREAD_ROWS;
            for (uint32_t term = 0; taken < LINEAR_THREAD_ROWS * column_count && term < round_blocks &&
                                    round_first + term < block_count;
                 term++) {
                const uint32_t place = term * LINEAR_THREAD_ROWS + taken_output;
                chains[chain] =
                    fmaf(sums[place * LINEAR_THREAD_SUMS_STRIDE + taken_column], powers[place], chains[chain]);
            }
        }
        __syncthreads();
    }
    /* the first slice's slots: the chains of the columns in use; the other slices' +0 */
#pragma unroll
    for (uint32_t chain = 0; chain < LINEAR_THREAD_CHAINS_MOST; chain++) {
        const uint32_t taken = group_thread + chain * group_threads;
        if (taken < LINEAR_THREAD_ROWS * column_count) {
            const uint32_t taken_output = taken % LINEAR_THREAD_ROWS, taken_column = taken / LINEAR_THREAD_ROWS;
            const uint32_t slot_lane =
                taken_output % OUTPUTS_PER_HALF_TILE * LANES_PER_LANE_GROUP + taken_column / COLUMNS_PER_LANE;
            chains_slots[slot_lane * ELEMENTS_PER_LANE + taken_output / OUTPUTS_PER_HALF_TILE * COLUMNS_PER_LANE +
                         taken_column % COLUMNS_PER_LANE] = chains[chain];
        }
    }
    if (group_thread >= THREADS_PER_WARP) {
#pragma unroll
        for (uint32_t element = 0; element < ELEMENTS_PER_LANE; element++) {
            chains_slots[(size_t)group_thread * ELEMENTS_PER_LANE + element] = 0.0f;
        }
    }
}
#endif

#endif
