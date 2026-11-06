/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * exlr8_kernels.cuh: what exlr8 v2 adds to the trellis decode of trellis_kernels.cuh, which includes this file and
 * holds everything the two formats share (the stream, the states, the instructions, the product, the rotations).
 * exlr8's codebook is EXL3's mcg with the fp16 sum then ROUNDED TO E4M3 (to nearest, a tie to the even code; below
 * 2^-6 the codes are 2^-9 apart): two roundings, in that order, and the encoder searched over the rounded values.
 * Here, for two weights at once (a pair is the two halves of one register of operand A):
 *   exlr8_mcg_pair            the two fp16 sums: the mcg codebook itself, which EXL3 has too
 *   exlr8_round_pair          the rounding by integer arithmetic on the two fp16 patterns: what a CMP 170HX has
 *   exlr8_place_pair          two E4M3 bytes (of a 64 KiB table by state) as the fp16 of their values times 2^-8
 *   sm_120a only: exlr8_bytes_of_pair, exlr8_pair_of_bytes: the rounding and its way back as one instruction each;
 *   exlr8_byte_of_value: a row's value as an E4M3 byte, for rows fed as bytes (FP8 rows)
 * The byte order of the two packed conversions is taken as element by element (the lower fp16 is the lower byte):
 * the probe's check of all 65,536 states is what says so on a card. Device code only. Measurement tool: not part
 * of the engine.
 */
#ifndef EXLR8_KERNELS_CUH
#define EXLR8_KERNELS_CUH
#include <cuda_runtime.h>
#include <stdint.h>

enum {
    EXLR8_TABLE_BYTES = 65536,                   /* an E4M3 byte per state */
    EXLR8_BYTE_MASK = 0xFF, EXLR8_LOW_HALVES = 0x5410, EXLR8_HIGH_HALVES = 0x7632,
    /* an fp16 pair: bit 15 of a half is its sign; 0x2400 is 2^-6, E4M3's least normal; 0xA400 is -2^-6 */
    EXLR8_SIGN_BIT = 15, EXLR8_LEAST_NORMAL = 0x2400, EXLR8_MINUS_LEAST_NORMAL = 0xA400,
    EXLR8_HALF_ONES = 0x00010001, EXLR8_DROPPED_BITS = 7, EXLR8_ROUNDING_BIAS = 0x003F003F,
    /* an E4M3 byte in the low byte of each half: its sign to bit 15, its seven magnitude bits to bits 7 .. 13 */
    EXLR8_BYTE_SIGNS = 0x00800080, EXLR8_BYTE_MAGNITUDES = 0x007F007F, EXLR8_BYTE_SIGN_SHIFT = 8,
    EXLR8_BYTE_MAGNITUDE_SHIFT = 7
};
#define EXLR8_SM_120A 120                        /* -DPROBE_SM of the sm_120a build, for the preprocessor */
#define EXLR8_MCG_MULTIPLIER 0xCBAC1FEDu
#define EXLR8_MCG_KEPT 0x8FFF8FFFu               /* of each half: its sign and ten mantissa bits */
#define EXLR8_MCG_FLIPPED 0x3B603B60u            /* an exponent field of 12 .. 15: a magnitude of 1/8 .. 2 */
#define EXLR8_SIGNS 0x80008000u
#define EXLR8_MAGNITUDES 0x7FFF7FFFu
#define EXLR8_LEAST_NORMALS 0x24002400u
#define EXLR8_KEPT_BITS 0xFF80FF80u              /* the exponent and the three mantissa bits an E4M3 keeps */
#define EXLR8_PLACED_SCALE 256.0f                /* what a sum of placed bytes is multiplied by: 2^8 */

/* Two fp16 sums in one instruction, each rounded to nearest; never merged into a multiply-add. */
__device__ __forceinline__ uint32_t exlr8_fp16_pair_add(uint32_t first, uint32_t second) {
    uint32_t sums;
    asm volatile("add.rn.f16x2 %0, %1, %2;\n" : "=r"(sums) : "r"(first), "r"(second));
    return sums;
}
/* The mcg codebook for two states: the lower halves of the two mixed products beside each other, the upper halves
 * likewise, and one addition: the lower half of the result is the value of low_state. */
__device__ __forceinline__ uint32_t exlr8_mcg_pair(uint32_t low_state, uint32_t high_state) {
    const uint32_t low = (low_state * EXLR8_MCG_MULTIPLIER & EXLR8_MCG_KEPT) ^ EXLR8_MCG_FLIPPED;
    const uint32_t high = (high_state * EXLR8_MCG_MULTIPLIER & EXLR8_MCG_KEPT) ^ EXLR8_MCG_FLIPPED;
    return exlr8_fp16_pair_add(__byte_perm(low, high, EXLR8_LOW_HALVES), __byte_perm(low, high, EXLR8_HIGH_HALVES));
}
/* Two of the codebook's fp16 sums rounded to E4M3, each returned as the fp16 of its rounded value (its sign kept,
 * of a zero too). A magnitude is rounded at its fourth significant bit by an addition into its pattern, the carry
 * entering the exponent field as it must; a half under 2^-6 is first lifted by 2^-6 into the binade whose codes
 * are 2^-9 apart, and lowered again after. Exact for the sums a codebook has (0, or 2^-13 .. 4 in multiples of
 * 2^-13: the lift loses no bit); it is no conversion of any fp16. */
__device__ __forceinline__ uint32_t exlr8_round_pair(uint32_t pair) {
    const uint32_t magnitudes = pair & EXLR8_MAGNITUDES;
    const uint32_t small = (~((magnitudes | EXLR8_SIGNS) - EXLR8_LEAST_NORMALS) >> EXLR8_SIGN_BIT) & EXLR8_HALF_ONES;
    const uint32_t lifted = exlr8_fp16_pair_add(magnitudes, small * EXLR8_LEAST_NORMAL);
    const uint32_t even = (lifted >> EXLR8_DROPPED_BITS) & EXLR8_HALF_ONES;        /* a tie goes to the even code */
    const uint32_t rounded = (lifted + EXLR8_ROUNDING_BIAS + even) & EXLR8_KEPT_BITS;
    return exlr8_fp16_pair_add(rounded, small * EXLR8_MINUS_LEAST_NORMAL) | (pair & EXLR8_SIGNS);
}
/* Two E4M3 bytes, one in the low byte of each half, as two fp16: each the byte's value times 2^-8, the bytes below
 * the normal range included (they become fp16 below the normal range, which the instruction multiplies exactly). */
__device__ __forceinline__ uint32_t exlr8_place_pair(uint32_t bytes) {
    return ((bytes & EXLR8_BYTE_SIGNS) << EXLR8_BYTE_SIGN_SHIFT)
           | ((bytes & EXLR8_BYTE_MAGNITUDES) << EXLR8_BYTE_MAGNITUDE_SHIFT);
}
#if PROBE_SM == EXLR8_SM_120A
/* The format's rounding as the card has it: two fp16 to two E4M3 bytes (bits 0 .. 15), and two bytes to two fp16. */
__device__ __forceinline__ uint32_t exlr8_bytes_of_pair(uint32_t pair) {
    uint32_t bytes;
    asm volatile("{ .reg .b16 packed;\n"
                 "cvt.rn.satfinite.e4m3x2.f16x2 packed, %1;\n"
                 "cvt.u32.u16 %0, packed; }\n" : "=r"(bytes) : "r"(pair));
    return bytes;
}
__device__ __forceinline__ uint32_t exlr8_pair_of_bytes(uint32_t bytes) {
    uint32_t pair;
    asm volatile("{ .reg .b16 packed, unused;\n"
                 "mov.b32 {packed, unused}, %1;\n"
                 "cvt.rn.f16x2.e4m3x2 %0, packed; }\n" : "=r"(pair) : "r"(bytes));
    return pair;
}
/* One fp32 as the nearest E4M3 byte, a tie to the even code, magnitudes above 448 as 448 (unit 3a held this
 * conversion to the definition on all 2^32 patterns): both operands are the value, so either byte is its code. */
__device__ __forceinline__ uint32_t exlr8_byte_of_value(float value) {
    uint32_t bytes;
    asm volatile("{ .reg .b16 packed;\n"
                 "cvt.rn.satfinite.e4m3x2.f32 packed, %1, %1;\n"
                 "cvt.u32.u16 %0, packed; }\n" : "=r"(bytes) : "f"(value));
    return bytes & EXLR8_BYTE_MASK;
}
#endif
#endif /* EXLR8_KERNELS_CUH */
