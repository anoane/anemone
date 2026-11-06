/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* random_cases.c: prints N random cases for mma_cases: form, C, then 16 pairs of exactly representable 16-bit values.
 *   random_cases <count> <form 0 bf16 | 1 f16> <spread> [A exponent offset] [B exponent offset]
 * A's exponents spread over `spread` binades around its offset, B's around its own: tiny A values (subnormals of the
 * format among them) times large B values give the products the carried form makes. One value in 16 is zero. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t state = 88172645463325252ull;
static uint64_t next_random(void)
{
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

/* The value rounded toward zero onto the format's grid: bf16 keeps the top 16 bits of a float32 (subnormals too);
 * f16 keeps 11 significant bits, and below 2^-14 multiples of 2^-24. */
static float representable(float value, uint32_t form)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    if (form == 0u) {
        bits &= 0xFFFF0000u;
        memcpy(&value, &bits, sizeof(value));
        return value;
    }
    const float magnitude = fabsf(value);
    const float unit = magnitude < ldexpf(1.0f, -14) ? ldexpf(1.0f, -24)
                                                     : ldexpf(1.0f, (int)floorf(log2f(magnitude)) - 10);
    return copysignf(truncf(magnitude / unit) * unit, value);
}

static float random_value(uint32_t form, int spread, int offset)
{
    const uint64_t word = next_random();
    if ((word & 0xF0u) == 0u) {
        return 0.0f;
    }
    const int exponent = (int)(word % (uint64_t)spread) - spread / 2 + offset;
    const float significand = 1.0f + (float)((uint32_t)(word >> 20) & 0x3FFu) / 1024.0f;
    return representable(((word >> 63) != 0 ? -1.0f : 1.0f) * ldexpf(significand, exponent), form);
}

int main(int argument_count, char **arguments)
{
    const int count = argument_count > 1 ? atoi(arguments[1]) : 100;
    const uint32_t form = argument_count > 2 ? (uint32_t)atoi(arguments[2]) : 0u;
    const int spread = argument_count > 3 ? atoi(arguments[3]) : 8;
    const int a_offset = argument_count > 4 ? atoi(arguments[4]) : 0;
    const int b_offset = argument_count > 5 ? atoi(arguments[5]) : 0;
    for (int index = 0; index < count; index++) {
        const float c = (next_random() & 1u) != 0 ? 0.0f
                      : ldexpf(random_value(0u, spread, 0), a_offset + b_offset + 2);   /* a float32 of the sum's size */
        printf("%u %a", form, (double)c);
        for (uint32_t term = 0; term < 16u; term++) {
            printf(" %a %a", (double)random_value(form, spread, a_offset), (double)random_value(form, spread, b_offset));
        }
        printf("\n");
    }
    return 0;
}
