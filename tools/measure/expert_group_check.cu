/* SPDX-License-Identifier: AGPL-3.0-or-later */
/*
 * expert_group_check: on a card of class 120, the routed experts' blocks of the exact profile two ways, bit for bit:
 * class_80_thread_group over the placed operands, as linear_thread_chains forms a block of the carried form (each
 * E2M1 code decoded by linear_thread_weight, each activation's bfloat16 operand packed and unpacked as its round
 * cache holds it), and class_80_e2m1_group over the whole numbers W = 2 w, A = 512 x, its result times 2^(c - 163).
 * Blocks of 4 groups of 8 products, drawn in modes: every code at random; large values of one sign (running sums
 * past 2^24: the 24-bit cut); sparse (zeros of both signs, E2M1 0.5, E4M3 subnormals); two groups of large products
 * of one sign, then odd products (odd weights by odd tiny activations) among large even ones: running sums past
 * 2^25 and the cut of each odd product to an even number; the block exponent c from
 * ACTIVATION_EXPONENT_SMALLEST_VALUE to ACTIVATION_CARRIED_EXPONENT_LARGEST. Prints, per mode, the blocks
 * checked, those that differ (the first one's codes), and how many took each cut.
 *
 *   expert_group_check [blocks per mode, millions]
 * Measurement tool: not part of the engine.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../source/kernel_linear_common.cuh"

#define CHECK_THREADS 256
#define CHECK_BLOCKS 1504                       /* 8 a multiprocessor of the PRO */
#define CHECK_GROUPS 4u
#define CHECK_PRODUCTS (CHECK_GROUPS * CLASS_80_GROUP_PRODUCTS)
#define CHECK_MODES 4
#define E4M3_LARGEST_NUMBER 0x7Eu               /* 448: the quantizer's largest code */
#define E4M3_MAGNITUDE_MASK 0x7Fu
#define E4M3_SIGN 0x80u
#define E4M3_MANTISSA_BITS 3u
#define E2M1_MAGNITUDE_MASK 0x7u
#define E2M1_SIGN 0x8u
#define BFLOAT16_MANTISSA_BITS 7u
#define BFLOAT16_SIGN 0x8000u
#define OPERAND_FIELD_OFFSET 100                /* bfloat16 field of x 2^(c - 27): its exponent + c + 100 */
#define LATTICE_FIELD_OFFSET 98                 /* A = (128 + m7) shifted by field - c - 98 */
#define PLACED_POWER_OFFSET 163                 /* a block sum in placed units: the whole numbers' sum times 2^(c - 163) */
#define MIX_MULTIPLIER 0x2545F4914F6CDD1Dull

struct check_result {
    unsigned long long checked, differing, evened, kept;
    uint32_t first_weights[CHECK_PRODUCTS], first_activations[CHECK_PRODUCTS], first_exponent;
    uint32_t first_reference, first_integer;
};

__device__ static uint64_t check_next(uint64_t *state)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return *state * MIX_MULTIPLIER;
}

/* The bfloat16 operand of E4M3 code `code` in a block of exponent c: x times 2^(c - 27), its sign kept. */
__device__ static uint32_t check_operand(uint32_t code, uint32_t exponent)
{
    const uint32_t magnitude = code & E4M3_MAGNITUDE_MASK, sign = (code & E4M3_SIGN) != 0 ? BFLOAT16_SIGN : 0u;
    if (magnitude == 0) {
        return sign;
    }
    uint32_t field = magnitude >> E4M3_MANTISSA_BITS, mantissa = magnitude & ((1u << E4M3_MANTISSA_BITS) - 1u);
    int32_t power = (int32_t)field - 7;          /* x = (1 + m / 8) 2^power, or a subnormal's m / 8 2^-6 */
    if (field == 0) {                           /* normalize the subnormal */
        power = -6;
        while ((mantissa & (1u << E4M3_MANTISSA_BITS)) == 0) {
            mantissa <<= 1;
            power--;
        }
        mantissa &= (1u << E4M3_MANTISSA_BITS) - 1u;
    }
    return sign | ((uint32_t)(power + (int32_t)exponent + OPERAND_FIELD_OFFSET) << BFLOAT16_MANTISSA_BITS) |
           (mantissa << (BFLOAT16_MANTISSA_BITS - E4M3_MANTISSA_BITS));
}

/* A of an operand (whole number: 512 x, signed) from its bfloat16 bits and the block's exponent. */
__device__ static int32_t check_lattice(uint32_t operand, uint32_t exponent)
{
    const uint32_t field = (operand >> BFLOAT16_MANTISSA_BITS) & CLASS_80_FIELD_MASK;
    const uint32_t significand = (1u << BFLOAT16_MANTISSA_BITS) | (operand & ((1u << BFLOAT16_MANTISSA_BITS) - 1u));
    const int32_t shift = (int32_t)field - (int32_t)exponent - LATTICE_FIELD_OFFSET;
    const int32_t magnitude = field == 0 ? 0
                              : shift >= 0 ? (int32_t)(significand << shift)
                                           : (int32_t)(significand >> -shift);
    return (operand & BFLOAT16_SIGN) != 0 ? -magnitude : magnitude;
}

/* W of an E2M1 code: 2 w, signed. */
__device__ static int32_t check_weight(uint32_t code)
{
    const uint32_t magnitude = code & E2M1_MAGNITUDE_MASK, field = magnitude >> 1, mantissa = magnitude & 1u;
    const int32_t value = field == 0 ? (int32_t)mantissa : (int32_t)((2u + mantissa) << (field - 1u));
    return (code & E2M1_SIGN) != 0 ? -value : value;
}

extern "C" __global__ void check_kernel(uint32_t mode, uint64_t seed, uint32_t rounds, struct check_result *result)
{
#if LINEAR_THREAD_SUMS                          /* the device pass for class 120; the host pass sees only the stub */
    uint64_t state = (seed + (uint64_t)(blockIdx.x * CHECK_THREADS + threadIdx.x) * 2654435761ull) | 1ull;
    unsigned long long checked = 0, differing = 0, evened = 0, kept = 0;
    for (uint32_t round = 0; round < rounds; round++) {
        uint32_t weights[CHECK_PRODUCTS], activations[CHECK_PRODUCTS];
	const uint32_t exponent =
		ACTIVATION_EXPONENT_SMALLEST_VALUE +
		(uint32_t)(check_next(&state) %
			   (ACTIVATION_CARRIED_EXPONENT_LARGEST -
			    ACTIVATION_EXPONENT_SMALLEST_VALUE + 1u));
	const uint32_t signs = (uint32_t)check_next(&state);
        for (uint32_t k = 0; k < CHECK_PRODUCTS; k++) {
            const uint64_t draw = check_next(&state);
            if (mode == 0) {                    /* every code */
                weights[k] = (uint32_t)draw & 0xFu;
                activations[k] = (uint32_t)(draw >> 8) % (E4M3_LARGEST_NUMBER + 1u) | ((uint32_t)(draw >> 16) & 1u) << 7;
            } else if (mode == 1) {             /* large and mostly of one sign: the running sum past the cuts */
                const uint32_t flip = (draw >> 40) % 16u == 0 ? 1u : 0u;
                weights[k] = (5u + (uint32_t)(draw % 3u)) | ((signs & 1u) ^ flip) << 3;
                activations[k] = (E4M3_LARGEST_NUMBER - (uint32_t)((draw >> 8) % 24u)) | ((signs >> 1) & 1u) << 7;
                if ((draw >> 20) % 4u == 0) {   /* odd products among them */
                    weights[k] = (3u | ((signs & 1u) << 3));
                    activations[k] = (0x78u | ((uint32_t)(draw >> 24) & 7u) | 1u) | ((signs >> 1) & 1u) << 7;
                }
            } else if (mode == 3) {             /* past 2^25 after two groups, then odd products: the even cut */
                if (k < 2u * CLASS_80_GROUP_PRODUCTS) {
                    weights[k] = (7u - (uint32_t)(draw % 2u)) | (signs & 1u) << 3;
                    activations[k] = (E4M3_LARGEST_NUMBER - (uint32_t)((draw >> 8) % 4u)) | ((signs >> 1) & 1u) << 7;
                } else {
                    weights[k] = (((draw >> 8) & 1u) != 0 ? 1u : 3u) | ((uint32_t)(draw >> 9) & 1u) << 3;
                    activations[k] = (1u + 2u * (uint32_t)((draw >> 12) % 8u)) | ((uint32_t)(draw >> 16) & 1u) << 7;
                    if ((draw >> 20) % 3u == 0) {   /* and some large even ones */
                        weights[k] = 6u | (signs & 1u) << 3;
                        activations[k] = (0x70u | (uint32_t)(draw >> 24) % 15u) | ((signs >> 1) & 1u) << 7;
                    }
                }
            } else {                            /* sparse: zeros of both signs, 0.5, subnormals */
                const uint32_t kind = (uint32_t)(draw % 4u);
                weights[k] = kind == 0 ? ((uint32_t)(draw >> 8) & 1u) << 3 : (kind == 1 ? 1u : (uint32_t)(draw >> 8) & 0xFu);
                activations[k] = kind == 2 ? ((uint32_t)(draw >> 12) & 7u) | ((uint32_t)(draw >> 16) & 1u) << 7
                                           : (uint32_t)(draw >> 12) % (E4M3_LARGEST_NUMBER + 1u);
            }
        }
        /* the reference: class_80_thread_group over the placed operands, as linear_thread_chains forms them */
        float running = 0.0f;
        int32_t integer = 0;
        for (uint32_t group = 0; group < CHECK_GROUPS; group++) {
            uint32_t row_magnitudes[CLASS_80_GROUP_PRODUCTS], column_magnitudes[CLASS_80_GROUP_PRODUCTS];
            int32_t row_exponents[CLASS_80_GROUP_PRODUCTS], row_signs[CLASS_80_GROUP_PRODUCTS];
            int32_t column_exponents[CLASS_80_GROUP_PRODUCTS], product_signs[CLASS_80_GROUP_PRODUCTS];
            int32_t whole_weights[CLASS_80_GROUP_PRODUCTS], whole_activations[CLASS_80_GROUP_PRODUCTS];
            for (uint32_t k = 0; k < CLASS_80_GROUP_PRODUCTS; k++) {
                const uint32_t index = group * CLASS_80_GROUP_PRODUCTS + k;
                linear_thread_weight<LINEAR_WEIGHTS_E2M1, LINEAR_FORM_CARRIED>(weights[index], &row_magnitudes[k],
                                                                              &row_exponents[k], &row_signs[k]);
                const uint32_t operand = check_operand(activations[index], exponent);
                const uint32_t word = linear_thread_packed<true>(operand, CLASS_80_BFLOAT16_COLUMN_SHIFT);
                column_magnitudes[k] = word & LINEAR_THREAD_PACKED_MAGNITUDE_MASK;
                column_exponents[k] =
                    (int32_t)((word >> LINEAR_THREAD_PACKED_EXPONENT_SHIFT) & LINEAR_THREAD_PACKED_EXPONENT_MASK) -
                    LINEAR_THREAD_PACKED_EXPONENT_BIAS;
                product_signs[k] = (row_signs[k] ^ ((int32_t)word >> (CLASS_80_WORD_BITS - 1u))) | 1;
                whole_weights[k] = check_weight(weights[index]);
                whole_activations[k] = check_lattice(operand, exponent);
            }
            running = class_80_thread_group(running, row_magnitudes, row_exponents, column_magnitudes,
                                            column_exponents, product_signs, CLASS_80_BFLOAT16_PAIR_BIAS);
            evened += abs(integer) >= CLASS_80_E2M1_EVEN_FROM ? 1u : 0u;
            const int32_t before = integer;
            int32_t products[CLASS_80_GROUP_PRODUCTS];
            for (uint32_t k = 0; k < CLASS_80_GROUP_PRODUCTS; k++) {
                products[k] = whole_weights[k] * whole_activations[k];
            }
            integer = class_80_e2m1_group(integer, products);
            (void)before;
            kept += abs(integer) >= CLASS_80_E2M1_KEPT_FROM ? 1u : 0u;
        }
        const float scaled = (float)integer * __uint_as_float((uint32_t)((int32_t)exponent - PLACED_POWER_OFFSET +
                                                                         CLASS_80_EXPONENT_BIAS)
                                                               << CLASS_80_FRACTION_BITS);
        checked++;
        if (__float_as_uint(scaled) != __float_as_uint(running)) {
            if (atomicAdd(&result->differing, 1ull) == 0) {
                for (uint32_t k = 0; k < CHECK_PRODUCTS; k++) {
                    result->first_weights[k] = weights[k];
                    result->first_activations[k] = activations[k];
                }
                result->first_exponent = exponent;
                result->first_reference = __float_as_uint(running);
                result->first_integer = __float_as_uint(scaled);
            }
            differing++;
        }
    }
    atomicAdd(&result->checked, checked);
    atomicAdd(&result->evened, evened);
    atomicAdd(&result->kept, kept);
    (void)differing;
#else
    (void)mode, (void)seed, (void)rounds, (void)result;
#endif
}

int main(int argc, char **argv)
{
    const double millions = argc > 1 ? atof(argv[1]) : 10.0;
    const uint32_t rounds = (uint32_t)(millions * 1e6 / ((double)CHECK_BLOCKS * CHECK_THREADS)) + 1u;
    struct check_result *result = NULL, host;
    if (cudaMalloc((void **)&result, sizeof(*result)) != cudaSuccess) {
        fprintf(stderr, "no card memory\n");
        return 1;
    }
    int failed = 0;
    for (uint32_t mode = 0; mode < CHECK_MODES; mode++) {
        cudaMemset(result, 0, sizeof(*result));
        check_kernel<<<CHECK_BLOCKS, CHECK_THREADS>>>(mode, 0x9E3779B97F4A7C15ull * (mode + 1u), rounds, result);
        if (cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(&host, result, sizeof(host), cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "the kernel failed: %s\n", cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        printf("mode %u: %llu blocks, %llu differ; groups from a running sum past 2^25: %llu, totals past 2^24: %llu\n",
               mode, host.checked, host.differing, host.evened, host.kept);
        if (host.differing != 0) {
            failed = 1;
            printf("  first: c %u, reference 0x%08X, integer 0x%08X\n  weights", host.first_exponent,
                   host.first_reference, host.first_integer);
            for (uint32_t k = 0; k < CHECK_PRODUCTS; k++) {
                printf(" %X", host.first_weights[k]);
            }
            printf("\n  activations");
            for (uint32_t k = 0; k < CHECK_PRODUCTS; k++) {
                printf(" %02X", host.first_activations[k]);
            }
            printf("\n");
        }
    }
    printf(failed ? "expert_group_check FAILED\n" : "expert_group_check passed\n");
    return failed;
}
