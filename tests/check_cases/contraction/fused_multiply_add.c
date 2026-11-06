/*
 * fused_multiply_add: says whether the compiler fused a multiplication and an addition. `make self_test` builds
 * it with the flags of release, where nothing may be fused, and once more with -ffp-contract=fast, where the
 * fusing must be seen. The exact square of 1 + 2^-27 is 1 + 2^-26 + 2^-54: rounded to a double it is 1 + 2^-26,
 * so the sum below is 0 when the product is rounded first and 2^-54 when it is not.
 */
#include <stdio.h>
#include <stdlib.h>

#define ONE_AND_A_LITTLE (1.0 + 0x1p-27)
#define MINUS_ITS_ROUNDED_SQUARE (-(1.0 + 0x1p-26))

int main(void)
{
    volatile double factor = ONE_AND_A_LITTLE;
    volatile double addend = MINUS_ITS_ROUNDED_SQUARE;
    double left = factor;
    double right = factor;
    double sum = left * right + addend;
    if (sum != 0.0) {
        (void)printf("the multiplication and the addition were fused: the sum is %g, not 0\n", sum);
        return EXIT_FAILURE;
    }
    (void)printf("nothing was fused\n");
    return EXIT_SUCCESS;
}
