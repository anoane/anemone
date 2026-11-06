/* Negative audit case only; never launched or linked into the engine. */
#include <stdio.h>

extern "C" __global__ void device_logging_canary(unsigned int value)
{
    printf("release_device_log %u\n", value);
}
