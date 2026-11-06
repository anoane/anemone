/*
 * register_hungry_kernel.cu: a kernel that holds more values in registers than a block of 1,024 threads may have,
 * and spills none: it states no launch bound. Named as the instance of gate and up of 16 slices, whose block is
 * 1,024 threads, it must be named by the register check of the engine's kernel build (Makefile, registers_fit).
 * `make self_test` shows that.
 */
#define HELD_VALUES 96

extern "C" __global__ void expert_gate_up_16(const float *input, float *output)
{
    float held[HELD_VALUES];
#pragma unroll
    for (int index = 0; index < HELD_VALUES; index++) {
        held[index] = input[threadIdx.x * HELD_VALUES + index];
    }
#pragma unroll
    for (int round = 0; round < HELD_VALUES; round++) {
#pragma unroll
        for (int index = 0; index < HELD_VALUES; index++) {
            held[index] = held[index] * held[(index + round) % HELD_VALUES] + held[(index + 1) % HELD_VALUES];
        }
    }
#pragma unroll
    for (int index = 0; index < HELD_VALUES; index++) {
        output[threadIdx.x * HELD_VALUES + index] = held[index];
    }
}
