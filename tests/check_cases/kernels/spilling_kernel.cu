/*
 * spilling_kernel.cu: a kernel made to spill. It keeps more values live than the registers its launch bound leaves a
 * thread, so the assembler must put some of them in local memory, and the engine's kernel build (Makefile,
 * NVCC_FLAGS) must refuse it. `make self_test` shows that refusal.
 */
#define HELD_VALUES 96
#define THREADS_PER_BLOCK 1024
#define BLOCKS_PER_PROCESSOR 2

extern "C" __global__ void __launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_PROCESSOR)
    spilling_kernel(const float *input, float *output)
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
