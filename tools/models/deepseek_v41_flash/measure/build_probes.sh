#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# build_probes.sh: compiles the model's standalone probes, the shapes of DeepSeek-V4.1-Flash's checkpoint: the CUDA ones
# for the three card generations of the servers so far (sm_80 CMP 170HX, sm_89 RTX 4090, sm_120 RTX PRO 6000) into
# tools/models/deepseek_v41_flash/measure/build/ (the machine's probes are tools/measure/build_probes.sh's).
set -eu
MEASURE_DIRECTORY=$(cd "$(dirname "$0")" && pwd)
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
ARCHITECTURES="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120"
mkdir -p "$MEASURE_DIRECTORY/build"
CUDA_PROBES="four_bit_expert_probe eight_bit_trunk_probe shared_rows_expert_probe wide_expert_probe"
for probe in $CUDA_PROBES; do
    libraries=""
    if [ "$probe" = wide_expert_probe ]; then libraries="-lcublas"; fi      # its bar is the library's product
    "$NVCC" -O3 -std=c++17 -Werror all-warnings -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -Werror -Xcompiler -pthread $ARCHITECTURES -o "$MEASURE_DIRECTORY/build/$probe" "$MEASURE_DIRECTORY/$probe.cu" $libraries
    echo "built $probe"
done
# Host probes: plain C for this machine's processor.
HOST_CC=${HOST_CC:-gcc}
for probe in cpu_expert_probe; do
    "$HOST_CC" -O3 -march=native -fopenmp -std=gnu11 -Wall -Wextra -Werror -o "$MEASURE_DIRECTORY/build/$probe" "$MEASURE_DIRECTORY/$probe.c" -lm
    echo "built $probe"
done
