#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# build_probes.sh: compiles the standalone probes: the CUDA ones for the three card generations of the servers so far
# (sm_80 CMP 170HX, sm_89 RTX 4090, sm_120 RTX PRO 6000) into tools/measure/build/.
set -eu
MEASURE_DIRECTORY=$(cd "$(dirname "$0")" && pwd)
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
ARCHITECTURES="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120"
mkdir -p "$MEASURE_DIRECTORY/build"
CUDA_PROBES="card_read_rate_probe mapped_host_exchange_probe four_bit_expert_probe eight_bit_trunk_probe \
host_cost_probe launch_path_probe ipc_holder_probe shared_rows_expert_probe wide_expert_probe kquant_expert_probe"
for probe in $CUDA_PROBES; do
    libraries=""
    if [ "$probe" = wide_expert_probe ]; then libraries="-lcublas"; fi      # its bar is the library's product
    "$NVCC" -O3 -std=c++17 -Werror all-warnings -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -Werror -Xcompiler -pthread $ARCHITECTURES -o "$MEASURE_DIRECTORY/build/$probe" "$MEASURE_DIRECTORY/$probe.cu" $libraries
    echo "built $probe"
done
# Host probes: plain C for this machine's processor.
HOST_CC=${HOST_CC:-gcc}
for probe in cpu_expert_probe row_table_probe; do
    "$HOST_CC" -O3 -march=native -fopenmp -std=gnu11 -Wall -Wextra -Werror -o "$MEASURE_DIRECTORY/build/$probe" "$MEASURE_DIRECTORY/$probe.c" -lm
    echo "built $probe"
done
