#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# run_with_port.sh <script.py> [arguments]: runs one measurement script against the frozen copy of the exllamav3
# port that serves today. The port's kernel-tuning files are copied once to a private folder, so the serving copy
# is never written. VISIBLE_CARDS orders the cards by PCI bus index (0 = RTX PRO 6000, 1 = CMP 170HX, 2 = RTX 4090).
set -u
PORT_RELEASE_DIRECTORY=$HOME/rnd/release/all
PORT_COMMIT=803aad57b4c4
MEASURE_DIRECTORY=$(cd "$(dirname "$0")" && pwd)
PRIVATE_TUNING_DIRECTORY=$HOME/anemone/port_tuning_copy_$PORT_COMMIT
[ -d "$PRIVATE_TUNING_DIRECTORY" ] || cp -r "$PORT_RELEASE_DIRECTORY/autotune-$PORT_COMMIT" "$PRIVATE_TUNING_DIRECTORY" || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${VISIBLE_CARDS:-1,0,2}
export EXLLAMAV3_TUNE_CACHE=$PRIVATE_TUNING_DIRECTORY
export PYTHONPATH=$PORT_RELEASE_DIRECTORY/src-$PORT_COMMIT:$PORT_RELEASE_DIRECTORY/native-$PORT_COMMIT
# the settings the serving layout uses for experts held in RAM
export EXL3_MOE_CPU_PIN=${EXL3_MOE_CPU_PIN:-1} EXL3_MOE_HOST_CORES=${EXL3_MOE_HOST_CORES:-0} EXL3_MOE_PINNED_ARENA=${EXL3_MOE_PINNED_ARENA:-1}
cd "$MEASURE_DIRECTORY" && exec "$HOME/workspace/ds41-serve-venv/bin/python" "$@"
