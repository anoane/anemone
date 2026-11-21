#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Reuse the release's CPU converters. Inputs are immutable; destination must be new.
set -eu
[ "$#" -eq 3 ] || { echo 'usage: PYTHON=python3 JOBS=2 sh' \
    'tools/models/deepseek_v41_flash/model_data.sh OFFICIAL EXL3 NEW_DATA_DIRECTORY' >&2; exit 2; }
script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository=$(CDPATH= cd -- "$script_directory/../../.." && pwd)
official=$(CDPATH= cd -- "$1" && pwd)
exl3=$(CDPATH= cd -- "$2" && pwd)
destination=$3
[ ! -e "$destination" ] || { echo 'destination exists; use a new directory' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python=${PYTHON:-python3}
jobs=${JOBS:-2}
(cd "$official" && sha256sum -c "$repository/docs/models/deepseek_v41_flash/model-data/official.sha256")
(cd "$exl3" && sha256sum -c "$repository/docs/models/deepseek_v41_flash/model-data/exl3.sha256")
mkdir -- "$destination"
destination=$(CDPATH= cd -- "$destination" && pwd)
convert=$repository/tools/convert
model_convert=$script_directory/convert
"$python" "$convert/front_end_tables.py" write "$official/tokenizer.json" "$destination/tokenizer.bin"
"$python" "$convert/make_pack.py" convert "$official" "$destination/original.anemone" --reference "$official" --jobs "$jobs"
"$python" "$convert/make_pack.py" verify "$destination/original.anemone" --jobs "$jobs"
(cd "$destination" && sha256sum -c "$repository/docs/models/deepseek_v41_flash/model-data/outputs.sha256")
"$python" "$model_convert/make_hybrid.py" "$official" "$exl3" "$destination/hybrid"
# The pinned sources make the map the engine lists as established (model_v41.h): its descriptor says so.
"$python" "$model_convert/make_virtual_pack.py" --native "$destination/original.anemone" --hybrid "$destination/hybrid" --exl3-config "$exl3/quantization_config.json" --quality-established "LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw @ 0be63c33, TP4_EXL3's default map: qualified by the release gates' TP4_EXL3 identity checks (RELEASE_NOTES.md) and its golden metrics (docs/results/GOLDEN_METRICS.md)" --output "$destination/model.exl3map"
(cd "$destination" && sha256sum tokenizer.bin original.anemone model.exl3map > SHA256SUMS)
echo "Prepared $destination; retain the original pack, hybrid links and EXL3 source shards at their absolute paths."
