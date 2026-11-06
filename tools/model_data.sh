#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Reuse the release's CPU converters. Inputs are immutable; destination must be new.
set -eu
[ "$#" -eq 3 ] || { echo 'usage: PYTHON=python3 JOBS=2 sh tools/model_data.sh OFFICIAL EXL3 NEW_DATA_DIRECTORY' >&2; exit 2; }
script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository=$(dirname "$script_directory")
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
(cd "$official" && sha256sum -c "$repository/docs/user/model-data/official.sha256")
(cd "$exl3" && sha256sum -c "$repository/docs/user/model-data/exl3.sha256")
mkdir -- "$destination"
destination=$(CDPATH= cd -- "$destination" && pwd)
convert=$script_directory/convert
"$python" "$convert/front_end_tables.py" write "$official/tokenizer.json" "$destination/tokenizer.bin"
"$python" "$convert/make_pack.py" convert "$official" "$destination/original.anemone" --reference "$official" --jobs "$jobs"
"$python" "$convert/make_pack.py" verify "$destination/original.anemone" --jobs "$jobs"
(cd "$destination" && sha256sum -c "$repository/docs/user/model-data/outputs.sha256")
"$python" "$convert/make_hybrid.py" "$official" "$exl3" "$destination/hybrid"
"$python" "$convert/make_virtual_pack.py" --native "$destination/original.anemone" --hybrid "$destination/hybrid" --output "$destination/model.exl3map"
(cd "$destination" && sha256sum tokenizer.bin original.anemone model.exl3map > SHA256SUMS)
echo "Prepared $destination; retain the original pack, hybrid links and EXL3 source shards at their absolute paths."
