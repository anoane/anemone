#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# make_pack_copies.sh: converts a model folder into one pack and puts a verified copy of it on each drive.
#
#   make_pack_copies.sh <source folder> <pack name> <destination folder> [<destination folder> ...]
#
# CONVERT_OPTIONS is handed to make_pack.py's convert: "--reference <checkpoint folder>" for a folder whose n-gram
# tables need the model's own code from elsewhere, "--without-tables" to leave them out.
# The pack is written into the first destination by make_pack.py and verified there; every other destination gets
# a copy made with direct reads and writes, and each copy is verified by recomputing every digest. The whole run
# sits in a memory-limited scope, so the page cache it fills is its own and is reclaimed from it: whatever else
# runs on the machine keeps the cache it depends on. Needs no card.
set -eu
[ $# -ge 3 ] || { echo "usage: $0 <source folder> <pack name> <destination folder> [<destination folder> ...]"; exit 2; }
CONVERT_DIRECTORY=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-$HOME/anemone/venv/bin/python}
SOURCE_FOLDER=$1
PACK_NAME=$2
shift 2
FIRST_DESTINATION=$1
MEMORY_HIGH=24G                                     # where the scope's page cache starts to be reclaimed
MEMORY_MAX=48G                                      # room for the workers' extents above that
COPY_BLOCK_BYTES=64M
JOBS=${JOBS:-$(nproc)}

if [ "${INSIDE_MEMORY_SCOPE:-no}" != yes ]; then
    exec sudo systemd-run --quiet --scope --uid="$(id -u)" --gid="$(id -g)" \
        -p MemoryHigh=$MEMORY_HIGH -p MemoryMax=$MEMORY_MAX \
        env INSIDE_MEMORY_SCOPE=yes HOME="$HOME" PYTHON="$PYTHON" JOBS="$JOBS" CONVERT_OPTIONS="${CONVERT_OPTIONS:-}" \
        "$0" "$SOURCE_FOLDER" "$PACK_NAME" "$@"
fi

for destination in "$@"; do mkdir -p "$destination"; done
FIRST_PACK=$FIRST_DESTINATION/$PACK_NAME
started=$(date +%s)
"$PYTHON" "$CONVERT_DIRECTORY/make_pack.py" convert "$SOURCE_FOLDER" "$FIRST_PACK" --jobs "$JOBS" ${CONVERT_OPTIONS:-}
echo "converted in $(( $(date +%s) - started )) s"
shift
for destination in "$@"; do
    started=$(date +%s)
    dd if="$FIRST_PACK" of="$destination/$PACK_NAME.partial" iflag=direct oflag=direct bs=$COPY_BLOCK_BYTES status=none
    mv "$destination/$PACK_NAME.partial" "$destination/$PACK_NAME"
    echo "copied to the next drive in $(( $(date +%s) - started )) s"
done
for destination in "$FIRST_DESTINATION" "$@"; do
    started=$(date +%s)
    "$PYTHON" "$CONVERT_DIRECTORY/make_pack.py" verify "$destination/$PACK_NAME" --jobs "$JOBS"
    echo "verified a copy in $(( $(date +%s) - started )) s"
done
