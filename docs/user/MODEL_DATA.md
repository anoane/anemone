<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Model data for Anemone v0.0.2-alpha

The release can prepare its model data on a CPU machine from two public,
revision-pinned Hugging Face snapshots. The converters are separate from the
inference binary. No GPU, CUDA toolkit, inference server or Hugging Face token
is needed for this preparation. The EXL3 step uses already quantized public
weights; it does not run a quantizer on the original weights.

After preparing these files, use [SERVER_PROFILES.md](../SERVER_PROFILES.md)
for the three complete serving commands and all startup/sampling parameters.

The reference workstation's tokenizer, hybrid index and complete EXL3 map were
reproduced byte for byte on 6 November. The original pack was checked by a
deterministic slice, described below. Full source-file hashes were independently
computed and matched against the public pinned Hugging Face tree metadata.

## Source identities and complete file inventories

| Source | Repository | Revision |
| --- | --- | --- |
| Official FP8/FP4 weights, BF16 embedding/head, vision, DSpark and Engram | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Quantized routed experts | [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/tree/0be63c3337efbb2a0061d5b8bb4006fa186d433f) | `0be63c3337efbb2a0061d5b8bb4006fa186d433f` |

The appendix lists every selected source filename, exact byte count and SHA256.
Machine-readable checks are [official.sha256](model-data/official.sha256) and
[exl3.sha256](model-data/exl3.sha256). There are 62 official files totaling
510,310,736,301 bytes and 50 EXL3 files totaling 210,818,726,352 bytes.
The weight shards are `model-00001-of-00048.safetensors` through
`model-00048-of-00048.safetensors`, and `model-00001-of-00043.safetensors`
through `model-00043-of-00043.safetensors`, respectively.

The official tokenizer and `inference/engram.py`, `inference/config.json` and
`tokenizer_config.json` are conversion inputs. Additional inference and encoding
Python sources are pinned for provenance. Anemone does not import the official
GPU inference implementation. The EXL3 metadata, template and non-routed shards
are included in the source inventory even when this map does not consume them.
Do not substitute a different revision or a similarly named 2.9-bpw checkpoint.

The pinned public revisions were checked again anonymously during the current
setup audit: all 62 official and 50 EXL3 selected filenames remain present, and
93 public LFS SHA256 records match the retained source manifests. Bounded HEAD
requests for the official tokenizer, EXL3 index and one weight shard per family
returned HTTP 200. No weight body was downloaded and no token or credential file
was used. The complete source-content hashing evidence below retains its own
recorded reproduction epoch.

## CPU environment and storage budget

The successful reference runs used Linux x86-64; Python 3.12.15 with NumPy
2.5.3, PyTorch 2.14.1, SymPy 1.14.0, tokenizers 0.23.3, transformers 5.19.0
and safetensors 0.8.0. The map producer also ran with Python 3.14.4 and NumPy
2.5.3. Python 3.10 or newer is required by the full original-pack path.
Install a CPU PyTorch wheel; Engram preparation uses CPU operations only.
Pin tokenizers because its Unicode regular-expression tables affect the output.

For example, in a new Python 3.12 virtual environment:

```sh
python3.12 -m venv /mnt/fastdisk/model-data-venv
/mnt/fastdisk/model-data-venv/bin/python -m pip install \
  torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
/mnt/fastdisk/model-data-venv/bin/python -m pip install \
  numpy==2.5.3 sympy==1.14.0 tokenizers==0.23.3 transformers==5.19.0 \
  safetensors==0.8.0 huggingface-hub==2.2.0
```

The measured commands used existing installed environments; this installation
recipe and wheel availability on other operating systems were not tested.
Use a filesystem supporting files larger than 100 GB. Budget at least 1.3 TB
free for the two selected source snapshots plus the 517,258,440,704-byte original
pack, and leave room for the Python environment and download temporary files.
The exact source-plus-output total is about 1.2384 TB (1.1263 TiB).
The map adds only 9,857,696 bytes; it references weight payloads in place.

Use `JOBS=2` and budget 16 GiB CPU RAM for conversion. This is a conservative
working budget, not a measured minimum or peak. A full conversion's peak depends
on worker count and the largest in-memory extent. The optional copies helper
uses a 24 GiB memory-high / 48 GiB memory-max systemd scope. Runtime Engram RAM
is a separate requirement: about 188.8 GiB before other serving allocations.

## Download and prepare

Run from the repository root. Choose a new directory on a suitably sized data
volume. These commands explicitly use anonymous public downloads and never
print authentication values:

```sh
export MODEL_DATA_ROOT=/mnt/fastdisk/anemone-model-data-v1
export PYTHON=/mnt/fastdisk/model-data-venv/bin/python
test ! -e "$MODEL_DATA_ROOT"
mkdir "$MODEL_DATA_ROOT"
"$PYTHON" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(os.environ['MODEL_DATA_ROOT'])
for family, repository, revision in [
    ('official', 'deepseek-ai/DeepSeek-V4.1-Flash-1017',
     '1bae0e1088993b9b82de08f59de5be5a34e42170'),
    ('exl3', 'LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw',
     '0be63c3337efbb2a0061d5b8bb4006fa186d433f'),
]:
    names = [line.split(None, 1)[1] for line in
             Path(f'docs/user/model-data/{family}.sha256').read_text().splitlines()]
    snapshot_download(repository, revision=revision, token=False,
                      local_dir=root / 'sources' / family, allow_patterns=names)
PY
JOBS=2 sh tools/model_data.sh \
  "$MODEL_DATA_ROOT/sources/official" "$MODEL_DATA_ROOT/sources/exl3" \
  "$MODEL_DATA_ROOT/data"
```

`tools/model_data.sh` checks every source SHA256, refuses an existing destination,
disables GPU visibility and network use during conversion, and runs the following
existing producer operations. It records final hashes in `data/SHA256SUMS`.
The full wrapper was not executed end to end on the reference workstation:
available NVMe space permitted the original-pack slice instead of a second full pack.

```sh
# All paths below are absolute, stable paths on your data volume.
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PYTHON" tools/convert/front_end_tables.py write "$OFFICIAL/tokenizer.json" "$DATA/tokenizer.bin"
"$PYTHON" tools/convert/make_pack.py convert "$OFFICIAL" "$DATA/original.anemone" --reference "$OFFICIAL" --jobs 2
"$PYTHON" tools/convert/make_pack.py verify "$DATA/original.anemone" --jobs 2
"$PYTHON" tools/convert/make_hybrid.py "$OFFICIAL" "$EXL3" "$DATA/hybrid"
"$PYTHON" tools/convert/make_virtual_pack.py --native "$DATA/original.anemone" --hybrid "$DATA/hybrid" --output "$DATA/model.exl3map"
```

For these individual commands, set `OFFICIAL`, `EXL3` and `DATA` to your selected
source and output directories. `DATA` must exist and its named outputs must not.
`make_pack.py` imports `kernel_layouts.py`, `front_end_tables.py` and
`engram_tables.py` from its own directory. It reads every safetensors file in
sorted filename order, derives tokenizer and Engram constants, lays out native
FP8/FP4 matrices, writes aligned extents and seals the pack index/header.
Do not use `--without-tables` for the release runtime.

`make_pack_copies.sh` is the existing optional alternative for converting and
copying the original pack across data drives. For example:

```sh
PYTHON="$PYTHON" JOBS=2 bash tools/convert/make_pack_copies.sh \
  "$OFFICIAL" original.anemone /mnt/drive-a/anemone /mnt/drive-b/anemone
```

That helper needs Linux systemd/sudo for its memory scope and verifies each
copy with `make_pack.py verify`. It is unnecessary for a single pack; it was
not run for this reproduction. `w4a16_pack_scan.py` is an optional analysis
tool requiring the full development checkout; it and its golden-analysis
dependencies are excluded from the release archive. It is not a producer
required by serve/generate.

## What the EXL3 map represents

The existing producer was found in the reference workstation's
`rnd/exl3_engine_v1/scripts/make_virtual_pack.py`; its index helper was
`rnd/curtis/make_hybrid.py`. The release copies accept explicit paths. The map
producer's tensor selection, layout and digest calculation are unchanged.
The unmodified reference map producer SHA256 is
`a45877f17b3539d322af74843c06c2ff46a98c789e13b863cb9e9c9d63308b41`;
the reference hybrid helper SHA256 is
`436c6e0b24ffc027cf60aee0ab182f73dfa5e06a82024966d7e202d62f06f1e6`.
The helper omits the old vLLM-only quantization-sidecar link; Anemone never
reads that sidecar. The six original `tools/convert/*` files retain the conversion
logic from engine main `a0ac3c64aa981efb3f8c020ec1a27cb1bee9f925`. The current
copies add SPDX declarations; their current file identities are listed below.
The pre-declaration source bytes remain preserved by the `snapshot-2026-11-06-v2` tag.

The hybrid index selects official non-routed tensors and LibertAI routed
experts for layers 0–39, preserving JSON insertion order and formatting.
Its 188,245 entries name 88 shard filenames; the index is 14,048,487 bytes. Its SHA256 is
`fa24746ffa8be36473aca6c99b432f3b0230f17237f0873f2db9be598d324e81`.
This generated index is not either upstream checkpoint index.

The map changes 46,080 routed matrices in 15,360 expert extents. Layers 18–22
use K2; layers 0–17 and 23–39 use K3. It validates the `0x83DCD12D` codebook,
tensor shapes and finite FP16 Hadamard vectors, then recomputes all expert
extent digests. It preserves the original pack's 434 non-routed extents,
including its BF16 head, original draft weights, vision and Engram.
The virtual pack is 424,522,379,264 bytes with 138,675 mapped segments.

The only external map sources are the original pack and LibertAI shards
`model-00002-of-00043.safetensors` through `model-00041-of-00043.safetensors`.
Source slot zero refers to the map itself; the other 41 source slots contain
absolute paths. The map contains its own pack header and index, not copies of
the complete weights. Its paths are limited to 511 encoded bytes each.

The loader in `source/storage_exl3_linux.c` selects this storage by the suffix
`.exl3map`. Use `model.exl3map`; an old example spelled `exl3.map`, which selects
the ordinary pack reader and fails. Regenerate the map whenever referenced
paths change. A relocated map legitimately has a different SHA256. Rebuilding
with unchanged source paths produces the reference hash below; changing only
the hybrid directory preserved all bytes outside the fixed-width path records.

## Runtime inventory and layout

| Runtime input | Producer / source | Required output identity |
| --- | --- | --- |
| `tokenizer.bin`, also called `front_end.tables` | `front_end_tables.py write` from the official `tokenizer.json` | 2,979,103 bytes; `a5ce9880ee5e314fe5c0892b3ff49fd7c2dbc36e14ce7236b63570eaf79df9c8` |
| `original.anemone` | `make_pack.py convert` from the 48 official shards and official tokenizer/Engram reference | 517,258,440,704 bytes; `b2494e0a1d3618a5a78a61cd0ca38fa7202a75f284c1be9f9b6fe8a262f519a1` |
| `model.exl3map` | `make_virtual_pack.py` from that original pack, the generated hybrid index and 40 EXL3 shards | 9,857,696 bytes; reference-path hash `bcad60a1c78a465ea71227c0afc1855d0421badb33de28636e4e74dcedcaa70b` |
| Engram rows and seven hash constants | `engram_tables.py`, called by `make_pack.py`, using official `inference/engram.py`, config and tokenizer | Embedded in the original pack, retained by the EXL3 map; slice hashes in [slices.json](model-data/slices.json) |
| Vision/aligner weights; `image_start`, `image_end`, `image_newline` | Official shards, packed by `make_pack.py` | Embedded in both weight families through the original pack |
| DSpark / draft / Markov-head weights, `mtp.*` | Official shards, packed by `make_pack.py` | Embedded in both families; no separately downloaded draft model |
| Chat template | `source/chat_template.c`, compiled into the binary | No runtime Jinja file; upstream encoding/template files are provenance/reference inputs |
| Placement layout | `--layout=` inline placement text, or compiled preset selection | No placement-layout file; e.g. INSTALL's PP5 layout is a command argument |
| Vision GEMM policy and RoPE correction tables | Full development checkout's numeric fixtures and `tools/build/embed_vision_*.py`; release builds use pinned `generated/` headers | Compiled into the executable/modules; hashes below; no extra runtime table file |
| Generate messages, `--prompt_token_ids` and image content | Caller-supplied messages/token JSON; image bytes carried by the request | Workload inputs, not checkpoint products |
| Optional `--draft_calibration=FILE` | User-supplied decimal calibration values | Not used by the release recipes; parser expects 2 × 5 × 8 probabilities per thousand |
| Optional replicas and session snapshots | `anemone replicate` and runtime session persistence | Generated for the installation, not source-model dependencies |

Authentication files are installation configuration, not model data. They were
not inspected or copied. Operating-system device/read-bound metadata and shared
libraries are environment dependencies, not model packs.

Arrange the binary and data as follows, or pass absolute paths:

```text
/mnt/fastdisk/anemone-model-data-v1/
  anemone
  sources/
    official/                 # pinned conversion source
    exl3/                     # pinned expert shards, retained at runtime
  data/
    tokenizer.bin
    original.anemone          # also retained for EXL3 non-expert extents
    hybrid/
      model.safetensors.index.json
      model-*-of-*.safetensors -> absolute source files
    model.exl3map
    model.exl3map.json         # producer receipt, not read by runtime
    SHA256SUMS
```

Run from `data/` with the serving/generation options in INSTALL.md or USAGE.md,
and pass `tokenizer.bin model.exl3map` for EXL3 or
`tokenizer.bin original.anemone` for original PP5 / TP4_PRO / exact profiles.
For generate, the messages filename precedes those two data arguments.
Keep the data and map source paths stable after generation.

## Executed reproduction and full verification

All reproduction writes went to
a separate reference reproduction directory.
Inputs were opened read-only. GPU visibility was disabled. The original engine
trees and the new repository were not edited.

| Operation | Result | Observed wall time |
| --- | --- | --- |
| Tokenizer producer | Identical to both benchmark `front_end.tables` files | 4.98 s |
| Hybrid index helper | Byte-identical index | Under 1 s, not separately timed |
| Complete map producer, unchanged source paths | Byte-identical; all expert digests recomputed | 111.16 s |
| Complete map producer, relocated hybrid paths | All bytes except absolute-path records identical | 100.20 s |
| Original-pack slice | Header, regenerated layout/index and 17 payload slices identical | 10.55 s |
| Two embedded vision generators | Generated headers byte-identical | Under 1 s combined, not separately timed |
| Full source and reference-file SHA256 inventory | 112 source files matched pinned public metadata; full reference pack hashed | 839.10 s (13 min 59 s, two hash workers) |

The slice regenerated all 48,504 tensor layout records and 15,794 extent records,
the 4,096-byte header and 5,394,432-byte index. Unselected extent content digests
were retained from the reference index: the omitted payloads were not converted
again. Thus index layout equality is independently checked, while omitted extent
digest values are not independent source-to-output verification.

The first and last tensor names in index order are `aligner.w1.bias` and
`vision.patch_embed.proj.weight`; both were reconstructed from official source
bytes and compared in full. Additional checks reconstructed first/last quantized
matrices, a draft tensor, the tokenizer, all seven Engram constants, and the
first and last blocks of both Engram row tensors, including padding. The final
physical payload tensor, `engram.token_map`, was also reconstructed in full. Exact
offsets, lengths and SHA256s are in [slices.json](model-data/slices.json).
The slice occupied about 24 MiB; all retained scratch was about 63 MiB before
cleanup. The volume had only 133 GB available at the start, below the 517 GB
needed for a second complete original pack. After collecting receipts, all generated
weight/table binaries, duplicate slices and hybrid symlinks were removed from
scratch; about 407 KB of scripts, copied tools and evidence remained.

For a full rerun, execute the documented `make_pack.py convert` command into a
new directory, then `make_pack.py verify --jobs 2` and check
[outputs.sha256](model-data/outputs.sha256) from that directory. This verifies
every extent against its index digest and then the entire output against the
reference SHA256. Full original conversion time was not measured in this run;
allow tens of minutes plus source/download/hash time, depending on CPU, worker
count and storage bandwidth. Download time depends on the connection.

The format's index stores tensor metadata and **64-bit per-extent digests**;
it does not store per-tensor SHA256 hashes. `make_pack.py verify` recomputes all
extent digests, including Engram. A per-tensor audit can read the indexed extent
offset plus tensor offset and hash each tensor's byte range; the slice receipt
does this for selected tensors. The full-file SHA256 is the cryptographic
identity check for a complete reproduced original pack.

The complete model-data wrapper, full original conversion and optional drive
copies remain unexecuted here. These limits do not affect the byte-identical
tokenizer/map results and are not claims of GPU inference or output qualification.

## Checks after adding license declarations

The current producer files passed 13 bounded CPU command checks. The tokenizer
was reproduced as 2,979,103 bytes with the same output SHA256 above; the hybrid
index was reproduced as 14,048,487 bytes and 188,245 entries with the same index
SHA256. Producer help, existing-output preservation, map-suffix refusal, wrapper
usage checks and the small original-pack self-test passed.

Those checks do not replace the recorded scope of the earlier full-map or
sampled original-pack reproduction. No full original pack or complete EXL3 map
was reconverted during this follow-up. Removing only the added SPDX comment
lines recovers the previous producer and generated-header bytes exactly; no
conversion arithmetic, model data or numeric fixture changed.

## Embedded vision table identities

Both the public checkout and source archive include `tools/development.mk`,
`tools/build/embed_vision_*.py` and the numeric fixtures under `tools/vision/fixtures/`.
The optional `make regenerate_vision` target uses Python to regenerate the pinned headers.
The normal release build uses the pinned headers in `generated/`, requires no Python
and does not regenerate these tables.

Run `make regenerate_vision` to update the headers, or `make check_generated` to
verify the pinned CLI and vision outputs. The recorded Linux checks reproduced
both vision headers exactly. The earlier CPU reproduction ran the generators
into scratch and compared their outputs with the checked-in headers (the commands
below require an existing `SCRATCH` directory):

```sh
python3 tools/build/embed_vision_gemm_policy.py tools/vision/fixtures/gemm_policy.json "$SCRATCH/vision_gemm_policy.h"
python3 tools/build/embed_vision_rope_corrections.py tools/vision/fixtures/rope_correction.hex "$SCRATCH/vision_rope_corrections.h"
cmp generated/vision_gemm_policy.h "$SCRATCH/vision_gemm_policy.h"
cmp generated/vision_rope_corrections.h "$SCRATCH/vision_rope_corrections.h"
```

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `tools/vision/fixtures/gemm_policy.json` | 4554 | `a88c90cd9b5e4dd593a01b9f8083ce05f486f48c20f1f6d0eb06c8df5099cc4b` |
| `generated/vision_gemm_policy.h` | 4261 | `8d17a43590d7430b6368e4aecbad2272ca06e6c6a70c5f49f485286ea6924953` |
| `tools/vision/fixtures/rope_correction.hex` | 82701 | `4b582bf26b1ec2c4583ffc33ccedfaf126e1eb1f4419e544e40eb9b5e73dbbed` |
| `generated/vision_rope_corrections.h` | 121016 | `b286aea36441a30c8c4ca5d4dbb9e0943da70e8e30b1901697716bf5ada127fe` |

## Producer file identities

These byte counts and SHA256s describe the current wrapper, adapters, local
converter imports and SPDX declarations. The optional `w4a16_pack_scan.py` row
records development-only provenance. The six original converter identities are
also recorded in [converter-sources.json](model-data/converter-sources.json).
The numeric fixtures and runtime model-input/output hashes are unchanged;
source-file hashes differ from the earlier v2 tag because of the added license comments.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `tools/convert/engram_tables.py` | 53757 | `83e27b47d42a8461eb2d30178cf73665a7c5cecd445f0de9ce4b4e65a7ab876d` |
| `tools/convert/front_end_tables.py` | 15163 | `1baff63328e93c6a8c864e4567584236d06c1b23a1a4140385331f5cdeb81f70` |
| `tools/convert/kernel_layouts.py` | 64498 | `6a91a9dd7025531d8fd496ce83e8acc3c3b83f9deab36c32cd1988a8102f23be` |
| `tools/convert/make_hybrid.py` | 2054 | `ae7d0f6b508a34f5c57bb43739631e86d33f6c531c572082d0d5881109401c28` |
| `tools/convert/make_pack.py` | 64286 | `3e59e81634bcb45f76c07dac42286cedf13f2059ecd54a80e7c770beb0564831` |
| `tools/convert/make_pack_copies.sh` | 2616 | `e0c7db7c788a0ee52a0a69d53b5efc7ce3c8202b7cbc633484bbdbfee879f60d` |
| `tools/convert/make_virtual_pack.py` | 10011 | `9e4772b6f4a8c468e76b8bab962295131dede12cd8f3017323fa4282639eda80` |
| `tools/convert/w4a16_pack_scan.py` | 63882 | `7e6d52cc115bf7525c6cab61f4f7120a12856d3d1eca100cd54ae34836fc7daf` |
| `tools/model_data.sh` | 1824 | `0b47667db64c247866fd66e53b1674b8dbf9efeaff05d0e53cccdc0efed862ec` |

## Complete source-file appendix

Each row uses the repository and revision in the source-identity table above.
Sizes are exact bytes. SHA256s were computed from the reference workstation's
file contents and checked against pinned public LFS hashes or Git blob identities.

### Official snapshot

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `model-00001-of-00048.safetensors` | 970533624 | `13bd121075f0b68cb7e8e8021a1011d42a81a83f557a2602fd9c07d52fddb0ec` |
| `model-00002-of-00048.safetensors` | 1323858272 | `43616eebd88dd4f1f1b642772356fc160478d1d9d04edd240431e7d065d2eda7` |
| `model-00003-of-00048.safetensors` | 7389759032 | `41611994fc9b27829b757d72257570d65a976fe65b8c6df8641adcd139d65ded` |
| `model-00004-of-00048.safetensors` | 7389759032 | `e8e901b3cb1ab46f32729c7eaab37bd774a09094f0a02c51a60e883f779f2c39` |
| `model-00005-of-00048.safetensors` | 7405953784 | `c5df71c02d6a22bcb20a0dbbcda9c6b65aa104f44415102a9a6a03bee92bb9de` |
| `model-00006-of-00048.safetensors` | 7389759032 | `a5d42afde2aee33619fb7a674b4c737c38d1f1d47a636a26f3edfb3b54164167` |
| `model-00007-of-00048.safetensors` | 7389759032 | `24856942add8f73f5e1fbaea1e9c9b6ff597fd1b4aa4651d789833dd3a5a2ab9` |
| `model-00008-of-00048.safetensors` | 7389759032 | `7bf5f7afb5d2bacad49feffb049989fb365d6dbf96986e76265bd7a7a3469564` |
| `model-00009-of-00048.safetensors` | 7389759032 | `e6ae47e9daca138fda8c04f833c8328eb21671c37d1c8e2c3f7a82aac26649f6` |
| `model-00010-of-00048.safetensors` | 7389759032 | `835e9a8b0a28cc571cb1cf9e6014b3d5eb84ac41990a8aa4519e1ac435d7e23c` |
| `model-00011-of-00048.safetensors` | 7405953784 | `5df968edefe59033e451ff8512bef57da2cf97b57b4e7a1380118489dd02915b` |
| `model-00012-of-00048.safetensors` | 7389759032 | `4f97bf292f58178ecdf593b55252a401d6e9102f18625a4fccc52b8ee57e7be0` |
| `model-00013-of-00048.safetensors` | 7389761368 | `1b8567854887396840eb5392ca763101308064a142cb36156159096e581659eb` |
| `model-00014-of-00048.safetensors` | 7389761368 | `e5ba6d7e4380728ad70b8208676075a898f80a64c3618060f1b81e4218a159d1` |
| `model-00015-of-00048.safetensors` | 7389761368 | `f4b29d936024e68f44fb14d0ffffb15b42b27be04b4a0296acdae695afa94c30` |
| `model-00016-of-00048.safetensors` | 7389761368 | `3aa67228351c80e708dab4ac7f6e56b19000649cce2cd655def3168a3de4115c` |
| `model-00017-of-00048.safetensors` | 7405956128 | `b704eec7b7aa1bad6c9d57fa7f61f2db28862e8e5fc2468fd9d872689c778c7d` |
| `model-00018-of-00048.safetensors` | 7389761368 | `1405cf03376739f46e72e6cdd4d404969f1b94c717bb01782a300e2fbbd17e29` |
| `model-00019-of-00048.safetensors` | 7389761368 | `a84beaded1913e8d6119ac111f25040c8ec42e59f200dbe04c3a991cac512669` |
| `model-00020-of-00048.safetensors` | 7389761368 | `038914f79cf5499da90f0bfb8e1b7761cc37d5f4dd132e2431a6d02c9fb9fdfe` |
| `model-00021-of-00048.safetensors` | 7389761368 | `2ba49f59c8cec144e41370f9f820f268899d0b5cd0501e881908ca65bccd2dc2` |
| `model-00022-of-00048.safetensors` | 7389761368 | `52f12cceca5e2fdd0191dfbd33b808b6c7ce844cf9351c119413c5ba3bcd28c8` |
| `model-00023-of-00048.safetensors` | 7400713088 | `d6e47e8b7147c64eb6619732694dda811e1a9f8af1894977919af9522724ca31` |
| `model-00024-of-00048.safetensors` | 7389761368 | `9d611d65374d37f5e02fdae14e680cb69bf55af8a8478802ac465b97c65be22b` |
| `model-00025-of-00048.safetensors` | 7389761368 | `8db532a7cbb23cd4b5aa775636ca13af95b758501290fb935074348f660a29cd` |
| `model-00026-of-00048.safetensors` | 7389761368 | `2d6d7261937d2a98c7398a55ed48dbf129d9b4f188aa9148e46ea9f720255115` |
| `model-00027-of-00048.safetensors` | 7395337384 | `8e76de6a5cdc7ac0923526d1c1a2e6459c55bbfa692793b29cdfe07b4f9fbc52` |
| `model-00028-of-00048.safetensors` | 7389761368 | `3c0e5ad6a4e8e811c1a55fd8f5b59c9fa329b241c671997c28fc7a01944d5a77` |
| `model-00029-of-00048.safetensors` | 7389761368 | `392a950bcb062b89f683abc52d49a17e92c66f80cd1cb6fb92b71091da75319b` |
| `model-00030-of-00048.safetensors` | 7389761368 | `0b44d1364e578fc26dc0cc86564e49a92871a51063fb7f2f3b51cb020de44ab0` |
| `model-00031-of-00048.safetensors` | 7395337384 | `d3230ffc32e11c5a34505c5acaeaff642aed5c770b297695e1f7a94071d361ac` |
| `model-00032-of-00048.safetensors` | 7389761368 | `96d284016ac885415859346d971c0de43bd9e7947337edfb53c34462866ef1d7` |
| `model-00033-of-00048.safetensors` | 7389761368 | `313953b3b6aaa6d4c60bbcfc177dda5d8d1380c9fe72d82560ba078f55278a8d` |
| `model-00034-of-00048.safetensors` | 7389761368 | `98eb27655774796bb3feadb35ec64a9849104b88c4c6ae7cabf6fc1ee317a056` |
| `model-00035-of-00048.safetensors` | 7395337384 | `4a321a28c048389039e4595641fe0b1f130428c61fc8f655545afe0083d37cd9` |
| `model-00036-of-00048.safetensors` | 7389761368 | `97c7910a8a14002dd941f45f6ddb2bd335eaf6fa05f4a76b2ff9285d6557b65d` |
| `model-00037-of-00048.safetensors` | 7389761368 | `d37ee013b805344619536e299c73adf92df03744aa0a242f2c5465590015e5be` |
| `model-00038-of-00048.safetensors` | 7389761368 | `7713bf7a78319eb1c6214b312e0a2da8cac88d77dce851d50f1113d78f8975a4` |
| `model-00039-of-00048.safetensors` | 7395337384 | `658a77d47d217995b9b78612e69ee0d0e33f6c2240f4ec629fecdca3ecb03f4c` |
| `model-00040-of-00048.safetensors` | 7389761368 | `6a9dbcc8d5894ba614b5e962764f45f6d0df7f14f1287b17f173d1f2cd762186` |
| `model-00041-of-00048.safetensors` | 7389761368 | `458308f08ed29b34850bd6b9d10bf769620729d3725c77f7bc0c1d14e01b47cc` |
| `model-00042-of-00048.safetensors` | 7389761368 | `3d283a4ed6eb2d5013bacc97c3500361926a3261957943434513304df52adfd7` |
| `model-00043-of-00048.safetensors` | 1323837624 | `4ae12b941c3facf8616ce62dd45bc6868076c2aeb82106f98123ea6c5bb64182` |
| `model-00044-of-00048.safetensors` | 2652728736 | `3e8ac315c2bcca3e83c5e6e25c5758a75c03732063a6ff0c1b0e24b9ff49f6e6` |
| `model-00045-of-00048.safetensors` | 2573998176 | `9c73b27c4343236790008979d5e0b565f687aa64440031b9e096fed1ab8039c5` |
| `model-00046-of-00048.safetensors` | 2706402896 | `1dccc19a3741972c661de734572dc7d4b899e6e99bfbdf3ae4e057c707e5b339` |
| `model-00047-of-00048.safetensors` | 101535150936 | `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5` |
| `model-00048-of-00048.safetensors` | 101537926640 | `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd` |
| `model.safetensors.index.json` | 7470294 | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |
| `config.json` | 3311 | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| `tokenizer.json` | 6367257 | `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |
| `tokenizer_config.json` | 801 | `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547` |
| `inference/convert.py` | 9458 | `035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76` |
| `inference/engram.py` | 8138 | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` |
| `inference/generate.py` | 8722 | `8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0` |
| `inference/image_processor.py` | 7699 | `482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272` |
| `inference/kernel.py` | 23790 | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` |
| `inference/model.py` | 61669 | `e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d` |
| `inference/vision.py` | 4457 | `5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c` |
| `inference/config.json` | 1982 | `2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809` |
| `encoding/encoding.py` | 37962 | `13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2` |
| `encoding/test_encoding.py` | 22449 | `288b79c9ea7d018ae03b0dd3e54a8220cc3a2df1ff97f59907e75ed8112765d6` |

### EXL3 snapshot

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `model-00001-of-00043.safetensors` | 1323827293 | `ba5ccb4d39882d08800e856150bce42b23584f3d9943520ae4cff5647d5a3045` |
| `model-00002-of-00043.safetensors` | 5322748742 | `11ed8a0cb267169a5f3ea0ceebb24ab09fb5b84b98e4d4c86dc2f06f1f5e0d40` |
| `model-00003-of-00043.safetensors` | 5223069939 | `1de02ee6b4d909eb5936e08e7c3825cbf2fa65693c5674deade2e70bcb773462` |
| `model-00004-of-00043.safetensors` | 5230054628 | `8e68a9e396d958d6645378b44ef3419a4c0b743da0e92b354d2163a4c8bf85ce` |
| `model-00005-of-00043.safetensors` | 5223069939 | `c14a66ba26f57e9577e1a656764b3e5828e09c7630ad7404b80c6e265d03a37d` |
| `model-00006-of-00043.safetensors` | 5223069939 | `2866602c20b535eec3bb1a737f62d5d6d2dfecbf362977d3a4f5f8e701580b6d` |
| `model-00007-of-00043.safetensors` | 5223069939 | `4d12814cd25cf07fe9c6791f0050a8afbc1038bae02fb201243f9c56de8171cd` |
| `model-00008-of-00043.safetensors` | 5223069939 | `cb88f50d0ddd81b98db00a6475c356511c3c21f843b2ccd9845fb4017c3f5d15` |
| `model-00009-of-00043.safetensors` | 5223069939 | `dbebd82073a27f79dd1353d3c309255f46e577eb1c2f9d8a10a52bc38f6d8bc2` |
| `model-00010-of-00043.safetensors` | 5230054628 | `8cde4e4124614502c2950e9f3a643027eceeeedbdf3c54cf786dbf125d3d536e` |
| `model-00011-of-00043.safetensors` | 5223069939 | `c96372d73271d2c59e5a959bb81551aaacb2abe8445f5b9ea9391e2b4a033601` |
| `model-00012-of-00043.safetensors` | 5223074621 | `337345b1ca174aebcff619bbc2c2a6e33881a42ad8f638956e69b5de4fe21d41` |
| `model-00013-of-00043.safetensors` | 5218650941 | `042b4274c48e55d09bfe51ac099915f8bc3bb0c7be9b0f9d5e75a9f5c2045497` |
| `model-00014-of-00043.safetensors` | 5218650941 | `bc27a27e60bebd88ad36f4896aa027911bc5f7fbb30ad9a25f95a59a6374d957` |
| `model-00015-of-00043.safetensors` | 5297522070 | `74d3f0d12e32a658d8f6c5f9a104826e473c53299267519b46d594619756ae66` |
| `model-00016-of-00043.safetensors` | 5225635649 | `0a1b0b3ecb817f448d9fb34ebf0757d4b00042220515ecb13f1b86908cb092f3` |
| `model-00017-of-00043.safetensors` | 5218650941 | `e4b4278a51ce9daf61290d4caba5b953f5a243832ffb5c8d29a0c259807facea` |
| `model-00018-of-00043.safetensors` | 5218650941 | `07e9d048f9cefb94b9a30f41e39b9390b6bf54a763c0b7e1dcd5d81144fc0849` |
| `model-00019-of-00043.safetensors` | 5218650941 | `2c6116ee6f584af8c4027cee8f5069d0d51a4e9b487dd73027127d29f4caf274` |
| `model-00020-of-00043.safetensors` | 3519956829 | `817d948752825a739073b95bc6171a35e4677c095ac617bd6772560ae0e4b5e4` |
| `model-00021-of-00043.safetensors` | 3519956829 | `4bdda16b5a00ac5ee0e39bc81271af6155caacb7c03928876b07344e20a4ba5f` |
| `model-00022-of-00043.safetensors` | 3525291432 | `00afb7bd831cd50added9b809cc2e9cec57ab44963f152f8bc42cd6d22f0a4e4` |
| `model-00023-of-00043.safetensors` | 3519956829 | `28ba9d401e2cd556d0145b31c03ed341f009e2849162e6d9f68438a067d74a59` |
| `model-00024-of-00043.safetensors` | 3519956829 | `1240d8744902914287da94cd8e2a5d9f17e7c39167bfab20c737a34d7605e50c` |
| `model-00025-of-00043.safetensors` | 5218650941 | `a053c3dc54c4ecfe295cfa4bd7fb44f1054ade1d596e46496f6791c87ffb60df` |
| `model-00026-of-00043.safetensors` | 5222266717 | `028c44abbea066c94da7ed0e64f07f8d36ae925a4ff437ef999f5a2b5ec6c9d4` |
| `model-00027-of-00043.safetensors` | 5218650941 | `69cbb8f2f00ddcb7a83b04d34ca9151c61f3319ec742d0ac5f06abf295a91eef` |
| `model-00028-of-00043.safetensors` | 5218650941 | `451c69700eb30a9e47ec80fe219aa537985c6afa3f7391db5699fba50d93e05b` |
| `model-00029-of-00043.safetensors` | 5218650941 | `906a9525cde8176d18066d27ae9d457ff55ee3134438dadea4ee63af8dce45ad` |
| `model-00030-of-00043.safetensors` | 5222266717 | `420ad929c1efa3f90ba4a39d28159d9a29175089cd9bd4fe32f25c6fb7e473f6` |
| `model-00031-of-00043.safetensors` | 5220125501 | `0312eb9c04dbe5f538efd94a21d1f76266a8cac9ea01250084aa6513657fc5df` |
| `model-00032-of-00043.safetensors` | 5223074621 | `4653d193e28120319fe5b3508ffed537d23f57b8d31b20f3f69371fd569e59b2` |
| `model-00033-of-00043.safetensors` | 5223074621 | `3a6bfc60065324e606082ae8ad68ff403285b24e7a720b3b2274e516a247252a` |
| `model-00034-of-00043.safetensors` | 5226690397 | `b1825ddd7016d3cefea0f13ec78aff33637c68e0eb721cdb12f83104c03b3760` |
| `model-00035-of-00043.safetensors` | 5223074621 | `9b513ca99a0e8403d172df623a4c0fe7ecaaa51e09015d9574efa90bd6414a32` |
| `model-00036-of-00043.safetensors` | 5223074621 | `206ca962082e2c4de968ba4cc82cc1837dccb0b8960f493e04891b281bbfe497` |
| `model-00037-of-00043.safetensors` | 5223074621 | `6ed69fdcd3bd32762ddefae671c7fce7f9b5a212cdbe1aaee64adaa5ef0eef2c` |
| `model-00038-of-00043.safetensors` | 5226690397 | `24a61b540656dc210b3bc3215b1713f0b4bee8b724176a4f5968e24b9540035c` |
| `model-00039-of-00043.safetensors` | 5223074621 | `29e5b40a97edcb91ed6a382b39bbb4d16ce40de2d1ddce2f75a8fd75e0ad609f` |
| `model-00040-of-00043.safetensors` | 5223074621 | `5c850a3950ce68a93a24336ccc417ac660fdbdb84bcc494aa6d57ea5c28246be` |
| `model-00041-of-00043.safetensors` | 5719789284 | `5a6f5791d3f5f9635e58db5a693f68c64c6c9e14357e8386a21e5fd232af032d` |
| `model-00042-of-00043.safetensors` | 4854649316 | `2f80f4a44e4608c6da139eccf2483122ac88a240046872965d546b4fa26ded06` |
| `model-00043-of-00043.safetensors` | 3500776125 | `c5b94d519d323921baacb34b6111e518ce2c36309d06c80fe4e42e3aa9662bc5` |
| `model.safetensors.index.json` | 15504210 | `778baeaf9fbcd351800536dc090ff1edddb72d102472b08de0edd9a52cb96656` |
| `config.json` | 3628 | `eb48f2216645b36f400036fa2d67d8ad6b48233b0f4a1bcb7e99fa6450dad253` |
| `tokenizer.json` | 6367257 | `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |
| `tokenizer_config.json` | 801 | `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547` |
| `quantization_config.json` | 54653012 | `696cdad521cdebea0642a3c22d01c9152dca4e5ccbf209b5e7f2b808f359bd27` |
| `exl3_k_map.json` | 1409 | `4359af687bdce7fccd4fc74a57c2f9154b42053447cdc6e0d1768e68fffbec0f` |
| `chat_template.jinja` | 34813 | `2ed8bc3372c9c68577f232229e78ebfe37a53ae7964694d845533e485ed738e8` |
