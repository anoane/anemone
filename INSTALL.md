<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Install and serve Anemone v0.0.2-alpha

Build first using [BUILD.md](BUILD.md). **Every shell command, service command
and configuration example on this page is not run in the snapshot test.**
That test compiles and runs host tests only; it does not load model weights,
start a model-serving endpoint, install a service or run the HTTP examples below.
Host tests do exercise synthetic HTTP fixtures inside an isolated network namespace.
The listed serving recipes were checked through the real axis resolver/parser;
that check does not start a model, allocate its memory or install a service.

The [final three-profile results](docs/results/COMPARISON_MATRIX.md) distinguish
balanced launch recipes from original exact. Exact remains one session with
capacity 512. Original balanced uses native PRO arithmetic, so its outputs
depend on assigned layers. The serve-only prompt-chunk identity result is
separate from the measured solo golden baseline and has no independent golden
quality score. The [snapshot record](docs/VALIDATION.md) identifies current packaging
and the versioned build/source identities; previous measurements retain their dates.
The current `spdx_04` executable was built from engine/help source
`1b0731a36b7482477b96d7a28b27c93e4dd49e3f`; [BUILD.md](BUILD.md) records its actual
expected SHA256. The earlier parser and benchmark receipts retain their own epochs.

## Model data

Prepare the public, revision-pinned model data on a CPU machine using
[MODEL_DATA.md](docs/user/MODEL_DATA.md). It lists the source files, SHA256s,
Python packages, disk budget, conversion commands and verified output hashes.
The release's converters preserve the original weights and build a read-only
EXL3 map from the published LibertAI EXL3 checkpoint.

After downloading and checking the two pinned snapshots as documented there:

```sh
PYTHON=/path/to/model-data-venv/bin/python JOBS=2 \
  sh tools/model_data.sh /absolute/sources/official /absolute/sources/exl3 /absolute/new-data
```

This runs the existing tokenizer and original-pack converters, verifies the
pack, creates the hybrid index, and builds `model.exl3map`. The destination must
be new. Full conversion needs about 1.24 TB for sources plus outputs; it needs
no GPU. The original pack is also required for EXL3's non-expert tensors,
Engram, vision and DSpark. Retain the map's referenced absolute paths, or
regenerate it after relocation. Use the `.exl3map` suffix: the storage loader
selects the EXL3 reader by that suffix.

Example layout:

```text
/opt/anemone/
  anemone
  sources/
    official/                 # pinned official snapshot, conversion input
    exl3/                     # pinned LibertAI snapshot, retained at runtime
  data/
    tokenizer.bin             # identical bytes to front_end.tables
    original.anemone          # original FP8/FP4 pack, required by both families
    hybrid/                   # generated index and source symlinks
    model.exl3map             # small map; embeds paths, not the weight payloads
```

Pass these paths explicitly in the recipes below, or run from `data/`.
Four expert parts are the reference EXL3 partition; eight is a distinct
arithmetic/order option. Original weights and EXL3 are different weight
families. Engram needs about 188.8 GiB of host RAM before other runtime
allocations. Replica files are optional validated copies.

## Card placement

Use `CUDA_DEVICE_ORDER=PCI_BUS_ID` and verify the engine's reported backend
ordinals on the target machine. The measured machine has CMPs at 0,1,3,4 and
the RTX PRO 6000 at 2. NVML indices or a visibility list need not match backend
ordering on a different machine.

| Placement | Weight family | Cards / order |
| --- | --- | --- |
| TP4_EXL3 | prepared EXL3 K3/K2 (2.9-bpw checkpoint) | four class-80 CMPs, automatically selected when order omitted |
| TP4_PRO | original FP8/FP4 | CMP TP4 plus native PRO; measured order 0,1,3,4,2 |
| PP5 | original FP8/FP4 | measured order 0,1,3,4,2; PRO last |

For the explicit PP5 layout below, logical cards 0–4 own layers 0–7, 8–14,
15–21, 22–28, 29–39. The PRO also owns the head and DSpark. Native PRO arithmetic
depends on layer placement; it is not an official equality guarantee. Emulation
is an explicit supported-recipe decision with a significant PRO cost, not a
general drop-in TP flag. TP5 remains refused in this main snapshot.

## Serving commands

Use the canonical [server profile guide](docs/SERVER_PROFILES.md) for the complete
runnable commands and explanations of every serve option. The commands are also
provided as exact argument arrays in
[SERVER_PROFILE_COMMANDS.json](docs/SERVER_PROFILE_COMMANDS.json), with health,
model-listing and tiny deterministic HTTP smoke requests. They passed the real
v0.0.2-alpha axis resolver/parser; that CPU check opens no models, GPUs or sockets.
GPU execution and qualification retain separate receipts in the matrix.

| Release recipe | Guide and limits |
| --- | --- |
| Original exact | [Bounded reference](docs/SERVER_PROFILES.md#original-exact): original PP THIN serial M1, one session, 512 total tokens. |
| Original balanced | [Native PRO PP5](docs/SERVER_PROFILES.md#original-balanced): explicit controls use numerics=exact, distinct from strict exactness=exact. |
| EXL3 balanced | [Four-CMP TP4 BATCH](docs/SERVER_PROFILES.md#exl3-balanced): explicit controls, separate quantized weight family; default mixed identity gate failed. |

The guide explains the separate opt-in EXL3 prompt-chunk guard, capacity versus
full-context qualification, raw graph default 0 versus recipe graph 1, and T0
fixtures versus T1 operating sampling. The balanced names here do not select
the held literal four-axis all-balanced tuple. All options precede files;
serve does not take generate's messages file.

## systemd example

This unit is **not run in the snapshot test**. It assumes you installed the
binary and original pack into the layout above and chose the bounded reference
profile deliberately. Adapt capacity and the admitted recipe for actual use.
Save as anemone.service through your normal deployment workflow.

```ini
[Unit]
Description=Anemone inference service
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/anemone/data
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
ExecStart=/opt/anemone/anemone serve --exactness=exact --temperature=1 --top_p=1 --bind_address=127.0.0.1 --port=8321 --model_alias=anemone tokenizer.bin original.anemone
Restart=on-failure
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

Deployment command example, **not run in the snapshot test**:

```sh
systemctl enable --now anemone.service
```

## HTTP API and health

HTTP/1.1 and HTTP/1.0 are accepted with one request per connection. The numeric
listener defaults to 127.0.0.1:8321; `--bind_address` and `--port` change it. TLS and remote
access policy belong to the operator's deployment. The OpenAI-compatible routes
are Chat Completions and stateless plaintext Responses; unsupported fields and
stored-response chains are refused rather than silently supported.

| Method / route | Purpose |
| --- | --- |
| GET /health | unauthenticated health; 503 on engine failure |
| GET /v1/models | model identity and alias |
| POST /v1/chat/completions | JSON or SSE chat, tools and reasoning |
| POST /v1/responses | stateless plaintext Responses adapter |
| GET /v1/anemone/stats | live counters, subject to configured authorization |

Examples below are **not run in the snapshot test**, for the local example with
model_alias=anemone and no key file:

```sh
curl -fsS http://127.0.0.1:8321/health
curl -fsS http://127.0.0.1:8321/v1/models
curl -fsS http://127.0.0.1:8321/v1/chat/completions -H 'Content-Type: application/json' --data-binary '{"model":"anemone","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}'
```

With an operator-managed key file, all routes except health require the configured
credential. Key material is absent from this archive; follow the manual's named
user/key-file format. A client environment variable does not configure server
authentication. Keep credentials out of commands and logs.

## Environment names after the identity-proven rename

| Previous name | Current name |
| --- | --- |
| ANEMONE_DIAG_DRAFT_PROBE | DIAG_DRAFT_PROBE |
| ANEMONE_DIAG_DRAFT_SKIP_FFN | DIAG_DRAFT_SKIP_FFN |
| ANEMONE_DIAG_DRAFT_STAGES | DIAG_DRAFT_STAGES |
| ANEMONE_DIAG_STATE_DIGESTS | DIAG_STATE_DIGESTS |
| ANEMONE_SOCKET_DESCRIPTOR | INFERENCE_SOCKET_FD |
| ANEMONE_TRANSPORT_CHECK_ONLY | DEVICE_TRANSPORT_CHECK |

The first four are production diagnostic interfaces, normally left unset; a
presence-tested name can be enabled even by an empty or zero value. The last
two belong to standalone measurement workers. ANEMONE_API_KEY is only a client
placeholder used by the full manual. The server does not read its key from it.
CUDA_DEVICE_ORDER and sanitizer variables belong to their external runtimes.
See [the complete environment reference](docs/USAGE.md#816-environment-variables-debugging-and-sanitizers).
