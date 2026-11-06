<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Server profiles for Anemone v0.0.2-alpha

Use this page for the three release launch recipes: original exact, original
balanced and EXL3 balanced. Each complete command below passed the real axis
resolver and argument parser at source
`6b9e276fe4da6750be46c874e5fb5c0ee2a46871`. Four separate functional smokes then
started these three servers plus the optional guard: all 12 health, model-listing
and tiny T0 Chat requests returned HTTP 200. Every Chat request completed eight
native output tokens with finish reason `length`. This is functional model/HTTP
evidence, with no speed, quality, concurrency or full-context claim.
The CPU parser check itself stops before model, GPU, socket and server startup.
The [comparison matrix](results/COMPARISON_MATRIX.md)
records GPU execution, identity, quality, timing and capacity separately, with
their actual source/binary epochs. A parser pass alone establishes none of those.

The commands are also available as exact argument arrays in
[SERVER_PROFILE_COMMANDS.json](SERVER_PROFILE_COMMANDS.json), with environment,
source pin and small HTTP smoke requests. The functional-smoke binary at source
`6b9e276`, installed as `spdx_02`, has SHA256
`da39965f79adf7883f92c8941e2b92d928706dc7b3f55671b89dce23b67c413e`.
This measured epoch remains immutable when help text changes. Consult
[BUILD.md](../BUILD.md) and [SNAPSHOT.md](VALIDATION.md) for the current release build.
Binary/data path substitutions are installation choices; retain and record the
numerical controls when comparing a measurement.

## Prepare the binary and data

1. Follow [BUILD.md](../BUILD.md) from a fresh release export on Linux x86-64.
   The tested CUDA 13.4/GCC 15.2 release build needs neither Git nor Python nor
   GPU devices. Check `./build/release/anemone --version` for `v0.0.2-alpha`.
2. Follow [MODEL_DATA.md](user/MODEL_DATA.md) for the separate CPU Python
   environment, all 112 pinned input files, SHA256 verification and every
   producer command. The official source is
   [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170);
   the EXL3 source is
   [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/tree/0be63c3337efbb2a0061d5b8bb4006fa186d433f).
   Both immutable revisions, their selected file inventories and public metadata
   were checked again anonymously; no weight download is implied by that check.
3. Download into new `sources/official` and `sources/exl3` directories, then run
   the documented `tools/model_data.sh` wrapper with a new output directory.
   It writes converted `tokenizer.bin`, `original.anemone`, `hybrid/` and
   `model.exl3map`. The individual tokenizer, pack conversion/verification,
   hybrid-index and map commands are in the same model-data guide. Budget at
   least 1.3 TB free. Current bounded checks reproduced the tokenizer and hybrid
   index; the original pack has the recorded deterministic slice proof, not a
   newly repeated full 517 GB conversion. The full wrapper was not rerun end to end.
4. Retain the original pack for both families: EXL3 replaces routed experts but
   still uses original non-routed tensors, vision, DSpark and Engram. The small
   map references the original pack and 40 LibertAI shards by absolute path;
   regenerate it after moving those files. Its suffix must be `.exl3map`.
   Raw upstream `tokenizer.json` is not the converted runtime tokenizer.
5. Use the actual backend card order on the target machine. The recorded machine
   has four CMP 170HX cards at backend ordinals `0,1,3,4`, and the RTX PRO 6000
   at `2`. With `CUDA_DEVICE_ORDER=PCI_BUS_ID`, verify the reported ordinals;
   another machine or visibility list can differ. Runtime Engram alone needs
   about 188.8 GiB host RAM, before other serving allocations. Startup must admit
   the combined GPU/host memory plan for the chosen slots and context.

The commands run from the built repository root and assume these prepared data
paths. Replace the two trailing paths if using another stable data root. No
service installation or privileged `/opt` copy is required by this example.

```text
/opt/anemone/data/tokenizer.bin       # converted official tokenizer; 2,979,103 bytes
/opt/anemone/data/original.anemone    # original FP8/FP4 pack; 517,258,440,704 bytes
/opt/anemone/data/model.exl3map       # EXL3 map; 9,857,696 bytes, references weights
```

## Choose a recipe

| Recipe | Arithmetic and scheduling | Qualified boundaries |
| --- | --- | --- |
| Original exact | Original PP THIN serial M1; `--exactness=exact` inserts the bounded reference controls. | One session, 512 total prompt-plus-completion tokens, text only. Official F32 logit and T0 fixture scopes are recorded; stochastic source RNG identity is unqualified. |
| Original balanced | Original PP5, native PRO, THIN rows 8, DSpark and wide prefill. | Uses `--numerics=exact` but does **not** select strict `--exactness=exact`. Native PRO output depends on its assigned layers. Eight allocated slots and 65,536-token contexts do not establish every workload at those capacities. |
| EXL3 balanced | Four-CMP TP4, BF16 wire, TC-MoE/head/trunk, DSpark and BATCH menu. | A separate weight family. Its shipped-default concurrent mixed identity gate failed; mixed rates are withheld. The separate guard below has narrower positive evidence. |

These names describe the recorded explicit launch recipes. The two balanced
commands do not select a literal all-balanced four-axis tuple; that tuple remains
held. There is no `--profile=NAME` switch and no implicit axis defaults. See
[the four-axis rules](USAGE.md#7-four-axis-configuration) and
[strict reference scope](USAGE.md#97-the-exactnessexact-tier-original-weight-reference-profile).

All three commands explicitly choose operating sampling defaults **T=1,
top_p=1**. Raw omitted CLI temperature remains 0, including when only the exact
axis is selected. The deterministic comparison requests override it with T=0.
At T>0 the engine uses the same admitted model logits, but its RNG/sampler order
differs from the pinned source; matching stochastic token IDs is unqualified.
These commands retain the recorded numerical flags while selecting example data
paths, the `anemone` alias and explicit operating sampling defaults.

### Original exact

The axis supplies `rows=1`, no speculation/wide prefill, serial pipeline 1,
reference HC/arithmetic, graphs 1, the measured PP5 map and disabled prefix
reuse. The context capacity and omitted-request output default are both 512;
a request's prompt still consumes part of the context. `reasoning_effort=0` disables template
reasoning. The listener/alias flags identify the local HTTP endpoint.

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID ./build/release/anemone serve \
  --placement=pp5 \
  --exactness=exact \
  --reasoning_effort=0 \
  --context_tokens=512 \
  --generated_tokens=512 \
  --session_slots=1 \
  --model_alias=anemone \
  --bind_address=127.0.0.1 \
  --port=8321 \
  --temperature=1 \
  --top_p=1 \
  /opt/anemone/data/tokenizer.bin /opt/anemone/data/original.anemone
```

### Original balanced

The explicit PP5 layout assigns logical cards 0–4 to target layers 0–7, 8–14,
15–21, 22–28 and 29–39. `card_order=0,1,3,4,2` puts the native PRO last; it also
holds the target head and DSpark. `pro_emulation=0` preserves native PRO math.
`linear_profile=0`, `linear_order=1` and `cmp_order=reference` pin the retained
CMP arithmetic; this does not turn native-PRO balanced execution into the
strict official-reference profile.

`rows=8` gives THIN capacity, `wide_rows=1024` prepares wide prefill, and
`full_steps=1`/`prefill_cone=1` enable the full/cone plan. One chunk is in flight;
the omitted PP5 ordinary pipeline resolves to 5 for this multi-slot recipe.
`step_graphs=1` explicitly enables CUDA graph replay; raw default is 0.
`speculative=2` and its agreeing `draft=dspark` alias select checkpoint DSpark,
with block capacity and maximum proposals both 5. The omitted draft policy
resolves to cost policy 1. `card_memory_kept_free_mib=0` adds no placement reserve;
the full memory plan must still fit. The remaining capacity, endpoint and
sampling flags have the same meanings as above, with eight slots and 65,536
total tokens per slot.

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID ./build/release/anemone serve \
  --placement=pp5 \
  --numerics=exact \
  --cmp_order=reference \
  --linear_order=1 \
  --linear_profile=0 \
  --pro_emulation=0 \
  --card_order=0,1,3,4,2 \
  '--layout=vision: card 0; embedding: card 0; layers 0..7: card 0; layers 8..14: card 1; layers 15..21: card 2; layers 22..28: card 3; layers 29..39: card 4; head: card 4; draft head: card 4' \
  --wide_rows=1024 \
  --rows=8 \
  --full_steps=1 \
  --prefill_cone=1 \
  --chunks_in_flight=1 \
  --step_graphs=1 \
  --speculative=2 \
  --draft=dspark \
  --draft_block=5 \
  --draft_positions=5 \
  --card_memory_kept_free_mib=0 \
  --reasoning_effort=0 \
  --context_tokens=65536 \
  --generated_tokens=65536 \
  --session_slots=8 \
  --model_alias=anemone \
  --bind_address=127.0.0.1 \
  --port=8321 \
  --temperature=1 \
  --top_p=1 \
  /opt/anemone/data/tokenizer.bin /opt/anemone/data/original.anemone
```

### EXL3 balanced

`placement=tp4_exl3` selects exactly four CMPs when card order is omitted.
`numerics=relaxed`, reference local CMP order and `tp_wire_bf16=1` select the
recorded BF16 recipe. Four expert parts retain the measured partition; eight
parts are a separate arithmetic option. TC-MoE, head/trunk MMA, ordered-wide
prefill, sharded/streamed head and sharded/TP draft layers are explicitly enabled
by the corresponding flags. `pro_emulation=0` is consistent with a CMP-only
placement; the PRO participates in neither the target nor draft model here.

The shared row/full/cone/graph/DSpark flags have the meanings above. The BATCH
row menu `8,8,12,12,16,24,32,48` pairs with verifier capacities
`1,2,2,4,4,4,8,8`, including distinct eight-row programs for one and two
verifiers. `batch_wide_head=1` and `batch_draft_rows=1` put verification/draft rows
in BATCH programs. This resolves ordinary pipeline depth to 1 and draft policy
to rate policy 2. Eight session slots and 65,536-token contexts allocate capacity;
the omitted-request output default is also 65,536. These settings do not guarantee arbitrary mixed traffic or full
context numerical qualification.

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID ./build/release/anemone serve \
  --placement=tp4_exl3 \
  --numerics=relaxed \
  --tp_wire_bf16=1 \
  --cmp_order=reference \
  --linear_order=1 \
  --linear_profile=0 \
  --pro_emulation=0 \
  --exl3_expert_parts=4 \
  --exl3_tc_moe=1 \
  --tp_head_mma=1 \
  --tp_trunk_mma=1 \
  --tp_ordered_wide=1 \
  --tp_head_shards=1 \
  --tp_head_stream=1 \
  --tp_draft_shards=1 \
  --rows=8 \
  --full_steps=1 \
  --prefill_cone=1 \
  --chunks_in_flight=1 \
  --step_graphs=1 \
  --speculative=2 \
  --draft=dspark \
  --draft_block=5 \
  --draft_positions=5 \
  --card_memory_kept_free_mib=0 \
  --wide_rows=1024 \
  --batch_rows=8,8,12,12,16,24,32,48 \
  --batch_verifiers=1,2,2,4,4,4,8,8 \
  --batch_wide_head=1 \
  --batch_draft_rows=1 \
  --tp_draft_layers=1 \
  --reasoning_effort=0 \
  --context_tokens=65536 \
  --generated_tokens=65536 \
  --session_slots=8 \
  --model_alias=anemone \
  --bind_address=127.0.0.1 \
  --port=8321 \
  --temperature=1 \
  --top_p=1 \
  /opt/anemone/data/tokenizer.bin /opt/anemone/data/model.exl3map
```

### Separate EXL3 prompt-chunk guard

The optional `exl3_guard` array in the command JSON is the EXL3 balanced command
plus `--prefill_prompt_chunks=1 --prefill_first=0 --prefill_order=0`. That arm
passed its recorded 96/96 mixed concurrent-versus-solo identities; the separate
default arm failed 36/96. The guard has no independent full-logit golden score,
so the baseline solo quality score does not establish guard quality equivalence.
It is opt-in, does not change a shipped default, and does not admit the held
balanced axis tuple. Read the [current matrix](results/COMPARISON_MATRIX.md)
before selecting it. One 1,048,576-token slot's readiness in either placement
also remains an allocation result, not full-context inference qualification.

## Request sampling and output controls

After one server is ready, the command JSON supplies these three small smoke
requests. They are examples; the CPU parser checks do not send them to a server.
All three requests passed in each of the four separately admitted functional smokes.
No key file is supplied by the recipes. If adding one, use the named-user server
key format in [the authentication manual](USAGE.md#811-http-server-authentication-and-queues)
and configure the client privately; health is the unauthenticated route.

```sh
curl -fsS http://127.0.0.1:8321/health
curl -fsS http://127.0.0.1:8321/v1/models
curl -fsS http://127.0.0.1:8321/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"anemone","messages":[{"role":"user","content":"Hello"}],"max_tokens":8,"temperature":0,"top_p":1,"seed":1,"reasoning_effort":0,"store":false}'
```

Startup flags allocate resources and choose default behavior. Request fields
below control each HTTP generation within that startup envelope. Detailed tool,
image, session, streaming and Responses behavior remains in
[the HTTP manual](USAGE.md#6-the-http-api).
For Pi, use the [model template and private user-key setup](USAGE.md#67-pi-configuration).

| Request field | Meaning / admission |
| --- | --- |
| `temperature` | Finite HTTP value 0–2; 0 is greedy. Omitted uses the startup default (explicitly 1 above). CLI accepts finite nonnegative floats with a wider bound. T0 fixture identity does not imply T1 source RNG identity. |
| `top_p` | Nucleus cutoff greater than 0 and at most 1; omitted uses startup (1 here). |
| `seed` | HTTP unsigned 64-bit, CLI unsigned 32-bit. An omitted serving seed follows request random-seed policy; the parser's stored zero is not proof of a fixed per-request seed. |
| `max_tokens`, `max_completion_tokens` | Integer Chat output limits; the smaller wins when both exist. Explicit request limits replace the `--generated_tokens` default; it is not a separate serving ceiling. Prompt-plus-output context capacity still bounds execution. Responses uses `max_output_tokens`. |
| `reasoning_effort` | Integer 0–100 or none/off=0, minimal/low=50, medium/high/xhigh=75, max=100. Startup sets 0 here. |
| `thinking` | Object with only `type: enabled` or `disabled`; disabled overrides effort, enabled with zero effort refuses. |
| `stop` | One nonempty string or up to four, at most 256 bytes each; matches content rather than reasoning. |
| `stream`, `stream_options.include_usage` | Boolean SSE selection and optional terminal usage chunk. |
| `n`, `store`, `service_tier` | Only `n=1`, `store=false`, and `service_tier=auto/default` are supported. |
| `anemone_draft_usage` | Per-response draft statistics: none, compact or full; overrides startup verbosity. |

## Every serve configuration option

The tables cover **169 distinct accepted configuration spellings** in the
current serve parser: 149 numeric/array options and 20 text/fraction/policy/axis
options. Actions and other-command flags follow separately. Each option links
to its detailed definition in the single canonical [USAGE manual](USAGE.md#15-compact-serve-flag-reference).
The shown ranges describe the parser vocabulary, not blanket admission of every
combination. Normal whole-recipe startup, program, weight-family, device and
memory gates still apply.

Values come from six real resolver/parser snapshots at the source pin above:
the three commands, the optional guard, bare original serve and bare explicit
EXL3 serve. All 149 numeric fields were captured in each snapshot. The **Raw**
column shows those no-axis omitted defaults: `O:` original and `E:` EXL3 where
they differ. Bare EXL3 defaults are a parser snapshot, not a runnable TP recipe:
TP startup additionally needs context at least 16,384, rows at most 8, DSpark or
hybrid, native arithmetic and reference CMP order. No axis defaults are invented.

- `I` after a value means its implementation path is inactive for that recipe
  (for example EXL3 controls under PP5, BATCH controls without BATCH, or draft
  policy with speculation off). The stored value is still shown. These labels
  identify direct placement/mode exclusions; an unmarked field can still be
  conditional on another disabled feature or a particular request/program shape.
- `R` means changing this binary switch to its opposite value was refused by
  the real parser while keeping the rest of that recipe fixed. There were 201
  opposite-value probes across 67 binary switches and three recipes. A missing
  `R` means only that the binary probe was accepted, or the option was not one
  of those probes; it never grants numerical/performance qualification.
- `off`/`unset` differ from an absent axis: axes are consumed by the resolver;
  their inserted low-level values are displayed elsewhere in the same tables.
- `PP5 map` means the exact layout printed in the original-balanced command.
  `auto` card order means backend selection, not a literal list of zero cards.
  Arrays omit unused trailing zero storage. `none` means no programs/list.
- Enum names are shown beside their numeric codes. The value columns report
  resolved parser configuration; runtime auto-selection and actual memory fit
  remain startup checks. Do not treat an inactive value as a kernel invocation.
  The per-user queue's stored zero is an omitted-value sentinel: `server.c`
  resolves it to half the global queue, at least one (8 for queue depth 16 here).

### Loader and turbo replicas

| Option, accepted vocabulary and effect | Raw O / E | Original exact | Original balanced | EXL3 balanced |
| --- | --- | --- | --- | --- |
| [--card_order](USAGE.md#knob-card_order): 0..15; comma list, at most16. Choose backend card ordinals in stage-fill order. | auto | 0,1,3,4,2 | 0,1,3,4,2 | auto |
| [--reader_threads_per_copy](USAGE.md#knob-reader_threads_per_copy): 1..16. Allocate this many asynchronous readers per pack copy. | 2 | 2 | 2 | 2 |
| [--tp_load_threads](USAGE.md#knob-tp_load_threads): 0..64. Load TP card weights concurrently; zero keeps the serial loader. | 32 | 32 (I) | 32 (I) | 32 |
| [--turbo_load](USAGE.md#knob-turbo_load): 0..1. Discover and read receipted replicas on other drives. | 1 | 1 | 1 | 1 |
| [--map_reads](USAGE.md#knob-map_reads): 0 buffered;1 drop;2 direct;3 direct+drop;4 auto. Choose buffered/direct EXL3-map reads and page-cache eviction policy. | 4 auto | 4 auto (I) | 4 auto (I) | 4 auto |
| [--tp_load_host_threads](USAGE.md#knob-tp_load_host_threads): 0..64. Load the host arena with this many workers; zero follows TP load threads. | 16 | 16 (I) | 16 (I) | 16 |
| [--tp_load_copy_split](USAGE.md#knob-tp_load_copy_split): 0..1. Read host chunks from copy zero and device extents from the last copy. | 1 | 1 (I) | 1 (I) | 1 |
| [--tp_load_reads_per_copy](USAGE.md#knob-tp_load_reads_per_copy): 0..64. Limit TP in-flight reads per copy; zero is unlimited. | 0 | 0 (I) | 0 (I) | 0 |

### Logging, profiling and diagnostic dumps

| Option, accepted vocabulary and effect | Raw O / E | Original exact | Original balanced | EXL3 balanced |
| --- | --- | --- | --- | --- |
| [--live_stats](USAGE.md#knob-live_stats): 0..60000. Print server counters at this interval in milliseconds; zero disables them. | 0 | 0 | 0 | 0 |
| [--draft_trace](USAGE.md#knob-draft_trace): 0 in release; 0 or 1 in debug/thread/profile. Print per-round draft anchors and proposals for diagnostics. | 0 | 0 (I) | 0 | 0 |
| [--dump_draft_ring](USAGE.md#knob-dump_draft_ring): path. Generate-only shutdown diagnostic; accepted as text by serve but its writer is inactive there. | unset | unset (I) | unset (I) | unset (I) |
| [--live_stats_format](USAGE.md#knob-live_stats_format): text / json / table. Choose live server statistics format. | text | text (I) | text (I) | text (I) |

### Memory, slots, context and 1M

| Option, accepted vocabulary and effect | Raw O / E | Original exact | Original balanced | EXL3 balanced |
| --- | --- | --- | --- | --- |
| [--card_memory_kept_free_mib](USAGE.md#knob-card_memory_kept_free_mib): 0..1048576. Subtract this many MiB from each card's placement budget. | 0 | 0 | 0 | 0 |
| [--session_slots](USAGE.md#knob-session_slots): 1..8. Allocate this many independent active session slots. | 1 | 1 | 8 | 8 |
| [--wide_rows](USAGE.md#knob-wide_rows): 0..4096. Prepare a wide prefill program of this many rows; zero disables it. | 0 | 0 | 1024 | 1024 |
| [--rows](USAGE.md#knob-rows): 1..8. Set ordinary THIN step row capacity (and proposal capacity minus one). | 1 | 1 | 8 | 8 |
| [--context_tokens](USAGE.md#knob-context_tokens): 1..1048576. Allocate the maximum prompt-plus-generation context per slot. | 16384 | 512 | 65536 | 65536 |
| [--layout](USAGE.md#knob-layout): &lt;=4096-byte statements joined by semicolon+space: &lt;units&gt;: card &lt;c&gt; or experts kept: &lt;units&gt;: &lt;n&gt;. Override PP5 unit placement. | unset | PP5 map | PP5 map | unset (I) |

### Original weights and PRO arithmetic

| Option, accepted vocabulary and effect | Raw O / E | Original exact | Original balanced | EXL3 balanced |
| --- | --- | --- | --- | --- |
| [--pro_emulation](USAGE.md#knob-pro_emulation): 0..1. Emulate class-80 sums on the PRO; zero uses class-120 native arithmetic. | 0 | 0 (R) | 0 | 0 (I) |
| [--tp_code_wire](USAGE.md#knob-tp_code_wire): 0..1. Transport original quantized codes/scales losslessly across TP ranks. | 0 | 0 (I, R) | 0 (I) | 0 (I, R) |
| [--original_dense_staged](USAGE.md#knob-original_dense_staged): parser0..16; admitted0 or16. Stage original TP4 BF16/FP32 dense products. | 0 | 0 (I) | 0 (I) | 0 (I) |
| [--original_wide_experts](USAGE.md#knob-original_wide_experts): 0..1. Run original TP4 expert jobs over complete chunks with lossless slab gathers. | 0 | 0 (I, R) | 0 (I, R) | 0 (I, R) |
| [--original_chunk_core](USAGE.md#knob-original_chunk_core): 0..1. Use original TP4 full-head attention core on whole eight-row slabs. | 0 | 0 (I, R) | 0 (I, R) | 0 (I, R) |
| [--original_batch](USAGE.md#knob-original_batch): 0..1. Admit the fixed original TP4 BATCH48/eight-verifier/eight-slot pilot. | 0 | 0 (I, R) | 0 (I, R) | 0 (I, R) |
| [--tp_pipeline_overlap](USAGE.md#knob-tp_pipeline_overlap): 0..1. Run the CMP prefix and native PRO tail on disjoint workers. | 0 | 0 (I, R) | 0 (I, R) | 0 (I, R) |

### MoE and expert products

| Option, accepted vocabulary and effect | Raw O / E | Original exact | Original balanced | EXL3 balanced |
| --- | --- | --- | --- | --- |
| [--exl3_expert_parts](USAGE.md#knob-exl3_expert_parts): 4 or8. Choose four reference or eight alternate expert product parts. | 4 | 4 (I) | 4 (I) | 4 |
| [--exl3_fast_products](USAGE.md#knob-exl3_fast_products): 0..1. Use wide-kernel EXL3 products instead of the retained chain. | 1 | 1 (I) | 1 (I) | 1 |
| [--exl3_wide_entry](USAGE.md#knob-exl3_wide_entry): 0..1. Run each chunk's EXL3 experts once over the full chunk. | 1 | 1 (I) | 1 (I) | 1 |
| [--exl3_tc_moe](USAGE.md#knob-exl3_tc_moe): 0..1. Use GNmac-style tensor-core EXL3 routed MoE products. | 0 | 0 (I) | 0 (I) | 1 |
| [--exl3_tc_product_blocks](USAGE.md#knob-exl3_tc_product_blocks): 0..592. Override the two EXL3 TC-MoE product grids in THIN/BATCH programs. | 0 | 0 (I) | 0 (I) | 0 |
| [--exl3_expert_rotation](USAGE.md#knob-exl3_expert_rotation): 0 fixed chunks;1 four-way rotation;2 three-way rotation. Rotate per-expert TC chunks across four ranks or three non-head ranks. | 0 | 0 (I) | 0 (I) | 0 |
| [--tp_router_mma](USAGE.md#knob-tp_router_mma): 0 dense;1 BATCH router MMA;2 also prefill routers. Use split-K tensor-core BATCH router products (two alternate modes). | 0 | 0 (I) | 0 (I) | 0 |
| [--exl3_tcp_two_blocks](USAGE.md#knob-exl3_tcp_two_blocks): 0..1. Launch prefill TC MoE products at two blocks per multiprocessor. | 1 | 1 (I) | 1 (I) | 1 |

### Attention, index and Engram

| Option, accepted vocabulary and effect | Raw O / E | Original exact | Original balanced | EXL3 balanced |
| --- | --- | --- | --- | --- |
| [--tp_attention_owner](USAGE.md#knob-tp_attention_owner): 0..1. Keep experimental TP4 attention state on its owning rank. | 0 | 0 (I, R) | 0 (I) | 0 (I, R) |
| [--tp_exact_dense](USAGE.md#knob-tp_exact_dense): 0..1. Use exact output-sharded attention dense products on the FP32 EXL3 recipe. | 1 | 1 (I) | 1 (I) | 1 (I) |
| [--tp_gather_balance](USAGE.md#knob-tp_gather_balance): 0 off;1 all chunk gathers;2 wide only;3 weighted by rank width. Assign chunk index/router work to narrower expert ranks (four policies). | 0 | 0 (I) | 0 (I) | 0 |
| [--tp_pool_chunk](USAGE.md#knob-tp_pool_chunk): 0..1. Prepare pooled entries/keys before chunk-slab scans. | 1 | 1 (I) | 1 (I) | 1 |
| [--tp_pool_gather](USAGE.md#knob-tp_pool_gather): 0..1. Partition pooled-source compressor rows over ranks and gather results. | 1 | 1 (I) | 1 (I) | 1 |
| [--tp_router_first_whole](USAGE.md#knob-tp_router_first_whole): 0..1. Compute first-half router rows on every rank without gathering. | 0 | 0 (I) | 0 (I) | 0 |
| [--tp_queries_piece](USAGE.md#knob-tp_queries_piece): 0..1. Project wide-program queries by piece alongside wq_b. | 1 | 1 (I) | 1 (I) | 1 |
| [--tp_index_span](USAGE.md#knob-tp_index_span): 0..1. Scan/select a span of chunk index rows per launch. | 1 | 1 (I) | 1 (I) | 1 |
| [--tp_core_compact_halves](USAGE.md#knob-tp_core_compact_halves): 0..1. Write split-half core results into compact wo_a rows. | 1 | 1 (I) | 1 (I) | 1 |
| [--tp_chunk_core](USAGE.md#knob-tp_chunk_core): 0..1. Run a chunk's attention core over the whole chunk. | 1 | 1 (I) | 1 (I) | 1 |
| [--tp_core_heads](USAGE.md#knob-tp_core_heads): 0,8,..64. Limit attention-core heads per block; zero uses the measured rule. | 0 | 0 (I) | 0 (I) | 0 |
| [--engram_shared](USAGE.md#knob-engram_shared): 0 or1. Gather identical TP Engram inputs once into a shared pinned block. | 0 | 0 (I) | 0 (I) | 0 |

### Precision, exactness, head and dense forms

| Option, accepted vocabulary and effect | Raw O / E | Original exact | Original balanced | EXL3 balanced |
| --- | --- | --- | --- | --- |
| [--tp_head_shards](USAGE.md#knob-tp_head_shards): 0..1. Shard target head products and sampling over TP4_EXL3 ranks. | 0 | 0 (I, R) | 0 (I, R) | 1 (R) |
| [--tp_head_stream](USAGE.md#knob-tp_head_stream): 0..1. Use the dense-stream target head products. | 0 | 0 (I, R) | 0 (I, R) | 1 |
| [--tp_head_mma](USAGE.md#knob-tp_head_mma): 0..1. Use tensor-core target-head products on the BF16 recipe. | 0 | 0 (I) | 0 (I) | 1 |
| [--tp_trunk_mma](USAGE.md#knob-tp_trunk_mma): 0..1. Use tensor-core BF16-row trunk products on the BF16 recipe. | 0 | 0 (I) | 0 (I) | 1 |
| [--tp_trunk_mma_fused](USAGE.md#knob-tp_trunk_mma_fused): 0..1. Sum trunk MMA split partials inside the 64-row kernel. | 0 | 0 (I) | 0 (I) | 0 |
| [--tp_trunk_mma_thin_splits](USAGE.md#knob-tp_trunk_mma_thin_splits): 0..1. Give 64-row trunk MMA products the THIN split geometry. | 0 | 0 (I) | 0 (I) | 0 |
| [--dense_skinny_bf16](USAGE.md#knob-dense_skinny_bf16): 0 retain default;1/4/8 outputs per block (other CLI-range values fail preparation). Tile BF16 skinny dense outputs per block above 48 rows; zero retains wide form. | 0 | 0 (I) | 0 (I) | 0 |
| [--dense_skinny_f32](USAGE.md#knob-dense_skinny_f32): 0 retain default;1/4/8 outputs per block (other CLI-range values fail preparation). Tile FP32-weight skinny dense outputs per block above 48 rows. | 0 | 0 (I) | 0 (I) | 0 |
| [--dense_staged](USAGE.md#knob-dense_staged): 0,8,16,32,64. Stage wide dense-product inputs with a selectable output-block width. | 16 | 16 (I) | 16 (I) | 16 |
| [--tp_front_branch](USAGE.md#knob-tp_front_branch): 0..2. Branch THIN attention mix coefficients; mode two also branches FFN coefficients. | 0 | 0 (I) | 0 (I) | 0 |
| [--tp_decode_round_once](USAGE.md#knob-tp_decode_round_once): 0..1. Sum published BF16 target partials in ascending-rank FP32 and round the result once. | 0 | 0 (I, R) | 0 (I, R) | 0 |
| [--tp_wire_bf16](USAGE.md#knob-tp_wire_bf16): 0..1. Round TP transport partials/sums to BF16; zero retains FP32 wire. | 0 | 0 (I) | 0 (I) | 1 |
| [--linear_profile](USAGE.md#knob-linear_profile): 0 precision;1 speed. Choose precision or speed linear-product preparation. | 0 | 0 | 0 | 0 |
| [--linear_order](USAGE.md#knob-linear_order): 0 fixed;1 reference. Choose fixed (zero) or source reference (one) product accumulation order. | 1 | 1 | 1 | 1 |
| [--official_hc_rows](USAGE.md#knob-official_hc_rows): parser0..8; admitted0/1/3/4/8. Select pinned HC arithmetic with an explicit reference call shape. | 0 | 1 | 0 | 0 (I) |
| [--official_reference](USAGE.md#knob-official_reference): 0 / 1. Select composed official-reference arithmetic through prepared original-weight programs. | 0 | 1 (R) | 0 (R) | 0 (I, R) |
| [--numerics](USAGE.md#knob-numerics): exact / relaxed / fast. Choose admitted startup numerical policy. | exact | exact | exact | relaxed |
| [--placement](USAGE.md#knob-placement): pp5 / tp4_pro / tp4_exl3 / tp2_pp2_pro / tp2_single_single_pro. Choose prepared weight/state placement. | O: pp5; E: tp4_exl3 | pp5 | pp5 | tp4_exl3 |
| [--cmp_order](USAGE.md#knob-cmp_order): reference / relaxed. Choose CMP trunk/expert sum order. | reference | reference | reference | reference |

### Draft and speculative policy

| Option, accepted vocabulary and effect | Raw O / E | Original exact | Original balanced | EXL3 balanced |
| --- | --- | --- | --- | --- |
| [--tp_draft_shards](USAGE.md#knob-tp_draft_shards): 0..1. Shard draft head, Markov projection and sampler with the target head. | 0 | 0 (I, R) | 0 (I, R) | 1 (R) |
| [--tp_draft_layers](USAGE.md#knob-tp_draft_layers): 0..1. Tensor-parallelize the draft model's three layers. | 0 | 0 (I, R) | 0 (I, R) | 1 |
| [--batch_draft_rows](USAGE.md#knob-batch_draft_rows): 0..1. Batch verifier draft rows into the BATCH program. | 0 | 0 (I, R) | 0 (I, R) | 1 (R) |
| [--batch_wide_head](USAGE.md#knob-batch_wide_head): 0..1. Use the wide target head for BATCH verification. | 0 | 0 (I, R) | 0 (I, R) | 1 (R) |
| [--batch_draft_slots_head](USAGE.md#knob-batch_draft_slots_head): 0..1. Run one verifier's draft head across eight slots at each position. | 0 | 0 (I) | 0 (I) | 0 |
| [--draft_bf16_rows](USAGE.md#knob-draft_bf16_rows): 0..1. Use BF16 rows for draft attention products. | 0 | 0 (I) | 0 | 0 |
| [--batch_draft_wide](USAGE.md#knob-batch_draft_wide): 0..3. Widen draft work: one sums/posts, two products, three cores. | 3 | 3 (I) | 3 (I) | 3 |
| [--draft_bf16_operands](USAGE.md#knob-draft_bf16_operands): 0..1. Use BF16 operands for class-80 draft quantized products. | 0 | 0 (I) | 0 | 0 |
| [--draft_learned_acceptance](USAGE.md#knob-draft_learned_acceptance): 0..1. Update the draft policy's shared confidence table from outcomes. | O: 0; E: 1 | 0 (I) | 0 | 1 |
| [--draft_session_acceptance](USAGE.md#knob-draft_session_acceptance): 0 shared table;1 session ratio;2 log-odds shift;3 Platt scale+shift;4 shift+hindsight;5 full depth while measured match&gt;=0.82;6 position-local EMA;7 middle scout;8 second-position scout;9 natural selector;10 conditional residual calibration;11 actual successful BATCH shape costs. Scale proposal acceptance by the current session's measured outcomes. | 0 | 0 (I) | 0 | 0 |
| [--batch_plan_ties](USAGE.md#knob-batch_plan_ties): 0..1. Cut planned drafts only between unequal predicted reach. | 0 | 0 (I) | 0 (I) | 0 |
| [--draft_start_economic](USAGE.md#knob-draft_start_economic): 0..1. Start DSpark at policy depth instead of a single-proposal probe. | 1 | 1 (I) | 1 | 1 |
| [--speculative](USAGE.md#knob-speculative): 0 off;1 lookup;2 DSpark;3 hybrid. Choose off, n-gram lookup, checkpoint DSpark or hybrid proposals. | 0 off | 0 off | 2 dspark | 2 dspark |
| [--draft_policy](USAGE.md#knob-draft_policy): 0 fixed;1 cost;2 rate;3 adaptive. Choose fixed, cost, rate or adaptive proposal-depth selection. | O: 1 cost; E: 0 fixed | 1 cost (I) | 1 cost | 2 rate |
| [--draft_threshold](USAGE.md#knob-draft_threshold): 0..1000. Set the minimum draft confidence in thousandths. | 100 | 100 (I) | 100 | 100 |
| [--draft_base_us](USAGE.md#knob-draft_base_us): 1..1000000000. Estimate the base verification-step cost in microseconds. | O: 12700; E: 14300 | 12700 (I) | 12700 | 14300 |
| [--draft_tail_us](USAGE.md#knob-draft_tail_us): 0..1000000000. Estimate fixed tail cost in microseconds. | O: 2200; E: 0 | 2200 (I) | 2200 | 0 |
| [--draft_row_us](USAGE.md#knob-draft_row_us): 1..1000000000. Estimate each additional verification row's cost in microseconds. | O: 2700; E: 690 | 2700 (I) | 2700 | 690 |
| [--draft_probe_rounds](USAGE.md#knob-draft_probe_rounds): 1..1048576. Probe full draft depth after this many policy rounds. | 8 | 8 (I) | 8 | 8 |
| [--hybrid_lookup_prior](USAGE.md#knob-hybrid_lookup_prior): 0..1000. Set lookup acceptance prior in thousandths. | 500 | 500 (I) | 500 (I) | 500 (I) |
| [--hybrid_lookup_min](USAGE.md#knob-hybrid_lookup_min): 1..16. Use lookup in hybrid mode only from this n-gram length. | 3 | 3 (I) | 3 (I) | 3 (I) |
| [--draft_block](USAGE.md#knob-draft_block): 5..7. Set the device draft block capacity. | 5 | 5 (I) | 5 | 5 |
| [--draft_positions](USAGE.md#knob-draft_positions): 1..7. Bound proposals verified per round. | 7 | 7 (I) | 5 | 5 |
| [--speculative_depth](USAGE.md#knob-speculative_depth): 1..7. Alias of draft_positions; both write the same proposal-depth field. | 7 | 7 (I) | 5 | 5 |
| [--lookup_ngram_max](USAGE.md#knob-lookup_ngram_max): 1..16. Set the longest lookup suffix searched. | 4 | 4 (I) | 4 (I) | 4 (I) |
| [--lookup_ngram_min](USAGE.md#knob-lookup_ngram_min): 1..16. Set the shortest lookup suffix searched. | 2 | 2 (I) | 2 (I) | 2 (I) |
| [--draft_calibration](USAGE.md#knob-draft_calibration): path to &lt;4096-byte file; exactly112 integers0..1000. Override draft acceptance priors in mode/position/bin order. | unset | unset (I) | unset | unset |
| [--draft](USAGE.md#knob-draft): lookup / dspark / hybrid. Alias lookup/dspark/hybrid for speculative=1/2/3; explicit spellings must agree. | off | off (I) | dspark | dspark |

### Tensor transport, graphs and overlap

| Option, accepted vocabulary and effect | Raw O / E | Original exact | Original balanced | EXL3 balanced |
| --- | --- | --- | --- | --- |
| [--tp_parallel_launch](USAGE.md#knob-tp_parallel_launch): 0..1. Launch each TP card's graph groups from a separate host thread. | 0 | 0 (I) | 0 (I) | 0 |
| [--tp_graph_parts](USAGE.md#knob-tp_graph_parts): 1..8. Partition each card's graph behind the prefill cone. | 1 | 1 (I) | 1 (I) | 1 |
| [--tp_ordered_wide](USAGE.md#knob-tp_ordered_wide): 0..1. Use ordered-wide attention trunk products for EXL3 prefill. | 0 | 0 (I) | 0 (I) | 1 |
| [--tp_graph_branches](USAGE.md#knob-tp_graph_branches): 0..2. Branch shared-expert work beside routed experts; mode two includes BATCH. | 1 | 1 (I) | 1 (I) | 1 |
| [--tp_piece_rings](USAGE.md#knob-tp_piece_rings): 0..1. Exchange BF16 wide results over 64-row pieces instead of eight-row slabs. | 0 | 0 (I) | 0 (I) | 0 |
| [--tp_trunk_shape_slices](USAGE.md#knob-tp_trunk_shape_slices): 0..1. Use the measured slice count per THIN trunk shard shape; zero forces four. | 1 | 1 (I) | 1 (I) | 1 |
| [--tp_bf16_slab_rings](USAGE.md#knob-tp_bf16_slab_rings): 0..1. Use BF16 slab-ring attention sums instead of whole-row FP32 accumulation. | 0 | 0 (I) | 0 (I) | 0 |
| [--tp_wide_exchange](USAGE.md#knob-tp_wide_exchange): 0 legacy;1 push;2 owner sums;3 owner rings;4 CE owner sums;5 CE rings. Choose wide transport: push, slab rings, owner sums or copy-engine rings. | 5 | 5 (I) | 5 (I) | 5 |
| [--tp_split_overlap](USAGE.md#knob-tp_split_overlap): 0 off;1 overlap;2 serial;3 attention;4 FFN;5 pipeline;6 joint;7 joint down. Overlap layer halves and wide exchange with a chosen split schedule. | 7 | 7 (I) | 7 (I) | 7 |
| [--tp_split_least_rows](USAGE.md#knob-tp_split_least_rows): 0..2048. Run chunks below this row count whole instead of split. | 0 | 0 (I) | 0 (I) | 0 |
| [--tp_launch_trims](USAGE.md#knob-tp_launch_trims): 0..1. Remove redundant ring-input copies and wo_b quantization launches. | 1 | 1 (I) | 1 (I) | 1 |
| [--tp_full_split_overlap](USAGE.md#knob-tp_full_split_overlap): 0 inherit;1..7 as tp_split_overlap. Override full-program split overlap; zero inherits tp_split_overlap. | 0 | 0 (I) | 0 (I) | 0 |
| [--tp_replicated_embedding](USAGE.md#knob-tp_replicated_embedding): 0..1. Embed input rows on every rank instead of handing off rank zero. | 1 | 1 (I) | 1 (I) | 1 |
| [--tp_ring_column_owners](USAGE.md#knob-tp_ring_column_owners): 0..1. Assign BF16 ring ownership by column quarter for batch-independent addition order. | 1 | 1 (I) | 1 (I) | 1 |
| [--step_graphs](USAGE.md#knob-step_graphs): 0..1. Capture/replay prepared CUDA step graphs instead of eager launches. | 0 | 1 (R) | 1 | 1 |
| [--pipeline_steps](USAGE.md#knob-pipeline_steps): 1..8. Bound ordinary step pipeline depth. | 1 | 1 | 5 | 1 |

### Decode and scheduling

| Option, accepted vocabulary and effect | Raw O / E | Original exact | Original balanced | EXL3 balanced |
| --- | --- | --- | --- | --- |
| [--batch_rows](USAGE.md#knob-batch_rows): 1..64; comma list, at most8. Prepare BATCH programs at these row capacities instead of THIN programs. | none | none (I) | none (I) | 8,8,12,12,16,24,32,48 |
| [--batch_context_slabs](USAGE.md#knob-batch_context_slabs): 0..1. Process BATCH draft context in THIN-sized eight-row slabs. | 0 | 0 (I, R) | 0 (I, R) | 0 |
| [--batch_thin_bits](USAGE.md#knob-batch_thin_bits): 0..1. Use THIN numerical splits in BATCH trunk/head/TC-MoE products. | 1 | 1 (I) | 1 (I) | 1 |
| [--batch_tc_paired](USAGE.md#knob-batch_tc_paired): 0..64. Pair TC MoE short forms from this many BATCH rows; zero disables pairing. | 0 | 0 (I) | 0 (I) | 0 |
| [--batch_row_sets](USAGE.md#knob-batch_row_sets): 0,8,16,24. Use trunk-product row sets above the grouped-row threshold. | 0 | 0 (I) | 0 (I) | 0 |
| [--batch_core_once](USAGE.md#knob-batch_core_once): 0..3. Compute BATCH attention core once; mode2 adds window writes and mode3 pools/deferred slots. | O: 2; E: 3 | 2 (I) | 2 (I) | 3 |
| [--batch_grouped_rows](USAGE.md#knob-batch_grouped_rows): 0..48. Use grouped THIN trunk forms through this BATCH row count. | 16 | 16 (I) | 16 (I) | 16 |
| [--batch_step_us](USAGE.md#knob-batch_step_us): 0 off, or distinct positive costs1..4294967295; never mix0 into a list; comma list, at most8. Supply per-program measured costs for the BATCH expected-token planner. | none | none (I) | none (I) | 19500,21500,28600,30000,32000,39000,46000,58000 |
| [--batch_verifiers](USAGE.md#knob-batch_verifiers): 1..8; comma list, at most8. Supply each BATCH program's drafting-session capacity. | none | none (I) | none (I) | 1,2,2,4,4,4,8,8 |
| [--prefill_first](USAGE.md#knob-prefill_first): 0..1. Prefer another session's wide/full prefill chunk before ordinary steps. | 0 | 0 (I) | 0 | 0 |
| [--prefill_prompt_chunks](USAGE.md#knob-prefill_prompt_chunks): 0..1. Keep runnable FULL/WIDE prompt chunks out of ordinary BATCH packing and retain decoder turns on forced landings. | 0 | 0 (I) | 0 | 0 |
| [--prefill_order](USAGE.md#knob-prefill_order): 0 round-robin chunks;1 command arrival order;2 shortest remaining prompt;3 shortest with concurrent short residuals. Choose round-robin chunks, command arrival order, shortest remaining prompt or opt-in shared short residuals. | 0 | 0 (I) | 0 | 0 |
| [--prefill_order_wait](USAGE.md#knob-prefill_order_wait): 0..1024. Bound shortest-order waiting in serving scheduler ticks. | 16 | 16 (I) | 16 | 16 |
| [--prefill_decode_every](USAGE.md#knob-prefill_decode_every): 0..64. Insert one decode round after this many head chunks; zero disables it. | 0 | 0 (I) | 0 | 0 |
| [--prefill_decode_waiters](USAGE.md#knob-prefill_decode_waiters): 0 or1. Keep whole chunks/one round while multiple prompts wait. | 0 | 0 (I) | 0 | 0 |
| [--prefill_decode_rounds](USAGE.md#knob-prefill_decode_rounds): 1..64. Run this many ordinary decode rounds between capped chunks. | 1 | 1 (I) | 1 | 1 |
| [--prefill_decode_rows](USAGE.md#knob-prefill_decode_rows): parser0..4096; nonzero multiple256 and &lt;=wide_rows; measured scope&lt;=2048. Cap prefill chunks while decoders are active. | 0 | 0 (I) | 0 | 0 |

### Prefill and chunk planning

| Option, accepted vocabulary and effect | Raw O / E | Original exact | Original balanced | EXL3 balanced |
| --- | --- | --- | --- | --- |
| [--full_steps](USAGE.md#knob-full_steps): 0..1. Use full programs behind the prefill front. | 1 | 0 (R) | 1 | 1 |
| [--chunks_in_flight](USAGE.md#knob-chunks_in_flight): 1..8. Bound chunk steps in flight; one uses the exact cone planner. | 1 | 1 | 1 | 1 |
| [--prefill_margin](USAGE.md#knob-prefill_margin): 1..3967. Keep this many full rows beyond the prefill cone. | 127 | 127 (I) | 127 | 127 |
| [--prefill_fast_plan](USAGE.md#knob-prefill_fast_plan): 0..1. Choose the largest fitting prefill program instead of the exact plan. | 0 | 0 (I) | 0 | 0 |
| [--prefill_prefix_stable](USAGE.md#knob-prefill_prefix_stable): 0 or1. Use fixed-capacity prefill forms with closed eight-row prefixes. | 0 | 0 (I, R) | 0 (I, R) | 0 (R) |
| [--prefill_chunk_rows](USAGE.md#knob-prefill_chunk_rows): parser0..4096; nonzero multiple256 and &lt;=wide_rows; measured scope&lt;=2048. Cap all exact-plan prefill chunks. | 0 | 0 (I) | 0 | 0 |
| [--prefill_cone_skip](USAGE.md#knob-prefill_cone_skip): 0 none;1 dead decoder core/MoE/wide trunk;2 last-row cone;3 mode1 with every trunk. Skip dead cone work; modes two/three extend the retained optimization. | 1 | 1 (I) | 1 | 1 |
| [--prefill_tail_rows](USAGE.md#knob-prefill_tail_rows): 0,128,320,448. Prepare 128/320/448-row full tail programs; zero disables them. | 0 | 0 (I) | 0 (I) | 128 |
| [--prefill_ladder_1024](USAGE.md#knob-prefill_ladder_1024): 0..1. Add 1024-row programs below larger wide programs. | 1 | 1 (I) | 1 | 1 |
| [--prefill_ladder_1536](USAGE.md#knob-prefill_ladder_1536): 0..1. Add 1536-row programs below larger wide programs. | 1 | 1 (I) | 1 | 1 |
| [--prefill_ladder_fine](USAGE.md#knob-prefill_ladder_fine): 0..1. Add 1792/1280/768-row exact-plan programs. | 0 | 0 (I) | 0 | 0 |
| [--prefill_front_tail](USAGE.md#knob-prefill_front_tail): 0..2. Use tail programs for the front region; two also uses the longer full tails. | 0 | 0 (I) | 0 (I) | 1 |
| [--prefill_tail_overlay](USAGE.md#knob-prefill_tail_overlay): 0..1. Overlay tail TC MoE workspace on prefill workspace. | 1 | 1 (I) | 1 (I) | 1 |
| [--prefill_tail_sets](USAGE.md#knob-prefill_tail_sets): 0..1. Run tail trunk products over their 64-row groups in one launch. | 1 | 1 (I) | 1 (I) | 1 |
| [--prefill_thin_tail](USAGE.md#knob-prefill_thin_tail): 0..1. Use 16..56-row programs for rows below a 64-row tail. | 0 | 0 (I, R) | 0 (I, R) | 0 |
| [--prefill_dead_halves](USAGE.md#knob-prefill_dead_halves): 0..2. Trim cone-dead sums; mode two also trims dead copies. | 2 | 2 (I) | 2 (I) | 2 |
| [--prefill_cone](USAGE.md#knob-prefill_cone): 0..1. Use a bounded front region with a prepared prefill cone. | 0 | 0 (I) | 1 (R) | 1 (R) |

### Cache, prefix reuse and snapshots

| Option, accepted vocabulary and effect | Raw O / E | Original exact | Original balanced | EXL3 balanced |
| --- | --- | --- | --- | --- |
| [--prefix_reuse](USAGE.md#knob-prefix_reuse): 0..1. Reuse eligible cached prompt prefixes instead of prefilling cold. | 1 | 0 (R) | 1 | 1 |
| [--prefix_reuse_exact](USAGE.md#knob-prefix_reuse_exact): 0 legacy;1 final-row repeat;2 cold-plan boundaries. Admit same-length cold-provenance prompt prefixes through repeat or cold-plan boundaries. | 0 | 0 | 0 | 0 |
| [--cache_scope](USAGE.md#knob-cache_scope): 0 session;1 user;2 global. Namespace reusable tokenizer/KV/snapshot state by session, user or global content. | 0 session | 0 session | 0 session | 0 session |
| [--session_fork](USAGE.md#knob-session_fork): 0 or1. Copy an eligible same-owner prefix into an independent child session slot. | 0 | 0 (I, R) | 0 (I, R) | 0 (R) |
| [--session_grace_seconds](USAGE.md#knob-session_grace_seconds): 0..3600. Prefer retaining a recently finished namespace when choosing an idle slot. | 30 | 30 | 30 | 30 |
| [--session_save_tokens](USAGE.md#knob-session_save_tokens): 1..1048576. Require this many processed tokens absent from a snapshot before saving. | 1024 | 1024 (I) | 1024 (I) | 1024 (I) |
| [--session_load_tokens](USAGE.md#knob-session_load_tokens): 1..1048576. Require this much reusable-token gain before loading a snapshot. | 256 | 256 (I) | 256 (I) | 256 (I) |
| [--session_idle_save_seconds](USAGE.md#knob-session_idle_save_seconds): 1..86400. Wait this long without work before starting an eligible idle snapshot save. | 120 | 120 (I) | 120 (I) | 120 (I) |
| [--session_directory](USAGE.md#knob-session_directory): path. Persist eligible session snapshots in this directory (requires prefix_reuse=1). | unset | unset | unset | unset |

### HTTP server, authentication and queues

| Option, accepted vocabulary and effect | Raw O / E | Original exact | Original balanced | EXL3 balanced |
| --- | --- | --- | --- | --- |
| [--queue_depth](USAGE.md#knob-queue_depth): 1..64. Bound all waiting server requests. | 16 | 16 | 16 | 16 |
| [--user_queue_depth](USAGE.md#knob-user_queue_depth): 1..64. Bound waiting requests per API user. | 0 auto → 8 | 0 auto → 8 | 0 auto → 8 | 0 auto → 8 |
| [--port](USAGE.md#knob-port): 1..65535. Bind the serve HTTP listener to this port. | 8321 | 8321 | 8321 | 8321 |
| [--bind_address](USAGE.md#knob-bind_address): numeric IPv4/IPv6. Bind the server listener. | 127.0.0.1 | 127.0.0.1 | 127.0.0.1 | 127.0.0.1 |
| [--key_file](USAGE.md#knob-key_file): path. Read server API-user authentication records from a private file; all routes except health then require a valid bearer key. | unset | unset | unset | unset |
| [--stats_users](USAGE.md#knob-stats_users): comma list of &lt;=64 usernames; each1..32 ASCII letters/digits/._-. Authorize named users to access the statistics endpoint. | unset | unset | unset | unset |
| [--model_alias](USAGE.md#knob-model_alias): 1..63 bytes. Add a request/listing model alias. | unset | anemone | anemone | anemone |

### Sampling, output and template controls

| Option, accepted vocabulary and effect | Raw O / E | Original exact | Original balanced | EXL3 balanced |
| --- | --- | --- | --- | --- |
| [--generated_tokens](USAGE.md#knob-generated_tokens): 1..4294967295. Set the omitted HTTP output-limit default; explicit request limits replace it within context capacity. | 4294967295 | 512 | 65536 | 65536 |
| [--seed](USAGE.md#knob-seed): 0..4294967295. CLI unsigned 32-bit sampler seed; omitted serve requests follow request seed policy. | omitted | omitted | omitted | omitted |
| [--reasoning_effort](USAGE.md#knob-reasoning_effort): 0..100. Set template reasoning effort from zero (off) through 100. | 0 | 0 | 0 | 0 |
| [--ignore_eos](USAGE.md#knob-ignore_eos): 0 / 1. Continue generation through end-token draws to the requested token/context limit. | 0 | 0 | 0 (R) | 0 (R) |
| [--temperature](USAGE.md#knob-temperature): finite float &gt;=0. Set sampling temperature; zero is greedy. | 0 | 1 | 1 | 1 |
| [--top_p](USAGE.md#knob-top_p): float &gt;0 and &lt;=1. Set nucleus probability cutoff. | 1 | 1 | 1 | 1 |
| [--draft_usage](USAGE.md#knob-draft_usage): none / compact / full. Choose draft statistics detail in responses. | compact | compact | compact | compact |

### Axes and branch feature boundaries

| Option, accepted vocabulary and effect | Raw O / E | Original exact | Original balanced | EXL3 balanced |
| --- | --- | --- | --- | --- |
| [--category](USAGE.md#knob-category): code / prose / structured / vision / balanced. Workload axis; no omitted default, subject to tuple qualification. | unset axis | unset axis | unset axis | unset axis |
| [--exactness](USAGE.md#knob-exactness): exact / balanced / fast. Numerical-tier axis; exact inserts bounded original-reference controls, distinct from numerics=exact. | unset axis | exact | unset axis | unset axis |
| [--concurrency](USAGE.md#knob-concurrency): single / multiple / balanced. Workload concurrency axis; this is not session_slots and has no omitted default. | unset axis | unset axis | unset axis | unset axis |
| [--focus](USAGE.md#knob-focus): balanced / ttft / decode / prefill. Objective axis; no omitted default and no latency guarantee. | unset axis | unset axis | unset axis | unset axis |

## Actions, aliases and unavailable controls

`anemone --version` is a standalone action: it prints version, author and license
identifier without opening a model. `anemone help axes` and `anemone help knobs`
are successful catalog actions; `anemone serve --help-scheduling` is a successful
scheduling-help action. Top-level `--help` and `serve --help` print usage with
exit 2 in this snapshot. They are not model-starting recipes.

`--speculative_depth` is the `--draft_positions` alias; the tables include both
accepted spellings. `--draft=lookup|dspark|hybrid` agrees with
`--speculative=1|2|3`; it is not an extra draft model. Axis expansion respects
these aliases when resolving explicit overrides. Explicit low-level values win
over inserted axis values in either argument order, while ordinary duplicate and
admission checks remain. Repeated axes refuse. There are no named profile aliases.

The catalog also documents controls from unavailable research branches. They
are **not current serve options**: `--original_dense_shards`,
`--original_queries_piece`, `--prefill_small_copy_rows`,
`--session_snapshot_format`, `--tp_core_queries`, `--tp5_expert_csf`,
`--tp5_expert_sidecar`, `--tp5_host_channels`, `--tp5_original_draft` and
`--tp5_pro_third`. TP5 itself remains refused. `--prompt_token_ids` belongs to
generate and refuses in serve. `--host`, `--interval_ms`, `--once`, `--format`,
`--user`, `--prune`, `--floor_gib` and `--rate_mib` belong to the separate
stats/keygen/replicate command interfaces; do not append them to serve.

The optional guard explicitly pins three controls relative to EXL3 balanced:
`prefill_prompt_chunks=1`, `prefill_first=0`, `prefill_order=0`. The latter two
already resolve to zero in this base recipe, so only `prefill_prompt_chunks`
changes its resolved value (0 → 1). The other 168 configuration spellings retain
their resolved values. Its snapshot and complete command were checked separately.

## Evidence and limits of this setup check

The documented command arrays are pinned to source `6b9e276`; all six parser
snapshots and 201 binary overrides used helper SHA256
`64b94814a8d30455dbf5e4f07368f32b726fbb3e6ccec91915d0dd4579bfc10e`.
It calls the real axis resolver and parser while omitting `run_program`, so it
opens no models, sockets, GPU backends or key files. Inactive-path labels also
use the current parser/implementation and detailed knob contract; they do not
pretend an inactive stored field was exercised by inference.

Four additional native Chat request-reader checks used an output default of 1:
omitting a request limit retained 1, explicit `max_tokens=8` returned 8,
`max_completion_tokens=8` returned 8, and both limits 8/5 returned 5. All passed
against the current prebuilt reader objects. This directly checks the serving
default-versus-request distinction without starting a model or HTTP listener.

The four separately admitted model/HTTP smokes used the exact frozen command
arrays and requests above, substituting only the executable and two data paths.
They passed on `6b9e276` / `spdx_02` / SHA256 `da39965f79adf7883f92c8941e2b92d928706dc7b3f55671b89dce23b67c413e`:
all 12 responses were HTTP 200; each tiny Chat response had eight native output
tokens and finish reason `length`. All owned servers/statistics/telemetry
processes retired, all five physical-card caps were restored, and each guard
reported the cards freed. These runs were functional only and establish no new
timing, quality, cross-profile identity, concurrency or long-context result.
They do not claim an end-to-end Pi agent run, and are not relabelled as a later
help-only binary. Detailed receipts are published through the comparison matrix.

The fresh candidate export built the source with an empty HOME and isolated
filesystem/network, no Git/Python execution or GPU devices: release build, exact
documented build command, version, storage probe, all 125 release host executables
and 594 header-alone compilations passed. Those synthetic host fixtures exercise
HTTP machinery; they do not start any of these model-serving commands. The
candidate binary matched the versioned binary hash above. The final delivered
tagged archive requires its own export check as recorded in [SNAPSHOT.md](VALIDATION.md).

Current producer checks covered tokenizer output, hybrid index, producer help,
overwrite/suffix refusals and the pack self-test. Full original conversion,
the full preparation wrapper, new Python-package installation and service
installation were not executed by this audit. Existing full-file, map and
deterministic original-slice evidence keeps its original epoch. Refer to
[MODEL_DATA.md](user/MODEL_DATA.md) for the exact scope, pinned hashes and paths.
