<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Release notes

## v0.0.6-alpha

Anemone v0.0.6-alpha makes the DSML tool-call parser tolerant, so that a tool call the model writes in any spelling
of DSML reaches the client as a tool call, and reorganises the source into module folders, with the model's code
under its own folder and prefix. Every reorganisation step builds the same programs as the step before it.

### Fixes
- **DSML tool calls (DSML-1).** The parser read only the exact reference bytes, so a call written with other spacing,
  an ASCII `|`, another keyword case or attribute order, or without a closing tag, or not after the blank line the
  reference writes before the block (for example straight after the reasoning), came back to the client as text. It
  now reads one grammar for chat completions, Anthropic messages and Responses: any whitespace inside a marker (at
  most 16 at each place of a head), `｜` or `|`, the keywords in any case, attributes in any order and quoting,
  closings implied by the next marker or the end. A closing tag written without its `<` is read as the closing
  (counted). A block opened by the reference's exact head is read by v0.0.5-alpha's rule first. A tool name is
  matched against the request's declared tools, exactly first, then ignoring case and `-`/`_`, never by its prefix; a
  name with no match goes out as parsed, so the client's unknown-tool error lets the model try again. At a token
  limit, a call that was cut off goes out as text, never as an executable call (counted cut_at_limit). Each repair is
  counted in `anemone_dsml_repairs_total`. Generation, sampling and every token are unchanged.
  `--dsml_parsing=reference` keeps v0.0.5-alpha's parser byte for byte. Speculative decoding was ruled out as a
  cause: with it on and off, production's settings give byte-identical output at T=0 and T=1 (254 turns).

### Reorganisation
- `source/`, `include/anemone/`, `generated/` and `tests/` are in module folders, and the model's tools and docs in
  `tools/models/` and `docs/models/`. Host files, headers and generated tables are named for what they hold; the
  model's kernel modules and test programs take its `ds41f_` prefix (two generic modules are renamed as well:
  `kernel_tp2_join` is `kernel_tp_join`, `kernel_head_offload` is `kernel_peer_row_copy`); the model's macros,
  enumerators, types and functions take its prefix; dead definitions are deleted; the renamed files are formatted
  with the Linux clang-format.
- Each step was checked on x86-64 against the step before it: the release programs, the 238 kernel modules and both
  kernel tables byte-identical (the module renames through a build with the renames put back, the prefix rename after
  the symbols are stripped), and the program dumps identical.
- The serve start prepares its programs, image programs and the head's banks through functions a test can call; the
  behaviour is unchanged (5,351,055 serve-start runs identical). `ds41f_split_program_dump_test` also covers the
  prefill caps (44 cases, 844 programs).

### Upgrade from v0.0.5-alpha
Replace the executable. The model data, the EXL3 maps and their descriptors, the storage profiles and the serve
commands of v0.0.5-alpha are unchanged. Source paths, test program names and the model's kernel module names have
changed: a script that builds or names them must follow the new paths.

### Limits
- Alpha software. The exact profile stays bounded to one session of 512 tokens. Vision is experimental. The comparison
  tables in the README keep their 6 November measurements, within the v0.0.2-alpha [measured scope](#measured-scope).
- The 2.68 bpw EXL3 map serves only on `--exl3_unrated_map=1`; its golden numbers are recorded, not a quality claim.
- Anthropic Messages: replaying a turn whose cut-off call went out as text after a tool_use returns 400 (the
  adapter's existing rule); replay the tool_use blocks only.

### Release check
The release executable ([VALIDATION.md](docs/VALIDATION.md)) was checked on the reference machine before the tag, on
21 Nov 2026.
- **CPU:** the full build in release, debug, thread and profile; the lints, cppcheck, clang-tidy and `make analysis`
  clean; every host suite in the four builds, the header check, the python tests and the release audit; the release
  objects equal the previous build's but for the two version texts; every resolver dry run equal but for the version
  and the help and usage text that names `--dsml_parsing`. DSML-1: the 525 variant cases in every build and 10,000
  fuzz cases in release and debug, 0 failures; `--dsml_parsing=reference` byte-identical to v0.0.5-alpha's parser on
  all of them (1,323,984 runs).
- **Identity:** on PP5, TP4_PRO and TP4_EXL3 (the 2.9 bpw map), one user and 15 streams a route, with each serve's
  given copies kept (`--turbo_load=0`), its outputs matched v0.0.5-alpha (source `d208e01`): PP5 15/15,
  TP4_PRO 15/15, TP4_EXL3 15/15; and v0.0.4-alpha (source `6ecf1bb`): PP5
  15/15, TP4_PRO 15/15, TP4_EXL3 15/15.
- **Tool calls:** on PP5 with production's serve options, 12 tool-loop scenarios, thinking off and on. At T=0 their
  24 first turns were identical with speculative decoding on and off, and under
  `--dsml_parsing=reference` identical to v0.0.5-alpha's; the tolerant parser's turns equalled the reference's on all
  24 it did not repair. At T=1, 72 tool loops in each of the three runs (tolerant
  and reference with speculative decoding, tolerant without) ran to their end with no request refused, and no DSML
  marker reached the text without a counted repair.
- **1M sessions:** on a PP5 serve with `--session_slots=max`, `--context_tokens=1048576`, `--generated_tokens=1048576`,
  `--temperature=1` and `--top_p=1` (the requests at T=0), the startup fit admitted 6 session slots of
  1,048,576 tokens each. Served: one fresh 1M-token session, HTTP 200, the needle found, prefill 3062 tok/s (1,022,969 prompt tokens in 334.1 s), decode 12.9 ms/token; four at once,
  HTTP 200 for all 8 requests, all 4 needles found, prefill about 765 tok/s each (1335.2 to 1338.6 s), decode 239.8 tok/s in all (4.17 ms/token).
- **Fraser:** the same serve with `--context_tokens=524288` and `--generated_tokens=524288`, T=0, thinking off, 2,048
  tokens, 1 user, 10 repetitions: median 172.8 tok/s
  (5.79 ms/token), best 183.0 tok/s (5.46 ms/token).
  v0.0.5-alpha in the same window, with the CPU quiet: median 171.9 tok/s; v0.0.6-alpha runs
  at 1.00 times its speed.

## v0.0.5-alpha

Anemone v0.0.5-alpha corrects v0.0.4-alpha's two out-of-date help texts and one TP4_EXL3 argument refusal, adds a
program dump that later source changes are checked against, and publishes golden scores for all 435 cases.

### Fixes
- **The help texts of v0.0.4-alpha's known issues.** `anemone help axes`, `tools/cli/AXES.md`, `tools/cli/KNOBS.md`
  and `tools/cli/axes.json` give PP5's `--concurrency=multiple` row the 6 sessions at 1M that v0.0.4-alpha measured
  on 17 Nov 2026 (they said v0.0.3-alpha's 5), and USAGE's section 9.3 dates its counts. `anemone help knobs
  points_per_run` and KNOBS.md give `--points_per_run`'s load-stage range as 0..78: its 35 arms twice, then its
  eight-point confirmation (they said 0..39).
- **An explicit `--tp_head_shards=0` on TP4_EXL3 runs.** The base inserts head and draft shards 1, and the parser
  refused the composed arguments (`--tp_draft_shards: needs --tp_head_shards=1`). The axes now bound the draft shards
  by the head shards and the draft layers by the draft shards, so the inserted values follow the explicit one. Only
  argument sets the parser refused before change.
- `model_v41_placement_test`'s verdict names the card over its budget, its weights and the bytes above, in every build
  (the release build compiles the engine's own error line out).
- `exl3_descriptor_test` also covers an established descriptor beside a pack the model lists as established.
- Comments and docs name what review and request labels stood for, or give their dates.

### Reorganisation, first part
- `tools/measure/request_front_end.mk` links again: its source list names `chat_images.c`.
- The serve start chooses its programs and describes its cards through functions a test can call; the behaviour is
  unchanged. Over 12,441,600 option sets the programs chosen are the earlier source's in every one.
- `model_v41_split_program_dump_test` dumps the programs each GPU split prepares (41 cases, 784 programs), so that two
  source versions can be shown to prepare the same programs.

### Quality
- **Golden scores for all 435 cases** (66,409 positions), captured with `anemone capture` on the three GPU splits'
  default routes at R = 6, R = 1 beside it: NLL 0.0867 on PP5, 0.0867 on TP4_PRO and 0.1178 on TP4_EXL3 with the 2.9
  bpw map, against the golden's own 0.0742; top-1 agreement 98.01 %, 98.01 % and 96.20 %
  ([GOLDEN_METRICS.md](docs/models/deepseek_v41_flash/results/GOLDEN_METRICS.md)). The 2.68 bpw map's numbers are recorded only.

### Upgrade from v0.0.4-alpha
Replace the executable. The model data, the EXL3 maps and their descriptors, the storage profiles and the serve
commands of v0.0.4-alpha are unchanged.

### Limits
- Alpha software. The exact profile stays bounded to one session of 512 tokens. Vision is experimental. The comparison
  tables in the README keep their 6 November measurements, within the v0.0.2-alpha [measured scope](#measured-scope).
- The 2.68 bpw EXL3 map serves only on `--exl3_unrated_map=1`; its golden numbers are recorded, not a quality claim.
- At 256K, PP5's 17th session leaves the first card about 7.7 MiB of its budget: a little less free memory there at
  startup gives 16.

### Release check
The release executable ([VALIDATION.md](docs/VALIDATION.md)) was checked on the reference machine before the tag, on
19 Nov 2026.
- **CPU:** the full build in release, debug, thread and profile; the lints, cppcheck, clang-tidy and `make analysis`
  clean; every host suite in the four builds, the header check, the python tests and the release audit; the release
  objects equal the previous build's but for the two version texts; every resolver dry run equal but for the version.
- **Identity:** on PP5, TP4_PRO and TP4_EXL3 (the 2.9 bpw map), one user and 15 streams a route, with each serve's
  given copies kept (`--turbo_load=0`), its outputs matched the previous build (source `73082c9`): PP5
  15/15, TP4_PRO 15/15, TP4_EXL3 15/15; and v0.0.4-alpha (source
  `6ecf1bb`): PP5 15/15, TP4_PRO 15/15, TP4_EXL3 15/15.
- **1M sessions:** on a PP5 serve with `--session_slots=max`, `--context_tokens=1048576`, `--generated_tokens=1048576`,
  `--temperature=1` and `--top_p=1` (the requests at T=0), the startup fit admitted 6 session slots of
  1,048,576 tokens each. Served: one fresh 1M-token session, HTTP 200, the needle found, prefill 3089 tok/s (1,022,969 prompt tokens in 331.1 s), decode 12.7 ms/token; four at once,
  HTTP 200 for all 8 requests, all 4 needles found, prefill about 767 tok/s each (1332.1 to 1335.5 s), decode 241.5 tok/s in all (4.14 ms/token).
- **Fraser:** the same serve with `--context_tokens=524288` and `--generated_tokens=524288`, T=0, thinking off, 2,048
  tokens, 1 user, 10 repetitions: median 171.5 tok/s
  (5.83 ms/token), best 183.7 tok/s (5.44 ms/token).

## v0.0.4-alpha

Anemone v0.0.4-alpha is a C11/CUDA engine for DeepSeek-V4.1-Flash-1017, on four CMP 170HX cards for EXL3, and on four
CMP 170HX plus an RTX PRO 6000 Blackwell for the original weights. It adds an Engram autotune stage, an engine-native
teacher-forced logits capture, EXL3 map descriptors with quality statuses and card numbering by PCI address, and PP5
fits one more 1M session.

### Serving
- **One more 1M session on PP5.** A PP plan sizes its attention-core scratch by a slab and its queries by a span, as
  its programs use them, which leaves 226.25 MiB more for sessions on every PP card; the programs and their outputs
  are unchanged. The release gate's PP5 serve fitted 6 sessions of 1,048,576 tokens (v0.0.3-alpha: 5).
  The engine's placement test at the same free memory gives 6 / 10 / 17 / 24 sessions at 1M / 512K / 256K / 128K
  (v0.0.3-alpha: 5 / 10 / 16 / 22). TP plans are unchanged.
- **Cards numbered by PCI address.** The CUDA backend numbers the visible cards by PCI domain, bus and device, whatever
  `CUDA_DEVICE_ORDER` or the order of a `CUDA_VISIBLE_DEVICES` list says, and prints one line when that differs from
  the driver's order ([INSTALL.md](INSTALL.md#card-placement)). `CUDA_DEVICE_ORDER=PCI_BUS_ID` is no longer needed
  for the PP5 fit.
- **A command-line refusal that was silent names the option and why,** in one `refused: OPTION: WHY` line before the
  usage. The refusals that already gave their reason keep their own wording, and some of them name no option.
- An inserted `--pipeline_steps` yields to an explicit `--batch_rows` (BATCH runs one pipeline step), and the
  `axes from` line says so.

### Loading and storage
- **The Engram autotune stage.** `anemone autotune PACK --stage=engram --tokens=FILE...` tunes serving's Engram step
  I/O (threads per copy, spin and pin: 32 arms) by replaying real token traces at seven levels, with cold reads and a
  byte-identity check of every point against the per-row path, and confirms the best challenger against the default
  with an A D D A. It writes the profile's own `engram` section, which serving applies (knob over profile over default)
  with a tagged `engram step io:` line; its parameters and state are separate from the load stage's. The earlier
  combined tuning is retired: `anemone autotune PACK` without `--stage` runs the copy, load and Engram stages in turn
  ([STORAGE_PROFILE.md](docs/user/STORAGE_PROFILE.md)).
- **Turbo load beside given copies.** Serve runs the replica search beside copies given on the command line too, not
  only beside a lone source. Before, one extra copy path turned the search off: on 15 Nov 2026, with the same binary,
  PP5 given the tables, the pack and one copy read 2 copies in 29.63 s, while the search from the tables and the pack
  alone read 4 in 16.94 s. A load given more than one copy prints one `turbo load: the load reads N copies:` line, and
  `--turbo_load=0` keeps exactly the given copies. A replica on a drive that holds a given copy is refused
  (`duplicate_physical_device`), and a given copy on no described drive (an anonymous `st_dev`, or a medium that names
  no drive: overlay, a loop or virtual block device, FUSE, a network filesystem) ends the search with one
  `unknown_physical_topology` line naming it; a copy on tmpfs or ramfs shares no drive.
- **The load stage in a balanced order.** Its 35 arms run in a fixed balanced order and then reversed, so a linear
  drift cancels in each arm's mean, with a warm-up before the first point, an eight-point A D D A D A A D confirmation
  and, before each point, a wait for every NVMe drive to cool below 60 °C (at most 180 s).
  `tools/measure/load_arena_probe` measures the stage's host read point by point.
- A stale replica receipt that the search passes over is named on a `storage: replica receipt` line.
- `anemone autotune` copes with an unset `HOME`: with neither `XDG_CONFIG_HOME` nor `HOME` it refuses by name for want
  of a default profile.

### EXL3
- **Map descriptors with quality statuses.** At startup the engine reads an EXL3 map's descriptor and checks it against
  the open pack, the model and the tables. The established 2.9 bpw map serves by default: `tools/model_data.sh`
  writes its descriptor with the quality `established`. A map of quality `none`, such as the 2.68 bpw map, serves only
  with `--exl3_unrated_map=1`. The descriptor's SHA-256 goes on the startup line.
- `--exl3_map=NAME` picks `NAME.exl3map` in `--model_dir`'s folder (default `model`).
- One table, `EXL3_WIDTHS`, drives every width set and kernel instantiation; every CUDA module is byte-identical.
- The engine carries a registry of its models (one entry, DeepSeek-V4.1-Flash): an EXL3 pack naming a model the build
  does not carry is refused in one line, and with `--exl3_tc_moe=1` the routed experts are checked against the
  tensor-core MoE's geometry before anything loads.

### Quality and code
- **`anemone capture`,** the engine-native teacher-forced logits capture: each golden case runs generate's whole path
  as a forced continuation, one or several rows a step as decode's verify steps run, and each drawing step's full F32
  head rows are read where the sampler read them, checked against their witness and written in the scorer's layout.
  `tools/golden/golden_capture_score.py` scores a capture against the
  [golden dataset](https://huggingface.co/datasets/anoane/anemone-ds41f1017-golden). The default path is unchanged.
- **REUSE compliance:** `reuse lint` passes on the tree. Every file states its copyright and licence, in its own
  header or through `REUSE.toml`, and every licence text is in `LICENSES/`. The Linux kernel's `.clang-format` keeps
  its header verbatim and is declared GPL-2.0-only.
- The [option catalogue](tools/cli/KNOBS.md) covers the `anemone autotune` and `anemone capture` options.
- `startup_axes_test` keeps its Engram store under `TMPDIR`, so it passes from any working directory, and the host
  tests free what they hold on their no-kernel paths.
- **Proven changes:** every merge had CPU proof on the reference machine: every program built in release, debug,
  thread and profile with GCC 14.2 (native and cross), the host suites under the sanitizers, mutants of the changed
  checks, object identity outside the intended changes and the dry-run sweep identical except its named texts, plus an
  adversarial review of the code. The [release gate](#release-gate)'s identity check compares the release with
  v0.0.3-alpha (source `f57d813`) on the three GPU splits' gated routes (PP5, TP4_PRO and TP4_EXL3; one user, 15
  streams each).

### Upgrade from v0.0.3-alpha
- **An EXL3 map made by v0.0.3-alpha's `tools/model_data.sh`** has a descriptor of quality `none` beside it, which
  v0.0.4-alpha refuses without `--exl3_unrated_map=1`. Rewrite the descriptor with the converter as v0.0.4-alpha's
  `tools/model_data.sh` calls it. The converter never writes over a map or a descriptor, so it writes into a new
  folder; the map it makes there is the same bytes as yours, and its descriptor then goes beside your map. `DATA` is
  the data folder your map is in, by the absolute path it was made with (the `tools/model_data.sh` destination),
  `EXL3` the EXL3 source folder, `NEW` a folder that does not exist yet and `PYTHON` the model-data environment's
  Python ([MODEL_DATA.md](docs/models/deepseek_v41_flash/MODEL_DATA.md)). From the v0.0.4-alpha repository root:

  ```sh
  mkdir "$NEW"
  "$PYTHON" tools/convert/make_virtual_pack.py --native "$DATA/original.anemone" --hybrid "$DATA/hybrid" \
    --exl3-config "$EXL3/quantization_config.json" \
    --quality-established "LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw @ 0be63c33, TP4_EXL3's default map: qualified by the release gates' TP4_EXL3 identity checks (RELEASE_NOTES.md) and its golden metrics (docs/results/GOLDEN_METRICS.md)" \
    --output "$NEW/model.exl3map"
  cmp "$NEW/model.exl3map" "$DATA/model.exl3map" &&
    mv "$DATA/model.exl3map.descriptor.json" "$DATA/model.exl3map.descriptor.json.v0.0.3-alpha" &&
    mv "$NEW/model.exl3map.descriptor.json" "$DATA/model.exl3map.descriptor.json"
  ```

  The converter only reads `DATA` and `EXL3` (the routed experts again: about 2 minutes on the reference machine) and
  writes only into `NEW`. If `cmp` reports a difference, the map was not made from these paths: nothing is moved, and
  v0.0.4-alpha's `tools/model_data.sh` makes the map again. The engine reads only `model.exl3map.descriptor.json`, so
  it ignores the kept v0.0.3-alpha descriptor, and startup prints `exl3 descriptor: ... quality established (...)`.
  `NEW` can then be removed.

### Limits
- Alpha software. The exact profile stays bounded to one session of 512 tokens. Vision is experimental. The comparison
  tables in the README keep their 6 November measurements, within the v0.0.2-alpha [measured scope](#measured-scope).
- The 2.68 bpw EXL3 map has no quality claim yet; it serves only on `--exl3_unrated_map=1`.
- At 256K, PP5's 17th session leaves the first card about 7.7 MiB of its budget: a little less free memory there at
  startup gives 16.
- Golden scores for all 435 cases, with `anemone capture`, follow in a later release.

### Known issues
Two help texts built into the executable are out of date; the next release corrects them:
- `anemone help axes` gives PP5's `--concurrency=multiple` row "5 at 1M", v0.0.3-alpha's count, and so do
  `tools/cli/AXES.md`, `tools/cli/KNOBS.md` and `tools/cli/axes.json`. On v0.0.4-alpha PP5 fits
  6 sessions at 1M ([USAGE.md](docs/USAGE.md)).
- `anemone help knobs points_per_run` and `tools/cli/KNOBS.md` give `--points_per_run`'s load-stage range as 0..39
  ("the load stage's grid of 35, then its A D D A's 4"). The load stage takes 0..78: its 35 arms run twice (70
  points), then its eight-point A D D A D A A D confirmation. The Engram stage's 0..72 is right.

### Release gate
The release executable ([VALIDATION.md](docs/VALIDATION.md)) was gated on the reference machine before the tag, on
17 Nov 2026.
- **Cards:** `backend_cuda_card_test` passed in debug and release under `CUDA_DEVICE_ORDER` unset, `FASTEST_FIRST` and
  `PCI_BUS_ID`, and the EXL3 rotation, EXL3 TCP forms and top-k card tests passed. The plain PP5 serve fitted the same
  sessions under the three orders, and the 2.68 bpw map was refused without `--exl3_unrated_map=1` and served with it.
- **Identity:** on PP5, TP4_PRO and TP4_EXL3, one user and 15 streams a route, with each serve's given copies kept
  (`--turbo_load=0`), its outputs matched v0.0.3-alpha (source `f57d813`): PP5 15/15, TP4_PRO
  15/15, TP4_EXL3 15/15; and the earlier production build (source `84858e7`): PP5
  15/15, TP4_PRO 15/15, TP4_EXL3 15/15. On PP5 with the replica search
  beside the same given copies (the default, no `--turbo_load=0`), its outputs matched v0.0.3-alpha's with the copies
  kept: 15/15.
- **1M sessions:** on a PP5 serve with `--session_slots=max`, `--context_tokens=1048576`, `--generated_tokens=1048576`,
  `--temperature=1` and `--top_p=1` (the requests at T=0), the startup fit admitted 6 session slots of
  1,048,576 tokens each. Served: one fresh 1M-token session, HTTP 200, the needle found, prefill 3076 tok/s (1,022,969 prompt tokens in 332.6 s), decode 12.8 ms/token; four at once,
  HTTP 200 for all 8 requests, all 4 needles found, prefill about 766 tok/s each (1333.4 to 1336.8 s), decode 238.8 tok/s in all (4.19 ms/token).
- **Fraser:** the same serve with `--context_tokens=524288` and `--generated_tokens=524288`, T=0, thinking off, 2,048
  tokens, 1 user, 10 repetitions: median 171.1 tok/s
  (5.84 ms/token), best 182.0 tok/s (5.49 ms/token).

## v0.0.3-alpha

Anemone v0.0.3-alpha is a C11/CUDA engine for DeepSeek-V4.1-Flash-1017, on four CMP 170HX cards for EXL3, and on four
CMP 170HX plus an RTX PRO 6000 Blackwell for the original weights. The [README](README.md) has the overview, the
1M-session table and the first-time setup.

### Serving
- **1M tokens for every session.** `--context_tokens` defaults to 1,048,576 and `--session_slots` to `max`: the
  startup fit serves as many full-context sessions as the cards hold, and is never shrunk automatically.
- **GPU split plus four axes.** `--placement` (PP5, TP4_PRO, TP4_EXL3) picks the split. `--category`, `--exactness`,
  `--concurrency` and `--focus` each carry a measured knob set per split. `--weights` and `--model_dir` name the weight
  family and the model folder.
- **Opt-in Anthropic `POST /v1/messages`**, with thinking controls and the same `/metrics` accounting as chat
  completions.
- **Opt-in Prometheus `/metrics`**, matching vLLM V1's metric names.

### Loading and storage
- **Turbo load.** Verified replicas on every local NVMe drive are found by their receipts and read in parallel with
  O_DIRECT and io_uring. The TP host arena steals work between copies. `anemone replicate --adopt` admits an existing
  copy after a full digest check.
- **Turbo Engram.** The Engram tables stay in RAM when they fit. Otherwise `auto` keeps the largest share that does and
  reads the rest from NVMe a step at a time on io_uring threads per copy, every block verified and a stalled read
  retried on another copy. The decision and its budget are printed at startup.
- **`anemone autotune` in stages.** `--stage=copy` prepares and verifies the replicas (24 workers by default);
  `--stage=load` tunes the TP host load into a profile section, enabled by `--qualify`.
- **New load defaults** ([STORAGE_PROFILE.md](docs/user/STORAGE_PROFILE.md)):
  - `anemone autotune --copy_hash_pairs=1` hashes the copy's source two chunks at a time, with the same digests:
    96 GiB to one target copied at 4.93 and 5.02 GB/s against 4.84 and 5.00 with one hash a chunk. `=0` is the
    earlier default.
  - The PP loader reads each copy with 4 readers (`--reader_threads_per_copy=4`, from 2): real PP5 serve loads took
    16.442 and 16.399 s against 16.854 and 17.207 s with 2, +3.58 % beyond a 2.07 % spread
    ([USAGE.md](docs/USAGE.md#knob-reader_threads_per_copy)).
  - `--load_piece_mib` stays 16 MiB on PP and TP: 64 MiB was no faster on PP5 loads, and TP4_PRO's tuned candidate
    loaded 1.54 % slower than the defaults ([KNOBS.md](tools/cli/KNOBS.md)).

### EXL3
- EXL3 widths K2, K2.5, K3 and K4, read per matrix from the pack. The 2.68 bpw map fits 2 sessions of 1M on TP4_EXL3;
  it is opt-in, with no quality claim yet.
- The EXL3 expert kernels are also built for the RTX PRO 6000.
- The converter admits a checkpoint by validation and writes a descriptor beside each map, with its provenance and a
  quality status.

### Quality and code
- **All linters clean, with no exceptions:** on a clean checkout `make analysis` (cppcheck at the exhaustive level,
  clang-tidy with every warning an error) exits 0, and the layout, literal and MUST_CHECK lints report nothing. There
  is no suppression and no list of accepted findings: the last named ones were fixed in code (the libFuzzer entry
  points are asm labels, a test counts allocations with its own allocator).
- **Lint cleanup, behaviour-neutral:** the source is rewrapped to 120 columns, the cppcheck and clang-tidy findings
  are fixed in code and padded structs are reordered. Every cleanup commit kept its objects identical, or its outputs
  identical on the three GPU splits' gated routes (PP5, TP4_PRO and TP4_EXL3; one user, 15 streams each).
- **`source/commands.c`:** the program's code moved out of `source/main.c` into `source/commands.c` and
  `source/commands.h`, a unit linked into the program and every test like any other source file. `main.c` keeps only
  `main()`, and no test includes it. This is the first slice of the source reorganisation.
- **Golden reference:**
  [anoane/anemone-ds41f1017-golden](https://huggingface.co/datasets/anoane/anemone-ds41f1017-golden)
  holds 435 cases and 66,409 positions, with the full F32 heads and a compact top-64 reference.
  `tools/golden_score.py --compact` scores NLL, perplexity, top-1 and top-k agreement and certain-token agreement
  against it.
- **Proven changes:** the lint and cleanup merges kept objects or outputs identical; every merge ran under
  sanitizers with an adversarial review; the default and feature changes are in their own sections above. The
  [release gate](#release-gate-1)'s identity check compares the release with the last GPU-gated build (source
  `288fa1e`) on the three GPU splits' gated routes (PP5, TP4_PRO and TP4_EXL3; one user, 15 streams each).
- [CONTRIBUTING.md](CONTRIBUTING.md) sets the bar for pull requests and issues.

### Limits
- Alpha software. The exact profile stays bounded to one session of 512 tokens. Vision is experimental. The comparison
  tables in the README keep their 6 November measurements, within the v0.0.2-alpha [measured scope](#measured-scope).
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` is required: without it the RTX PRO 6000 can be enumerated first and the PP5 startup
  fit fails with a message that does not name the cause ([INSTALL.md](INSTALL.md#card-placement)). The engine does
  not yet order the cards by itself.
- `make host_tests` needs `HOME` set: under `env -i` without it, a case of `replica_prepare_test` fails, so
  [BUILD.md](BUILD.md)'s command gives it an empty folder.

### Release gate
The release executable ([VALIDATION.md](docs/VALIDATION.md)) was gated on the reference machine before the tag.
- **Identity:** on PP5, TP4_PRO and TP4_EXL3, one user and 15 streams a route, its outputs matched the last gated build
  before the release commit (source `288fa1e`): PP5 15/15, TP4_PRO 15/15, TP4_EXL3 15/15; and the earlier production
  build (source `84858e7`): PP5 15/15, TP4_PRO 15/15, TP4_EXL3 15/15.
- **1M sessions:** on a PP5 serve with `--session_slots=max`, `--context_tokens=1048576`, `--generated_tokens=1048576`,
  `--temperature=1` and `--top_p=1` (the requests at T=0), the startup fit admitted 5 session slots of 1,048,576 tokens
  each. Served: one fresh 1M-token session, HTTP 200, the needle found, prefill 3087 tok/s (1,022,969 prompt tokens in
  331.3 s), decode 12.9 ms/token; four at once, HTTP 200 for all 8 requests, all 4 needles found, prefill about 766
  tok/s each (1333.5 to 1336.9 s), decode 238.1 tok/s in all (4.20 ms/token).
- **Fraser:** the same serve with `--context_tokens=524288` and `--generated_tokens=524288`, T=0, thinking off, 2,048
  tokens, 1 user, 10 repetitions: median 173.2 tok/s (5.78 ms/token), best 179.4 tok/s (5.57 ms/token).

## v0.0.2-alpha

Anemone provides a C11/CUDA DeepSeek-V4.1-Flash-1017 engine targeting four CMP
170HX cards for EXL3, and four CMP 170HX cards plus an RTX PRO 6000 Blackwell
for original weights. See [README](README.md), [BUILD](BUILD.md),
[INSTALL](INSTALL.md), [model preparation](docs/models/deepseek_v41_flash/MODEL_DATA.md) and
[the complete server recipes](docs/models/deepseek_v41_flash/SERVER_PROFILES.md).

This public layout keeps the full comparison matrix, concise benchmark methods,
source-review notes and curated machine-readable results. Internal handoffs,
development diaries, raw workstation receipts and duplicate historical documents
are archived separately. Complete build/generator source is included; executable
Python tools use explicit `.py` filenames, and build outputs are not shipped.
This snapshot contains no Git metadata or history.

Licensing metadata, version/startup attribution and five corrected help entries
preserve the existing inference defaults and CUDA payloads. All serve options
are documented. The Pi template uses the requested model display name and endpoint/
key placeholders, with private user-list creation and Linux getrandom guidance.
Required author attribution and the original license terms remain intact.

### Measured scope

The [full matrix](docs/models/deepseek_v41_flash/results/COMPARISON_MATRIX.md) retains all dated comparisons
and source/binary identities. Current non-golden coverage is 42 attempts from an
86-job catalogue: 40 passes, one TP4 mixed identity failure and one unqualified
capacity cleanup attempt, with 44 jobs unattempted. Both failed/unqualified rates
are withheld. The separate prompt-chunk guard and PP5 mixed tests pass 96/96.

Six Curtis jobs, four functional snapshot jobs, two actual 100K-class pilots and
two Fraser reasoning jobs pass their bounded protocols. The 100K prompts contain
99,972 native input tokens plus one output, with two fresh repetitions per
balanced format. Q timings with CPU overlap are not performance results.
Fraser's short diagnostics and one-token restoration do not establish full
quality or logical-state equivalence. Exact is one session and 512 total tokens;
original balanced output depends on native PRO layer placement.

Current full golden replay remains 0/435 cases per balanced format, and actual
Pi remains unrun. Historical 54-case conditional-head scores keep their date
and denominator. Needles, soak, seeded isolation, other unexecuted protocols,
fixture gates and unsupported capabilities remain explicit. Source reviews are
not GPU tests. No extra HumanEval/GSM8K acquisition or evaluation is included.

The public-package companion manifest records actual source-only build checks,
archive identities and executable hashes; [VALIDATION.md](docs/VALIDATION.md)
explains their scope. No inference optimization or new GPU benchmark is implied
by this repository cleanup.
