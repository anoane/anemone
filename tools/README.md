<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Source tools

Every program in this directory is readable source. No prebuilt ELF, Mach-O,
Windows executable, library or CUDA binary is checked in. Executable Python
scripts have shebangs and `.py` names; C/CUDA helpers are compiled into `build/`.
Hex and JSON fixture inputs are data with their generator source included.

| Directory or program | Purpose and dependencies |
| --- | --- |
| `build/` | Native C kernel embedder and host-fixture writer; optional Python regeneration helpers. |
| `cli/` | Axis/option registries and explicit Python help generators. |
| `check/` | Optional Python lint and release instrumentation audits. |
| `convert/`, `model_data.sh` | CPU model/tokenizer/pack preparation; dependencies in MODEL_DATA. |
| `golden_score.py` | Portable aligned F32 conditional-head scorer; Python and NumPy. |
| `golden/` | Optional capture/fixture preparation source; caller-supplied models/reference data. |
| `measure/` | Optional C/CUDA/Python measurement probes, built from their source. |
| `oracle/` | Optional CPU reference and capture comparisons; Python/NumPy/PyTorch as used by each tool. |
| `prefill_cone/` | Optional prefill comparison source. |
| `vision/` | Vision fixture generation/comparison and pinned numeric input data. |
| `development.mk` | Explicit development targets included by the main Makefile. |

Use [MODEL_DATA.md](../docs/user/MODEL_DATA.md) for model preparation and
[golden_score.md](golden_score.md) for score input contracts. For example:

```sh
python3 tools/golden_score.py GOLDEN_CASE CANDIDATE_CASE --output=NEW_SCORE_JSON
python3 tools/golden_score_test.py
```

Release compilation and inference do not depend on these optional Python
analysis tools. Model weights and captures are caller-supplied external data;
no private corpus or workstation-specific publication tool is included.

GPU probes require reserved devices and an operator-controlled execution guard.
They are separate from native host tests and are never started by the source
build. Source build, generators and checks are described in
[DEVELOPMENT.md](../docs/DEVELOPMENT.md). Keep outputs in ignored `build/` paths.
