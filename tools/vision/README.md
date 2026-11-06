<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
The fixtures call the unmodified official `inference/vision.py` with BF16 activations and synthetic weights.
The numerical target is the official GPU default in the recorded PyTorch environment, pinned explicitly to
`SDPBackend.FLASH_ATTENTION`. Generation first runs the unpinned official call and requires identical BF16 output
from the pinned call. No checkpoint is needed for the small fixtures.

The dispatch probe records actual operators and CUDA kernel names, PyTorch version/commit, device, input strides,
backend controls, and production RoPE frequency/trigonometric words. On the measured PyTorch 2.14.1+cu130 build
(`5c4886908584029761b579af026dcfb627c84070`), CPU chooses
`aten::_scaled_dot_product_flash_attention_for_cpu`; both CMP 170HX and RTX PRO 6000 choose CUDA Flash. The small
head width 8 is padded to kernel traits `<32,128,128,4>`; production head width 64 uses `<64,128,128,4>`.
Installed PyTorch dispatch normalizes the official rank-three inputs to a singleton batch for fused backends.
Rank three therefore does not imply a MATH fallback in this environment.

CPU Flash and CUDA Flash have distinct arithmetic. In the original 4x5 case, CPU produces -0.00982666015625 at
projector element 25, while both GPUs produce -0.0113525390625. The latter agrees with the C kernel-rounding
oracle. All four GPU fixture outputs match between CMP and PRO bit for bit; the CPU/GPU 4x5 outputs
differ in 40 of 64 BF16 words. Pinning MATH selects a different numerical target. Its files remain a separate
mathematical control.

The complete two-layer fixtures have width 16, two heads, intermediate width 12, language width 16, patch size 2
and downsample ratio 3. Grids are 1x1, 2x3, 4x5 and 11x12. The last has 132 patches and crosses the observed
128-key boundary. It exercises reverse key-tile traversal, online maximum/output rescaling, four FP32 denominator
partials, log2 exponentials and per-tile BF16 probability rounding. The tensor recipe is shared with
`tests/vision_fixture.c`. Case 1 exercises packed BF16 norm weights widened to FP32; other cases use FP32 norm
weights. Every stored norm value is identical across the two layouts.

Both `vision_test` and `vision_card_test` require bitwise agreement with the small GPU Flash goldens. All four
cases passed that stronger boundary on CMP and PRO before it was enabled. PyTorch's BF16 dtype bound, relative
0.016 plus absolute 0.00001, remains the separate original-weight diagnostic bound. The host test also runs
the independent double mathematical control against MATH files at relative 0.0078125 plus absolute
0.000244140625, and corrupts a primary output word to prove the required comparator refuses it. Passing small
fixtures is one gate; the original-weight scope is recorded below.

The CUDA implementation follows the measured GPU arithmetic. Flash uses native BF16 QK/PV MMA, reverse
128-key tiles, online FP32 state, BF16 probability rounding and an explicit fused first denominator update.
Dense products follow the pinned default GEMM dispatch: fresh FP32 accumulation per K partition, then a BF16
carry between serial partitions. K16 is the usual instruction; first-projector odd cell counts select the
observed s1688 family's pair of ascending BF16 K8 instructions. Partition widths round to the measured K tile,
normalized to K32 wherever that gives the same width; the PRO eleven-part projector needs K64 and width 896.
Bias belongs to the first or final partition according to the recorded epilogue family. In either case it is
added to that partition's fresh FP32 accumulator before the previous BF16 carry. Residual K prefixes run first.

RMSNorm reproduces the reference's vector-four accumulation and reduction order. Validated BF16 norm weights
are widened exactly before FP32 multiplication; F32 norm storage remains supported. All 65 original-pack norms
are BF16. RoPE uses the measured FP32 inverse frequencies and a bounded F64 trigonometric helper followed by
packed signed-ULP correction data. The complete default-theta domain is positions 0–3,062, sixteen frequencies
and both sine/cosine: 98,016 FP32 values, all bitwise equal to the reference on both cards. This data describes
function inputs independently of image pixels, model parameters and output fixtures.

`fixtures/gemm_policy_manifest.json` pins four dispatch-map hashes, the numeric JSON hash, reference source and
PyTorch epoch. The six-product tables cover 69,004 device/role/row groups; independent reconstruction checks
103,506 class/role/row selections, including K8, with zero differences. This is dispatch metadata coverage.
Original-weight numerical captures cover grids 39x39 (1,521 patches/169 cells) and 42x216 (9,072 patches/1,008
cells) on CMP and PRO. All 1,432 captured passes, comprising 14,853,476,352 BF16 intermediate words, and all four
normal final outputs match bit for bit. The shared digest of all 263 tower/projector parameters is recorded in
the manifest. These captures qualify those complete inputs; they are separate from the synthetic fixtures.

GEMM boundaries and RoPE corrections are committed as numeric data. Build tools validate their names, ranges
and counts, then emit only numeric declarations under `build/`. Policy control and descriptors remain
hand-written C; generated code is absent from the source tree. Regenerate and requalify the data after changing
the reference/library epoch, reduction options, trigonometric helper or frequencies.

Run all CUDA payloads inside the owner's runner, exposing one card at a time. Use absolute packet paths because
the runner changes its working directory:

```sh
GPU_GUARD=/absolute/path/to/your-gpu-guard
"$GPU_GUARD" python3 tools/vision/make_fixtures.py \
  --device cuda --reference /path/to/official/inference \
  --output build/vision/reference --trace-directory build/vision/traces
```

Use `probe_attention.py --device cuda --reference reference --output dispatch_cmp` in the same guarded context.
CPU probes/controls execute outside that runner with `CUDA_VISIBLE_DEVICES=` and a separate output directory.
Never overwrite the GPU primary fixtures with CPU outputs. Fixture hex and manifests are ordinary source data;
the manifest records the producing card, dtype, reference hash, PyTorch version and output hashes.

`vision_test --trace` prints the C Flash-rounding passes; `--trace-math` prints the mathematical control.
`vision_card_test --trace` captures corresponding device passes. The ordinary card test checks unchanged
inputs/weights and requires exactly one approved CMP or PRO with `UNDER_GPU_RUN=1`.

The optional original-weight check loads only the tower/projector stage from the pack:

1. With cards hidden, run `original_fixtures.py prepare --reference reference --image IMAGE --output PATCHES`.
   This uses official image preprocessing and writes grid dimensions followed by BF16 patches.
2. Inside the operator’s GPU guard, run `vision_card_test --original PACK PATCHES CANDIDATE`. It warms up once and records
   three full encoder forwards, flushing the fixed scratch range before each timed forward.
3. Inside the operator’s GPU guard, run `original_fixtures.py reference --device cuda --backend flash --reference reference
   --pack PACK --patches PATCHES --output REFERENCE --candidate CANDIDATE`. It checks pinned/default equality,
   reports every output mismatch against the original BF16 dtype bound, and refuses failures.

For diagnosis, `original_fixtures.py reference --trace-directory DIR` saves every official pass and requires
the traced final output to equal the unmodified call. Set `VISION_TRACE_DIRECTORY` to an existing directory for
the C helper's replay after timing. `compare_traces.py --reference REF_DIR --candidate C_DIR` is a host-only
comparison reporting the first bitwise difference separately from the first dtype-tolerance failure.
The reference trace records operation names, shapes, byte counts and hashes. Trace transfers are outside timing.
