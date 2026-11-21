#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Guarded direct-GEMM dispatch map using original weights and captured real ViT inputs.

No reference flags are changed. Ordinary linear inputs are contiguous; aligner input preserves official
unfold's transposed stride (1, rows). --full profiles every row count through the fixed encoder capacities.
All GPU execution must be inside gpu_run.sh with exactly one exposed card.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import struct

import original_fixtures as original

PATCH_ROW_CAPACITY = 9216
CELL_ROW_CAPACITY = 1024
PROFILE_BATCH_ROWS = 64
DEADLINE_SECONDS = 900


def parse_rows(value):
    rows = set()
    for part in value.split(","):
        bounds = [int(item) for item in part.split(":")]
        if len(bounds) == 1:
            rows.add(bounds[0])
        elif len(bounds) in (2, 3):
            step = bounds[2] if len(bounds) == 3 else 1
            if step <= 0:
                raise ValueError("row step must be positive")
            rows.update(range(bounds[0], bounds[1] + 1, step))
        else:
            raise ValueError("rows require comma-separated numbers or inclusive start:stop:step ranges")
    return sorted(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mlp-rows", default="1510,1521,2048,4096,8192,9189")
    parser.add_argument("--aligner-rows", default="168,169,256,512,1021")
    parser.add_argument("--ordinary-rows", default="1022,1521,4096,9189")
    parser.add_argument("--aligner-output-rows", default="164,169,512,1021")
    parser.add_argument("--kinds", default="mlp,aligner",
                        help="comma-separated patch,qkv,attention_output,mlp_up,mlp,aligner,aligner_output")
    parser.add_argument("--patches", type=Path, help="official patch file, required only for the patch kind")
    parser.add_argument("--full", action="store_true")
    arguments = parser.parse_args()
    if os.environ.get("UNDER_GPU_RUN") != "1":
        raise RuntimeError("direct GEMM probe requires gpu_run.sh")
    signal.alarm(DEADLINE_SECONDS)
    import torch
    import torch.nn.functional as functional
    from torch.profiler import ProfilerActivity, profile, record_function
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one card")
    properties = torch.cuda.get_device_properties(0)
    if "170HX" not in properties.name and "RTX PRO 6000" not in properties.name:
        raise RuntimeError("unsupported physical card")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    definitions = {
        "patch": {"weight": "vision.patch_embed.proj.weight", "bias": "vision.patch_embed.proj.bias",
                  "input": None, "row_argument": arguments.ordinary_rows, "capacity": PATCH_ROW_CAPACITY},
        "qkv": {"weight": "vision.blocks.0.attn.wqkv.weight", "bias": "vision.blocks.0.attn.wqkv.bias",
                "input": "blocks.0.norm1", "row_argument": arguments.ordinary_rows, "capacity": PATCH_ROW_CAPACITY},
        "attention_output": {"weight": "vision.blocks.0.attn.wo.weight", "bias": "vision.blocks.0.attn.wo.bias",
                             "input": "blocks.0.attn.attention", "row_argument": arguments.ordinary_rows,
                             "capacity": PATCH_ROW_CAPACITY},
        "mlp_up": {"weight": "vision.blocks.0.mlp.w1.weight", "input": "blocks.0.norm2",
                   "row_argument": arguments.ordinary_rows, "capacity": PATCH_ROW_CAPACITY},
        "mlp": {"weight": "vision.blocks.0.mlp.w2.weight", "input": "blocks.0.mlp.silu_up",
                "row_argument": arguments.mlp_rows, "capacity": PATCH_ROW_CAPACITY},
        "aligner": {"weight": "aligner.w1.weight", "bias": "aligner.w1.bias", "input": "unfold",
                    "row_argument": arguments.aligner_rows, "capacity": CELL_ROW_CAPACITY, "column_input": True},
        "aligner_output": {"weight": "aligner.w2.weight", "bias": "aligner.w2.bias", "input": "aligner.gelu",
                           "row_argument": arguments.aligner_output_rows, "capacity": CELL_ROW_CAPACITY}}
    kinds = arguments.kinds.split(",")
    if not kinds or len(set(kinds)) != len(kinds) or any(kind not in definitions for kind in kinds):
        raise ValueError("unknown, empty or duplicate matrix kind")
    if "patch" in kinds and arguments.patches is None:
        raise ValueError("the patch kind requires --patches")
    specifications = {kind: definitions[kind] for kind in kinds}
    for specification in specifications.values():
        specification["rows"] = (list(range(1, specification["capacity"] + 1)) if arguments.full
                                  else parse_rows(specification["row_argument"]))
    selected = {item[field] for item in specifications.values() for field in ("weight", "bias") if field in item}
    weights, weight_hashes = {}, {}
    for name, dtype, shape, payload in original.pack_parameters(arguments.pack):
        if name not in selected:
            continue
        if dtype != 1:
            raise ValueError(f"expected original BF16 matrix/bias: {name}")
        weights[name] = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(shape).to("cuda")
        weight_hashes[name] = hashlib.sha256(payload).hexdigest()
    if set(weights) != selected:
        raise ValueError("required matrix weights missing from pack")
    manifest = json.loads((arguments.reference_trace / "manifest.json").read_text())
    result = {"torch": torch.__version__, "torch_git": torch.version.git_version,
              "cuda": torch.version.cuda, "device": properties.name,
              "capability": list(torch.cuda.get_device_capability(0)),
              "multiprocessors": properties.multi_processor_count, "dtype": "bfloat16",
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "matmul_allow_bf16_reduced_precision_reduction":
                  torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "reference_flags_changed": False, "weight_sha256": weight_hashes,
              "reference_sha256": manifest["reference_sha256"], "input_sha256": {},
              "kernel_launches": [], "chrome_traces": [], "coverage": {}, "matrix_layouts": {}}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for kind, specification in specifications.items():
        rows = specification["rows"]
        if not rows or rows[0] <= 0 or rows[-1] > specification["capacity"]:
            raise ValueError(f"row count outside fixed {kind} capacity")
        weight = weights[specification["weight"]]
        bias = weights[specification["bias"]] if "bias" in specification else None
        if kind == "patch":
            patch_file = arguments.patches.read_bytes()
            if len(patch_file) < 8 or hashlib.sha256(patch_file).hexdigest() != manifest["patch_sha256"]:
                raise ValueError("patch header or reference digest mismatch")
            height, width = struct.unpack_from("<II", patch_file)
            if (not height or not width or height * width > PATCH_ROW_CAPACITY or
                    len(patch_file) != 8 + height * width * weight.shape[1] * 2):
                raise ValueError("incompatible patch payload")
            payload = patch_file[8:]
        else:
            entry = next(item for item in manifest["passes"] if item["operation"] == specification["input"])
            payload = (arguments.reference_trace / entry["file"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError("reference input digest mismatch")
        result["input_sha256"][kind] = hashlib.sha256(payload).hexdigest()
        result["matrix_layouts"][kind] = {"weight_shape": list(weight.shape), "weight_stride": list(weight.stride()),
            "bias_shape": list(bias.shape) if bias is not None else None,
            "bias_stride": list(bias.stride()) if bias is not None else None,
            "source_layout": "column_major" if specification.get("column_input") else "row_major"}
        source = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(-1, weight.shape[1]).to("cuda")
        repeated = source.repeat(((rows[-1] + source.shape[0] - 1) // source.shape[0], 1))[:rows[-1]]
        column_storage = (torch.empty(rows[-1] * weight.shape[1], dtype=torch.bfloat16, device="cuda")
                          if specification.get("column_input") else None)
        mapped_rows = set()
        row_layouts = {}
        for batch_begin in range(0, len(rows), PROFILE_BATCH_ROWS):
            batch = rows[batch_begin:batch_begin + PROFILE_BATCH_ROWS]
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as captured:
                for count in batch:
                    if specification.get("column_input"):
                        column_matrix = column_storage[:count * weight.shape[1]].view(weight.shape[1], count)
                        column_matrix.copy_(repeated[:count].transpose(0, 1))
                        source_matrix = column_matrix.transpose(0, 1)
                    else:
                        source_matrix = repeated[:count]
                    with record_function(f"vision_dispatch:{kind}:{count}"):
                        output = functional.linear(source_matrix, weight, bias)
                    row_layouts[count] = {"input_stride": list(source_matrix.stride()),
                        "input_shape": list(source_matrix.shape), "output_stride": list(output.stride()),
                        "output_shape": list(output.shape), "input_dtype": str(source_matrix.dtype),
                        "input_alignment_mod256": source_matrix.data_ptr() % 256,
                        "weight_alignment_mod256": weight.data_ptr() % 256}
                    del output
                torch.cuda.synchronize()
            prefix = arguments.output.with_name(f"{arguments.output.stem}.{kind}.{batch_begin:05d}.json")
            trace_path, launches = original.profile_launches(captured, prefix)
            result["chrome_traces"].append(trace_path)
            for launch in launches:
                matrices = [shape for shape in launch.get("input_shapes") or [] if len(shape) == 2]
                if (len(matrices) != 2 or matrices[0][1] != weight.shape[1] or
                        matrices[1] != [weight.shape[1], weight.shape[0]] or matrices[0][0] not in batch):
                    continue
                count = matrices[0][0]
                mapped_rows.add(count)
                result["kernel_launches"].append({"kind": kind, "rows": count,
                    **row_layouts[count], **launch})
        result["coverage"][kind] = {"requested": rows, "mapped": sorted(mapped_rows),
                                    "complete": mapped_rows == set(rows)}
        del repeated, source, column_storage
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    complete = all(item["complete"] for item in result["coverage"].values())
    print(json.dumps({"output": str(arguments.output), "kernel_launches": len(result["kernel_launches"]),
                      "chrome_traces": len(result["chrome_traces"]), "complete": complete}))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
