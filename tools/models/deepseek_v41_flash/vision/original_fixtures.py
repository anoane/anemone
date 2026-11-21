#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Original-weight ViT only: official preprocessing, pinned official forward, candidate output comparison.

prepare runs on the host with CUDA_VISIBLE_DEVICES=. Every CUDA invocation must run inside gpu_run.sh.
No language-model weights are loaded, and no source checkpoint or pack is modified.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace

PACK_HEADER_BYTES = 128
PACK_BLOCK_BYTES = 4096
MAX_TABLE_BYTES = 1024 ** 3
MAX_TOWER_TENSOR_BYTES = 128 * 1024 ** 2
MAX_PATCHES = 9216
MAX_IMAGE_TOKENS = 1024


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def event_kernel_names(event):
    names = [kernel.name for kernel in getattr(event, "kernels", ())]
    for child in getattr(event, "cpu_children", ()):
        names.extend(event_kernel_names(child))
    return names


def profile_launches(profiler, output):
    trace_path = output.with_suffix(output.suffix + ".profile.json")
    profiler.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text())["traceEvents"]
    matrix = [event for event in events if event.get("cat") == "cpu_op" and
              event.get("name") in ("aten::mm", "aten::addmm", "aten::bmm")]
    linear_scopes = [event for event in events if event.get("name", "").startswith("vision_linear:")]
    by_external = {str(event["args"]["External id"]): event for event in matrix
                   if event.get("args", {}).get("External id") is not None}
    runtime = {str(event.get("args", {}).get("correlation")): event for event in events
               if event.get("cat") in ("cuda_runtime", "cuda_driver") and
               event.get("args", {}).get("correlation") is not None}
    launches = []
    for event in events:
        if event.get("cat") != "kernel":
            continue
        fields = event.get("args", {})
        owner = by_external.get(str(fields.get("External id")))
        if owner is None:
            launch = runtime.get(str(fields.get("correlation")))
            if launch is not None:
                parents = [item for item in matrix if item.get("pid") == launch.get("pid") and
                           item.get("tid") == launch.get("tid") and
                           item["ts"] <= launch["ts"] <= item["ts"] + item.get("dur", 0)]
                if parents:
                    owner = min(parents, key=lambda item: item.get("dur", float("inf")))
        scope = None
        if owner is not None:
            parents = [item for item in linear_scopes if item.get("pid") == owner.get("pid") and
                       item.get("tid") == owner.get("tid") and
                       item["ts"] <= owner["ts"] <= item["ts"] + item.get("dur", 0)]
            if parents:
                scope = min(parents, key=lambda item: item.get("dur", float("inf")))
        launches.append({"kernel": event["name"], "grid": fields.get("grid"), "block": fields.get("block"),
                         "external_id": fields.get("External id"), "correlation": fields.get("correlation"),
                         "operator": owner["name"] if owner is not None else None,
                         "input_shapes": owner.get("args", {}).get("Input Dims") if owner is not None else None,
                         "linear_parameter": scope["name"].removeprefix("vision_linear:")
                                             if scope is not None else None})
    return str(trace_path), launches


def settings(reference):
    config = json.loads((reference / "config.json").read_text())
    vision = config["vision_config"]
    return SimpleNamespace(vision_n_layers=vision["num_hidden_layers"], vision_dim=vision["hidden_size"],
        vision_n_heads=vision["num_attention_heads"], vision_inter_dim=vision["intermediate_size"],
        vision_patch_size=vision["patch_size"], vision_rope_theta=vision["rope_theta"],
        vision_downsample_ratio=vision["downsample_ratio"], vision_max_n_token=vision["max_image_tokens"],
        vision_min_pixels=vision["min_pixels"], vision_max_wh_ratio=vision["max_wh_ratio"],
        dim=config["text_config"]["hidden_size"])


def prepare(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("host preparation requires CUDA_VISIBLE_DEVICES=")
    import torch
    processor = load_module("original_image_processor", args.reference / "inference/image_processor.py")
    patches, height, width, llm_height, llm_width = processor.load_image({"data": args.image.read_bytes()},
                                                                       settings(args.reference))
    payload = struct.pack("<II", height, width) + patches.contiguous().view(torch.uint16).numpy().tobytes()
    args.output.write_bytes(payload)
    print(json.dumps({"image": str(args.image), "grid": [height, width], "cells": [llm_height, llm_width],
                      "patch_file": str(args.output), "patch_sha256": hashlib.sha256(payload).hexdigest()}))


def pack_parameters(path):
    with path.open("rb") as handle:
        header_bytes = handle.read(PACK_HEADER_BYTES)
        if len(header_bytes) != PACK_HEADER_BYTES:
            raise ValueError("short pack header")
        header = struct.unpack("<8sII64sQQQIIIIQ", header_bytes)
        (magic, version, block, _, file_bytes, table_bytes, _, extent_count,
         unit_count, tensor_count, name_bytes, _) = header
        if magic != b"ANEMONE\0" or version != 2 or block != PACK_BLOCK_BYTES or table_bytes > MAX_TABLE_BYTES:
            raise ValueError("incompatible pack metadata")
        tensor_start = extent_count * 32 + unit_count * 8
        name_start = tensor_start + tensor_count * 64
        if (file_bytes != os.fstat(handle.fileno()).st_size or not extent_count or not unit_count or
                unit_count > extent_count or not table_bytes or table_bytes % block or
                name_start + name_bytes > table_bytes or block + table_bytes > file_bytes):
            raise ValueError("inconsistent pack table bounds")
        handle.seek(block)
        tables = handle.read(table_bytes)
        if len(tables) != table_bytes:
            raise ValueError("short pack tables")
        seen = set()
        for index in range(tensor_count):
            row = struct.unpack_from("<QQIIIII4IIII", tables, tensor_start + index * 64)
            offset, count, extent, name_offset, name_count, dtype, dimensions = row[:7]
            if (not name_count or name_offset + name_count > name_bytes or extent >= extent_count or
                    not 1 <= dimensions <= 4 or any(row[7 + dimensions:11])):
                raise ValueError("invalid tensor record bounds")
            name = tables[name_start + name_offset:name_start + name_offset + name_count].decode("ascii")
            if not name.startswith(("vision.", "aligner.")):
                continue
            if name in seen:
                raise ValueError(f"duplicate tower tensor {name}")
            seen.add(name)
            if any(row[11:]) or dtype not in (1, 2):
                raise ValueError(f"unsupported tower tensor {name}")
            shape = row[7:7 + dimensions]
            elements = 1
            for dimension in shape:
                elements *= dimension
            extent_offset, extent_count_bytes = struct.unpack_from("<QQ", tables, extent * 32)
            if (not elements or count != elements * (2 if dtype == 1 else 4) or
                    count > MAX_TOWER_TENSOR_BYTES or offset % 64 or
                    extent_offset < block + table_bytes or extent_offset + extent_count_bytes > file_bytes or
                    offset + count > extent_count_bytes):
                raise ValueError(f"invalid tower tensor extent {name}")
            handle.seek(extent_offset + offset)
            payload = handle.read(count)
            if len(payload) != count:
                raise ValueError(f"short tensor {name}")
            yield name, dtype, shape, payload


def run_reference(args):
    if args.device == "cuda" and os.environ.get("UNDER_GPU_RUN") != "1":
        raise RuntimeError("CUDA reference requires gpu_run.sh")
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(args.device)
    module = load_module("original_vision", args.reference / "inference/vision.py")
    tower, aligner = module.ViT(settings(args.reference)), module.Aligner(settings(args.reference))
    parameters = {"vision." + name: value for name, value in tower.named_parameters()}
    parameters.update({"aligner." + name: value for name, value in aligner.named_parameters()})
    copied = set()
    parameter_digest = hashlib.sha256()
    with torch.inference_mode():
        for name, dtype, shape, payload in pack_parameters(args.pack):
            if name not in parameters:
                raise ValueError(f"unexpected parameter {name}")
            parameter = parameters[name]
            source = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16 if dtype == 1 else torch.float32)
            if tuple(parameter.shape) != tuple(shape):
                raise ValueError(f"incompatible parameter {name}: {shape}")
            parameter.copy_(source.reshape(shape).to(device=args.device, dtype=parameter.dtype))
            copied.add(name)
            parameter_digest.update(name.encode() + b"\0" + struct.pack("<I", dtype))
            parameter_digest.update(struct.pack("<" + "I" * len(shape), *shape))
            parameter_digest.update(payload)
        if copied != set(parameters):
            raise ValueError(f"missing parameters: {sorted(set(parameters) - copied)}")
        payload = args.patches.read_bytes()
        if len(payload) < 8:
            raise ValueError("short patch header")
        height, width = struct.unpack_from("<II", payload)
        patch_size = settings(args.reference).vision_patch_size
        ratio = settings(args.reference).vision_downsample_ratio
        cells_height, cells_width = (height + ratio - 1) // ratio, (width + ratio - 1) // ratio
        if (not height or not width or height * width > MAX_PATCHES or
                cells_height * (cells_width + 1) + 2 > MAX_IMAGE_TOKENS or
                len(payload) != 8 + height * width * 3 * patch_size * patch_size * 2):
            raise ValueError("incompatible patch payload bounds")
        patches = torch.frombuffer(bytearray(payload[8:]), dtype=torch.bfloat16).reshape(
            height * width, 3, settings(args.reference).vision_patch_size,
            settings(args.reference).vision_patch_size).to(args.device)

        def forward():
            return aligner(tower(patches, height, width), height, width)

        activities = [torch.profiler.ProfilerActivity.CPU]
        linear_calls = []
        functional = torch.nn.functional
        original_linear = functional.linear
        parameter_names = {id(value): name for name, value in parameters.items()}

        def capture_linear(source_values, weight, bias=None):
            name = parameter_names.get(id(weight), "unnamed")
            with torch.profiler.record_function("vision_linear:" + name):
                value = original_linear(source_values, weight, bias)
            linear_calls.append({"parameter": name, "input_shape": list(source_values.shape),
                "input_stride": list(source_values.stride()), "input_dtype": str(source_values.dtype),
                "input_storage_offset": source_values.storage_offset(),
                "input_alignment_mod256": source_values.data_ptr() % 256,
                "weight_shape": list(weight.shape), "weight_stride": list(weight.stride()),
                "weight_dtype": str(weight.dtype), "weight_alignment_mod256": weight.data_ptr() % 256,
                "bias_shape": list(bias.shape) if bias is not None else None,
                "bias_stride": list(bias.stride()) if bias is not None else None,
                "output_shape": list(value.shape), "output_stride": list(value.stride()),
                "output_dtype": str(value.dtype)})
            return value

        if args.profile_kernels:
            if args.device != "cuda":
                raise ValueError("--profile-kernels requires --device cuda")
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            functional.linear = capture_linear
        try:
            with torch.profiler.profile(activities=activities, record_shapes=args.profile_kernels) as profiler:
                default = forward()
        finally:
            functional.linear = original_linear
        operator_names = sorted({item.key for item in profiler.key_averages()
                                 if "scaled_dot_product" in item.key or "flash_attention" in item.key})
        backend = {"flash": SDPBackend.FLASH_ATTENTION, "math": SDPBackend.MATH,
                   "efficient": SDPBackend.EFFICIENT_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}[args.backend]
        if args.device == "cuda":
            torch.cuda.synchronize()
        began = time.monotonic_ns()
        with sdpa_kernel(backend):
            output = forward()
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic_ns() - began
        if not torch.equal(default, output):
            raise ValueError(f"pinned {args.backend} differs from default official forward")
        result = {"torch": torch.__version__, "device": args.device,
                  "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
                  "reference_sha256": hashlib.sha256((args.reference / "inference/vision.py").read_bytes()).hexdigest(),
                  "patch_sha256": hashlib.sha256(payload).hexdigest(), "parameters": len(copied),
                  "parameter_sha256": parameter_digest.hexdigest(),
                  "grid": [height, width], "shape": list(output.shape), "attention_backend": args.backend,
                  "default_operators": operator_names, "pinned_default_bitwise_equal": True,
                  "reference_forward_ns_unflushed": elapsed}
        if args.device == "cuda":
            result["multiprocessors"] = torch.cuda.get_device_properties(0).multi_processor_count
        if args.profile_kernels:
            result["linear_calls"] = linear_calls
            result["cuda_kernels"] = sorted({item.key for item in profiler.key_averages()
                                            if str(item.device_type).lower().endswith("cuda")})
            result["matrix_operators"] = [{"operator": item.name, "input_shapes": item.input_shapes,
                "kernels": event_kernel_names(item), "event_id": item.id,
                "cpu_parent": item.cpu_parent.name if item.cpu_parent is not None else None}
                for item in profiler.events() if item.name in ("aten::mm", "aten::addmm", "aten::bmm")]
            result["chrome_trace"], result["cuda_kernel_launches"] = profile_launches(profiler, args.output)
        if args.trace_directory:
            tracer = load_module("original_vision_tracer", Path(__file__).parent / "make_fixtures.py")
            with sdpa_kernel(backend):
                stages = tracer.traced_forward(module, tower, aligner, patches, height, width)
            if not torch.equal(stages[-1], output):
                raise ValueError("traced full tower differs from unmodified official forward")
            args.trace_directory.mkdir(parents=True, exist_ok=True)
            block_passes = ("norm1", "attn.wqkv", "attn.rope", "attn.attention", "attn.wo", "attn.residual",
                            "norm2", "mlp.w1", "mlp.silu_up", "mlp.w2", "mlp.residual")
            names = ["patch_embed"] + [f"blocks.{layer}.{name}" for layer in range(len(tower.blocks))
                                       for name in block_passes] + ["norm", "unfold", "aligner.w1",
                                                                   "aligner.gelu", "aligner.w2"]
            if len(names) != len(stages):
                raise ValueError("incompatible trace pass inventory")
            traces = []
            for index, stage in enumerate(stages):
                name = f"pass_{index:03d}.bf16"
                data = stage.contiguous().cpu().view(torch.uint16).numpy().tobytes()
                (args.trace_directory / name).write_bytes(data)
                traces.append({"pass": index, "operation": names[index], "file": name, "shape": list(stage.shape),
                               "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            (args.trace_directory / "manifest.json").write_text(json.dumps({
                "reference_sha256": result["reference_sha256"], "patch_sha256": result["patch_sha256"],
                "torch": torch.__version__, "torch_git": torch.version.git_version,
                "device": args.device, "backend": args.backend, "traced_default_bitwise_equal": True,
                "passes": traces}, indent=2) + "\n")
            result["trace"] = {"directory": str(args.trace_directory), "passes": len(stages),
                               "bytes": sum(item["bytes"] for item in traces)}
        expected = output.contiguous().cpu()
        args.output.write_bytes(expected.view(torch.uint16).numpy().tobytes())
        if args.candidate:
            candidate = torch.frombuffer(bytearray(args.candidate.read_bytes()), dtype=torch.bfloat16).reshape(
                expected.shape).to("cpu")
            expected_float, candidate_float = expected.float(), candidate.float()
            difference = (candidate_float - expected_float).abs()
            tolerance = expected_float.abs() * 0.016 + 0.00001
            failed = (~torch.isfinite(difference)) | (difference > tolerance)
            result["comparison"] = {"elements": expected.numel(),
                "bitwise_differences": int((candidate.view(torch.uint16) != expected.view(torch.uint16)).sum()),
                "outside_bf16_tolerance": int(failed.sum()), "rtol": 0.016, "atol": 0.00001,
                "maximum_absolute": float(difference.max()),
                "relative_rms": float(torch.sqrt(torch.mean(difference.square()) /
                                                 torch.mean(expected_float.square()))),
                "passed": not bool(failed.any())}
        print(json.dumps(result, indent=2))
        if args.candidate and not result["comparison"]["passed"]:
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--image", type=Path, required=True)
    reference_parser = commands.add_parser("reference")
    reference_parser.add_argument("--pack", type=Path, required=True)
    reference_parser.add_argument("--patches", type=Path, required=True)
    reference_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    reference_parser.add_argument("--backend", choices=("flash", "math", "efficient", "cudnn"), default="flash")
    reference_parser.add_argument("--candidate", type=Path)
    reference_parser.add_argument("--trace-directory", type=Path)
    reference_parser.add_argument("--profile-kernels", action="store_true")
    for command in (prepare_parser, reference_parser):
        command.add_argument("--reference", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run_reference)(args)


if __name__ == "__main__":
    main()
