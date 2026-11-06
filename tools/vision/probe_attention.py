#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Record the unmodified official ViT's actual SDPA dispatch; no checkpoint is opened.

CPU: CUDA_VISIBLE_DEVICES= python probe_attention.py --device cpu --reference ../reference
GPU: gpu_run.sh env CUDA_VISIBLE_DEVICES=<one card> python probe_attention.py --device cuda ...
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.device == "cuda":
        import os
        if not os.environ.get("UNDER_GPU_RUN"):
            raise RuntimeError("GPU probe must run inside gpu_run.sh")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one card")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(arguments.device)
    source = arguments.reference / "inference" / "vision.py"
    spec = importlib.util.spec_from_file_location("official_vision", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_sdpa = module.F.scaled_dot_product_attention
    calls = []

    def capture(query, key, value, *args, **kwargs):
        calls.append({"shape": list(query.shape), "query_stride": list(query.stride()),
                      "key_stride": list(key.stride()), "value_stride": list(value.stride()),
                      "dtype": str(query.dtype), "device": str(query.device)})
        return original_sdpa(query, key, value, *args, **kwargs)

    module.F.scaled_dot_product_attention = capture
    configuration = SimpleNamespace(vision_n_layers=2, vision_dim=16, vision_n_heads=2,
                                    vision_inter_dim=12, vision_patch_size=2, vision_rope_theta=10000.0,
                                    vision_downsample_ratio=3, dim=16)
    tower, projector = module.ViT(configuration), module.Aligner(configuration)
    tensor_id = 0

    def fill(shape, norm=False):
        nonlocal tensor_id
        count = 1
        for dimension in shape:
            count *= dimension
        values = ((torch.arange(count, dtype=torch.int64) * 17 + tensor_id * 11) % 31 - 15).float() / 64
        tensor_id += 1
        return ((1 + values) if norm else values.to(torch.bfloat16)).reshape(shape)

    output = {"torch": torch.__version__, "torch_git": torch.version.git_version,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "device": arguments.device, "default_dtype": str(torch.get_default_dtype()),
              "cpu_capability": torch.backends.cpu.get_cpu_capability(),
              "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
              "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
              "memory_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
              "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "matmul_allow_bf16_reduced_precision_reduction":
                  torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    if arguments.device == "cuda":
        output.update(card=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                      cuda=torch.version.cuda,
                      multiprocessors=torch.cuda.get_device_properties(0).multi_processor_count)
    with torch.inference_mode():
        tower.patch_embed.proj.weight.copy_(fill(tower.patch_embed.proj.weight.shape))
        tower.patch_embed.proj.bias.copy_(fill(tower.patch_embed.proj.bias.shape))
        for block in tower.blocks:
            for tensor, norm in ((block.norm1.weight, True), (block.attn.wqkv.weight, False),
                                 (block.attn.wqkv.bias, False), (block.attn.wo.weight, False),
                                 (block.attn.wo.bias, False), (block.norm2.weight, True),
                                 (block.mlp.w1.weight, False), (block.mlp.w2.weight, False)):
                tensor.copy_(fill(tensor.shape, norm))
        tower.norm.weight.copy_(fill(tower.norm.weight.shape, True))
        for tensor in (projector.w1.weight, projector.w1.bias, projector.w2.weight, projector.w2.bias):
            tensor.copy_(fill(tensor.shape))
        for _ in range(3):
            fill((configuration.dim,))
        input_id = tensor_id
        output["cases"] = []
        arguments.output.mkdir(parents=True, exist_ok=True)
        activities = [ProfilerActivity.CPU]
        if arguments.device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        for case_index, (height, width) in enumerate(((1, 1), (2, 3), (4, 5))):
            tensor_id = input_id
            patches = fill((height * width, 3, 2, 2))
            # Warm up before collecting the official call and the dispatch operators.
            projector(tower(patches, height, width), height, width)
            calls.clear()
            with profile(activities=activities) as captured:
                default = projector(tower(patches, height, width), height, width)
                if arguments.device == "cuda":
                    torch.cuda.synchronize()
            case = {"case": case_index, "grid": [height, width], "calls": calls.copy(),
                    "attention_operators": [event.key for event in captured.key_averages()
                                            if "attention" in event.key or "softmax" in event.key
                                            or "flash" in event.key]}
            for name, backend in (("math", SDPBackend.MATH), ("flash", SDPBackend.FLASH_ATTENTION),
                                  ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                                  ("cudnn", SDPBackend.CUDNN_ATTENTION)):
                try:
                    with sdpa_kernel(backend):
                        pinned = projector(tower(patches, height, width), height, width)
                    case[name] = {"bitwise_default": bool(torch.equal(default, pinned)),
                                  "max_absolute": float((default.float() - pinned.float()).abs().max())}
                except RuntimeError as error:
                    case[name] = {"unsupported": str(error).splitlines()[0]}
            words = default.contiguous().view(torch.uint16).cpu().flatten().tolist()
            (arguments.output / f"vit_{arguments.device}_{case_index}.hex").write_text(
                "".join(f"{word:04x}\n" for word in words))
            case["words"] = words
            output["cases"].append(case)
        # The production tower has head width 64. Test the same non-contiguous rank-three call layout.
        query = fill((20, 16, 64)).transpose(0, 1)
        with profile(activities=activities) as captured:
            original_sdpa(query, query, query)
            if arguments.device == "cuda":
                torch.cuda.synchronize()
        output["production_head_attention_operators"] = [event.key for event in captured.key_averages()
                                                        if "attention" in event.key or "softmax" in event.key
                                                        or "flash" in event.key]
        inverse_frequency = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
        cosine, sine = module.get_vision_cos_sin(3, 3063, 32, 10000.0)
        sample_rows = torch.tensor([0, 1, 2, 3, 127, 128, 1520, 3062, 3063, 6125, 9188], dtype=torch.int64)
        output["production_rope"] = {
            "grid": [3, 3063], "rope_dimension": 32, "theta": 10000.0,
            "inverse_frequency_f32_words": inverse_frequency.contiguous().view(torch.uint32).cpu().tolist(),
            "sample_rows": sample_rows.cpu().tolist(),
            "cosine_f32_words": cosine.index_select(0, sample_rows).contiguous().view(torch.uint32).cpu().tolist(),
            "sine_f32_words": sine.index_select(0, sample_rows).contiguous().view(torch.uint32).cpu().tolist()}
    (arguments.output / "dispatch.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
