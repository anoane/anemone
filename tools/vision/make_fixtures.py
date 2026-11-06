#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pinned official vision.py, actual Flash SDPA, small synthetic BF16 tower/projector fixtures.

CUDA_VISIBLE_DEVICES= python3 tools/vision/make_fixtures.py --reference /path/to/deepseek_v41_official_1017
The C fixture uses this same integer tensor recipe. No checkpoint is opened.
CUDA generation is permitted only inside gpu_run.sh with one exposed card.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel


def traced_forward(module, tower, projector, patches, height, width):
    stages = []
    def save(value):
        stages.append(value.detach().clone())
        return value
    value = save(tower.patch_embed(patches))
    cosine, sine = module.get_vision_cos_sin(height, width, tower.rope_dim, tower.rope_theta)
    for block in tower.blocks:
        normalized = save(block.norm1(value))
        qkv = save(block.attn.wqkv(normalized))
        query, key, values = (part.view(height * width, block.attn.n_heads, block.attn.head_dim)
                              for part in qkv.chunk(3, dim=-1))
        query = module.apply_rotary(query, cosine, sine)
        key = module.apply_rotary(key, cosine, sine)
        save(torch.cat((query.flatten(1), key.flatten(1), values.flatten(1)), dim=-1))
        attended = save(functional.scaled_dot_product_attention(query.transpose(0, 1), key.transpose(0, 1),
                         values.transpose(0, 1)).transpose(0, 1).reshape(height * width, -1))
        projected = save(block.attn.wo(attended))
        value = save(value + projected)
        normalized = save(block.norm2(value))
        gates = save(block.mlp.w1(normalized))
        gate, up = gates.chunk(2, dim=-1)
        activated = save(functional.silu(gate) * up)
        projected = save(block.mlp.w2(activated))
        value = save(value + projected)
    normalized = save(tower.norm(value))
    ratio = projector.downsample_ratio
    grid = normalized.view(height, width, -1).permute(2, 0, 1)
    grid = functional.pad(grid, (0, -width % ratio, 0, -height % ratio))
    unfolded = save(functional.unfold(grid.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1))
    projected = save(projector.w1(unfolded))
    activated = save(functional.gelu(projected))
    save(projector.w2(activated))
    return stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "fixtures")
    parser.add_argument("--trace-directory", type=Path)
    arguments = parser.parse_args()
    if arguments.device == "cuda":
        if not os.environ.get("UNDER_GPU_RUN") or torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA fixtures require gpu_run.sh and exactly one exposed card")
    source = arguments.reference / "inference" / "vision.py"
    spec = importlib.util.spec_from_file_location("official_vision", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(arguments.device)
    args = SimpleNamespace(vision_n_layers=2, vision_dim=16, vision_n_heads=2, vision_inter_dim=12,
                           vision_patch_size=2, vision_rope_theta=10000.0, vision_downsample_ratio=3, dim=16)
    tower, projector = module.ViT(args), module.Aligner(args)
    tensor_id = 0

    def fill(shape, norm=False):
        nonlocal tensor_id
        count = 1
        for dimension in shape:
            count *= dimension
        values = ((torch.arange(count, dtype=torch.int64) * 17 + tensor_id * 11) % 31 - 15).float() / 64
        tensor_id += 1
        return ((1 + values) if norm else values.to(torch.bfloat16)).reshape(shape)

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
            fill((args.dim,))  # learned delimiter embeddings, not inputs to the encoder
        input_id = tensor_id
        arguments.output.mkdir(parents=True, exist_ok=True)
        manifests = []
        for case, (height, width) in enumerate(((1, 1), (2, 3), (4, 5), (11, 12))):
            tensor_id = input_id
            patches = fill((height * width, 3, args.vision_patch_size, args.vision_patch_size))
            default_output = projector(tower(patches, height, width), height, width)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output = projector(tower(patches, height, width), height, width)
            if not torch.equal(default_output, output):
                raise ValueError("pinned Flash SDPA differs from the official default path")
            data = output.contiguous().view(torch.uint16).cpu().numpy().tobytes()
            path = arguments.output / f"vit_{case}.hex"
            words = output.contiguous().view(torch.uint16).flatten().tolist()
            path.write_text("".join(f"{word:04x}\n" for word in words))
            with sdpa_kernel(SDPBackend.MATH):
                math_output = projector(tower(patches, height, width), height, width)
            math_words = math_output.contiguous().view(torch.uint16).flatten().tolist()
            math_path = arguments.output / f"vit_math_{case}.hex"
            math_path.write_text("".join(f"{word:04x}\n" for word in math_words))
            if arguments.trace_directory is not None:
                arguments.trace_directory.mkdir(parents=True, exist_ok=True)
                stages = traced_forward(module, tower, projector, patches, height, width)
                if not torch.equal(stages[-1], output):
                    raise ValueError("traced and original official output differ")
                for stage_index, stage in enumerate(stages):
                    trace = arguments.trace_directory / f"case_{case}_pass_{stage_index}.hex"
                    stage_words = stage.contiguous().view(torch.uint16).flatten().tolist()
                    trace.write_text(" ".join(f"{word:04x}" for word in stage_words) + "\n")
            math_data = math_output.contiguous().view(torch.uint16).cpu().numpy().tobytes()
            manifests.append({"case": case, "grid": [height, width], "shape": list(output.shape),
                              "sha256": hashlib.sha256(data).hexdigest(),
                              "math_sha256": hashlib.sha256(math_data).hexdigest()})
        manifest = {"reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "torch": torch.__version__, "torch_git": torch.version.git_version,
                    "dtype": "bfloat16", "device": arguments.device,
                    "primary_test_attention": "SDPBackend.FLASH_ATTENTION",
                    "primary_comparison": "bitwise bfloat16 words",
                    "default_equals_pinned": True,
                    "retained_diagnostic_attention": "SDPBackend.MATH",
                    "primary_bfloat16_rtol": 0.0, "primary_bfloat16_atol": 0.0,
                    "reference_bfloat16_rtol": 0.016, "reference_bfloat16_atol": 1e-5,
                    "math_bfloat16_rtol": 0.0078125, "math_bfloat16_atol": 0.000244140625,
                    "cases": manifests}
        if arguments.device == "cuda":
            manifest["card"] = torch.cuda.get_device_name(0)
            manifest["capability"] = list(torch.cuda.get_device_capability(0))
        (arguments.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
