#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Re-evaluate pinned original HC methods on saved actual streams at declared row shapes.

This is an isolated operator reference. It calls the imported source methods,
without loading a complete model or changing any saved full-forward capture.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import reference_oracle as oracle
from official_ops import SOURCE_PINS
from strict_capture_compare import Capture, compare


def tensor(capture, name, device='cuda'):
    metadata = capture.records[name]
    value = torch.from_numpy(np.array(capture.read(name), copy=True)).to(device)
    return value.to(torch.bfloat16) if metadata.get('reference_dtype') == 'bfloat16' else value


@torch.inference_mode()
def capture_shape(reference, source, folder, rows):
    writer = oracle.CaptureWriter(str(folder))
    writer.manifest['conditions'] = dict(scope='isolated original HC operations on actual saved streams',
        rows=rows, reference_files_sha256=reference.file_digests, torch_version=torch.__version__,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        float32_precision=torch.get_float32_matmul_precision(),
        source_manifest_sha256=source.manifest_sha256)
    complete = False
    try:
        writer.add('input.token_ids', tensor(source, 'input.token_ids', 'cpu')[:, :rows], None, 'Saved fixture tokens.')
        parameters = {}
        for name in source.records:
            if name.startswith('layer_0.parameter.'):
                parameters[name.removeprefix('layer_0.parameter.')] = tensor(source, name)
                writer.add(name, parameters[name.removeprefix('layer_0.parameter.')], 0, 'Saved original parameter.')
        previous_pre = torch.zeros((1, rows, 4), dtype=torch.float32, device='cuda')
        previous_pre[..., 0] = 1
        for site in ('attn', 'ffn'):
            # FFN uses the preceding attention pre from the same saved full forward.
            if site == 'ffn':
                previous_pre = torch.stack([tensor(source, f'layer_0.position_{position}.attn_site.pre')
                                            for position in range(rows)]).unsqueeze(0)
            streams = torch.stack([tensor(source, f'layer_0.position_{position}.{site}_site.streams')
                                   for position in range(rows)]).unsqueeze(0)
            weight, scale, base = [parameters[f'hc_{site}_{part}'] for part in ('fn', 'scale', 'base')]
            block = SimpleNamespace(norm_eps=1e-20, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6)
            observed = {}
            original_sinkhorn = reference.model.hc_split_sinkhorn

            def sinkhorn(mixes, *arguments, **keywords):
                observed['mixes'] = mixes.detach().clone()
                return original_sinkhorn(mixes, *arguments, **keywords)

            reference.model.hc_split_sinkhorn = sinkhorn
            try:
                pre, post, comb = reference.model.Block.hc_mixes(block, streams, weight, scale, base)
            finally:
                reference.model.hc_split_sinkhorn = original_sinkhorn
            collapsed = reference.model.Block.hc_pre(block, streams, previous_pre)
            norm = reference.model.RMSNorm(5120, 1e-20).to('cuda')
            norm.weight.copy_(parameters[site + '_norm'])
            normed = norm(collapsed)
            raw = torch.nn.functional.linear(streams.flatten(2).float(), weight)
            mean = streams.flatten(2).float().square().mean(-1, keepdim=True)
            inverse = torch.rsqrt(mean + 1e-20)
            outputs = dict(streams=streams, mixes=observed['mixes'], pre=pre, post=post, comb=comb,
                           collapsed=collapsed, normed=normed, raw_projection=raw,
                           mean=mean, inverse=inverse)
            for part, values in outputs.items():
                for position in range(rows):
                    writer.add(f'layer_0.position_{position}.{site}_site.{part}', values[0, position], 0,
                               'Actual result of isolated pinned original method.', position=position)
        complete = True
    finally:
        writer.close(complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--checkpoint', required=True)
    arguments = parser.parse_args()
    if os.environ.get('UNDER_GPU_RUN') != '1':
        raise RuntimeError('Run original HC operator capture through gpu_run.')
    source = Capture(arguments.source)
    reference = oracle.import_reference(arguments.checkpoint)
    if any(reference.file_digests.get(name) != digest for name, digest in SOURCE_PINS.items()):
        raise RuntimeError('isolated HC reference requires the pinned 1017 source')
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    arguments.output.mkdir()
    for rows in (1, 3, 4, 8):
        for repetition in ('first', 'second'):
            capture_shape(reference, source, arguments.output/f'rows{rows}_{repetition}', rows)
        first, second = [Capture(arguments.output/f'rows{rows}_{repetition}')
                         for repetition in ('first', 'second')]
        result = compare(first, second)
        (arguments.output/f'rows{rows}_repeat.json').write_text(json.dumps(result, indent=2)+'\n')
        if not result['exact']:
            raise RuntimeError(f'rows {rows}: isolated original methods did not repeat')
        if rows == 8:
            names = [name for name in first.records if name in source.records]
            result = compare(source, first, names)
            (arguments.output/'eight_row_control.json').write_text(json.dumps(result, indent=2)+'\n')
            if not result['exact']:
                raise RuntimeError('isolated eight-row methods did not preserve full-forward outputs')


if __name__ == '__main__':
    main()
