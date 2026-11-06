#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Isolate original RMSNorm on actual captured L0 inputs, retaining its actual mean/inverse.

No complete model is loaded. The original module runs once per call. Its mean and
rsqrt wrappers retain the returned operands without changing their computation.
Eight-row outputs must also preserve the corresponding original forward records.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from official_hc_reference import tensor
from official_ops import SOURCE_PINS
import reference_oracle as oracle
from strict_capture_compare import Capture, compare


def norm_statistics(norm, values):
    original_mean, original_inverse = torch.Tensor.mean, torch.rsqrt
    code = type(norm).forward.__code__
    observed = {}

    def mean(inputs, *arguments, **keywords):
        output = original_mean(inputs, *arguments, **keywords)
        if sys._getframe(1).f_code is code:
            if 'mean' in observed:
                raise RuntimeError('original norm called mean twice')
            observed['mean'] = output.detach().clone()
        return output

    def inverse(inputs, *arguments, **keywords):
        output = original_inverse(inputs, *arguments, **keywords)
        if sys._getframe(1).f_code is code:
            if 'inverse' in observed:
                raise RuntimeError('original norm called rsqrt twice')
            observed['inverse'] = output.detach().clone()
        return output

    torch.Tensor.mean, torch.rsqrt = mean, inverse
    try:
        output = norm(values)
    finally:
        torch.Tensor.mean, torch.rsqrt = original_mean, original_inverse
    if set(observed) != {'mean', 'inverse'}:
        raise RuntimeError('original RMSNorm operations were not observed')
    return output, observed['mean'], observed['inverse']


@torch.inference_mode()
def capture_shape(reference, source, weights, folder, rows):
    writer = oracle.CaptureWriter(str(folder))
    writer.manifest['conditions'] = dict(scope=__doc__, rows=rows, source_manifest_sha256=source.manifest_sha256,
        reference_files_sha256=reference.file_digests, torch_version=torch.__version__,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32)
    complete = False
    names = dict(attn=('attn_site.collapsed', 'attn_site.normed'),
                 ffn=('ffn_site.collapsed', 'ffn_site.normed'),
                 qr=('attn.wq_a', 'attn.qr'), kv=('attn.wkv', 'attn.kv_normed'))
    preservation = []
    try:
        writer.add('input.token_ids', tensor(source, 'input.token_ids', 'cpu')[:, :rows], None, 'Saved fixture IDs.')
        for site, (input_name, output_name) in names.items():
            inputs = torch.stack([tensor(source, f'layer_0.position_{position}.{input_name}')
                                  for position in range(rows)]).unsqueeze(0)
            if inputs.dtype != torch.bfloat16:
                raise RuntimeError('original norm replay input must retain BF16')
            norm = reference.model.RMSNorm(inputs.shape[-1], 1e-20).to('cuda')
            norm.weight.copy_(weights[site])
            output, mean, inverse = norm_statistics(norm, inputs)
            writer.add('norm.'+site+'.weight', norm.weight, 0, 'Actual original module weight.')
            for position in range(rows):
                for part, values in dict(input=inputs, output=output, mean=mean, inverse=inverse).items():
                    writer.add(f'norm.{site}.position_{position}.{part}', values[0, position], 0,
                               'Actual original RMSNorm operand/result.', position=position)
                if rows == 8:
                    old = tensor(source, f'layer_0.position_{position}.{output_name}')
                    preservation.append(oracle.same_bits(old, output[0, position]))
        writer.results['eight_row_forward_output_preservation'] = all(preservation) if rows == 8 else None
        if rows == 8 and not all(preservation):
            raise RuntimeError('isolated original eight-row norm differs from the captured forward')
        complete = True
    finally:
        writer.close(complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    arguments = parser.parse_args()
    if os.environ.get('UNDER_GPU_RUN') != '1':
        raise RuntimeError('Run original norm capture through gpu_run.')
    reference = oracle.import_reference(str(arguments.checkpoint))
    if any(reference.file_digests.get(name) != digest for name, digest in SOURCE_PINS.items()):
        raise RuntimeError('norm reference requires pinned 1017 sources')
    config = json.loads((arguments.checkpoint/'inference/config.json').read_text())
    if config.get('norm_eps', reference.model.ModelArgs.norm_eps) != 1e-20:
        raise RuntimeError('norm epsilon differs from the candidate contract')
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    source = Capture(arguments.source)
    paths, checkpoint = oracle.read_shard_headers(str(arguments.checkpoint))
    checkpoint = {oracle.converted_name(name): value for name, value in checkpoint.items()}
    weights = {}
    for site, suffix in dict(attn='attn_norm.weight', ffn='ffn_norm.weight',
                             qr='attn.q_norm.weight', kv='attn.kv_norm.weight').items():
        parameter = checkpoint['layers.0.'+suffix]
        if parameter.dtype_name != 'BF16':
            raise RuntimeError('unexpected norm parameter storage dtype')
        with open(paths[parameter.shard_index], 'rb') as stream:
            stream.seek(parameter.byte_offset)
            raw = stream.read(parameter.byte_count)
        if len(raw) != parameter.byte_count:
            raise RuntimeError('truncated norm parameter')
        weights[site] = torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).view(torch.bfloat16).to('cuda')
    for rows in (1, 3, 4, 8):
        for repeat in ('first', 'second'):
            capture_shape(reference, source, weights, arguments.output/f'rows{rows}_{repeat}', rows)
        result = compare(Capture(arguments.output/f'rows{rows}_first'), Capture(arguments.output/f'rows{rows}_second'))
        (arguments.output/f'rows{rows}_repeat.json').write_text(json.dumps(result, indent=2)+'\n')
        if not result['exact']:
            raise RuntimeError('original norm repeat differs')


if __name__ == '__main__':
    main()
