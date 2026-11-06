#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Aggregate strict per-record comparisons into a layer-zero operation table.

Missing operations remain visible. RMS is over all finite compared elements of
an operation, weighted by element count. Represented bits, including signed zero,
determine equality. This does not certify raw BF16 NaN payloads.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from strict_capture_compare import Capture, compare


OPERATIONS = (
    ('Input token IDs', 'input.token_ids'),
    ('Embedding', 'input.embeddings'),
    ('Attention mHC raw projection', 'attn_site.raw_projection'),
    ('Attention mHC projection and RMS', 'attn_site.mixes'),
    ('Attention mHC pre', 'attn_site.pre'),
    ('Attention mHC post', 'attn_site.post'),
    ('Attention mHC Sinkhorn', 'attn_site.comb'),
    ('Attention collapse', 'attn_site.collapsed'),
    ('Attention norm', 'attn_site.normed'),
    ('Attention input FP8 codes', 'layer_0.quant_0.codes'),
    ('Attention input FP8 scales', 'layer_0.quant_0.scales'),
    ('Query rank projection', 'attn.wq_a'),
    ('Query rank norm', 'attn.qr'),
    ('Query rank FP8 codes', 'layer_0.quant_1.codes'),
    ('Query rank FP8 scales', 'layer_0.quant_1.scales'),
    ('Query head projection', 'attn.wq_b'),
    ('KV projection', 'attn.wkv'),
    ('KV norm', 'attn.kv_normed'),
    ('Query RoPE', 'rope_0.output'),
    ('KV RoPE', 'rope_1.output'),
    ('Window KV FP8 round-trip', 'layer_0.quant_3.output'),
    ('Sparse attention queries', 'attn.q'),
    ('Sparse attention indices', 'attn.topk_idxs'),
    ('Sparse attention core', 'attn.output_rotated'),
    ('Attention output FP8 round-trip', 'layer_0.quant_4.output'),
    ('Attention grouped output projection', 'attn.wo_a'),
    ('Attention output projection', 'attn.wo_b'),
    ('Attention residual post-mix', 'ffn_site.streams'),
    ('FFN mHC raw projection', 'ffn_site.raw_projection'),
    ('FFN mHC projection and RMS', 'ffn_site.mixes'),
    ('FFN mHC pre', 'ffn_site.pre'),
    ('FFN mHC post', 'ffn_site.post'),
    ('FFN mHC Sinkhorn', 'ffn_site.comb'),
    ('FFN collapse', 'ffn_site.collapsed'),
    ('FFN norm', 'ffn_site.normed'),
    ('Router logits', 'router.logits'),
    ('Router affinities', 'router.scores'),
    ('Router biased scores', 'router.biased_scores'),
    ('Routed expert IDs', 'layer_0.mlp.routed_expert_ids'),
    ('Routed expert weights', 'layer_0.mlp.route_weights'),
    ('Routed expert sum', 'experts.row_sum'),
    ('Shared expert', 'shared_expert.output'),
    ('Combined MoE', 'experts.combined'),
    ('Layer-zero residual', 'layer_0.block.output'),
    ('Layer-zero carried pre', 'layer_0.block.output_collapse_weights'),
    ('All-vocabulary logits', 'logits.all_rows'),
)


def aggregate(reference, candidate):
    table = []
    positions = int(reference.read('input.token_ids').size)
    for operation, suffix in OPERATIONS:
        names = [suffix] if suffix in reference.records or suffix in candidate.records else [
            f'layer_0.position_{position}.{suffix}' for position in range(positions)]
        comparison = compare(reference, candidate, names)
        row = dict(operation=operation, names=names, records=len(names), exact=comparison['exact'])
        errors = [record for record in comparison['records'] if 'error' in record]
        if errors:
            row.update(status='missing or incompatible records', missing_or_incompatible=errors,
                       elements=None, differing_elements=None, max_absolute_difference=None, rms_difference=None)
        else:
            count = different = finite_count = 0
            maximum = square_sum = 0.0
            for name, record in zip(names, comparison['records']):
                left = reference.read(name).reshape(-1)
                right = candidate.read(name).reshape(-1)
                count += record['elements']
                different += record['differing_elements']
                finite = np.isfinite(left) & np.isfinite(right)
                delta = left[finite].astype(np.float64) - right[finite].astype(np.float64)
                finite_count += int(finite.sum())
                if delta.size:
                    maximum = max(maximum, float(np.abs(delta).max()))
                    square_sum += float(np.dot(delta, delta))
            row.update(status='equal' if not different else 'different', elements=count,
                       differing_elements=different, finite_elements=finite_count,
                       max_absolute_difference=maximum,
                       rms_difference=math.sqrt(square_sum/finite_count) if finite_count else None)
        table.append(row)
    return dict(scope='declared layer-zero operations; missing records do not pass',
                reference_manifest_sha256=reference.manifest_sha256,
                candidate_manifest_sha256=candidate.manifest_sha256, operations=table)


def render(result):
    def number(value):
        return 'n/a' if value is None else f'{value:.9g}'

    rows = ['| Operation | Differing / elements | Max absolute | RMS |',
            '|---|---:|---:|---:|']
    for row in result['operations']:
        if row['elements'] is None:
            rows.append(f"| {row['operation']} | Missing / incompatible | — | — |")
        else:
            rows.append(f"| {row['operation']} | {row['differing_elements']:,} / {row['elements']:,} | "
                        f"{number(row['max_absolute_difference'])} | {number(row['rms_difference'])} |")
    return '\n'.join(rows)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenwise-parent', type=Path,
                        help='map step_N captures from this folder into the reference prompt layouts')
    arguments = parser.parse_args()
    reference = Capture(arguments.reference)
    if arguments.tokenwise_parent:
        from compare_official_schedule import TokenwiseView
        reference = TokenwiseView(reference, [Capture(arguments.tokenwise_parent/f'step_{position}')
            for position in range(reference.read('input.token_ids').size)])
    result = aggregate(reference, Capture(arguments.candidate))
    result['reference_schedule'] = 'one token per forward' if arguments.tokenwise_parent else 'captured prompt'
    arguments.output.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    arguments.output.with_suffix('.md').write_text(render(result))
    print(render(result), end='')


if __name__ == '__main__':
    main()
