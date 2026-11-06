#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare repeated tokenwise captures and explicit prompt-position views.

The mapped view includes stored per-position layer-zero records and declared
token axes in input/logit, block, route, ngram and the first six attention
quantizer records. Shared KV arrays, physical sparse-pick arrays, named linear call batches and parameters
are outside this comparison. A shared-key extent can happen to equal the prompt
length; shape alone never makes it a token axis. No values are recomputed or
cast. Local step position zero maps to the actual prompt position.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from strict_capture_compare import Capture, compare, logical_shape


def position_axis(name, shape, positions):
    """Only named contracts identify axes; reject coincidental cache extents."""
    if re.fullmatch(r'layer_\d+\.mlp\.(routed_expert_ids|route_weights)', name):
        return 0 if shape and shape[0] == positions else None
    known = (name in ('input.token_ids', 'input.embeddings', 'logits.all_rows')
             or re.fullmatch(r'layer_\d+\.block\.(input|output)(_collapse_weights)?', name)
             or re.fullmatch(r'layer_\d+\.ngram\.row_ids', name)
             or re.fullmatch(r'layer_0\.quant_[0-5]\.(input|output|codes|scales)', name))
    return 1 if known and shape[:2] == [1, positions] else None


def at_position(values, axis, position):
    return values[position:position+1] if axis == 0 else values[:, position:position+1]


class PositionView:
    def __init__(self, prompt, position, positions):
        if not 0 <= position < positions or prompt.read('input.token_ids').size != positions:
            raise ValueError('invalid prompt position/count')
        self.records, self.sources = {}, {}
        self.prompt, self.position = prompt, position
        self.manifest_sha256 = hashlib.sha256(
            f'{prompt.manifest_sha256}:position={position}:positions={positions}:declared_axes_v2'.encode()).hexdigest()
        for name, record in prompt.records.items():
            match = re.fullmatch(r'layer_0\.position_(\d+)\.(.*)', name)
            if match:
                if int(match[1]) != position or match[2] == 'attn.topk_idxs':
                    continue
                target, shape, axis = f'layer_0.position_0.{match[2]}', record['shape'], None
            else:
                axis = position_axis(name, record['shape'], positions)
                if axis is None:
                    continue
                target, shape = name, list(record['shape'])
                shape[axis] = 1
            self.records[target] = dict(record, name=target, shape=shape)
            self.sources[target] = (name, axis)

    def read(self, name):
        source, axis = self.sources[name]
        values = self.prompt.read(source)
        return at_position(values, axis, self.position) if axis is not None else values


class TokenwiseView:
    """Expose step captures in the prompt's declared layouts, with no arithmetic or casts."""

    def __init__(self, prompt, steps):
        self.records, self.sources, self.steps, self.axes = {}, {}, steps, {}
        positions = prompt.read('input.token_ids').size
        if len(steps) != positions:
            raise ValueError('one complete step capture per prompt position is required')
        self.manifest_sha256 = hashlib.sha256(':'.join(
            ['declared_axes_v2', prompt.manifest_sha256]+[step.manifest_sha256 for step in steps]).encode()).hexdigest()
        for position, step in enumerate(steps):
            if step.manifest['conditions'].get('start_pos') != position:
                raise ValueError('tokenwise capture has the wrong start_pos')
        for name, record in prompt.records.items():
            match = re.fullmatch(r'layer_0\.position_(\d+)\.(.*)', name)
            if match:
                if match[2] == 'attn.topk_idxs':
                    continue
                position = int(match[1])
                local = f'layer_0.position_0.{match[2]}'
                if position >= positions or local not in steps[position].records:
                    continue
                sources = [(position, local)]
                expected_shape = record['shape']
                axis = None
            elif ((axis := position_axis(name, record['shape'], positions)) is not None
                  and all(name in step.records for step in steps)):
                sources = [(position, name) for position in range(positions)]
                expected_shape = list(record['shape'])
                expected_shape[axis] = 1
            else:
                continue
            for position, local in sources:
                metadata = steps[position].records[local]
                if (metadata['dtype'] != record['dtype']
                        or logical_shape(metadata['shape']) != logical_shape(expected_shape)):
                    raise ValueError('incompatible tokenwise view record: '+local)
            self.records[name] = dict(record)
            self.sources[name] = sources
            self.axes[name] = axis

    def read(self, name):
        sources = self.sources[name]
        values = [self.steps[position].read(local) for position, local in sources]
        if len(sources) == 1:
            return values[0]
        return np.concatenate(values, axis=self.axes[name])


def compare_campaign(folder):
    folder = Path(folder)
    prompt, repeated = Capture(folder/'first'), Capture(folder/'second')
    positions = prompt.read('input.token_ids').size
    result = dict(prompt_repeat=compare(prompt, repeated), tokenwise_repeat=[], prompt_vs_tokenwise=[])
    for position in range(positions):
        first = Capture(folder/'tokenwise_first'/f'step_{position}')
        second = Capture(folder/'tokenwise_second'/f'step_{position}')
        if any(capture.manifest['conditions'].get('start_pos') != position for capture in (first, second)):
            raise ValueError('tokenwise capture has the wrong start_pos')
        view = PositionView(prompt, position, positions)
        result['tokenwise_repeat'].append(dict(position=position, comparison=compare(first, second)))
        result['prompt_vs_tokenwise'].append(dict(position=position, comparison=compare(view, first, view.records)))
    result['repeat_exact'] = result['prompt_repeat']['exact'] and all(
        row['comparison']['exact'] for row in result['tokenwise_repeat'])
    result['schedule_exact_in_mapped_scope'] = all(
        row['comparison']['exact'] for row in result['prompt_vs_tokenwise'])
    result['mapping_scope'] = __doc__
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    result = compare_campaign(arguments.folder)
    with arguments.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if not result['repeat_exact']:
        raise RuntimeError('original schedule repeats failed')
