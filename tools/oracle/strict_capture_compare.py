#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare named oracle records with zero differing represented bits.

This is a strict, explicitly scoped comparison, independent of the existing
tolerance gates. BF16 widened to FP32 is compared in its stored representation;
this does not establish raw BF16 NaN-payload identity. Leading singleton axes
may be removed to map the oracle and engine's documented logical layouts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


DTYPES = {name: np.dtype('<' + code) for name, code in {
    'float16': 'f2', 'float32': 'f4', 'float64': 'f8',
    'int8': 'i1', 'uint8': 'u1', 'int16': 'i2', 'uint16': 'u2',
    'int32': 'i4', 'uint32': 'u4', 'int64': 'i8', 'uint64': 'u8',
}.items()}


def logical_shape(shape):
    shape = tuple(shape)
    while len(shape) > 1 and shape[0] == 1:
        shape = shape[1:]
    return shape


class Capture:
    def __init__(self, folder):
        self.folder = Path(folder)
        manifest_path = self.folder / 'manifest.json'
        raw = manifest_path.read_bytes()
        self.manifest = json.loads(raw)
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if self.manifest.get('complete') is not True:
            raise ValueError(f'{self.folder}: incomplete capture')
        self.path = self.folder / 'tensors.bin'
        self.records = {}
        end = 0
        records = self.manifest['tensors']
        if not isinstance(records, list) or not records:
            raise ValueError(f'{self.folder}: no tensor records')
        for record in sorted(records, key=lambda item: item['byte_offset']):
            name, shape = record['name'], record['shape']
            dtype = DTYPES.get(record['dtype'])
            if (not isinstance(name, str) or not name or name in self.records or dtype is None
                    or not isinstance(shape, list)
                    or any(type(size) is not int or size < 0 for size in shape)
                    or type(record['byte_offset']) is not int or record['byte_offset'] != end
                    or type(record['byte_count']) is not int
                    or record['byte_count'] != math.prod(shape) * dtype.itemsize):
                raise ValueError(f'{self.folder}: invalid/duplicate/noncontiguous record {name!r}')
            self.records[name] = record
            end += record['byte_count']
        if self.path.stat().st_size != end:
            raise ValueError(f'{self.folder}: payload size does not match manifest')

    def read(self, name):
        record = self.records[name]
        if not record['byte_count']:
            return np.empty(record['shape'], dtype=DTYPES[record['dtype']])
        return np.memmap(self.path, mode='r', dtype=DTYPES[record['dtype']],
                         offset=record['byte_offset'], shape=tuple(record['shape']))


def record_order(name):
    match = re.match(r'layer_(\d+)\.', name)
    return (int(match[1]) if match else (-1 if name.startswith('input.') else 1000000), name)


def compare(reference, candidate, names=None):
    names = sorted(set(names) if names is not None else reference.records.keys() | candidate.records.keys(),
                   key=record_order)
    if not names:
        raise ValueError('comparison scope is empty')
    records = []
    for name in names:
        row = {'name': name}
        match = re.match(r'layer_(\d+)\.', name)
        row['layer'] = int(match[1]) if match else None
        if name not in reference.records or name not in candidate.records:
            row.update(exact=False, error='missing required record',
                       reference_present=name in reference.records, candidate_present=name in candidate.records)
        else:
            left_meta, right_meta = reference.records[name], candidate.records[name]
            if (left_meta['dtype'] != right_meta['dtype']
                    or logical_shape(left_meta['shape']) != logical_shape(right_meta['shape'])):
                row.update(exact=False, error='dtype or logical shape mismatch',
                           reference_dtype=left_meta['dtype'], candidate_dtype=right_meta['dtype'],
                           reference_shape=left_meta['shape'], candidate_shape=right_meta['shape'])
            else:
                left, right = reference.read(name).reshape(-1), candidate.read(name).reshape(-1)
                unequal = np.any(left.view(np.uint8).reshape(-1, left.dtype.itemsize)
                                 != right.view(np.uint8).reshape(-1, right.dtype.itemsize), axis=1)
                count = int(np.count_nonzero(unequal))
                row.update(exact=count == 0, elements=int(left.size), differing_elements=count)
                if count:
                    first = int(np.flatnonzero(unequal)[0])
                    row['first_differing_flat_index'] = first
                    row['first_reference_bits'] = left[first:first+1].tobytes().hex()
                    row['first_candidate_bits'] = right[first:first+1].tobytes().hex()
                    if left.dtype.kind == 'f':
                        finite = np.isfinite(left) & np.isfinite(right)
                        delta = np.abs(left[finite].astype(np.float64) - right[finite].astype(np.float64))
                        row['max_finite_absolute_difference'] = float(delta.max()) if delta.size else None
                        row['rms_finite_difference'] = float(np.sqrt(np.mean(delta * delta))) if delta.size else None
                        row['nonfinite_elements'] = int(np.count_nonzero(~finite))
                    if name.startswith('logits.'):
                        shape = logical_shape(left_meta['shape'])
                        left_rows, right_rows = left.reshape(-1, shape[-1]), right.reshape(-1, shape[-1])
                        row['top1_equal_rows'] = int(np.count_nonzero(left_rows.argmax(-1) == right_rows.argmax(-1)))
                        row['logit_rows'] = int(left_rows.shape[0])
        records.append(row)
    differences = [row for row in records if not row['exact']]
    differing_layers = [row['layer'] for row in differences if row['layer'] is not None]
    return dict(exact=not differences, scope='named stored representations; leading singleton axes mapped',
                required_records=len(records), differing_records=len(differences),
                first_differing_record=differences[0]['name'] if differences else None,
                first_differing_layer=min(differing_layers) if differing_layers else None,
                reference_manifest_sha256=reference.manifest_sha256,
                candidate_manifest_sha256=candidate.manifest_sha256, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('candidate')
    parser.add_argument('--names', help='comma-separated required names; default requires the union of both captures')
    parser.add_argument('--output', type=Path)
    arguments = parser.parse_args()
    try:
        result = compare(Capture(arguments.reference), Capture(arguments.candidate),
                         arguments.names.split(',') if arguments.names else None)
        status = 0 if result['exact'] else 1
    except (ValueError, KeyError, OSError, TypeError) as failure:
        result, status = dict(exact=False, invalid_capture=str(failure)), 2
    rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if arguments.output:
        with arguments.output.open('x') as stream:
            stream.write(rendered)
    print(rendered, end='')
    return status


if __name__ == '__main__':
    sys.exit(main())
