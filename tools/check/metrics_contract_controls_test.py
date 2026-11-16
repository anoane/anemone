#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Causal controls: reject valid syntax whose label values or le spelling drift."""
import argparse
import importlib.util
import json
from pathlib import Path


def run(exposition, checker, context):
    specification = importlib.util.spec_from_file_location('checked_contract', checker)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.ROOT = Path(__file__).resolve().parents[2]
    text = exposition.read_text()
    _, samples = module.parse(text)
    encoded_model = next(sample['labels']['model_name'] for sample in samples if 'model_name' in sample['labels'])
    identity = {'model': json.loads('"' + encoded_model + '"')} if 'model' in module.compare.__code__.co_varnames else {}
    module.compare(text, context, False, **identity)
    examples = {}
    for sample in samples:
        if not sample['name'].startswith('anemone_'):
            for key, value in sample['labels'].items():
                examples.setdefault(key, value)
    invalid = {'model_name': 'wrong', 'finished_reason': 'stopped', 'reason': 'other',
               'source': 'other', 'status': '9xx', 'method': 'invented', 'handler': '/unknown',
               'enable_prefix_caching': 'invalid', 'kv_cache_size_tokens': str(context + 1),
               'kv_cache_max_concurrency': str(int(float(examples['kv_cache_max_concurrency'])))}
    mutations = [(f'{key}="{examples[key]}"', f'{key}="{value}"') for key, value in invalid.items()]
    mutations.append(('le="1.0"', 'le="1"'))
    accepted = []
    for before, after in mutations:
        if before not in text:
            raise ValueError(f'control fixture lacks {before}')
        try:
            module.compare(text.replace(before, after), context, False, **identity)
        except ValueError:
            print('rejected:', after)
        else:
            accepted.append(after)
            print('FAILED accepted:', after)
    if accepted:
        raise ValueError(f'{len(accepted)} invalid contract controls accepted')
    print(f'{len(mutations)} independent label-value/spelling controls rejected')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exposition', type=Path)
    parser.add_argument('--checker', type=Path, default=Path(__file__).with_name('metrics_vllm_test.py'))
    parser.add_argument('--context', type=int, default=4096)
    args = parser.parse_args()
    run(args.exposition, args.checker, args.context)
