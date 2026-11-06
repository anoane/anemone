#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Capture repeated prompt and tokenwise forwards in one pinned original-oracle session.

This retains the existing oracle's host staging and quantizer-definition checks.
Only the call schedule changes. The original FP8/FP4 attention state stays in use;
model buffers are reset before a sequence, never between its one-token steps.
Each step is a separate capture with local position zero and explicit start_pos.
"""
import copy
import hashlib
import os
from pathlib import Path
import sys

import torch

import official_ops
from official_kernel_manifest import compiled_kernel_manifest
import reference_oracle as oracle


def capture_steps(session, folder, token_ids, conditions):
    oracle.forget_earlier_runs(session)
    for position, token in enumerate(token_ids):
        writer = oracle.CaptureWriter(str(Path(folder)/f'step_{position}'))
        complete = False
        try:
            writer.manifest['conditions'] = copy.deepcopy(conditions)
            writer.manifest['conditions'].update(schedule='one token per forward', start_pos=position)
            inputs = torch.tensor([[token]], dtype=torch.long, device=oracle.CARD_DEVICE)
            writer.add('input.token_ids', inputs, None, 'One saved fixture token at start_pos.')
            with oracle.observed_pass(session, writer, True, True) as observation:
                started, writer_seconds = oracle.synchronized_clock(), writer.seconds
                output_ids, logits, _ = session.model.forward(inputs, position)
                seconds = oracle.synchronized_clock()-started-(writer.seconds-writer_seconds)
            if (len(observation.layers) != len(session.model.layers)
                    or tuple(logits.shape) != (1, session.model_arguments.vocab_size)
                    or not bool(torch.isfinite(logits).all())):
                raise RuntimeError('tokenwise original forward returned invalid layers/logits')
            writer.add('logits.last_row', logits.view(1, 1, -1), None, 'Original forward logits.')
            every_row = session.model.head(observation.head_input, full_logits=True)
            writer.add('logits.all_rows', every_row, None, 'Original head over the actual head input.')
            writer.results.update(positions=1, start_pos=position, next_token_id=int(output_ids[0]),
                                  layers=observation.layers, seconds_of_the_pass=seconds,
                                  last_of_all_rows_has_the_bits_of_last_row=oracle.same_bits(every_row[:, -1], logits),
                                  quantizer_counts_cumulative=dict(session.quantization_counts))
            complete = True
        finally:
            writer.close(complete)


def capture_campaign_body(session, writer, token_ids, arguments):
    if arguments.modes != ('forward',) or not arguments.all_logits or len(token_ids) < 2:
        raise RuntimeError('schedule control requires only forward mode, all logits and at least two tokens')
    if arguments.weights_gguf is not None:
        raise RuntimeError('schedule control requires the original checkpoint weights')
    folder = Path(arguments.output_directory)
    conditions = writer.manifest['conditions']
    for name in ('first', 'second'):
        prompt = oracle.CaptureWriter(str(folder/name))
        complete = False
        try:
            prompt.manifest['conditions'] = copy.deepcopy(conditions)
            prompt.manifest['conditions'].update(schedule='whole prompt', start_pos=0)
            oracle.capture_pass(session, prompt, token_ids, arguments)
            complete = True
        finally:
            prompt.close(complete)
    for name in ('tokenwise_first', 'tokenwise_second'):
        capture_steps(session, folder/name, token_ids, conditions)
    writer.add('input.token_ids', torch.tensor([token_ids], dtype=torch.long), None, 'Saved fixture IDs.')
    writer.results.update(schedule_captures=['first', 'second', 'tokenwise_first', 'tokenwise_second'],
                          tokenwise_steps=len(token_ids),
                          state_format='original; no u16 or alternate attention patch')


def capture_campaign(session, writer, token_ids, arguments):
    folder = Path(arguments.output_directory)/'compiled_kernels'
    with compiled_kernel_manifest(session.reference, folder) as manifest:
        capture_campaign_body(session, writer, token_ids, arguments)
    writer.results['compiled_kernel_manifest'] = dict(path=str(folder/'manifest.json'), complete=manifest['complete'],
        scope='actual parent FP8/FP4 factory objects; helper attention and cuBLAS are separate',
        sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    if os.environ.get('UNDER_GPU_RUN') != '1' and not any(arg in sys.argv for arg in ('--help', '-h')):
        raise RuntimeError('Run original schedule capture through gpu_run.')
    oracle.observed_pass = official_ops.observed_pass
    oracle.CAPTURES['forward'] = capture_campaign
    sys.exit(oracle.main())
