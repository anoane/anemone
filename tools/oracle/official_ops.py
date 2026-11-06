#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Observe layer zero of the existing official-1017 oracle without replacing its arithmetic.

The oracle still owns model loading, quantizer checks and the complete forward.
Wrappers call their saved original exactly once. Point records use the existing
teacher harness's names; additional records retain projection and rounding sites.
No u16 state switch or alternative attention implementation is installed here.
"""
import contextlib
import functools
import os
import sys

import torch

import reference_oracle as oracle
from official_statistics import layer_zero_statistics

SOURCE_PINS = {
    'model.py': 'e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d',
    'kernel.py': '1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455',
}


@contextlib.contextmanager
def layer_zero_moe_points(session, writer):
    """Read the actual gate operands and the actual routed accumulator, without recomputation.

    The Python caller-code checks distinguish the pinned Gate/MoE operations
    from quantizer helpers or another module using the same torch functions.
    Every wrapper calls its saved original exactly once and returns it unchanged.
    """
    reference = session.reference.model
    moe = session.model.layers[0].ffn
    gate = moe.gate
    gate_code, moe_code = type(gate).forward.__code__, type(moe).forward.__code__
    handles, saved = [], []
    active = gate_active = False
    accumulator = None
    expected_shape = None
    counts = dict.fromkeys(('linear', 'topk', 'gather', 'zeros', 'sum'), 0)

    def points(suffix, values):
        for position, row in enumerate(values):
            writer.add(f'layer_0.position_{position}.{suffix}', row, 0,
                       'Actual operand/result from the original Gate/MoE call.', position=position)

    def replace(owner, name, wrapper):
        original = getattr(owner, name)
        saved.append((owner, name, original))
        setattr(owner, name, wrapper(original))

    def moe_begins(module, arguments):
        nonlocal active, expected_shape
        active = True
        values = arguments[0]
        expected_shape = (values.numel()//values.shape[-1], values.shape[-1])

    def moe_ends(module, arguments, output):
        nonlocal active
        active = False
        if any(count != 1 for count in counts.values()):
            raise RuntimeError('original Gate/MoE observation did not see each required operation exactly once')

    def gate_begins(module, arguments):
        nonlocal gate_active
        gate_active = True

    def gate_ends(module, arguments, output):
        nonlocal gate_active
        gate_active = False
        points('router.route_weights', output[0])
        points('router.expert_ids', output[1])

    def linear_wrapper(original):
        def wrapped(values, weight, *arguments, **keywords):
            output = original(values, weight, *arguments, **keywords)
            if gate_active:
                counts['linear'] += 1
                points('router.logits', output)
                points('router.input', values)
                writer.add('layer_0.parameter.router', weight, 0, 'Actual F32 weight operand of original gate.')
            return output
        return wrapped

    def topk_wrapper(original):
        def wrapped(values, *arguments, **keywords):
            if gate_active and sys._getframe(1).f_code is gate_code:
                counts['topk'] += 1
                points('router.biased_scores', values)
            return original(values, *arguments, **keywords)
        return wrapped

    def gather_wrapper(original):
        def wrapped(values, *arguments, **keywords):
            output = original(values, *arguments, **keywords)
            if gate_active and sys._getframe(1).f_code is gate_code:
                counts['gather'] += 1
                points('router.scores', values)
                points('router.selected_scores', output)
            return output
        return wrapped

    def zeros_wrapper(original):
        def wrapped(values, *arguments, **keywords):
            nonlocal accumulator
            output = original(values, *arguments, **keywords)
            if active and sys._getframe(1).f_code is moe_code:
                counts['zeros'] += 1
                if output.dtype != torch.float32 or tuple(output.shape) != expected_shape:
                    raise RuntimeError('original routed accumulator has an unexpected dtype or shape')
                accumulator = output
            return output
        return wrapped

    def before_shared(module, arguments):
        if not active or accumulator is None:
            raise RuntimeError('original shared expert ran before the routed accumulator was observed')
        counts['sum'] += 1
        points('experts.row_sum', accumulator)

    handles.extend((moe.register_forward_pre_hook(moe_begins), moe.register_forward_hook(moe_ends),
                    gate.register_forward_pre_hook(gate_begins), gate.register_forward_hook(gate_ends),
                    moe.shared_experts.register_forward_pre_hook(before_shared)))
    replace(reference, 'linear', linear_wrapper)
    replace(torch.Tensor, 'topk', topk_wrapper)
    replace(torch.Tensor, 'gather', gather_wrapper)
    replace(torch, 'zeros_like', zeros_wrapper)
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)


@contextlib.contextmanager
def layer_zero_points(session, writer):
    reference = session.reference.model
    block = session.model.layers[0]
    handles, saved = [], []
    active = False
    site = None

    def points(suffix, tensor, flattened=False):
        if not active or tensor is None:
            return
        rows = tensor.reshape(-1, *tensor.shape[1:]) if flattened else tensor[0]
        for position, row in enumerate(rows):
            writer.add(f'layer_0.position_{position}.{suffix}', row, 0,
                       'Observed output of the pinned official operation.', position=position)

    def replace(owner, name, function):
        original = getattr(owner, name)
        saved.append((owner, name, original))
        setattr(owner, name, function(original))

    def beginning(module, arguments):
        nonlocal active
        active = True

    def ending(module, arguments, output):
        nonlocal active
        active = False

    def mix_wrapper(original):
        @functools.wraps(original)
        def wrapped(values, weight, scale, base):
            nonlocal site
            site = 'attn_site' if weight is block.hc_attn_fn else 'ffn_site'
            points(site + '.streams', values)
            output = original(values, weight, scale, base)
            for part, tensor in zip(('pre', 'post', 'comb'), output):
                points(site + '.' + part, tensor)
            return output
        return wrapped

    def sinkhorn_wrapper(original):
        def wrapped(mixes, *arguments, **keywords):
            points(site + '.mixes', mixes)
            return original(mixes, *arguments, **keywords)
        return wrapped

    def hook_output(module, suffix, flatten=False):
        handles.append(module.register_forward_hook(
            lambda unused, arguments, output: points(suffix, output, flatten)))

    def hook_input(module, suffix):
        handles.append(module.register_forward_pre_hook(
            lambda unused, arguments: points(suffix, arguments[0])))

    def quant_wrapper(original):
        calls = 0

        def wrapped(values, *arguments, **keywords):
            nonlocal calls
            recording = active
            call = calls
            if recording:
                calls += 1
                writer.add(f'layer_0.quant_{call}.input', values, 0,
                           'Actual quantizer input; expert calls retain their gathered row order.')
            output = original(values, *arguments, **keywords)
            if recording:
                if isinstance(output, tuple):
                    for part, tensor in zip(('codes', 'scales'), output):
                        writer.add(f'layer_0.quant_{call}.' + part, tensor, 0, 'Actual quantizer result.')
                else:
                    writer.add(f'layer_0.quant_{call}.output', values, 0, 'Actual in-place quantizer result.')
            return output
        return wrapped

    def sparse_wrapper(original):
        def wrapped(query, keys, sink, picks, scale):
            points('attn.q', query)
            # The keys are shared by all queries and keep their full logical shape.
            if active:
                writer.add('layer_0.attn.kv', keys, 0, 'Actual input to sparse_attn.')
                writer.add('layer_0.attn.sink', sink, 0, 'Actual attention sink.')
            points('attn.topk_idxs', picks)
            output = original(query, keys, sink, picks, scale)
            points('attn.output_rotated', output)
            return output
        return wrapped

    def rotary_wrapper(original):
        calls = 0

        def wrapped(values, *arguments, **keywords):
            nonlocal calls
            call = calls
            if active:
                calls += 1
                points(f'rope_{call}.input', values)
                if arguments and isinstance(arguments[0], torch.Tensor) and arguments[0].is_complex():
                    writer.add(f'layer_0.rope_{call}.frequencies', torch.view_as_real(arguments[0]), 0,
                               'Actual complex frequency operand, viewed as F32 real/imaginary pairs.')
            output = original(values, *arguments, **keywords)
            if active:
                points(f'rope_{call}.output', values)
            return output
        return wrapped

    handles.append(block.register_forward_pre_hook(beginning))
    handles.append(block.register_forward_hook(ending))
    for module, suffix in ((block.attn_norm, 'attn_site'), (block.ffn_norm, 'ffn_site')):
        hook_input(module, suffix + '.collapsed')
        hook_output(module, suffix + '.normed')
    attention = block.attn
    for module, suffix in ((attention.wq_a, 'attn.wq_a'), (attention.q_norm, 'attn.qr'),
                           (attention.wq_b, 'attn.wq_b'), (attention.wkv, 'attn.wkv'),
                           (attention.kv_norm, 'attn.kv_normed'), (attention.wo_b, 'attn.wo_b')):
        hook_output(module, suffix)
    hook_input(attention.wo_b, 'attn.wo_a')
    hook_output(block.ffn.shared_experts, 'shared_expert.output', True)
    hook_output(block.ffn, 'experts.combined')
    replace(block, 'hc_mixes', mix_wrapper)
    replace(reference, 'hc_split_sinkhorn', sinkhorn_wrapper)
    replace(reference, 'act_quant', quant_wrapper)
    replace(reference, 'sparse_attn', sparse_wrapper)
    replace(reference, 'apply_rotary_emb', rotary_wrapper)
    # Small actual parameters permit later isolated arithmetic probes without reloading the model.
    for name in ('hc_attn_fn', 'hc_attn_scale', 'hc_attn_base', 'hc_ffn_fn', 'hc_ffn_scale', 'hc_ffn_base'):
        writer.add('layer_0.parameter.' + name, getattr(block, name), 0, 'Loaded official parameter.')
    for name, module in (('attn_norm', block.attn_norm), ('ffn_norm', block.ffn_norm)):
        writer.add('layer_0.parameter.' + name, module.weight, 0, 'Loaded official norm weight.')
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)


original_observed_pass = oracle.observed_pass


@contextlib.contextmanager
def observed_pass(session, writer, record_streams, keep_head_input):
    if not record_streams:
        raise RuntimeError('official_ops requires the forward capture mode')
    if any(session.reference.file_digests.get(name) != digest for name, digest in SOURCE_PINS.items()):
        raise RuntimeError('official operation observer requires the pinned 1017 source')
    with original_observed_pass(session, writer, record_streams, keep_head_input) as observation:
        with layer_zero_statistics(session, writer), layer_zero_points(session, writer), layer_zero_moe_points(session, writer):
            recorder = oracle.LinearRecorder(session, writer, [0], 64, observation.routing)
            with recorder.installed():
                yield observation
            if not recorder.calls[0]:
                raise RuntimeError('official layer-zero quantized linears were not observed')
            writer.results['layer_0_quantized_linear_calls'] = recorder.calls[0]
            writer.results['layer_0_routed_accumulator_capture'] = 'actual F32 buffer before shared expert'


if __name__ == '__main__':
    if os.environ.get('UNDER_GPU_RUN') != '1' and not any(option in sys.argv for option in ('--help', '-h')):
        raise RuntimeError('Run official GPU capture through gpu_run.')
    oracle.observed_pass = observed_pass
    sys.exit(oracle.main())
