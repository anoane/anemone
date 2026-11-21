# SPDX-License-Identifier: AGPL-3.0-or-later
"""Observe actual L0 HC raw products and HC/RMS mean/inverse results, without recomputation."""
import contextlib
import sys

import torch


@contextlib.contextmanager
def layer_zero_statistics(session, writer):
    block = session.model.layers[0]
    hc_code = type(block).hc_mixes.__code__
    norm_code = type(block.attn_norm).forward.__code__
    norms = {id(block.attn_norm): 'attn_site.norm', id(block.ffn_norm): 'ffn_site.norm',
             id(block.attn.q_norm): 'attn.qr_norm', id(block.attn.kv_norm): 'attn.kv_norm'}
    expected = {site+'.'+part for site in ('attn_site', 'ffn_site') for part in ('mean', 'inverse', 'raw_projection')}
    expected.update(prefix+'.'+part for prefix in norms.values() for part in ('mean', 'inverse'))
    counts = dict.fromkeys(expected, 0)
    active = False
    original_mean, original_inverse, original_linear = torch.Tensor.mean, torch.rsqrt, torch.nn.functional.linear

    def prefix(frame):
        if not active:
            return None
        owner = frame.f_locals.get('self')
        if frame.f_code is hc_code and owner is block:
            weight = frame.f_locals.get('hc_fn')
            if weight is block.hc_attn_fn:
                return 'attn_site'
            if weight is block.hc_ffn_fn:
                return 'ffn_site'
            raise RuntimeError('original HC used an unknown weight')
        return norms.get(id(owner)) if frame.f_code is norm_code else None

    def record(suffix, values):
        counts[suffix] += 1
        for position, row in enumerate(values[0]):
            writer.add(f'layer_0.position_{position}.{suffix}', row, 0,
                       'Actual intermediate returned to the original HC/RMSNorm forward.', position=position)

    def mean(values, *arguments, **keywords):
        output = original_mean(values, *arguments, **keywords)
        name = prefix(sys._getframe(1))
        if name is not None:
            record(name+'.mean', output)
        return output

    def inverse(values, *arguments, **keywords):
        output = original_inverse(values, *arguments, **keywords)
        name = prefix(sys._getframe(1))
        if name is not None:
            record(name+'.inverse', output)
        return output

    def linear(values, *arguments, **keywords):
        output = original_linear(values, *arguments, **keywords)
        name = prefix(sys._getframe(1))
        if name in ('attn_site', 'ffn_site'):
            record(name+'.raw_projection', output)
        return output

    def begins(module, arguments):
        nonlocal active
        active = True

    def ends(module, arguments, output):
        nonlocal active
        active = False
        if any(count != 1 for count in counts.values()):
            raise RuntimeError('original L0 statistic/product observation missed or repeated an operation')

    handles = (block.register_forward_pre_hook(begins), block.register_forward_hook(ends))
    torch.Tensor.mean, torch.rsqrt, torch.nn.functional.linear = mean, inverse, linear
    try:
        yield
    finally:
        torch.Tensor.mean, torch.rsqrt, torch.nn.functional.linear = original_mean, original_inverse, original_linear
        for handle in handles:
            handle.remove()
