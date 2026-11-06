# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU preservation and exceptional cleanup for actual primitive-statistic taps."""
import types
import unittest

import torch

from official_statistics import layer_zero_statistics


class Norm(torch.nn.Module):
    def forward(self, values):
        values = values.float()
        return values*torch.rsqrt(values.square().mean(-1, keepdim=True)+1e-20)


class Block(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.hc_attn_fn = torch.nn.Parameter(torch.ones(4, 32))
        self.hc_ffn_fn = torch.nn.Parameter(torch.ones(4, 32)*2)
        self.attn_norm, self.ffn_norm = Norm(), Norm()
        self.attn = types.SimpleNamespace(q_norm=Norm(), kv_norm=Norm())
        self.fail = fail

    def hc_mixes(self, values, hc_fn):
        values = values.flatten(2).float()
        inverse = torch.rsqrt(values.square().mean(-1, keepdim=True)+1e-20)
        if self.fail:
            raise RuntimeError('injected original failure')
        return torch.nn.functional.linear(values, hc_fn)*inverse

    def forward(self, values):
        first = self.hc_mixes(values, self.hc_attn_fn)
        collapsed = values[:, :, 0]
        for norm in (self.attn_norm, self.ffn_norm, self.attn.q_norm, self.attn.kv_norm):
            collapsed = norm(collapsed)
        return first+self.hc_mixes(values, self.hc_ffn_fn)+collapsed[..., :4]


class Writer:
    def __init__(self):
        self.records = {}

    def add(self, name, values, *arguments, **keywords):
        if name in self.records:
            raise RuntimeError('duplicate actual primitive')
        self.records[name] = values.detach().clone()


class StatisticsTests(unittest.TestCase):
    def test_actual_values_and_output_bits(self):
        block, writer = Block(), Writer()
        values = torch.arange(96, dtype=torch.bfloat16).reshape(1, 3, 4, 8)
        session = types.SimpleNamespace(model=types.SimpleNamespace(layers=[block]))
        expected = block(values)
        with layer_zero_statistics(session, writer):
            actual = block(values)
        self.assertTrue(torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)))
        self.assertEqual(len(writer.records), 14*3)
        raw = torch.nn.functional.linear(values.flatten(2).float(), block.hc_attn_fn)
        self.assertTrue(torch.equal(writer.records['layer_0.position_0.attn_site.raw_projection'], raw[0, 0]))
        mean = values.flatten(2).float().square().mean(-1, keepdim=True)
        self.assertTrue(torch.equal(writer.records['layer_0.position_0.attn_site.mean'], mean[0, 0]))

    def test_failure_restores_methods_and_hooks(self):
        block = Block(fail=True)
        session = types.SimpleNamespace(model=types.SimpleNamespace(layers=[block]))
        originals = torch.Tensor.mean, torch.rsqrt, torch.nn.functional.linear
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with layer_zero_statistics(session, Writer()):
                block(torch.ones(1, 1, 4, 8))
        self.assertEqual((torch.Tensor.mean, torch.rsqrt, torch.nn.functional.linear), originals)
        self.assertFalse(block._forward_hooks)
        self.assertFalse(block._forward_pre_hooks)


if __name__ == '__main__':
    unittest.main()
