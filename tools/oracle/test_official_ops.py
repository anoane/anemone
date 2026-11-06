# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU checks of observer preservation, row mapping and wrapper cleanup."""
import types
import unittest

import torch

from official_ops import layer_zero_points, layer_zero_moe_points
import reference_oracle as oracle


class Writer:
    def __init__(self):
        self.records = {}

    def add(self, name, value, *arguments, **keywords):
        if name in self.records:
            raise RuntimeError('duplicate capture name: ' + name)
        self.records[name] = value.detach().clone()


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('wq_a', 'q_norm', 'wq_b', 'wkv', 'kv_norm', 'wo_b'):
            setattr(self, name, torch.nn.Identity())

    def forward(self, value):
        for name in ('wq_a', 'q_norm', 'wq_b', 'wkv', 'kv_norm', 'wo_b'):
            value = getattr(self, name)(value)
        return value


class FeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_experts = torch.nn.Identity()

    def forward(self, value):
        return self.shared_experts(value.flatten(0, 1)).view_as(value)


class Block(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.reference = reference
        self.attn_norm = torch.nn.LayerNorm(8)
        self.ffn_norm = torch.nn.LayerNorm(8)
        self.attn = Attention()
        self.ffn = FeedForward()
        for name in ('hc_attn_fn', 'hc_attn_scale', 'hc_attn_base',
                     'hc_ffn_fn', 'hc_ffn_scale', 'hc_ffn_base'):
            setattr(self, name, torch.nn.Parameter(torch.ones(4)))

    def hc_mixes(self, value, weight, scale, base):
        mixes = value[..., :4]
        return self.reference.hc_split_sinkhorn(mixes, scale, base)

    def forward(self, value):
        self.hc_mixes(value, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        value = self.attn(self.attn_norm(value))
        self.hc_mixes(value, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        return self.ffn(self.ffn_norm(value))


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.reference = types.SimpleNamespace(
            hc_split_sinkhorn=lambda mixes, *arguments: (mixes, mixes, mixes),
            act_quant=lambda values, *arguments, **keywords: values,
            sparse_attn=lambda query, *arguments: query,
            apply_rotary_emb=lambda values, *arguments, **keywords: values)
        self.block = Block(self.reference)
        self.session = types.SimpleNamespace(reference=types.SimpleNamespace(model=self.reference),
            model=types.SimpleNamespace(layers=[self.block]))
        self.values = torch.arange(24, dtype=torch.float32).view(1, 3, 8)

    def test_output_bits_and_row_mapping(self):
        baseline = self.block(self.values)
        writer = Writer()
        original_sinkhorn = self.reference.hc_split_sinkhorn
        with layer_zero_points(self.session, writer):
            observed = self.block(self.values)
        self.assertTrue(torch.equal(baseline.view(torch.uint8), observed.view(torch.uint8)))
        for position in range(3):
            name = f'layer_0.position_{position}.shared_expert.output'
            self.assertEqual(writer.records[name].shape, (8,))
            self.assertTrue(torch.equal(writer.records[name], baseline[0, position]))
        self.assertIs(self.reference.hc_split_sinkhorn, original_sinkhorn)
        self.assertFalse(self.block._forward_hooks)
        self.assertFalse(self.block._forward_pre_hooks)

    def test_exception_restores_wrappers(self):
        original_quant = self.reference.act_quant
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with layer_zero_points(self.session, Writer()):
                raise RuntimeError('deliberate')
        self.assertIs(self.reference.act_quant, original_quant)
        self.assertFalse(self.block._forward_hooks)
        self.assertFalse(self.block._forward_pre_hooks)

    def test_nested_observers_preserve_linear_wrappers(self):
        self.reference.linear = torch.nn.functional.linear
        self.block.ffn = MoE(self.reference)
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([self.block])
        self.session.model, self.session.stages = model, []
        baseline = self.block(self.values)
        writer = Writer()
        with layer_zero_points(self.session, writer), layer_zero_moe_points(self.session, writer):
            recorder = oracle.LinearRecorder(self.session, writer, [0], 64, {})
            with recorder.installed():
                observed = self.block(self.values)
        self.assertTrue(torch.equal(baseline.view(torch.uint8), observed.view(torch.uint8)))
        self.assertIs(self.reference.linear, torch.nn.functional.linear)


class Gate(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.reference = reference
        self.weight = torch.nn.Parameter(torch.ones(3, 8))
        self.bias = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))

    def forward(self, values):
        scores = self.reference.linear(values.float(), self.weight.float())
        scores = torch.nn.functional.softplus(scores).sqrt()
        indices = (scores + self.bias).topk(2, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
        return weights*1.5, indices


class MoE(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.gate = Gate(reference)
        self.shared_experts = torch.nn.Identity()

    def forward(self, values):
        shape = values.shape
        values = values.view(-1, shape[-1])
        self.gate(values)
        accumulator = torch.zeros_like(values, dtype=torch.float32)
        accumulator += values*2
        accumulator += self.shared_experts(values)
        return accumulator.view(shape)


class MoEObserverTests(unittest.TestCase):
    def setUp(self):
        self.reference = types.SimpleNamespace(linear=torch.nn.functional.linear)
        self.moe = MoE(self.reference)
        self.session = types.SimpleNamespace(reference=types.SimpleNamespace(model=self.reference),
            model=types.SimpleNamespace(layers=[types.SimpleNamespace(ffn=self.moe)]))
        self.values = torch.arange(1, 25, dtype=torch.float32).view(1, 3, 8)

    def test_actual_accumulator_and_output_bits(self):
        baseline = self.moe(self.values)
        writer = Writer()
        with layer_zero_moe_points(self.session, writer):
            observed = self.moe(self.values)
        self.assertTrue(torch.equal(baseline.view(torch.uint8), observed.view(torch.uint8)))
        for position in range(3):
            actual = writer.records[f'layer_0.position_{position}.experts.row_sum']
            self.assertTrue(torch.equal(actual, self.values[0, position]*2))
        self.assertIn('layer_0.position_0.router.logits', writer.records)
        self.assertIn('layer_0.position_0.router.biased_scores', writer.records)
        self.assertIn('layer_0.position_0.router.scores', writer.records)

    def test_exception_restores_torch_methods(self):
        before = torch.Tensor.topk, torch.Tensor.gather, torch.zeros_like, self.reference.linear
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with layer_zero_moe_points(self.session, Writer()):
                raise RuntimeError('deliberate')
        after = torch.Tensor.topk, torch.Tensor.gather, torch.zeros_like, self.reference.linear
        self.assertEqual(before, after)

if __name__ == '__main__':
    unittest.main()
