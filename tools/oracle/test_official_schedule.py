# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU checks that schedule observation preserves caches between original steps."""
import contextlib
import types
import unittest
from unittest import mock

import numpy as np
import torch

import official_schedule as schedule
from compare_official_schedule import PositionView, TokenwiseView


class ScheduleTests(unittest.TestCase):
    def test_original_state_is_reset_once_and_carried_through_steps(self):
        writers, resets, positions = [], [], []
        cache = [999]

        class Writer:
            def __init__(self, folder):
                self.folder, self.manifest, self.results, self.seconds = folder, {}, {}, 0.0
                self.values = {}
                writers.append(self)

            def add(self, name, values, *arguments):
                self.values[name] = values.clone()

            def close(self, complete):
                self.complete = complete

        def reset(session):
            resets.append(True)
            cache.clear()

        def forward(inputs, position):
            self.assertEqual(position, len(cache))
            positions.append(position)
            cache.append(int(inputs[0, 0]))
            logits = torch.tensor([[float(sum(cache)), 0.0]])
            observation.head_input = logits.view(1, 1, 2)
            return logits.argmax(-1), logits, None

        observation = types.SimpleNamespace(layers=[{}], head_input=None)
        session = types.SimpleNamespace(model=types.SimpleNamespace(layers=[0], forward=forward,
            head=lambda values, **keywords: values), model_arguments=types.SimpleNamespace(vocab_size=2),
            quantization_counts={})
        with (mock.patch.object(schedule.oracle, 'CaptureWriter', Writer),
              mock.patch.object(schedule.oracle, 'forget_earlier_runs', reset),
              mock.patch.object(schedule.oracle, 'observed_pass', lambda *args: contextlib.nullcontext(observation)),
              mock.patch.object(schedule.oracle, 'synchronized_clock', return_value=0.0),
              mock.patch.object(schedule.oracle, 'CARD_DEVICE', 'cpu')):
            schedule.capture_steps(session, '/unused', [3, 5, 7], {'source': 'fixture'})
        self.assertEqual(len(resets), 1)
        self.assertEqual(positions, [0, 1, 2])
        self.assertEqual([writer.manifest['conditions']['start_pos'] for writer in writers], [0, 1, 2])
        self.assertTrue(all(writer.complete for writer in writers))
        self.assertEqual([writer.values['logits.last_row'][0, 0, 0].item() for writer in writers], [3, 8, 15])

    def test_position_view_maps_only_declared_values_without_casting(self):
        values = {'input.token_ids': np.array([[4, 9]], dtype=np.int64),
                  'input.embeddings': np.array([[[0.0], [-0.0]]], dtype=np.float32),
                  'layer_0.position_1.attn.q': np.array([-0.0], dtype=np.float32),
                  'layer_0.position_0.attn.q': np.array([1.0], dtype=np.float32),
                  'layer_0.linear.expert': np.ones((2, 4), dtype=np.float32)}
        prompt = types.SimpleNamespace(read=lambda name: values[name], manifest_sha256='source',
            records={name: dict(name=name, shape=list(value.shape), dtype=str(value.dtype))
                     for name, value in values.items()})
        view = PositionView(prompt, 1, 2)
        self.assertEqual(set(view.records), {'input.token_ids', 'input.embeddings', 'layer_0.position_0.attn.q'})
        self.assertEqual(view.read('input.token_ids')[0, 0], 9)
        self.assertEqual(view.read('input.embeddings').tobytes(), values['input.embeddings'][:, 1:2].tobytes())
        self.assertEqual(view.read('layer_0.position_0.attn.q').view(np.uint32)[0], 0x80000000)
        with self.assertRaises(ValueError):
            PositionView(prompt, 2, 2)

    def test_tokenwise_view_concatenates_original_storage_and_rejects_casts(self):
        def capture(values, position=None):
            return types.SimpleNamespace(read=lambda name: values[name], manifest_sha256=str(position),
                manifest={'conditions': {'start_pos': position}},
                records={name: dict(name=name, shape=list(value.shape), dtype=str(value.dtype))
                         for name, value in values.items()})

        prompt = capture({'input.token_ids': np.array([[3, 5]], dtype=np.int64),
                          'input.embeddings': np.ones((1, 2, 1), dtype=np.float32),
                          'layer_0.position_1.attn.q': np.ones(1, dtype=np.float32)})
        steps = [capture({'input.token_ids': np.array([[token]], dtype=np.int64),
                          'input.embeddings': np.array([[[-0.0]]], dtype=np.float32),
                          'layer_0.position_0.attn.q': np.array([-0.0], dtype=np.float32)}, position)
                 for position, token in enumerate((3, 5))]
        view = TokenwiseView(prompt, steps)
        self.assertEqual(view.read('input.token_ids').tolist(), [[3, 5]])
        self.assertTrue(np.all(view.read('input.embeddings').view(np.uint32) == 0x80000000))
        self.assertEqual(view.read('layer_0.position_1.attn.q').view(np.uint32)[0], 0x80000000)
        steps[1].records['input.embeddings']['dtype'] = 'float16'
        with self.assertRaises(ValueError):
            TokenwiseView(prompt, steps)

    def test_shared_key_extent_is_not_a_token_axis_and_routes_use_axis_zero(self):
        def capture(values, position=None):
            return types.SimpleNamespace(read=lambda name: values[name], manifest_sha256=str(position),
                manifest={'conditions': {'start_pos': position}},
                records={name: dict(name=name, shape=list(value.shape), dtype=str(value.dtype))
                         for name, value in values.items()})

        prompt = capture({'input.token_ids': np.array([[3, 5]], dtype=np.int64),
            'layer_0.attn.kv': np.ones((1, 2, 8), dtype=np.float32),
            'layer_0.position_1.attn.topk_idxs': np.zeros(2, dtype=np.int64),
            'layer_0.linear.expert': np.ones((1, 2, 8), dtype=np.float32),
            'layer_0.mlp.routed_expert_ids': np.array([[4, 6], [1, 8]], dtype=np.int64)})
        steps = [capture({'input.token_ids': np.array([[token]], dtype=np.int64),
            'layer_0.attn.kv': np.ones((1, 128, 8), dtype=np.float32),
            'layer_0.position_0.attn.topk_idxs': np.zeros(128, dtype=np.int64),
            'layer_0.linear.expert': np.ones((1, 3, 8), dtype=np.float32),
            'layer_0.mlp.routed_expert_ids': np.array([routes], dtype=np.int64)}, position)
                 for position, (token, routes) in enumerate(((3, [4, 6]), (5, [1, 8])))]
        view = TokenwiseView(prompt, steps)
        self.assertEqual(set(view.records), {'input.token_ids', 'layer_0.mlp.routed_expert_ids'})
        self.assertEqual(view.read('layer_0.mlp.routed_expert_ids').tolist(), [[4, 6], [1, 8]])
        position = PositionView(prompt, 1, 2)
        self.assertEqual(set(position.records), set(view.records))
        self.assertEqual(position.read('layer_0.mlp.routed_expert_ids').tolist(), [[1, 8]])


if __name__ == '__main__':
    unittest.main()
