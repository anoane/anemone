# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU checks of actual norm-statistic observation and exceptional cleanup."""
import unittest

import torch

from official_norm_reference import norm_statistics


class Norm(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail

    def forward(self, values):
        original_dtype = values.dtype
        values = values.float()
        variance = values.square().mean(-1, keepdim=True)
        if self.fail:
            raise RuntimeError('injected original norm failure')
        return (values*torch.rsqrt(variance+1e-20)).to(original_dtype)


class NormReferenceTests(unittest.TestCase):
    def test_original_output_preserved_and_actual_statistics_retained(self):
        values = torch.arange(24, dtype=torch.bfloat16).reshape(1, 3, 8)
        norm = Norm()
        original_mean, original_inverse = torch.Tensor.mean, torch.rsqrt
        expected = norm(values)
        output, mean, inverse = norm_statistics(norm, values)
        self.assertTrue(torch.equal(expected.view(torch.uint8), output.view(torch.uint8)))
        self.assertTrue(torch.equal(mean, values.float().square().mean(-1, keepdim=True)))
        self.assertTrue(torch.equal(inverse, torch.rsqrt(mean+1e-20)))
        self.assertIs(torch.Tensor.mean, original_mean)
        self.assertIs(torch.rsqrt, original_inverse)

    def test_original_failure_restores_torch_methods(self):
        original_mean, original_inverse = torch.Tensor.mean, torch.rsqrt
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            norm_statistics(Norm(fail=True), torch.ones(1, 1, 8, dtype=torch.bfloat16))
        self.assertIs(torch.Tensor.mean, original_mean)
        self.assertIs(torch.rsqrt, original_inverse)


if __name__ == '__main__':
    unittest.main()
