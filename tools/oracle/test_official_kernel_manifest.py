# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU tests that generated-code observation preserves factory calls/objects."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from official_kernel_manifest import compiled_kernel_manifest


class KernelManifestTests(unittest.TestCase):
    def test_objects_calls_deduplication_and_restoration(self):
        compiled = types.SimpleNamespace(get_kernel_source=lambda: 'generated original', execution_backend='fixture')
        fp8, fp4 = mock.Mock(return_value=compiled), mock.Mock(return_value=compiled)
        module = types.ModuleType('fixture_original_kernel')
        module.fp8_gemm_kernel, module.fp4_gemm_kernel = fp8, fp4
        function = types.SimpleNamespace(__module__=module.__name__)
        reference = types.SimpleNamespace(model=types.SimpleNamespace(fp8_gemm=function), file_digests={'pin': 'source'})
        with mock.patch.dict(sys.modules, {module.__name__: module}), tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'generated'
            with compiled_kernel_manifest(reference, folder) as manifest:
                self.assertIs(module.fp8_gemm_kernel(1280, 5120), compiled)
                self.assertIs(module.fp8_gemm_kernel(1280, 5120), compiled)
                self.assertIs(module.fp4_gemm_kernel(2304, 5120), compiled)
            self.assertIs(module.fp8_gemm_kernel, fp8)
            self.assertIs(module.fp4_gemm_kernel, fp4)
            self.assertEqual(fp8.call_count, 2)
            self.assertEqual(fp4.call_count, 1)
            saved = json.loads((folder/'manifest.json').read_text())
            self.assertTrue(saved['complete'])
            self.assertEqual([row['calls'] for row in saved['kernels']], [2, 1])
            self.assertEqual(len(list(folder.glob('*.cu'))), 1)
            self.assertTrue(manifest['complete'])

    def test_failure_restores_factories_and_records_incomplete(self):
        module = types.ModuleType('fixture_original_kernel')
        module.fp8_gemm_kernel, module.fp4_gemm_kernel = mock.Mock(), mock.Mock()
        original = module.fp8_gemm_kernel
        reference = types.SimpleNamespace(model=types.SimpleNamespace(fp8_gemm=types.SimpleNamespace(
            __module__=module.__name__)), file_digests={})
        with mock.patch.dict(sys.modules, {module.__name__: module}), tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'generated'
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with compiled_kernel_manifest(reference, folder):
                    raise RuntimeError('injected')
            self.assertIs(module.fp8_gemm_kernel, original)
            self.assertFalse(json.loads((folder/'manifest.json').read_text())['complete'])


if __name__ == '__main__':
    unittest.main()
