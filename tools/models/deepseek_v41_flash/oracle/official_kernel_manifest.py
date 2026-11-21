# SPDX-License-Identifier: AGPL-3.0-or-later
"""Retain generated source from the actual parent-process TileLang GEMM factories.

Factories are called exactly once and their original compiled objects are
returned unchanged. This observes FP8/FP4 GEMMs, not the separate attention
helper or cuBLAS. Saved generated source is evidence, never implementation code
to copy into Anemone. Caller owns the guarded, single-threaded capture process.
"""
import contextlib
import hashlib
import json
from pathlib import Path
import sys


@contextlib.contextmanager
def compiled_kernel_manifest(reference, folder):
    folder = Path(folder)
    folder.mkdir()
    module = sys.modules[reference.model.fp8_gemm.__module__]
    originals, seen, records = {}, {}, []
    tilelang = sys.modules.get('tilelang')
    manifest = dict(complete=False, scope=__doc__, reference_files_sha256=reference.file_digests,
                    tilelang_version=str(getattr(tilelang, '__version__', 'unreported')), kernels=records)

    def wrap(name, original):
        def factory(*arguments, **keywords):
            compiled = original(*arguments, **keywords)
            argument_text = tuple(str(value) for value in arguments)
            keyword_text = tuple((key, str(value)) for key, value in sorted(keywords.items()))
            key = (name, id(compiled), argument_text, keyword_text)
            if key in seen:
                seen[key]['calls'] += 1
                return compiled
            generated = compiled.get_kernel_source()
            if not isinstance(generated, str) or not generated:
                raise RuntimeError('compiled original GEMM has no generated source')
            digest = hashlib.sha256(generated.encode()).hexdigest()
            path = folder/(digest+'.cu')
            if path.exists():
                if path.read_text() != generated:
                    raise RuntimeError('generated-source digest collision')
            else:
                with path.open('x') as stream:
                    stream.write(generated)
            record = dict(factory=name, arguments=list(argument_text), keywords=dict(keyword_text), calls=1,
                          source_sha256=digest, source_file=path.name,
                          execution_backend=str(getattr(compiled, 'execution_backend', 'unreported')))
            seen[key] = record
            # Retain object identity until the observation ends; avoid ID reuse.
            record_index = len(records)
            records.append(record)
            retained[record_index] = compiled
            return compiled
        return factory

    retained = {}
    try:
        for name in ('fp8_gemm_kernel', 'fp4_gemm_kernel'):
            originals[name] = getattr(module, name)
            setattr(module, name, wrap(name, originals[name]))
        yield manifest
        if not records or not all(any(row['factory'] == name for row in records) for name in originals):
            raise RuntimeError('both original GEMM factory kinds must actually run')
        manifest['complete'] = True
    finally:
        for name, original in originals.items():
            setattr(module, name, original)
        with (folder/'manifest.json').open('x') as stream:
            stream.write(json.dumps(manifest, indent=2)+'\n')
