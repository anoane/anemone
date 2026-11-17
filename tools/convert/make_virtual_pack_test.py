#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""make_virtual_pack.py on a small synthetic model: a native pack and an EXL3 hybrid checkpoint of 2 layers of 2
experts. The map builds from config.json and quantization_config.json alone (an unknown index with its provenance),
twice to the same bytes, with its descriptor; each disagreement is refused, and nothing is written then."""
import hashlib
import json
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_pack as pack  # noqa: E402

TOOL = HERE / 'make_virtual_pack.py'
BLOCK = 4096
HIDDEN, INTERMEDIATE, LAYERS, EXPERTS = 32, 16, 2, 2
KINDS = (('w1', INTERMEDIATE, HIDDEN), ('w2', HIDDEN, INTERMEDIATE), ('w3', INTERMEDIATE, HIDDEN))
TILE_WORDS = {0: 48, 1: 40}  # layer 0 at K3, layer 1 at K2.5
NATIVE_EXPERT_FORMAT = 3
MUL1_MULTIPLIER = 0x83DCD12D
HALF_ONE = b'\x00\x3c'
SOURCE = 'example/synthetic-exl3 @ 0123abcd'
REFERENCE = 'the synthetic qualification, its records'
GEOMETRY = {'hidden_size': HIDDEN, 'moe_intermediate_size': INTERMEDIATE, 'n_routed_experts': EXPERTS,
            'num_hidden_layers': LAYERS}


def native_pack(path):
    """Block 0, then the tables: an embedding extent, then one extent an expert of its 3 matrices."""
    rows = [('embed.weight', 0, (HIDDEN, HIDDEN), 0)]
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            for kind, outputs, inputs in KINDS:
                rows.append((f'layers.{layer}.ffn.experts.{expert}.{kind}.weight', 1 + layer * EXPERTS + expert,
                             (outputs, inputs), NATIVE_EXPERT_FORMAT))
    extent_count, names = 1 + LAYERS * EXPERTS, b''.join(name.encode() for name, *_ in rows)
    tables = [pack.EXTENT_RECORD.pack(BLOCK * (index + 1), BLOCK, 0, index, 0) for index in range(extent_count)]
    tables += [pack.UNIT_RECORD.pack(0, index) for index in range(extent_count)]
    name_offset = 0
    for name, extent, dimensions, weight_format in rows:
        tables.append(pack.TENSOR_RECORD.pack(0, 2 * dimensions[0] * dimensions[1], extent, name_offset, len(name),
                                              0, 2, *dimensions, 0, 0, weight_format, 0, 0))
        name_offset += len(name)
    table_bytes = b''.join(tables) + names
    tables_byte_count = (len(table_bytes) + BLOCK - 1) // BLOCK * BLOCK
    header = pack.HEADER_RECORD.pack(b'ANEMONE\0', 2, BLOCK, b'synthetic', BLOCK + tables_byte_count, tables_byte_count,
                                     0, extent_count, extent_count, len(rows), len(names), 0)
    path.write_bytes(header.ljust(BLOCK, b'\0') + table_bytes.ljust(tables_byte_count, b'\0'))


def hybrid_checkpoint(folder, config_text, storage_bits):
    """config.json, one safetensors shard with every expert's trellis, suh, svh and mul1, and its index."""
    folder.mkdir()
    entries, payload, weight_map, storage = {}, bytearray(), {}, {}
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            for kind, outputs, inputs in KINDS:
                prefix = f'layers.{layer}.ffn.experts.{expert}.{kind}'
                tiles = [inputs // 16, outputs // 16, TILE_WORDS[layer]]
                for suffix, dtype, shape, data in (
                        ('trellis', 'I16', tiles, bytes(2 * tiles[0] * tiles[1] * tiles[2])),
                        ('suh', 'F16', [inputs], HALF_ONE * inputs), ('svh', 'F16', [outputs], HALF_ONE * outputs),
                        ('mul1', 'I32', [], struct.pack('<I', MUL1_MULTIPLIER))):
                    entries[prefix + '.' + suffix] = {'dtype': dtype, 'shape': shape,
                                                      'data_offsets': [len(payload), len(payload) + len(data)]}
                    weight_map[prefix + '.' + suffix] = 'experts.safetensors'
                    payload += data
                storage[prefix] = {'quant_format': 'exl3', 'mul1_multiplier': MUL1_MULTIPLIER,
                                   'bits_per_weight': storage_bits.get(prefix, TILE_WORDS[layer] / 16)}
    header = json.dumps(entries).encode()
    (folder / 'experts.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + payload)
    (folder / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {}, 'weight_map': weight_map}))
    (folder / 'config.json').write_text(json.dumps({'model_type': 'deepseek_v41', 'text_config': config_text}))
    return {'tensor_storage': storage}


class VirtualPackTest(unittest.TestCase):
    def setUp(self):
        self.folder = pathlib.Path(tempfile.mkdtemp(prefix='make_virtual_pack_test.'))
        self.addCleanup(shutil.rmtree, self.folder)
        native_pack(self.folder / 'original.anemone')

    def build(self, output_name, *options, checkpoint='checkpoint', config_text=None, storage_bits=None):
        """The tool's run on CHECKPOINT (made on first use) into OUTPUT_NAME.exl3map."""
        hybrid, quantization_path = self.folder / checkpoint, self.folder / (checkpoint + '.quantization_config.json')
        if not hybrid.exists():
            quantization = hybrid_checkpoint(hybrid, dict(GEOMETRY, **(config_text or {})), storage_bits or {})
            quantization_path.write_text(json.dumps(quantization))
        output = self.folder / (output_name + '.exl3map')
        done = subprocess.run([sys.executable, '-I', str(TOOL), '--native', str(self.folder / 'original.anemone'),
                               '--hybrid', str(hybrid), '--output', str(output), '--exl3-config',
                               str(quantization_path), *options], capture_output=True, text=True)
        return done, output

    def refused(self, name, message, *options, **checkpoint):
        done, output = self.build(name, *options, checkpoint=name, **checkpoint)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn(message, done.stderr)
        self.assertFalse(output.exists() or output.with_name(output.name + '.descriptor.json').exists())

    def test_builds_from_config_with_descriptor(self):
        record = self.folder / 'capture.json'
        record.write_text('{}')
        first, output = self.build('first', '--exl3-source', SOURCE)
        self.assertEqual(first.returncode, 0, first.stderr)
        second, again = self.build('second', '--exl3-source', SOURCE, '--quality-record', str(record))
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(output.read_bytes(), again.read_bytes())
        descriptor = json.loads(output.with_name(output.name + '.descriptor.json').read_text())
        self.assertEqual(descriptor['quantiser']['source'], SOURCE)
        self.assertEqual(descriptor['map']['sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(descriptor['experts']['matrices'], {kind: [rows, columns] for kind, rows, columns in KINDS})
        self.assertEqual(descriptor['widths']['1']['w2'], {'K2.5': EXPERTS})
        self.assertEqual(descriptor['widths_summary'],
                         {'matrices': {'K3': 6, 'K2.5': 6}, 'layers': {'K3': 1, 'K2.5': 1}})
        self.assertEqual(descriptor['quality'], {'status': 'none'})
        recorded = json.loads(again.with_name(again.name + '.descriptor.json').read_text())['quality']
        self.assertEqual(recorded['status'], 'recorded')
        self.assertEqual(recorded['sha256'], hashlib.sha256(b'{}').hexdigest())
        self.assertNotEqual(self.build('first', '--exl3-source', SOURCE)[0].returncode, 0)  # epochs are kept
        third, established = self.build('third', '--exl3-source', SOURCE, '--quality-established', REFERENCE)
        self.assertEqual(third.returncode, 0, third.stderr)
        self.assertEqual(output.read_bytes(), established.read_bytes())
        other = json.loads(established.with_name(established.name + '.descriptor.json').read_text())
        self.assertEqual(other.pop('quality'), {'status': 'established', 'reference': REFERENCE})
        self.assertEqual(other['map'].pop('file'), 'third.exl3map')
        del descriptor['quality'], descriptor['map']['file']
        self.assertEqual(other, descriptor)

    def test_config_option(self):
        config = self.folder / 'config.json'
        config.write_text(json.dumps({'text_config': GEOMETRY}))
        self.refused('other', 'not a routed expert', '--exl3-source', SOURCE, config_text={'n_routed_experts': 1})
        done, output = self.build('given', '--exl3-source', SOURCE, '--config', str(config), checkpoint='other')
        self.assertEqual(done.returncode, 0, done.stderr)
        descriptor = json.loads(output.with_name(output.name + '.descriptor.json').read_text())
        self.assertEqual(descriptor['architecture']['config_sha256'], hashlib.sha256(config.read_bytes()).hexdigest())

    def test_refusals(self):
        self.refused('unknown', 'give its provenance with --exl3-source')
        self.refused('both', 'recorded or established, not both', '--exl3-source', SOURCE, '--quality-record',
                     str(self.folder / 'capture.json'), '--quality-established', REFERENCE)
        self.refused('empty', 'needs the reference', '--exl3-source', SOURCE, '--quality-established', ' ')
        self.refused('sourceless', 'needs the checkpoint\'s provenance', '--exl3-source', ' \t')
        self.refused('wide', 'expert geometry', '--exl3-source', SOURCE, config_text={'hidden_size': 2 * HIDDEN})
        self.refused('short', 'not a routed expert', '--exl3-source', SOURCE, config_text={'n_routed_experts': 1})
        self.refused('more', 'layers of 3 experts', '--exl3-source', SOURCE, config_text={'n_routed_experts': 3})
        self.refused('nameless', 'no positive integer num_hidden_layers', '--exl3-source', SOURCE,
                     config_text={'num_hidden_layers': None})
        self.refused('bits', 'quantization_config disagrees with the trellis', '--exl3-source', SOURCE,
                     storage_bits={'layers.1.ffn.experts.0.w1': 3.0})


if __name__ == '__main__':
    unittest.main()
