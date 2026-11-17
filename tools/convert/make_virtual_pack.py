#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build a read-only scatter manifest, preserving native extents and indexed EXL3 bytes.
CPU-only. No checkpoint payloads are modified. No complete weight copy is made.

Any EXL3 checkpoint of the model is admitted when every routed expert checks out against the checkpoint's own
config.json and quantization_config.json: each matrix's shape, dtypes, stored width, mul1 codebook, and the experts
and matrices of every layer. Beside the map, <map>.descriptor.json records its provenance, width table and quality
status: none (served only on an explicit opt-in), recorded (an NLL/PPL capture) or established (a qualification).
"""
import argparse
import concurrent.futures
import functools
import hashlib
import json
import os
import pathlib
import re
import struct
import sys
import time
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_pack as pack

NATIVE = HYBRID = DEST = None
# The provenance of the hybrid indexes make_hybrid.py wrote for the checkpoints built so far (from the official index
# and each EXL3 checkpoint's); another checkpoint's is given with --exl3-source. Provenance, not an allowlist.
KNOWN_INDEXES = {
    'fa24746ffa8be36473aca6c99b432f3b0230f17237f0873f2db9be598d324e81':
        'LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw @ 0be63c33',
    '9ebd8f1fb7e92ef49d043f3b8cf01711299208bfb22d745e49b9b8fa78c93032':
        'turboderp/DeepSeek-V4.1-Flash-1017-exl3 2.68bpw @ 608c237e',
}
# The pack's stored width (include/anemone/exl3_records.h) of each trellis tile width the engine decodes: a tile is
# 16 K int16; a whole K stores K, a half K stores 16 K (K 2, 2.5, 3, 4).
STORED_WIDTHS = {32: 2, 40: 40, 48: 3, 64: 4}
TILE_WORDS_PER_K = 16
MUL1_MULTIPLIER = 0x83DCD12D
# A routed expert's matrices in the native pack: layers.<layer>.ffn.experts.<expert>.<kind>.weight, two of them
# (intermediate, hidden) and one (hidden, intermediate) in (outputs, inputs).
EXPERT_NAME = re.compile(r'layers\.(\d+)\.ffn\.experts\.(\d+)\.(\w+)\.weight')
EXPERT_MATRICES = 3
WIDENING_MATRICES = 2  # (intermediate, hidden): gate and up
GEOMETRY_KEYS = ('hidden_size', 'moe_intermediate_size', 'n_routed_experts', 'num_hidden_layers')
DESCRIPTOR_FORMAT_VERSION = 1
SPACES = ' \t\n\v\f\r'  # a text of only these is none, as the engine's descriptor reader holds (exl3_descriptor.c)
HASH_CHUNK_BYTES = 64 * 1024 * 1024
HEADER = struct.Struct('<8sQII40s')
SEGMENT = struct.Struct('<QQQII')
PATH_BYTES = 512
BLOCK = 4096


def align(value, multiple):
    return (value + multiple - 1) // multiple * multiple


@functools.lru_cache(maxsize=100)
def shard_header(shard):
    with (HYBRID / shard).open('rb') as stream:
        count = struct.unpack('<Q', stream.read(8))[0]
        if count > 64 * 1024 * 1024:
            raise ValueError('unbounded safetensors header')
        return count + 8, json.loads(stream.read(count))


def digest(data):
    # Existing canonical pack digest, unchanged.
    return pack.digest_of(data)


def file_sha256(path):
    checksum = hashlib.sha256()
    with open(path, 'rb') as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            checksum.update(chunk)
    return checksum.hexdigest()


def expert_geometry(config):
    """The routed experts' geometry from config.json: its text_config's when it has one."""
    text = config.get('text_config', config)
    missing = [key for key in GEOMETRY_KEYS if not isinstance(text.get(key), int) or text[key] <= 0]
    if missing:
        raise ValueError(f'config.json: no positive integer {", ".join(missing)}')
    return {key: text[key] for key in GEOMETRY_KEYS}


def width_name(tile_words):
    return f'K{tile_words / TILE_WORDS_PER_K:g}'


def main():
    global NATIVE, HYBRID, DEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', required=True, type=pathlib.Path)
    parser.add_argument('--hybrid', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    parser.add_argument('--exl3-config', required=True, type=pathlib.Path,
                        help="the EXL3 checkpoint's quantization_config.json: each matrix's bits_per_weight and mul1 "
                             "multiplier must agree with its trellis")
    parser.add_argument('--exl3-source', help='the EXL3 checkpoint\'s provenance, "repository @ revision"; required '
                                              'for a checkpoint the tool has not built before')
    parser.add_argument('--config', type=pathlib.Path,
                        help="the model's config.json, for the experts' geometry (default: the hybrid's own)")
    parser.add_argument('--front-end-tables', type=pathlib.Path,
                        help='the front_end.tables the map is served with: its sha256 goes into the descriptor')
    parser.add_argument('--quality-record', type=pathlib.Path,
                        help="the NLL/PPL capture that covers this map: without one, the map's descriptor says it "
                             "has no quality record, and it is served only on an explicit opt-in")
    parser.add_argument('--quality-established', metavar='REFERENCE',
                        help='what established the map\'s quality without a capture (a qualification, by its records)')
    args = parser.parse_args()
    if args.quality_record is not None and args.quality_established is not None:
        parser.error('a map\'s quality is recorded or established, not both')
    if args.quality_established is not None and not args.quality_established.strip(SPACES):
        parser.error('--quality-established needs the reference that established the quality')
    if args.exl3_source is not None and not args.exl3_source.strip(SPACES):
        parser.error('--exl3-source needs the checkpoint\'s provenance, "repository @ revision"')
    NATIVE, HYBRID, DEST = (path.absolute() for path in (args.native, args.hybrid, args.output))
    if DEST.suffix != '.exl3map':
        parser.error('the runtime requires an .exl3map output suffix')
    descriptor_path = DEST.with_name(DEST.name + '.descriptor.json')
    if DEST.exists() or descriptor_path.exists():
        raise ValueError('destination exists; preserve previous epochs before replacing')
    started = time.monotonic()
    index_bytes = (HYBRID / 'model.safetensors.index.json').read_bytes()
    index_sha256 = hashlib.sha256(index_bytes).hexdigest()
    checkpoint = KNOWN_INDEXES.get(index_sha256, args.exl3_source)
    if checkpoint is None or args.exl3_source not in (None, checkpoint):
        raise ValueError(f'checkpoint index {index_sha256}: give its provenance with --exl3-source '
                         f'"repository @ revision"' if checkpoint is None else
                         f'--exl3-source {args.exl3_source!r} disagrees with the index\'s {checkpoint!r}')
    print(f'Checkpoint: {checkpoint}', flush=True)
    index = json.loads(index_bytes)['weight_map']
    config_bytes = (args.config or HYBRID / 'config.json').read_bytes()
    config = json.loads(config_bytes)
    geometry = expert_geometry(config)
    hidden, intermediate = geometry['hidden_size'], geometry['moe_intermediate_size']
    layer_count, expert_count = geometry['num_hidden_layers'], geometry['n_routed_experts']
    quantization_bytes = args.exl3_config.read_bytes()
    storage = json.loads(quantization_bytes)['tensor_storage']
    expert_matrices = {}  # (layer, expert) -> {kind: (outputs, inputs)}
    widths = {}  # layer -> kind -> width name -> matrices
    with NATIVE.open('rb') as stream:
        head = list(pack.HEADER_RECORD.unpack(stream.read(128)))
        stream.seek(0)
        tables = bytearray(stream.read(BLOCK + head[5]))
    if head[0] != b'ANEMONE\0' or head[1] != 2:
        raise ValueError('native pack version')
    native_header_digest = head[11]
    extent_count, unit_count, tensor_count = head[7:10]
    extent_begin = BLOCK
    unit_begin = extent_begin + extent_count * 32
    tensor_begin = unit_begin + unit_count * 8
    name_begin = tensor_begin + tensor_count * 64
    extents = [list(pack.EXTENT_RECORD.unpack_from(tables, extent_begin + i * 32))
               for i in range(extent_count)]
    tensors = [list(pack.TENSOR_RECORD.unpack_from(tables, tensor_begin + i * 64))
               for i in range(tensor_count)]
    names = [bytes(tables[name_begin+t[3]:name_begin+t[3]+t[4]]).decode() for t in tensors]
    grouped = [[] for _ in extents]
    for i, tensor in enumerate(tensors):
        grouped[tensor[2]].append(i)
    paths = [str(DEST), str(NATIVE)]
    path_index = {str(NATIVE): 1}
    pieces = []
    new_offset = BLOCK + head[5]
    jobs = []
    changed = 0
    for extent_index, extent in enumerate(extents):
        selected = grouped[extent_index]
        is_expert = bool(selected) and all(names[i].startswith('layers.') and '.ffn.experts.' in names[i]
                                           for i in selected)
        if not is_expert:
            pieces.append((new_offset, extent[1], extent[0], 1, 0))
            extent[0] = new_offset
            new_offset += extent[1]
            continue
        matrix_pieces = []
        extent_bytes = 0
        for tensor_index in selected:
            tensor = tensors[tensor_index]
            name = names[tensor_index]
            if not name.endswith('.weight') or tensor[11] != 3:
                raise ValueError(('unexpected native expert tensor', name, tensor))
            prefix = name[:-len('.weight')]
            outputs, inputs = tensor[7:9]
            parts = EXPERT_NAME.fullmatch(name)
            if parts is None or int(parts[1]) >= layer_count or int(parts[2]) >= expert_count:
                raise ValueError(f'{name}: not a routed expert of the {layer_count} layers and {expert_count} experts '
                                 f'config.json gives')
            if (outputs, inputs) not in ((intermediate, hidden), (hidden, intermediate)):
                raise ValueError(f'{name}: expert geometry {(outputs, inputs)}, config.json gives hidden {hidden} '
                                 f'and intermediate {intermediate}')
            expert_matrices.setdefault((int(parts[1]), int(parts[2])), {})[parts[3]] = (outputs, inputs)
            trellis_shard = index[prefix + '.trellis']
            tile_words = shard_header(trellis_shard)[1][prefix + '.trellis']['shape'][-1]
            if tile_words not in STORED_WIDTHS:
                raise ValueError(f'{prefix}: EXL3 K={tile_words / 16} is not supported (supported: 2, 2.5, 3, 4)')
            entry = storage.get(prefix, {})
            if (entry.get('quant_format'), (entry.get('bits_per_weight') or 0) * 16,
                    entry.get('mul1_multiplier')) != ('exl3', tile_words, MUL1_MULTIPLIER):
                raise ValueError(('quantization_config disagrees with the trellis', prefix, tile_words, entry))
            layer_widths = widths.setdefault(int(parts[1]), {}).setdefault(parts[3], {})
            layer_widths[width_name(tile_words)] = layer_widths.get(width_name(tile_words), 0) + 1
            extent_bytes = align(extent_bytes, 64)
            tensor[0] = extent_bytes
            tensor[11], tensor[12] = 2, STORED_WIDTHS[tile_words]
            specs = [('trellis', 'I16', [inputs//16, outputs//16, tile_words]),
                     ('suh', 'F16', [inputs]), ('svh', 'F16', [outputs])]
            marker_name = prefix + '.mul1'
            marker_shard = index[marker_name]
            payload_begin, marker_header = shard_header(marker_shard)
            marker = marker_header[marker_name]
            with (HYBRID / marker_shard).open('rb') as stream:
                stream.seek(payload_begin + marker['data_offsets'][0])
                if (marker['dtype'], marker['shape'], stream.read(4)) != ('I32', [], struct.pack('<I', MUL1_MULTIPLIER)):
                    raise ValueError('unsupported codebook')
            matrix_bytes = 0
            for suffix, dtype, shape in specs:
                key = prefix + '.' + suffix
                shard = index[key]
                payload_begin, header = shard_header(shard)
                desc = header[key]
                if desc['dtype'] != dtype or desc['shape'] != shape:
                    raise ValueError(('tensor metadata mismatch', key, desc))
                path = str(HYBRID / shard)
                if path not in path_index:
                    path_index[path] = len(paths)
                    paths.append(path)
                begin, end = desc['data_offsets']
                length = end - begin
                if suffix != 'trellis':
                    with open(path, 'rb') as stream:
                        stream.seek(payload_begin + begin)
                        values = np.frombuffer(stream.read(length), dtype='<f2')
                    if not np.all(np.isfinite(values)):
                        raise ValueError('nonfinite Hadamard vectors')
                matrix_pieces.append((extent_bytes + matrix_bytes, length, payload_begin + begin,
                                      path_index[path], 0))
                matrix_bytes += length
            expected = outputs * inputs * tile_words // 128 + 2 * (inputs + outputs)
            if matrix_bytes != expected:
                raise ValueError('matrix length')
            tensor[1] = matrix_bytes
            extent_bytes += matrix_bytes
            changed += 1
        extent[0] = new_offset
        extent[1] = align(extent_bytes, BLOCK)
        absolute_pieces = [(new_offset+pos, count, source, file_id, zero)
                           for pos, count, source, file_id, zero in matrix_pieces]
        pieces.extend(absolute_pieces)
        jobs.append((extent_index, new_offset, extent[1], absolute_pieces))
        new_offset += extent[1]
    kinds = dict(next(iter(expert_matrices.values()), {}))
    widening = sum(shape == (intermediate, hidden) for shape in kinds.values())
    if (len(expert_matrices) != layer_count * expert_count or len(kinds) != EXPERT_MATRICES or
            widening != WIDENING_MATRICES or any(matrices != kinds for matrices in expert_matrices.values()) or
            changed != layer_count * expert_count * EXPERT_MATRICES):
        raise ValueError(f'{changed} EXL3 matrices at {len(expert_matrices)} experts: config.json gives {layer_count} '
                         f'layers of {expert_count} experts, each with the same {EXPERT_MATRICES} matrices '
                         f'({WIDENING_MATRICES} of them (intermediate, hidden))')
    print(f'Validated {changed} EXL3 matrices, digesting {len(jobs)} expert extents', flush=True)
    source_fds = [None] + [os.open(path, os.O_RDONLY) for path in paths[1:]]
    def extent_digest(job):
        extent_index, base, count, fragments = job
        data = bytearray(count)
        for pos, length, offset, file_id, _ in fragments:
            payload = os.pread(source_fds[file_id], length, offset)
            if len(payload) != length:
                raise ValueError('short source read')
            data[pos-base:pos-base+length] = payload
        return extent_index, digest(data)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for done, (extent_index, checksum) in enumerate(pool.map(extent_digest, jobs), 1):
            extents[extent_index][2] = checksum
            if done % 1024 == 0:
                print(f'digested {done}/{len(jobs)}, {time.monotonic()-started:.1f}s', flush=True)
    for fd in source_fds[1:]:
        os.close(fd)
    for i, extent in enumerate(extents):
        pack.EXTENT_RECORD.pack_into(tables, extent_begin + i*32, *extent)
    for i, tensor in enumerate(tensors):
        pack.TENSOR_RECORD.pack_into(tables, tensor_begin + i*64, *tensor)
    head[4] = new_offset
    head[6] = digest(tables[BLOCK:])
    head[11] = 0
    packed_header = bytearray(pack.HEADER_RECORD.pack(*head)).ljust(BLOCK, b"\0")
    head[11] = digest(packed_header)
    tables[:128] = pack.HEADER_RECORD.pack(*head)
    manifest_tables_offset = HEADER.size + len(paths)*PATH_BYTES + (len(pieces)+1)*SEGMENT.size
    pieces.insert(0, (0, len(tables), manifest_tables_offset, 0, 0))
    temporary = DEST.with_suffix('.tmp')
    with temporary.open('xb') as stream:
        stream.write(HEADER.pack(b'ANEXL3\0\0', new_offset, len(paths), len(pieces), bytes(40)))
        for i, path in enumerate(paths):
            encoded = b'' if i == 0 else path.encode()
            if len(encoded) >= PATH_BYTES:
                raise ValueError('path bound')
            stream.write(encoded + bytes(PATH_BYTES-len(encoded)))
        for piece in pieces:
            stream.write(SEGMENT.pack(*piece))
        if stream.tell() != manifest_tables_offset:
            raise ValueError('manifest metadata size')
        stream.write(tables)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(DEST)
    result = {'changed_matrices': changed, 'expert_extents': len(jobs), 'virtual_bytes': new_offset,
              'manifest_bytes': DEST.stat().st_size, 'sources': paths[1:], 'segments': len(pieces),
              'sha256': hashlib.sha256(DEST.read_bytes()).hexdigest(), 'elapsed_seconds': time.monotonic()-started}
    DEST.with_suffix('.exl3map.json').write_text(json.dumps(result, indent=2)+'\n')
    matrices_by_width, layers_by_width = {}, {}
    for layer_widths in widths.values():
        names = {name for counts in layer_widths.values() for name in counts}
        layer_name = names.pop() if len(names) == 1 else 'mixed'
        layers_by_width[layer_name] = layers_by_width.get(layer_name, 0) + 1
        for counts in layer_widths.values():
            for name, count in counts.items():
                matrices_by_width[name] = matrices_by_width.get(name, 0) + count
    quality = {'status': 'none'}
    if args.quality_record is not None:
        quality = {'status': 'recorded', 'record': str(args.quality_record.absolute()),
                   'sha256': file_sha256(args.quality_record)}
    if args.quality_established is not None:
        quality = {'status': 'established', 'reference': args.quality_established}
    descriptor = {
        'format_version': DESCRIPTOR_FORMAT_VERSION,
        'map': {'file': DEST.name, 'sha256': result['sha256'], 'manifest_bytes': result['manifest_bytes'],
                'logical_bytes': new_offset, 'pack_header_digest': head[11],
                'native_pack_header_digest': native_header_digest},
        'architecture': {'model_type': config.get('model_type'), 'architectures': config.get('architectures'),
                         'config_sha256': hashlib.sha256(config_bytes).hexdigest()},
        'quantiser': {'source': checkpoint, 'index_sha256': index_sha256,
                      'quantization_config_sha256': hashlib.sha256(quantization_bytes).hexdigest()},
        'codebook': {'mul1_multiplier': f'0x{MUL1_MULTIPLIER:08X}'},
        'front_end_tables_sha256': None if args.front_end_tables is None else file_sha256(args.front_end_tables),
        'experts': {'layers': layer_count, 'experts_per_layer': expert_count, 'hidden': hidden,
                    'intermediate': intermediate,
                    'matrices': {kind: list(shape) for kind, shape in sorted(kinds.items())}},
        'widths': {str(layer): widths[layer] for layer in sorted(widths)},
        'widths_summary': {'matrices': matrices_by_width, 'layers': layers_by_width},
        'quality': quality,
    }
    temporary = descriptor_path.with_name(descriptor_path.name + '.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(descriptor, indent=1) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(descriptor_path)
    print(json.dumps({k:v for k,v in result.items() if k != 'sources'}), flush=True)
    print(f'Descriptor: {descriptor_path} ({hashlib.sha256(descriptor_path.read_bytes()).hexdigest()})', flush=True)

if __name__ == '__main__':
    main()
