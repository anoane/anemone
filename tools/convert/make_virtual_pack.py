#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build a read-only scatter manifest, preserving native extents and indexed EXL3 bytes.
CPU-only. No checkpoint payloads are modified. No complete weight copy is made.
"""
import argparse
import concurrent.futures
import functools
import hashlib
import json
import os
import pathlib
import struct
import sys
import time
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_pack as pack

NATIVE = HYBRID = DEST = None
EXPECTED_INDEX = 'fa24746ffa8be36473aca6c99b432f3b0230f17237f0873f2db9be598d324e81'
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


def main():
    global NATIVE, HYBRID, DEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', required=True, type=pathlib.Path)
    parser.add_argument('--hybrid', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    NATIVE, HYBRID, DEST = (path.absolute() for path in (args.native, args.hybrid, args.output))
    if DEST.suffix != '.exl3map':
        parser.error('the runtime requires an .exl3map output suffix')
    if DEST.exists():
        raise ValueError('destination exists; preserve previous epochs before replacing')
    started = time.monotonic()
    index_bytes = (HYBRID / 'model.safetensors.index.json').read_bytes()
    if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_INDEX:
        raise ValueError('checkpoint index identity mismatch')
    index = json.loads(index_bytes)['weight_map']
    with NATIVE.open('rb') as stream:
        head = list(pack.HEADER_RECORD.unpack(stream.read(128)))
        stream.seek(0)
        tables = bytearray(stream.read(BLOCK + head[5]))
    if head[0] != b'ANEMONE\0' or head[1] != 2:
        raise ValueError('native pack version')
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
            layer = int(name.split('.')[1])
            bits = 2 if 18 <= layer <= 22 else 3
            outputs, inputs = tensor[7:9]
            if (outputs, inputs) not in ((2304, 5120), (5120, 2304)):
                raise ValueError('expert geometry')
            extent_bytes = align(extent_bytes, 64)
            tensor[0] = extent_bytes
            tensor[11], tensor[12] = 2, bits
            specs = [('trellis', 'I16', [inputs//16, outputs//16, 16*bits]),
                     ('suh', 'F16', [inputs]), ('svh', 'F16', [outputs])]
            marker_name = prefix + '.mul1'
            marker_shard = index[marker_name]
            payload_begin, marker_header = shard_header(marker_shard)
            marker = marker_header[marker_name]
            with (HYBRID / marker_shard).open('rb') as stream:
                stream.seek(payload_begin + marker['data_offsets'][0])
                if (marker['dtype'], marker['shape'], stream.read(4)) != ('I32', [], struct.pack('<I', 0x83DCD12D)):
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
            expected = outputs * inputs * bits // 8 + 2 * (inputs + outputs)
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
    if changed != 40 * 384 * 3:
        raise ValueError(('wrong changed tensor count', changed))
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
    print(json.dumps({k:v for k,v in result.items() if k != 'sources'}), flush=True)

if __name__ == '__main__':
    main()
