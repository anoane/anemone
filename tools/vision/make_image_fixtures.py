#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU-only decoder/patch fixtures from the pinned official image processor.

CUDA_VISIBLE_DEVICES= python3 tools/vision/make_image_fixtures.py \
  --reference /path/to/reference/deepseek_v41_official_1017 --output build/image_fixtures.bin

The reference source is imported unmodified; no model weights or CUDA are used.
The emitted stream contains input bytes, Pillow RGB bytes, the official grid and
BF16 patches, making the decode/resize/normalization gates independently visible.
"""
import argparse
import importlib.util
import io
import json
import os
import struct
import sys
import types
import zlib
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from PIL import Image

torch.set_num_threads(1)
SIGNATURE = b'ANIMG001'
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
PASSES = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8),
          (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))


def chunk(kind, payload):
    return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))


def png_case(width, height, color, depth, interlace, compression=6, filter_type=0, strategy=None, multiblock=False):
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    limit = (1 << depth) - 1
    yy, xx, cc = np.indices((height, width, channels))
    samples = ((xx * 43 + yy * 67 + cc * 89 + xx * yy * 3) & limit).astype(np.uint16)
    if color == 3:
        samples %= min(256, 1 << depth)
    raw = bytearray()
    for start_x, start_y, step_x, step_y in PASSES if interlace else ((0, 0, 1, 1),):
        if start_x >= width:
            continue
        previous = None
        for row in range(start_y, height, step_y):
            values = samples[row, start_x::step_x].reshape(-1)
            if depth == 16:
                packed = values.astype('>u2').tobytes()
            elif depth == 8:
                packed = values.astype(np.uint8).tobytes()
            else:
                bits = ''.join(f'{int(value):0{depth}b}' for value in values)
                bits += '0' * (-len(bits) % 8)
                packed = bytes(int(bits[offset:offset+8], 2) for offset in range(0, len(bits), 8))
            selected_filter = (row // step_y) % 5 if filter_type is None else filter_type
            filtered = bytearray(len(packed))
            pixel_bytes = max(1, (channels * depth + 7) // 8)
            for offset, value in enumerate(packed):
                left = packed[offset - pixel_bytes] if offset >= pixel_bytes else 0
                above = previous[offset] if previous is not None else 0
                corner = previous[offset - pixel_bytes] if previous is not None and offset >= pixel_bytes else 0
                prediction = left + above - corner
                distances = abs(prediction - left), abs(prediction - above), abs(prediction - corner)
                paeth = left if distances[0] <= min(distances[1:]) else above if distances[1] <= distances[2] else corner
                predictor = (0, left, above, (left + above) // 2, paeth)[selected_filter]
                filtered[offset] = (value - predictor) & 255
            raw.append(selected_filter)
            raw.extend(filtered)
            previous = packed
    header = struct.pack('>IIBBBBB', width, height, depth, color, 0, 0, int(interlace))
    payload = PNG_SIGNATURE + chunk(b'IHDR', header)
    if color == 3:
        palette = bytes(value for index in range(min(256, 1 << depth))
                        for value in ((index * 67) & 255, (index * 131) & 255, (index * 251) & 255))
        payload += chunk(b'PLTE', palette)
    compressor = zlib.compressobj(compression, zlib.DEFLATED, 15, 8,
        zlib.Z_DEFAULT_STRATEGY if strategy is None else strategy)
    if multiblock:
        middle = len(raw) // 2
        compressed = (compressor.compress(bytes(raw[:middle])) + compressor.flush(zlib.Z_SYNC_FLUSH) +
                      compressor.compress(bytes(raw[middle:])) + compressor.flush())
    else:
        compressed = compressor.compress(bytes(raw)) + compressor.flush()
    # Split at awkward byte positions to exercise cross-chunk deflate continuation.
    split = len(compressed) // 2
    return payload + chunk(b'IDAT', compressed[:split]) + chunk(b'IDAT', compressed[split:]) + chunk(b'IEND', b'')


def cases():
    for color, depths in ((0, (1, 2, 4, 8, 16)), (2, (8, 16)), (3, (1, 2, 4, 8)),
                          (4, (8, 16)), (6, (8, 16))):
        for depth in depths:
            for interlace in (False, True):
                yield f'png_c{color}_d{depth}_i{int(interlace)}', png_case(17, 11, color, depth, interlace)
    yield 'png_stored', png_case(9, 7, 2, 8, False, 0)
    for selected_filter in range(1, 5):
        yield f'png_filter_{selected_filter}', png_case(33, 21, 2, 8, False, filter_type=selected_filter)
    yield 'png_all_filters_adam7', png_case(65, 43, 6, 16, True, filter_type=None)
    yield 'png_fixed_huffman', png_case(91, 79, 2, 8, False, strategy=zlib.Z_FIXED, filter_type=None)
    yield 'png_multiblock', png_case(321, 245, 6, 16, True, filter_type=None, multiblock=True)
    yield 'png_indexed_tiny_adam7', png_case(1, 1, 3, 1, True)
    yield 'png_grey_tiny_adam7', png_case(2, 3, 0, 4, True)
    yy, xx, cc = np.indices((31, 37, 3))
    rgb = ((xx * 17 + yy * 29 + cc * 71 + xx * yy) & 255).astype(np.uint8)
    for subsampling in (0, 1, 2):
        image = Image.fromarray(rgb)
        stream = io.BytesIO()
        image.save(stream, format='JPEG', quality=87, subsampling=subsampling)
        yield f'jpeg_rgb_sub{subsampling}', stream.getvalue()
    for width, height in ((1, 1), (2, 2), (3, 3), (5, 7), (9, 11)):
        for subsampling in (1, 2):
            stream = io.BytesIO()
            Image.fromarray(rgb[:height, :width]).save(stream, format='JPEG', quality=91,
                subsampling=subsampling, optimize=True)
            yield f'jpeg_tiny_{width}x{height}_sub{subsampling}', stream.getvalue()
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format='JPEG', quality=91, keep_rgb=True)
    yield 'jpeg_rgb_direct', stream.getvalue()
    for mode in ('L', 'CMYK'):
        image = Image.fromarray(rgb).convert(mode)
        stream = io.BytesIO()
        image.save(stream, format='JPEG', quality=87)
        yield f'jpeg_{mode}', stream.getvalue()
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format='JPEG', quality=91, subsampling=2, restart_marker_blocks=2)
    yield 'jpeg_restart', stream.getvalue()
    for width, height in ((1, 1), (1, 301), (301, 1), (544, 544), (1401, 999), (1, 32768), (32768, 1)):
        yield f'geometry_{width}x{height}', png_case(width, height, 2, 8, False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.reference / 'inference' / 'image_processor.py'
    specification = importlib.util.spec_from_file_location('image_processor_reference', source)
    reference = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = reference
    specification.loader.exec_module(reference)
    config = json.loads((args.reference / 'config.json').read_text())
    vision = config['vision_config']
    model_args = types.SimpleNamespace(vision_patch_size=vision['patch_size'],
        vision_downsample_ratio=vision['downsample_ratio'], vision_min_pixels=vision['min_pixels'],
        vision_max_n_token=vision['max_image_tokens'], vision_max_wh_ratio=vision['max_wh_ratio'])
    entries = list(cases())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as output:
        output.write(SIGNATURE + struct.pack('<I', len(entries)))
        for name, encoded in entries:
            image = Image.open(io.BytesIO(encoded)).convert('RGB')
            patches, vit_h, vit_w, llm_h, llm_w = reference.load_image({'data': encoded}, model_args)
            expected = patches.view(torch.uint16).numpy().astype('<u2').tobytes()
            fields = (len(encoded), image.width, image.height, vit_h, vit_w, llm_h, llm_w, len(expected) // 2)
            output.write(name.encode().ljust(64, b'\0'))
            output.write(struct.pack('<8I', *fields))
            output.write(encoded)
            output.write(image.tobytes())
            output.write(expected)
            print(name, image.size, (vit_h, vit_w), (llm_h, llm_w), flush=True)
    print(f'{len(entries)} fixtures: {args.output} ({args.output.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
