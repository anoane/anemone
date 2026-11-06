#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Create CPU-only image chat token fixtures from the unmodified official encoder and image processor.

    CUDA_VISIBLE_DEVICES= python3 tools/golden/make_image_chat_golden.py --reference REF --output FILE

The fixture contains messages JSON, the exact reference prompt, expanded token ids and each image's token/grid
metadata. It uses synthetic PNG bytes only and loads no model weights. The checked C test consumes these records
with the converted official tokenizer tables: image_chat_golden_test TABLES FILE.
"""
import argparse
import base64
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace


def load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def image_part(width, height, color):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": url}}


def write_bytes(handle, value):
    handle.write(struct.pack("<I", len(value)))
    handle.write(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    from tokenizers import Tokenizer
    encoder = load_module("encoding", arguments.reference / "encoding/encoding.py")
    processor = load_module("image_processor", arguments.reference / "inference/image_processor.py")
    settings = json.loads((arguments.reference / "config.json").read_text())
    config = settings["vision_config"]
    args = SimpleNamespace(vision_patch_size=config["patch_size"], vision_downsample_ratio=config["downsample_ratio"],
                           vision_max_n_token=config["max_image_tokens"], vision_min_pixels=config["min_pixels"],
                           vision_max_wh_ratio=config["max_wh_ratio"], image_token_id=settings["image_token_id"],
                           vision_enabled=True)
    tokenizer = Tokenizer.from_file(str(arguments.reference / "tokenizer.json"))

    class ReferenceTokenizer:
        unk_token_id = None

        def convert_tokens_to_ids(self, spelling):
            return tokenizer.token_to_id(spelling)

        def encode(self, prompt):
            return tokenizer.encode(prompt, add_special_tokens=False).ids

    first, second = image_part(7, 9, (19, 83, 211)), image_part(73, 31, (209, 47, 11))
    calls = [{"id": name, "type": "function", "function": {"name": name, "arguments": "{}"}}
             for name in ("b", "a")]
    cases = [
        [{"role": "user", "content": [{"type": "text", "text": "Describe this image."}, first]}],
        [{"role": "user", "content": [first, {"type": "text", "text": "Compare."}, second]}],
        [{"role": "assistant", "content": None, "tool_calls": calls},
         {"role": "tool", "tool_call_id": "a", "content": [first]},
         {"role": "tool", "tool_call_id": "b", "content": [second]}]
    ]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("wb") as handle:
        handle.write(b"ANEMVLG1" + struct.pack("<I", len(cases)))
        for messages in cases:
            message_json = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()
            prompt, media = encoder.encode_messages(messages, thinking_mode="chat", reasoning_effort=75,
                                                     return_multi_modal_data=True)
            tokens, types, images = processor.prepare_vl_inputs(prompt, media["images"], ReferenceTokenizer(), args)
            write_bytes(handle, message_json)
            write_bytes(handle, prompt.encode())
            handle.write(struct.pack("<I", len(tokens)))
            handle.write(struct.pack("<" + "I" * len(tokens), *tokens))
            handle.write(struct.pack("<I", len(images)))
            for image in images:
                handle.write(struct.pack("<IIII", image.start, image.types.numel(), image.n_vit_h, image.n_vit_w))
    print(f"wrote {len(cases)} official image conversations to {arguments.output}")


if __name__ == "__main__":
    main()
