# SPDX-License-Identifier: AGPL-3.0-or-later
# CPU-only adaptation of the reference hybrid-index producer.
# Only paths and output refusal changed; the vLLM-only sidecar link is omitted.
# make_hybrid.py <libertai pack dir> <out dir>: the DeepSeek-V4.1-Flash-1017 hybrid checkpoint as Curtis's recipe takes
# it, by links only: official 1017 tensors, the pack's routed experts of layers 0-39, his quantization_config.json
import argparse, json, os, re, sys
parser = argparse.ArgumentParser(description='Build the pinned routed-expert index by symlinks, CPU only.')
parser.add_argument('official')
parser.add_argument('exl3')
parser.add_argument('output')
args = parser.parse_args()
OFF, pack, out = map(os.path.abspath, (args.official, args.exl3, args.output))
if os.path.lexists(out):
    parser.error('output exists; preserve previous epochs')
routed = re.compile(r"^layers\.(\d+)\.ffn\.experts\.\d+\.")
def is_main_routed(k):
    m = routed.match(k)
    return m is not None and int(m[1]) < 40
off = json.load(open(f"{OFF}/model.safetensors.index.json"))["weight_map"]
lib = json.load(open(f"{pack}/model.safetensors.index.json"))["weight_map"]
merged = {k: v for k, v in off.items() if not is_main_routed(k)}
dropped = len(off) - len(merged)
add = {k: v for k, v in lib.items() if is_main_routed(k)}
assert add and not (set(add) & set(merged)) and not (set(off.values()) & set(add.values()))
merged.update(add)
os.makedirs(out, exist_ok = True)
for d, names in ((OFF, os.listdir(OFF)), (pack, sorted(set(add.values())))):
    for n in names:
        if n in ("model.safetensors.index.json", ".cache") or (d == OFF and n == "quantization_config.json"):
            continue
        dst = os.path.join(out, n)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(d, n), dst)
json.dump({"metadata": {}, "weight_map": merged}, open(f"{out}/model.safetensors.index.json", "w"), indent = 1)
print(out, "official keys dropped", dropped, "routed expert keys added", len(add), "from", len(set(add.values())), "shards, total", len(merged))
