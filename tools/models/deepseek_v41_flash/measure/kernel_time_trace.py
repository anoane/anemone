#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where the time of today's engine goes, per card: the baseline every Anemone kernel has to beat.

Runs the exllamav3 port (the frozen copy that serves today) and reports, for each card:
  * wall time per layer, split into attention, experts and the rest of the block, at 1 to 8 rows (decode and
    draft-verify shapes) and at one prefill chunk;
  * GPU kernel time and the number of kernel launches per step, taken from the profiler (instrumented, not counted
    from the source).

    run_with_port.sh kernel_time_trace.py --part forward   --layers-per-device 12,18,10 --ram-layers 33..39
    run_with_port.sh kernel_time_trace.py --part generator --layers-per-device 12,18,10 --ram-layers 33..39

"forward" drives the model directly with teacher-forced tokens; "generator" measures the real decode loop.
Measurement tool: not part of the engine.
"""
import argparse
import glob
import json
import os
import sys
import time

import torch

DEFAULT_MODEL_DIRECTORY = os.path.expanduser("~/workspace/DeepSeek-V4.1-Flash-EXL3-3.0bpw")
DEFAULT_CORPUS_DIRECTORY = os.path.expanduser("~/dsv41f-min/upstream")
CACHE_PAGE_TOKENS = 256                      # the port's cache page size: batch shapes are multiples of it
PREFILL_CHUNK_TOKENS = 4096
PREFILL_CHUNKS_MEASURED = 2                  # after one warm-up chunk
DECODE_ROW_COUNTS = (1, 2, 4, 6, 8)          # 1 = plain decode, 2..8 = draft-verify windows
STEPS_PER_ROW_COUNT = 48
WARM_UP_STEPS = 8                            # left out of every mean: graph captures and kernel picks
PROFILED_STEPS = 16
GENERATOR_PROMPT_TOKENS = 2000
GENERATOR_NEW_TOKENS = 200
TOP_KERNELS_PRINTED = 12
MILLISECONDS_PER_SECOND = 1000.0
NANOSECONDS_PER_MILLISECOND = 1.0e6


def synchronize_all_devices():
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)


class ModuleTimer:
    """Wall time of each wrapped module's forward between two device synchronisations, switchable."""

    def __init__(self):
        self.enabled = False
        self.seconds_by_module = {}
        self.calls_by_module = {}
        self.description_by_module = {}

    def wrap(self, module, module_name, kind, device_name):
        inner_forward = module.forward
        self.seconds_by_module[module_name] = 0.0
        self.calls_by_module[module_name] = 0
        self.description_by_module[module_name] = (kind, device_name)

        def timed_forward(*positional, **named):
            if not self.enabled:
                return inner_forward(*positional, **named)
            synchronize_all_devices()
            started = time.perf_counter()
            result = inner_forward(*positional, **named)
            synchronize_all_devices()
            self.seconds_by_module[module_name] += time.perf_counter() - started
            self.calls_by_module[module_name] += 1
            return result
        module.forward = timed_forward

    def reset(self):
        for module_name in self.seconds_by_module:
            self.seconds_by_module[module_name] = 0.0
            self.calls_by_module[module_name] = 0

    def milliseconds_per_layer_by_card(self, steps):
        """{card: {kind: [total ms per step, modules]}} over the steps since the last reset."""
        table = {}
        for module_name, seconds in self.seconds_by_module.items():
            kind, device_name = self.description_by_module[module_name]
            entry = table.setdefault(device_name, {}).setdefault(kind, [0.0, 0])
            entry[0] += MILLISECONDS_PER_SECOND * seconds / steps
            entry[1] += 1
        return table


def wrap_transformer_blocks(model, module_timer):
    for module in model.modules:
        if type(module).__name__ != "TransformerBlock":
            continue
        device_name = str(module.device)
        experts_in_ram = bool(getattr(module.mlp, "cpu_offload", False))
        experts_split = getattr(module.mlp, "cpu_split_first", None) is not None
        expert_kind = "experts in RAM" if experts_in_ram else "experts split VRAM/RAM" if experts_split else "experts in VRAM"
        module_timer.wrap(module, module.key, "whole block, " + expert_kind, device_name)
        module_timer.wrap(module.attn, module.key + ".attn", "attention", device_name)
        module_timer.wrap(module.mlp, module.key + ".mlp", expert_kind, device_name)


def print_layer_table(title, table):
    print(title)
    for device_name in sorted(table):
        kinds = table[device_name]
        attention_ms, attention_count = kinds.get("attention", [0.0, 0])
        for kind in sorted(kinds):
            if not kind.startswith("whole block"):
                continue
            block_ms, block_count = kinds[kind]
            expert_kind = kind.split(", ", 1)[1]
            expert_ms, _ = kinds.get(expert_kind, [0.0, 0])
            attention_share = attention_ms * block_count / max(attention_count, 1)
            print(f"  {device_name} {expert_kind:24s} {block_count:2d} layers: block {block_ms / block_count:8.3f} ms per layer"
                  f" = attention {attention_share / block_count:7.3f} + experts {expert_ms / block_count:7.3f}"
                  f" + rest {(block_ms - attention_share - expert_ms) / block_count:7.3f}")


def profile_kernels(run_steps, steps):
    """GPU kernel time and launches per step and per card, from the profiler's device events."""
    from torch.profiler import profile, ProfilerActivity
    with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        run_steps()
        synchronize_all_devices()
    per_device = {}
    for event in profiler.profiler.kineto_results.events():
        if "cuda" not in str(event.device_type()).lower():
            continue
        duration_ns = event.duration_ns() if hasattr(event, "duration_ns") else event.duration_us() * 1000
        name = event.name()
        entry = per_device.setdefault(int(event.device_index()), {"kernel_ms": 0.0, "launches": 0, "copy_ms": 0.0, "copies": 0, "by_name": {}})
        is_copy = name.lower().startswith(("memcpy", "memset"))
        entry["copy_ms" if is_copy else "kernel_ms"] += duration_ns / NANOSECONDS_PER_MILLISECOND
        entry["copies" if is_copy else "launches"] += 1
        if not is_copy:
            by_name = entry["by_name"].setdefault(name, [0.0, 0])
            by_name[0] += duration_ns / NANOSECONDS_PER_MILLISECOND
            by_name[1] += 1
    for entry in per_device.values():
        for key in ("kernel_ms", "copy_ms"):
            entry[key] /= steps
        for key in ("launches", "copies"):
            entry[key] /= steps
        entry["by_name"] = {name: [ms / steps, count / steps] for name, (ms, count) in entry["by_name"].items()}
    return per_device


def print_kernel_profile(title, per_device):
    print(title)
    total_kernel_ms, total_launches = 0.0, 0.0
    for device_index in sorted(per_device):
        entry = per_device[device_index]
        total_kernel_ms += entry["kernel_ms"]
        total_launches += entry["launches"]
        print(f"  card {device_index} {torch.cuda.get_device_name(device_index)}: kernels {entry['kernel_ms']:9.3f} ms in "
              f"{entry['launches']:8.1f} launches per step; copies {entry['copy_ms']:7.3f} ms in {entry['copies']:6.1f}")
        ranked = sorted(entry["by_name"].items(), key = lambda item: -item[1][0])[:TOP_KERNELS_PRINTED]
        for name, (ms, count) in ranked:
            print(f"      {ms:9.4f} ms {count:8.1f} x  {name[:110]}")
    print(f"  all cards: kernels {total_kernel_ms:.3f} ms in {total_launches:.1f} launches per step")
    return {"kernel_ms_per_step": total_kernel_ms, "launches_per_step": total_launches}


def load_corpus_tokens(tokenizer, corpus_directory, token_count):
    source_files = sorted(path for extension in ("py", "cu", "cuh", "cpp", "h")
                          for path in glob.glob(os.path.join(corpus_directory, "**", f"*.{extension}"), recursive = True))
    pieces, collected = [], 0
    for path in source_files:
        with open(path, encoding = "utf8", errors = "replace") as handle:
            piece = tokenizer.encode(handle.read(), add_bos = False, encode_special_tokens = False)[0]
        pieces.append(piece[:token_count - collected])
        collected += pieces[-1].numel()
        if collected >= token_count:
            break
    if collected < token_count:
        raise SystemExit(f"corpus too small: {collected} of {token_count} tokens")
    return torch.cat(pieces).unsqueeze(0)


def load_model(arguments):
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.model.config import moe_cpu_layers, moe_cpu_split_sizes
    config = Config.from_directory(arguments.model)
    config.infer_params.moe_cpu_offload = moe_cpu_layers(arguments.ram_layers)
    if arguments.ram_split != "0":
        config.infer_params.moe_cpu_split = moe_cpu_split_sizes(arguments.ram_split)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = arguments.cache_tokens)
    load_options = {}
    if arguments.layers_per_device:
        load_options["layers_per_device"] = [int(value) for value in arguments.layers_per_device.split(",")]
    if arguments.gigabytes_per_device:
        load_options["use_per_device"] = [float(value) for value in arguments.gigabytes_per_device.split(",")]
    load_started = time.time()
    model.load(progressbar = False, max_chunk_size = PREFILL_CHUNK_TOKENS, **load_options)
    blocks = [module for module in model.modules if type(module).__name__ == "TransformerBlock"]
    print(f"loaded in {time.time() - load_started:.0f} s")
    print("placement (layer:card, r = experts in RAM, s = split): " + " ".join(
        f"{block.key.split('.')[-1]}:{str(block.device)[-1]}"
        f"{'r' if getattr(block.mlp, 'cpu_offload', False) else 's' if getattr(block.mlp, 'cpu_split_first', None) is not None else ''}"
        for block in blocks))
    for device_index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        print(f"card {device_index} {torch.cuda.get_device_name(device_index)}: free {free_bytes / 2**30:.2f} of {total_bytes / 2**30:.2f} GiB")
    print("environment: " + json.dumps({key: value for key, value in sorted(os.environ.items()) if key.startswith("EXL3_")}))
    return config, model, cache, Tokenizer.from_config(config)


def measure_direct_forward(arguments, results):
    config, model, cache, tokenizer = load_model(arguments)
    module_timer = ModuleTimer()
    wrap_transformer_blocks(model, module_timer)
    prefill_tokens = PREFILL_CHUNK_TOKENS * (1 + PREFILL_CHUNKS_MEASURED)
    decode_tokens = sum(rows * (STEPS_PER_ROW_COUNT + PROFILED_STEPS) for rows in DECODE_ROW_COUNTS)
    token_ids = load_corpus_tokens(tokenizer, arguments.corpus, prefill_tokens + decode_tokens)
    batch_rows = -(-token_ids.shape[1] // CACHE_PAGE_TOKENS) * CACHE_PAGE_TOKENS
    state = cache.get_new_state()
    position = 0

    def forward(row_count):
        nonlocal position
        model.forward(token_ids[:, position:position + row_count],
                      {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, batch_rows), "past_len": position,
                       "recurrent_states": [state]})
        position += row_count

    try:
        forward(PREFILL_CHUNK_TOKENS)                                            # warm-up chunk
        module_timer.reset()
        module_timer.enabled = True
        synchronize_all_devices()
        started = time.perf_counter()
        for _ in range(PREFILL_CHUNKS_MEASURED - 1):
            forward(PREFILL_CHUNK_TOKENS)
        synchronize_all_devices()
        chunk_seconds = (time.perf_counter() - started) / (PREFILL_CHUNKS_MEASURED - 1)
        module_timer.enabled = False
        table = module_timer.milliseconds_per_layer_by_card(PREFILL_CHUNKS_MEASURED - 1)
        print_layer_table(f"\n== {PREFILL_CHUNK_TOKENS} rows (one prefill chunk), synchronised around every module: "
                          f"{MILLISECONDS_PER_SECOND * chunk_seconds:.1f} ms per chunk", table)
        profile = profile_kernels(lambda: forward(PREFILL_CHUNK_TOKENS), 1)
        totals = print_kernel_profile(f"== {PREFILL_CHUNK_TOKENS} rows, GPU kernel time of one chunk", profile)
        results[f"rows_{PREFILL_CHUNK_TOKENS}"] = {"wall_ms_synchronised": MILLISECONDS_PER_SECOND * chunk_seconds, "layers": table, "kernels": profile, **totals}

        for row_count in DECODE_ROW_COUNTS:
            for _ in range(WARM_UP_STEPS):
                forward(row_count)
            synchronize_all_devices()
            started = time.perf_counter()
            for _ in range(STEPS_PER_ROW_COUNT - 2 * WARM_UP_STEPS):
                forward(row_count)
            synchronize_all_devices()
            plain_ms = MILLISECONDS_PER_SECOND * (time.perf_counter() - started) / (STEPS_PER_ROW_COUNT - 2 * WARM_UP_STEPS)
            module_timer.reset()
            module_timer.enabled = True
            for _ in range(WARM_UP_STEPS):
                forward(row_count)
            module_timer.enabled = False
            table = module_timer.milliseconds_per_layer_by_card(WARM_UP_STEPS)
            print_layer_table(f"\n== {row_count} row(s) per step: {plain_ms:.2f} ms per step unsynchronised; per layer, synchronised:", table)
            profile = profile_kernels(lambda: [forward(row_count) for _ in range(PROFILED_STEPS)], PROFILED_STEPS)
            totals = print_kernel_profile(f"== {row_count} row(s) per step, GPU kernel time", profile)
            results[f"rows_{row_count}"] = {"wall_ms_plain": plain_ms, "layers": table, "kernels": profile, **totals}
    finally:
        cache.release_state(state)


def measure_generator(arguments, results):
    from exllamav3 import Generator, Job
    from exllamav3.generator.sampler import ArgmaxSampler
    config, model, cache, tokenizer = load_model(arguments)
    module_timer = ModuleTimer()
    wrap_transformer_blocks(model, module_timer)
    head = tokenizer.encode("<｜begin▁of▁sentence｜>You are a helpful AI assistant.<｜User｜>", add_bos = False, encode_special_tokens = True)
    tail = tokenizer.encode("\n\nExplain in detail, step by step, what the code above does and how its parts fit together.<｜Assistant｜></think>",
                            add_bos = False, encode_special_tokens = True)
    body = load_corpus_tokens(tokenizer, arguments.corpus, GENERATOR_PROMPT_TOKENS - head.shape[1] - tail.shape[1])
    prompt_ids = torch.cat([head, body, tail], dim = 1)
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer, max_batch_size = 1, max_chunk_size = PREFILL_CHUNK_TOKENS)

    def generate(on_first_token = None, new_tokens = GENERATOR_NEW_TOKENS):
        generator.pagetable.reset_page_table()
        generator.enqueue(Job(input_ids = prompt_ids, max_new_tokens = new_tokens, sampler = ArgmaxSampler(), stop_conditions = []))
        stamps, produced = [], 0
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if result.get("stage") == "error":
                    raise result["error"]
                token_ids = result.get("token_ids")
                if token_ids is not None:
                    if not produced and on_first_token is not None:
                        on_first_token()
                    produced += token_ids.numel()
                    stamps.append((time.perf_counter(), produced))
        synchronize_all_devices()
        (first_time, first_count), (last_time, last_count) = stamps[min(WARM_UP_STEPS, len(stamps) - 2)], stamps[-1]
        return (last_time - first_time) / (last_count - first_count), produced

    generate()
    seconds_per_token, produced = generate()
    print(f"\n== generator, greedy, no drafting: {1.0 / seconds_per_token:.1f} tok/s ({MILLISECONDS_PER_SECOND * seconds_per_token:.2f} ms per token)")

    def start_timing():
        module_timer.reset()
        module_timer.enabled = True
    timed_seconds_per_token, produced = generate(start_timing)
    module_timer.enabled = False
    table = module_timer.milliseconds_per_layer_by_card(produced - 1)
    print_layer_table(f"== generator, synchronised around every module: {MILLISECONDS_PER_SECOND * timed_seconds_per_token:.2f} ms per token", table)

    profile_holder = {}

    def profiled_generation():
        profile_holder["produced"] = generate(new_tokens = PROFILED_STEPS + WARM_UP_STEPS)[1]
    profile = profile_kernels(profiled_generation, 1)
    print_kernel_profile(f"== generator, GPU kernel time of one whole request ({GENERATOR_PROMPT_TOKENS} prompt tokens + "
                         f"{PROFILED_STEPS + WARM_UP_STEPS} new): divide by hand, the prompt is inside", profile)
    results["generator"] = {"ms_per_token_plain": MILLISECONDS_PER_SECOND * seconds_per_token,
                            "ms_per_token_synchronised": MILLISECONDS_PER_SECOND * timed_seconds_per_token,
                            "layers": table, "kernels_whole_request": profile}


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices = ("forward", "generator"), required = True)
    parser.add_argument("--model", default = DEFAULT_MODEL_DIRECTORY)
    parser.add_argument("--corpus", default = DEFAULT_CORPUS_DIRECTORY)
    parser.add_argument("--ram-layers", default = "0", help = "layers whose experts stay in RAM, as the port's -mcl takes them")
    parser.add_argument("--ram-split", default = "0", help = "experts per layer kept in RAM, as the port's -mcs takes them")
    parser.add_argument("--layers-per-device", default = None)
    parser.add_argument("--gigabytes-per-device", default = None)
    parser.add_argument("--cache-tokens", type = int, default = 65536)
    parser.add_argument("--json", default = None, help = "also write every number to this file")
    arguments = parser.parse_args()
    results = {"part": arguments.part, "arguments": vars(arguments),
               "cards": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]}
    with torch.inference_mode():
        if arguments.part == "forward":
            measure_direct_forward(arguments, results)
        else:
            measure_generator(arguments, results)
    if arguments.json:
        with open(arguments.json, "w") as handle:
            json.dump(results, handle, indent = 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
