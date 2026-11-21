#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a decode token of today's engine costs after a long prompt, and what of it is index selection.

    run_with_port.sh port_long_decode.py --prompt-tokens 1040000 --layers-per-device 11,18,11 --ram-layers 28..28
        [--cache-tokens 1048576] [--break new_tokens | --break selection_timed]

Runs the exllamav3 port (the frozen copy) through its Generator, greedy, no drafting: the first --prompt-tokens
tokens of the corpus, prefilled in Generator calls of 65,536 tokens (the port's pipelined prefill), then 364 new
tokens; three times, each from an emptied page table. It prints
  DECODE   per repeat: the seconds to the first new token, and the milliseconds a token from the 9th new token to
           the 300th, with nothing waited for;
  SELECT   per layer that scores its pool: the wall time inside the port's _indexer_topk between two waits for
           the layer's card, over the 64 tokens that follow the last repeat's 300. Inside it are the query and
           weight projections, the scores, the candidate masks and the top-k;
  VERDICT  the median of the repeats, the selection of one token (the layers' calls added up: a layer selects
           once a token), and each card's highest allocation since its last module loaded.
Checked in every run, each shown failing by --break <name> (exit status 0 only if that check alone reported):
  new_tokens       every job gave the tokens it was asked for (the break asks each job for one fewer);
  selection_timed  every timed layer selected at least once (the break times the layers that never select).
Refused before the load: a prompt that does not fit --cache-tokens with its new tokens (exit status 2).
What it does not show: drafted or sampled decoding; selection without the waits (a part that is waited for is
longer than its share of a plain token); the kernels inside selection (kernel_time_trace.py names them, at short
context); another layout: the load's placement line belongs to every figure. Exit status 1 after a failed check.
Measurement tool: not part of the engine.
"""
import argparse
import sys
import time

import torch

from kernel_time_trace import DEFAULT_CORPUS_DIRECTORY, DEFAULT_MODEL_DIRECTORY, load_corpus_tokens, load_model

GENERATOR_CHUNK_TOKENS = 65536               # above the load's 4,096: the port prefills pipelined across the cards
DEFAULT_CACHE_TOKENS = 1048576
REPEATS = 3
NEW_TOKENS = 300                             # of each repeat, timed with nothing waited for
WARM_UP_TOKENS = 8                           # left out of every figure: first-use captures and kernel picks
SYNCHRONISED_TOKENS = 64                     # after the last repeat's NEW_TOKENS, with selection waited for
JOB_TOKENS = NEW_TOKENS + SYNCHRONISED_TOKENS    # of every repeat: the port tidies its cache in a job's last call
SELECTING_MODE = "full"                      # the port's indexer_mode of a layer that scores its pool itself
CHECK_NAMES = ("new_tokens", "selection_timed")
NOT_AVAILABLE = "na"
MILLISECONDS_PER_SECOND = 1000.0
BYTES_PER_GIB = 2.0 ** 30
EXIT_CODE_FAILURE = 1


def time_selection(blocks, seconds, calls):
    """Puts each block's _indexer_topk between two waits for the card of its input, and adds up what it took."""
    for block in blocks:
        def timed(x, *positional, inner = block.attn._indexer_topk, layer = block.attn.layer_idx, **named):
            torch.cuda.synchronize(x.device)
            started = time.perf_counter()
            result = inner(x, *positional, **named)
            torch.cuda.synchronize(x.device)
            seconds[layer] += time.perf_counter() - started
            calls[layer] += 1
            return result
        block.attn._indexer_topk = timed


def generate(generator, prompt_ids, new_tokens, timed_from, start_timing):
    """One greedy job from an emptied page table: its start, and a (time, new tokens so far) stamp for every result
    that carried tokens. start_timing is called once timed_from tokens are out; never where that is None."""
    from exllamav3 import Job
    from exllamav3.generator.sampler import ArgmaxSampler
    generator.pagetable.reset_page_table()
    generator.enqueue(Job(input_ids = prompt_ids, max_new_tokens = new_tokens, sampler = ArgmaxSampler(),
                          stop_conditions = []))
    started, stamps, produced = time.perf_counter(), [], 0
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            if result.get("stage") == "error":
                raise result["error"]
            if result.get("token_ids") is None:
                continue
            produced += result["token_ids"].numel()
            stamps.append((time.perf_counter(), produced))
            if timed_from is not None and produced >= timed_from:
                start_timing()
                timed_from = None
    return started, stamps


def shown(value):
    return NOT_AVAILABLE if value is None else f"{value:.3f}"


def measure(arguments):
    from exllamav3 import Generator
    _, model, cache, tokenizer = load_model(arguments)
    prompt_ids = load_corpus_tokens(tokenizer, DEFAULT_CORPUS_DIRECTORY, arguments.prompt_tokens)
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer, max_batch_size = 1,
                          max_chunk_size = GENERATOR_CHUNK_TOKENS)
    selecting = arguments.broken_check != "selection_timed"          # the break times the layers that never select
    timed_blocks = [module for module in model.modules if type(module).__name__ == "TransformerBlock"
                    and (module.attn.indexer_mode == SELECTING_MODE) == selecting]
    seconds = {block.attn.layer_idx: 0.0 for block in timed_blocks}
    calls = dict.fromkeys(seconds, 0)
    label, short_jobs, rates = f"prompt_tokens={arguments.prompt_tokens}", 0, []
    for repeat in range(1, REPEATS + 1):
        started, stamps = generate(generator, prompt_ids, JOB_TOKENS - (arguments.broken_check == "new_tokens"),
                                   NEW_TOKENS if repeat == REPEATS else None,
                                   lambda: time_selection(timed_blocks, seconds, calls))
        short_jobs += stamps[-1][1] != JOB_TOKENS
        timed = [stamp for stamp in stamps if WARM_UP_TOKENS <= stamp[1] <= NEW_TOKENS]
        rates.append(MILLISECONDS_PER_SECOND * (timed[-1][0] - timed[0][0]) / (timed[-1][1] - timed[0][1]))
        print(f"DECODE {label} repeat={repeat} first_token_s={stamps[0][0] - started:.3f} "
              f"new_tokens={stamps[-1][1]} ms_per_token={rates[-1]:.3f}", flush = True)

    selection = 0.0
    for block in timed_blocks:
        layer = block.attn.layer_idx
        per_call = MILLISECONDS_PER_SECOND * seconds[layer] / calls[layer] if calls[layer] else None
        selection += per_call or 0.0
        print(f"SELECT {label} layer={layer} device={block.device} rate={block.attn.compress_rate} "
              f"candidates={block.attn.candidate_mode} calls={calls[layer]} ms_per_call={shown(per_call)}")
    differing = {"new_tokens": (short_jobs, REPEATS),
                 "selection_timed": (sum(1 for count in calls.values() if not count), len(calls))}
    failed = False
    for name in CHECK_NAMES:
        count, of = differing[name]
        must_report = arguments.broken_check == name
        verdict = "FAILED" if (count == 0 and of > 0) == must_report else "reported" if must_report else "ok"
        failed |= verdict == "FAILED"
        print(f"CHECK {label} name={name} differing={count} of={of} verdict={verdict}")
    peaks = "/".join(f"{torch.cuda.max_memory_allocated(card) / BYTES_PER_GIB:.2f}"
                     for card in range(torch.cuda.device_count()))
    print(f"VERDICT figure=port_decode {label} ms_per_token={sorted(rates)[REPEATS // 2]:.3f}"
          f" selection_ms_per_token={shown(selection)} peak_allocated_gib={peaks} "
          f"verdict={'not_a_result' if failed or arguments.broken_check else 'measured'}")
    return EXIT_CODE_FAILURE if failed else 0


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-tokens", type = int, required = True, help = "taken from the corpus")
    parser.add_argument("--layers-per-device", default = None)
    parser.add_argument("--ram-layers", default = "0", help = "layers whose experts stay in RAM, as the port's -mcl")
    parser.add_argument("--cache-tokens", type = int, default = DEFAULT_CACHE_TOKENS)
    parser.add_argument("--break", dest = "broken_check", choices = CHECK_NAMES, help = "the check that must report")
    parser.set_defaults(model = DEFAULT_MODEL_DIRECTORY, ram_split = "0", gigabytes_per_device = None)  # load_model's
    arguments = parser.parse_args()
    if not 0 < arguments.prompt_tokens < arguments.cache_tokens - JOB_TOKENS:    # the port keeps one token more
        parser.error(f"--prompt-tokens, its {JOB_TOKENS} new tokens and one more must fit --cache-tokens")
    with torch.inference_mode():
        return measure(arguments)


if __name__ == "__main__":
    sys.exit(main())
