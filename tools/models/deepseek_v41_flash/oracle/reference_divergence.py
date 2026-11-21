#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Finds where two passes of DeepSeek's reference over one prompt first differ, for reference_oracle.py.

The oracle's captures are only a ground truth if the reference repeats itself. When two greedy runs give other
tokens, this tool says where the difference is born: it runs the prompt's pass several times in one process and
records, for every call of every module of the model and of every function of the reference's model, kernel and
engram files, a position-weighted checksum of what went in, of what came out, and of the arguments after the call
(a kernel writes into one). The first record that differs between two passes names the call; whether it is an
"in", an "out" or an "after" record says whether the call itself is not repeatable (same inputs, other output) or
something between two recorded calls is.

    reference_divergence.py --checkpoint <folder> --prompt-json FILE --output-directory <folder> [--passes N]
                            [--settle] [--compare-with <earlier output folder>] [--attention-card N]

Checksums are queued on the card and read only when a pass has ended, so that recording does not make the host
wait at every call; --settle waits for the card before every record instead (a difference that then disappears
depended on timing). The first pass's records are saved; --compare-with reads those of another process.
Measurement tool: not part of the engine.
"""
import argparse
import os
import sys
import types

import torch

import reference_oracle as oracle

TRACE_FILE_NAME = "divergence_trace.pt"
DEFAULT_PASSES = 2
CHECKSUM_MODULUS = 65521                     # the weights of a checksum repeat with this period: a prime below 2^16
CHECKSUM_CHUNK_BYTES = 1 << 24               # bytes weighted at a time, so that a large tensor needs no large copy
CONTEXT_RECORDS = 6                          # records printed before the first one that differs
DIFFERING_NAMES_PRINTED = 12
TRACED_REFERENCE_MODULES = ("model", "kernel", "engram")
RECORD_IN, RECORD_OUT, RECORD_AFTER = "in", "out", "after"
EXIT_CODE_SAME, EXIT_CODE_DIFFERENT, EXIT_CODE_FAILURE = 0, 1, 2


def tensors_of(value, found = None):
    """Every tensor inside nested tuples, lists and dictionaries, in order."""
    found = [] if found is None else found
    if isinstance(value, torch.Tensor):
        found.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            tensors_of(item, found)
    elif isinstance(value, dict):
        for item in value.values():
            tensors_of(item, found)
    return found


class Trace:
    """The records of one pass: (call name, kind of record, shapes and types, checksums)."""

    def __init__(self, settle):
        self.settle, self.records, self.weights = settle, [], {}

    def checksum(self, tensor):
        """A 0-dimensional tensor on the tensor's device: the sum of its bytes, each times a weight that depends on
        its position, so that two tensors holding the same values in another order differ."""
        data = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
        total = torch.zeros((), dtype = torch.int64, device = data.device)
        for start in range(0, data.numel(), CHECKSUM_CHUNK_BYTES):
            chunk = data[start:start + CHECKSUM_CHUNK_BYTES]
            if chunk.device not in self.weights:
                self.weights[chunk.device] = torch.arange(CHECKSUM_CHUNK_BYTES, dtype = torch.int64,
                                                          device = chunk.device) % CHECKSUM_MODULUS + 1
            weighted = chunk.to(torch.int64) * self.weights[chunk.device][:chunk.numel()]
            total = total + weighted.sum() * (start // CHECKSUM_CHUNK_BYTES + 1)
        return total

    def note(self, name, kind, value):
        tensors = tensors_of(value)
        if not tensors:
            return
        if self.settle:
            torch.cuda.synchronize()
        with torch.no_grad():
            self.records.append((name, kind, " ".join(f"{tuple(tensor.shape)}:{str(tensor.dtype)[6:]}"
                                                      for tensor in tensors),
                                 [self.checksum(tensor) for tensor in tensors]))

    def finished(self):
        """The records with their checksums read from the card."""
        torch.cuda.synchronize()
        every = [checksum.to(oracle.CARD_DEVICE) for record in self.records for checksum in record[3]]
        values, finished, taken = (torch.stack(every).tolist() if every else []), [], 0
        for name, kind, shapes, checksums in self.records:
            finished.append((name, kind, shapes, tuple(values[taken:taken + len(checksums)])))
            taken += len(checksums)
        return finished


def traced_pass(session, token_ids, settle):
    """One pass over the prompt from position 0, as the oracle's captures run it, with every call recorded."""
    reference_modules = {name: sys.modules[name] for name in TRACED_REFERENCE_MODULES}
    trace, handles, replaced = Trace(settle), [], []

    def watch(name, module):
        handles.append(module.register_forward_pre_hook(
            lambda _, arguments: trace.note(name, RECORD_IN, arguments)))
        handles.append(module.register_forward_hook(
            lambda _, arguments, output: (trace.note(name, RECORD_OUT, output),
                                          trace.note(name, RECORD_AFTER, arguments)) and None))

    def wrap(module_name, function_name, function):
        name = f"{module_name}.{function_name}()"

        def traced(*positional, **named):
            trace.note(name, RECORD_IN, (positional, named))
            result = function(*positional, **named)
            trace.note(name, RECORD_OUT, result)
            trace.note(name, RECORD_AFTER, (positional, named))
            return result
        return traced

    for name, module in session.model.named_modules():
        watch(name or "model", module)
    for stage_index, stage in enumerate(session.stages):                   # the reference's Expert on the card
        for name, module in stage.expert.named_modules():
            watch(f"staged_expert_{stage_index}.{name}" if name else f"staged_expert_{stage_index}", module)
    for module_name, module in reference_modules.items():
        for function_name, function in list(vars(module).items()):
            if isinstance(function, types.FunctionType):                   # the module's own and the ones it imported
                replaced.append((module, function_name, function))
                setattr(module, function_name, wrap(module_name, function_name, function))
    try:
        oracle.forget_earlier_runs(session)
        input_ids = torch.tensor([token_ids], dtype = torch.long, device = oracle.CARD_DEVICE)
        with torch.inference_mode():
            _, logits, _ = session.model.forward(input_ids, 0)
            trace.note("logits of the last position", RECORD_OUT, logits)
            return trace.finished(), int(logits.argmax(dim = -1)[0])
    finally:
        for module, function_name, function in replaced:
            setattr(module, function_name, function)
        for handle in handles:
            handle.remove()


def compare(first, second, first_name, second_name):
    """Prints where two lists of records first differ. True when they are the same."""
    shared = min(len(first), len(second))
    differing = [index for index in range(shared) if first[index] != second[index]]
    if not differing and len(first) == len(second):
        print(f"{first_name} against {second_name}: the same {len(first)} records")
        return True
    if not differing:
        print(f"{first_name} against {second_name}: {len(first)} and {len(second)} records, the first {shared} "
              "the same")
        return False
    index = differing[0]
    print(f"{first_name} against {second_name}: {len(differing)} of {shared} records differ, the first is record "
          f"{index}")
    for earlier in range(max(0, index - CONTEXT_RECORDS), index):
        name, kind, shapes, _ = first[earlier]
        print(f"    same      {earlier:7d} {kind:5s} {name}  {shapes}")
    for which, records in ((first_name, first), (second_name, second)):
        name, kind, shapes, checksums = records[index]
        print(f"    DIFFERENT {index:7d} {kind:5s} {name}  {shapes}  in {which}: {checksums}")
    names = []
    for later in differing:
        label = f"{first[later][1]} {oracle.NUMBER_IN_A_NAME.sub('*', first[later][0])}"
        if label not in names:
            names.append(label)
    print(f"    kinds of record that differ, in order of first appearance: {names[:DIFFERING_NAMES_PRINTED]}")
    return False


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev = False)
    parser.add_argument("--checkpoint", required = True, type = os.path.abspath)
    parser.add_argument("--output-directory", required = True, type = os.path.abspath)
    parser.add_argument("--prompt-json", required = True, type = os.path.abspath)
    parser.add_argument("--card", type = int, default = 0)
    parser.add_argument("--attention-card", default = None)
    parser.add_argument("--tokens", type = int, default = None, help = "use only the first N tokens of the prompt")
    parser.add_argument("--passes", type = int, default = DEFAULT_PASSES)
    parser.add_argument("--settle", action = "store_true", help = "wait for the card before every record")
    parser.add_argument("--compare-with", default = None, type = os.path.abspath)
    arguments = parser.parse_args()
    if arguments.passes < 1 or arguments.card < 0 or (arguments.tokens is not None and arguments.tokens < 1):
        parser.error("--passes, --card or --tokens is out of range")
    if arguments.passes <= 1 and arguments.compare_with is None:
        parser.error("one pass and no --compare-with: nothing would be compared, and that is not 'the same'")
    arguments.modes, arguments.layer_list, arguments.self_test = ("forward",), [0], False
    arguments.tokens_json, arguments.sequence, arguments.new_tokens = None, 0, None
    arguments.thinking_mode = oracle.DEFAULT_THINKING_MODE
    arguments.maximum_sequence_tokens = oracle.DEFAULT_MAXIMUM_SEQUENCE_TOKENS
    try:
        os.makedirs(arguments.output_directory, exist_ok = True)
        earlier = None if arguments.compare_with is None \
            else torch.load(os.path.join(arguments.compare_with, TRACE_FILE_NAME))   # before this run saves its own
        session, token_ids = oracle.open_session(arguments, {})
        passes = []
        for pass_number in range(1, arguments.passes + 1):
            records, next_token_id = traced_pass(session, token_ids, arguments.settle)
            print(f"pass {pass_number}: {len(records)} records over {len(token_ids)} tokens; the next token would be "
                  f"{next_token_id}", flush = True)
            passes.append(records)
        torch.save(passes[0], os.path.join(arguments.output_directory, TRACE_FILE_NAME))
        same = all([compare(passes[0], later, "pass 1", f"pass {number}")
                    for number, later in enumerate(passes[1:], start = 2)])
        if earlier is not None:
            same = compare(earlier, passes[0], "the earlier process's pass 1", "pass 1") and same
    except Exception as failure:
        import traceback
        traceback.print_exc()
        print(f"reference_divergence FAILED: {type(failure).__name__}: {failure}", file = sys.stderr)
        return EXIT_CODE_FAILURE
    return EXIT_CODE_SAME if same else EXIT_CODE_DIFFERENT


if __name__ == "__main__":
    sys.exit(main())
