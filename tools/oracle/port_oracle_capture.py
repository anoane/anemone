#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference captures of today's engine: what every Anemone piece is checked against.

Loads the exllamav3 port (the frozen copy that serves today) once, in the layout given, and performs one capture:
  * layers      a corpus prompt run from an empty cache with direct model.forward calls; for the FIRST chunk only and
                for every transformer block: the streams that enter and leave it, the input and output of its
                attention and of its MLP, and what crosses layers inside one forward (the collapse weights of the
                previous site, the pool entries an indexer selected, the candidate blocks); then the attention
                state and the last row's logits after that chunk, as "state" records them;
  * state       the attention state after a corpus prompt: the pool entries the sequence wrote, the window rows and
                compressor buffers of every layer, the token ids the n-gram layers keep, the last row's logits;
  * tokens      greedy continuations of the requests of a JSON file, [{"name": ..., "token_ids": [...]}, ...],
                through the Generator, every request more than once, and whether the runs agree;
  * likelihood  teacher-forced negative log-likelihood of a corpus text, per token, in float64 on the CPU; two runs
                of it give the noise floor of the quality gate;
  * logits      one request of a JSON file run from an empty cache with direct model.forward calls, and the logits
                of every position in float32, under the record names of reference_oracle.py's forward capture, so
                that compare_logit_captures.py measures this engine and its weights against the reference's.

    run_with_port.sh ../oracle/port_oracle_capture.py LAYOUT --output-directory FOLDER --capture layers --tokens 4096
    run_with_port.sh ../oracle/port_oracle_capture.py LAYOUT --output-directory FOLDER --capture layers --tokens 8 --chunk-tokens 8
    run_with_port.sh ../oracle/port_oracle_capture.py LAYOUT --output-directory FOLDER --capture state --tokens 100000 --cache-tokens 131072
    run_with_port.sh ../oracle/port_oracle_capture.py LAYOUT --output-directory FOLDER --capture tokens --requests requests.json --new-tokens 200
    run_with_port.sh ../oracle/port_oracle_capture.py LAYOUT --output-directory FOLDER --capture likelihood --tokens 8192
    run_with_port.sh ../oracle/port_oracle_capture.py LAYOUT --output-directory FOLDER --capture logits \
                     --requests chunks.json --request-index 0

LAYOUT is kernel_time_trace.py's: --layers-per-device 12,18,10 --ram-layers 33..39, for instance. FOLDER receives
tensors.bin (raw little-endian tensor bytes, one tensor after another) and manifest.json (name, dtype, shape,
byte_offset and byte_count of every tensor, the conditions of the run and its results); it must not hold a capture
already. It is created, and tensors.bin opened, before the model is loaded; a run that fails after that leaves a
manifest that says "complete": false. A relative FOLDER or requests file is taken from the working directory, and
run_with_port.sh changes into its own folder first; the absolute FOLDER is printed and recorded. The layers capture
of 4096 rows takes about 37 GB. For tokens, --chunk-tokens is a whole number of cache pages, as the Generator
prefills. The environment is whatever the caller exports; its EXL3_*, EXLLAMA*, CUDA_*, TORCH_* and TRITON_*
variables are recorded, with the arguments as resolved.
Measurement tool: not part of the engine.
"""
import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import traceback

import torch

sys.dont_write_bytecode = True               # nothing is written outside the output directory, compiled imports included
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "measure"))
from kernel_time_trace import (CACHE_PAGE_TOKENS, DEFAULT_CORPUS_DIRECTORY, DEFAULT_MODEL_DIRECTORY,
                               PREFILL_CHUNK_TOKENS, load_corpus_tokens, load_model)

DEFAULT_CACHE_TOKENS = 65536                 # as kernel_time_trace.py; a longer prompt needs more
TENSORS_FILE_NAME = "tensors.bin"
MANIFEST_FILE_NAME = "manifest.json"
REQUIRED_BYTE_ORDER = "little"               # tensors.bin holds the host's bytes as they are
NUMPY_NAME_BY_DTYPE = {torch.float16: "float16", torch.float32: "float32", torch.float64: "float64",
                       torch.int8: "int8", torch.uint8: "uint8", torch.int16: "int16", torch.int32: "int32",
                       torch.int64: "int64"}
RECORDED_ENVIRONMENT_PREFIXES = ("EXL3_", "EXLLAMA", "CUDA_", "TORCH_", "TRITON_")   # the port's switches and tuning
                                             # folder, the visible cards and their order, torch's and Triton's switches
PLACEMENT_LINE_START = "placement"           # first word of the placement line load_model prints
LIKELIHOOD_SLAB_ROWS = 256                   # rows of logits scored at a time: 265 MB in float64 at 129,280 columns
GENERATION_RUNS = 2                          # times every request is generated, to see whether the runs agree
LAYER_CAPTURE_BYTES_PER_VALUE = 44           # layers capture, per block, row and hidden value: 4 fp32 streams in and
                                             # 4 out (32), fp16 inputs (4) and fp32 outputs (8) of attention and MLP
DIRECT_FORWARD_SEQUENCES = 1                 # the direct model.forward calls run one sequence
GENERATOR_SEQUENCES = 1                      # one job at a time, as the measure tools run the Generator
GENERATOR_RESERVED_TOKENS = 1                # cache tokens a job takes beyond its prompt and its new tokens (the port's
                                             # generator/job.py: max_rq_tokens = max_new_tokens + 1 + draft tokens)
GENERATOR_EMPTY_STEPS_LIMIT = 16             # steps in a row that return nothing before a job counts as stuck: a job
                                             # that moves reports its start, its prefill progress or a token every step
MANIFEST_INDENT_SPACES = 1
EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1
POOL_TENSOR_DESCRIPTIONS = {                 # CacheLayer_dsa attributes; the ones a cache does not use are None
    "pool_c": "pool entries, plain part",
    "pool_q": "pool entries, plain part in the packed form: int32 words",
    "pool_s": "pool entries, plain part in the packed form: group scales",
    "pool_r": "pool entries, rotated part",
    "pool_idx": "index keys of the pool entries",
}
STATE_TENSOR_DESCRIPTIONS = {                # tensors of the per-layer recurrent states, by attribute name
    "ring": "window rows (one rotated key = value row per token): row r is absolute position window_begin + r; "
            "the newest window - 1 of them are what the next token reads",
    "comp_buf_kv": "compressor buffer, projected kv rows, at index absolute position modulo the buffer length",
    "comp_buf_gate": "compressor buffer, projected gate rows, at index absolute position modulo the buffer length",
    "comp_ovl": "compressor overlap snapshots, at index (entry number - 1) modulo their count",
    "idx_buf_kv": "indexer compressor buffer, projected kv rows, at index absolute position modulo the buffer length",
    "idx_buf_gate": "indexer compressor buffer, projected gate rows, at index absolute position modulo the buffer length",
    "idx_ovl": "indexer compressor overlap snapshots, at index (entry number - 1) modulo their count",
    "ids": "compressed token ids the n-gram layer keeps, at index absolute position modulo the ring length; only the "
           "newest positions_held positions (position - positions_held to position - 1) are the sequence's, the "
           "other entries are what earlier forwards left in the slot",
}


class CaptureWriter:
    """tensors.bin, written tensor by tensor as each is captured, and manifest.json, written when the capture ends.
    The conditions of the run are set in manifest["conditions"] once the model is loaded."""

    def __init__(self, output_directory):
        os.makedirs(output_directory, exist_ok = True)
        self.manifest_path = os.path.join(output_directory, MANIFEST_FILE_NAME)
        self.tensors_path = os.path.join(output_directory, TENSORS_FILE_NAME)
        self.tensors_file = open(self.tensors_path, "xb")                  # refuses to overwrite a capture
        self.records, self.results, self.names, self.byte_offset = [], {}, set(), 0
        self.manifest = {"complete": False, "conditions": None, "results": self.results, "tensors": self.records}

    def add(self, name, tensor, layer, description, **facts):
        if name in self.names:
            raise RuntimeError(f"tensor recorded twice: {name}")
        if tensor.dtype == torch.bool:
            tensor = tensor.to(torch.uint8)
        if tensor.dtype not in NUMPY_NAME_BY_DTYPE:
            raise RuntimeError(f"{name}: no numpy name for {tensor.dtype}")
        values = tensor.detach().cpu().contiguous().numpy()
        values.tofile(self.tensors_file)
        self.records.append({"name": name, "dtype": NUMPY_NAME_BY_DTYPE[tensor.dtype], "shape": list(values.shape),
                             "byte_offset": self.byte_offset, "byte_count": values.nbytes, "layer": layer,
                             "description": description, **facts})
        self.names.add(name)
        self.byte_offset += values.nbytes

    def close(self, complete):
        self.tensors_file.close()
        bytes_on_disk = os.path.getsize(self.tensors_path)
        self.manifest["complete"] = complete and bytes_on_disk == self.byte_offset
        with open(self.manifest_path, "x", encoding = "utf8") as handle:
            json.dump(self.manifest, handle, indent = MANIFEST_INDENT_SPACES)
        if bytes_on_disk != self.byte_offset:
            raise RuntimeError(f"{self.tensors_path} holds {bytes_on_disk} bytes, {self.byte_offset} were recorded")


class PrintedTextCopy:
    """Stands in for the standard output: everything goes through, and a copy of the text is kept."""

    def __init__(self, stream):
        self.stream, self.pieces = stream, []

    def write(self, text):
        self.pieces.append(text)
        return self.stream.write(text)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def cache_rows_for(token_count):
    """Rows of the cache a sequence of token_count tokens takes: whole pages."""
    return -(-token_count // CACHE_PAGE_TOKENS) * CACHE_PAGE_TOKENS


def transformer_blocks(model):
    return [module for module in model.modules if type(module).__name__ == "TransformerBlock"]


def describe_run(arguments, model, placement):
    import exllamav3
    layers = []
    for block in transformer_blocks(model):
        attention, mlp = block.attn, block.mlp
        layers.append({
            "layer": block.layer_idx, "card": str(block.device),
            "experts": "RAM" if getattr(mlp, "cpu_offload", False)
                       else "split VRAM/RAM" if getattr(mlp, "cpu_split_first", None) is not None else "VRAM",
            "attention_kind": attention.layer_type, "tokens_per_pool_entry": attention.compress_rate,
            "pool_of_layer": attention.kv_source, "indexer": attention.indexer_mode,
            "candidate_blocks": attention.candidate_mode,
        })
    return {
        "capture": arguments.capture,
        "command_line": sys.argv,
        "arguments": {name: value for name, value in sorted(vars(arguments).items()) if name != "request_list"},
        "placement": placement,
        "environment": {name: value for name, value in sorted(os.environ.items())
                        if name.startswith(RECORDED_ENVIRONMENT_PREFIXES)},
        "torch_version": torch.__version__,
        "cards": [torch.cuda.get_device_name(card) for card in range(torch.cuda.device_count())],
        "port_directory": os.path.dirname(os.path.abspath(exllamav3.__file__)),
        "vocabulary_size": model.config.vocab_size,
        "layers": layers,
    }


def record_corpus_tokens(arguments, tokenizer, writer, token_count):
    token_ids = load_corpus_tokens(tokenizer, arguments.corpus, token_count)
    writer.add("input.token_ids", token_ids, None, "token ids of the corpus text, in order")
    return token_ids


def forward_in_chunks(model, cache, token_ids, chunk_tokens, after_chunk):
    """Runs token_ids (1, n) through model.forward from an empty cache, chunk by chunk, the way the measure tools
    drive the port: every module sees the chunk's rows, the head included. after_chunk(position, logits, parameters,
    state) sees every forward before the next one starts."""
    token_count = token_ids.shape[1]
    state = cache.get_new_state()
    try:
        for position in range(0, token_count, chunk_tokens):
            chunk = token_ids[:, position:position + chunk_tokens]
            parameters = {"attn_mode": "flash_attn", "cache": cache, "past_len": position, "recurrent_states": [state],
                          "batch_shape": (DIRECT_FORWARD_SEQUENCES, cache_rows_for(token_count))}
            logits = model.forward(chunk, parameters)
            if state.position != position + chunk.shape[1]:
                raise RuntimeError(f"the forward at {position} left the sequence at {state.position}")
            after_chunk(position, logits, parameters, state)
            del logits
    finally:
        cache.release_state(state)


def record_sequence_state(writer, cache, state, block_table):
    """What the port holds for one sequence between forwards: its pool entries, found through its block table
    (entry e is row e % entries_per_page of page block_table[e // entries_per_page]), and its slot of every
    per-layer recurrent tensor."""
    slot, position, window_begin = state.slot, state.position, state.window_beg
    block_table = block_table.cpu()
    for (layer, _), pool_layer in cache.layers.items():
        entries = torch.arange(position // pool_layer.compress_rate)
        pool_rows = block_table[0, entries // pool_layer.epp].long() * pool_layer.epp + entries % pool_layer.epp
        for name, description in POOL_TENSOR_DESCRIPTIONS.items():
            pool = getattr(pool_layer, name)
            if pool is not None:
                writer.add(f"layer_{layer}.pool.{name}", pool.view(-1, pool.shape[-1])[pool_rows.to(pool.device)],
                           layer, description + ": row e is entry e of the sequence",
                           tokens_per_entry = pool_layer.compress_rate)
    for (layer_number, _), layer_state in cache.get_all_recurrent_layers().items():
        # the port numbers the n-gram layer in front of block i as -(i + 1)
        layer, kind = (layer_number, "attention_state") if layer_number >= 0 else (-layer_number - 1, "ngram_state")
        for name, tensor in sorted(vars(layer_state).items()):
            if not isinstance(tensor, torch.Tensor):
                continue
            slot_tensor, facts = tensor[slot], {}
            if name == "ring":
                rows_held = position - window_begin
                if not 0 <= rows_held <= len(slot_tensor):
                    raise RuntimeError(f"layer {layer}: {rows_held} window rows held, the ring has {len(slot_tensor)}")
                slot_tensor, facts = slot_tensor[:rows_held], {"window_begin": window_begin}
            elif name == "ids":                                            # a new sequence does not clear them
                facts = {"positions_held": min(position, len(slot_tensor))}
            writer.add(f"layer_{layer}.{kind}.{name}", slot_tensor, layer,
                       STATE_TENSOR_DESCRIPTIONS.get(name, f"recurrent state tensor {name}"), **facts)
    writer.results["sequence_state"] = {"slot": slot, "position": position, "window_begin": window_begin}


def record_last_row_logits(writer, model, logits):
    writer.add("logits.last_row", logits[:, -1:], None, "logits of the last row, computed with the rows of its chunk, "
               f"as the head returns them: the first {model.config.vocab_size} columns are the vocabulary")


def install_layer_recorders(blocks, writer):
    """Replaces the forward of every block, of its attention and of its MLP by one that records what goes in and
    what comes out. Returns the modules changed and the calls seen per kind, for the caller to undo and to check."""
    from exllamav3.modules.hyperconnections import hc_flush
    changed, calls = [], {"block": 0, "attention": 0, "mlp": 0, "stream_updates_finished_here": 0}

    def shared_by_layers(parameters):
        """What the port's attention layers hand to each other inside one forward, for the sequence of this run."""
        return (parameters.get("dsv4_shared") or {}).get(parameters["recurrent_states"][0].slot, {})

    def record_collapse_weights(name, layer, parameters, description):
        if parameters.get("hc_pre") is not None:
            writer.add(name, parameters["hc_pre"], layer, description)

    def install(block):
        layer, name = block.layer_idx, f"layer_{block.layer_idx}"
        attention, mlp = block.attn, block.mlp
        block_forward, attention_forward, mlp_forward = block.forward, attention.forward, mlp.forward

        def recording_block_forward(streams, parameters, *positional, **named):
            calls["block"] += 1
            writer.add(f"{name}.block.input", streams, layer, "residual streams entering the block")
            record_collapse_weights(f"{name}.block.input_collapse_weights", layer, parameters,
                                    "weights of the site before this block, which collapse the entering streams "
                                    "for its attention (absent for the first block: it takes stream 0)")
            streams_out = block_forward(streams, parameters, *positional, **named)
            if "hc_pending" in parameters:           # the port left the block's last stream update to the next site
                hc_flush(parameters)
                calls["stream_updates_finished_here"] += 1
            writer.add(f"{name}.block.output", streams_out, layer, "residual streams leaving the block")
            record_collapse_weights(f"{name}.block.output_collapse_weights", layer, parameters,
                                    "weights computed at this block's MLP site, which collapse the streams at the "
                                    "next site")
            return streams_out

        def recording_attention_forward(attention_input, parameters, *positional, **named):
            calls["attention"] += 1
            writer.add(f"{name}.attention.input", attention_input, layer,
                       "attention input: the collapsed and normalised rows")
            shared_before = dict(shared_by_layers(parameters))
            attention_output = attention_forward(attention_input, parameters, *positional, **named)
            writer.add(f"{name}.attention.output", attention_output, layer,
                       "attention output, before the stream update")
            shared = shared_by_layers(parameters)
            selection, dropped_blocks = shared.get("topk"), shared.get("cand")
            if attention.indexer_mode == "full" and selection is not None and selection is not shared_before.get("topk"):
                writer.add(f"{name}.attention.selected_entries", selection[0], layer,
                           "pool entries this layer's indexer selected for every row, -1 where unused; this layer "
                           "and the layers after it without an indexer attend to them",
                           selected_per_row = int(selection[1]))
            if attention.candidate_mode == "source" and dropped_blocks is not None \
                    and dropped_blocks is not shared_before.get("cand"):
                writer.add(f"{name}.attention.dropped_candidate_blocks", dropped_blocks, layer,
                           "per row and block of pool entries: 1 = not a candidate for the later indexers, 0 = kept",
                           entries_per_block = attention.candidate_block_size)
            return attention_output

        def recording_mlp_forward(mlp_input, parameters, *positional, **named):
            calls["mlp"] += 1
            writer.add(f"{name}.mlp.input", mlp_input, layer, "MLP input: the collapsed and normalised rows")
            mlp_output = mlp_forward(mlp_input, parameters, *positional, **named)
            writer.add(f"{name}.mlp.output", mlp_output, layer, "MLP output (routed and shared experts), before the "
                       "stream update")
            return mlp_output

        block.forward, attention.forward, mlp.forward = \
            recording_block_forward, recording_attention_forward, recording_mlp_forward
        changed.extend((block, attention, mlp))

    for block in blocks:
        install(block)
    return changed, calls


def capture_layers(arguments, model, cache, tokenizer, writer):
    blocks = transformer_blocks(model)
    first_chunk_rows = min(arguments.tokens, arguments.chunk_tokens)
    bytes_needed = len(blocks) * first_chunk_rows * model.config.hidden_size * LAYER_CAPTURE_BYTES_PER_VALUE
    bytes_free = shutil.disk_usage(arguments.output_directory).free
    if bytes_free < bytes_needed:
        raise RuntimeError(f"about {bytes_needed} bytes are needed in {arguments.output_directory}, {bytes_free} are free")
    token_ids = record_corpus_tokens(arguments, tokenizer, writer, arguments.tokens)
    changed, calls = install_layer_recorders(blocks, writer)

    def after_chunk(position, logits, parameters, state):
        if position:
            return                                                         # only the first chunk is recorded
        for module in changed:
            del module.forward                                             # back to the port's own forward
        if not calls["block"] == calls["attention"] == calls["mlp"] == len(blocks):
            raise RuntimeError(f"every module should have run once in the first chunk: {calls}, {len(blocks)} blocks")
        record_sequence_state(writer, cache, state, parameters["block_table"])
        record_last_row_logits(writer, model, logits)
        writer.results.update(rows_recorded = state.position, blocks = len(blocks),
                              stream_updates_finished_by_this_tool = calls["stream_updates_finished_here"])

    forward_in_chunks(model, cache, token_ids, arguments.chunk_tokens, after_chunk)


def capture_state(arguments, model, cache, tokenizer, writer):
    token_ids = record_corpus_tokens(arguments, tokenizer, writer, arguments.tokens)

    def after_chunk(position, logits, parameters, state):
        if state.position == arguments.tokens:                             # the last chunk
            record_sequence_state(writer, cache, state, parameters["block_table"])
            record_last_row_logits(writer, model, logits)

    forward_in_chunks(model, cache, token_ids, arguments.chunk_tokens, after_chunk)


def capture_likelihood(arguments, model, cache, tokenizer, writer):
    token_ids = record_corpus_tokens(arguments, tokenizer, writer, arguments.tokens + 1)   # + the last row's target
    vocabulary_size = model.config.vocab_size
    values = torch.empty(arguments.tokens, dtype = torch.float64)

    def after_chunk(position, logits, parameters, state):
        row_count = state.position - position
        if tuple(logits.shape[:-1]) != (DIRECT_FORWARD_SEQUENCES, row_count) or logits.shape[-1] < vocabulary_size:
            raise RuntimeError(f"logits of shape {tuple(logits.shape)} for {row_count} rows, {vocabulary_size} tokens")
        for first_row in range(0, row_count, LIKELIHOOD_SLAB_ROWS):
            slab = logits[0, first_row:first_row + LIKELIHOOD_SLAB_ROWS, :vocabulary_size].cpu().double()
            first_position, end_position = position + first_row, position + first_row + slab.shape[0]
            next_tokens = token_ids[0, first_position + 1:end_position + 1]
            values[first_position:end_position] = \
                torch.logsumexp(slab, dim = -1) - slab.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

    forward_in_chunks(model, cache, token_ids[:, :arguments.tokens], arguments.chunk_tokens, after_chunk)
    writer.add("likelihood.negative_log_likelihood", values, None,
               "value i is -ln p(token i + 1 | tokens 0 .. i) under teacher forcing")
    values_not_finite = int((~torch.isfinite(values)).sum())
    if values_not_finite:
        raise RuntimeError(f"{values_not_finite} of {arguments.tokens} values are not finite")
    mean = values.mean().item()
    writer.results.update(tokens_scored = arguments.tokens, mean_negative_log_likelihood = mean,
                          exponent_of_the_mean = math.exp(mean))
    print(f"mean negative log-likelihood of {arguments.tokens} tokens: {mean!r}; its exponent: {math.exp(mean)!r}")


def capture_logits(arguments, model, cache, tokenizer, writer):
    request = arguments.request_list[arguments.request_index]
    vocabulary_size = model.config.vocab_size
    if max(request["token_ids"]) >= vocabulary_size:
        raise RuntimeError(f"request {request['name']}: token id {max(request['token_ids'])} is outside the "
                           f"vocabulary of {vocabulary_size}")
    token_ids = torch.tensor([request["token_ids"]], dtype = torch.long)
    writer.add("input.token_ids", token_ids, None, f"token ids of request {request['name']}, in order")
    every_row = torch.empty(DIRECT_FORWARD_SEQUENCES, token_ids.shape[1], vocabulary_size, dtype = torch.float32)
    dtypes_returned = set()

    def after_chunk(position, logits, parameters, state):
        row_count = state.position - position
        if tuple(logits.shape[:-1]) != (DIRECT_FORWARD_SEQUENCES, row_count) or logits.shape[-1] < vocabulary_size:
            raise RuntimeError(f"logits of shape {tuple(logits.shape)} for {row_count} rows, {vocabulary_size} tokens")
        dtypes_returned.add(str(logits.dtype))
        every_row[:, position:position + row_count] = logits[..., :vocabulary_size].float().cpu()

    forward_in_chunks(model, cache, token_ids, arguments.chunk_tokens, after_chunk)
    values_not_finite = int((~torch.isfinite(every_row)).sum())
    if values_not_finite:
        raise RuntimeError(f"{values_not_finite} logits are not finite")
    writer.add("logits.all_rows", every_row, None, "logits of every position, each computed with the rows of its "
               "chunk, as the head returns them and widened to float32: [sequence, position, vocabulary]")
    writer.results.update(request = request["name"], positions = token_ids.shape[1],
                          logits_dtypes_the_port_returned = sorted(dtypes_returned))
    print(f"request {request['name']}: logits of {token_ids.shape[1]} positions, returned as {sorted(dtypes_returned)}")


def generate_greedy(generator, request, new_tokens):
    """One greedy job from an empty prompt cache. Returns the new_tokens generated ids and the job, for its cache
    counters."""
    from exllamav3 import Job
    from exllamav3.generator.sampler import ArgmaxSampler
    generator.pagetable.reset_page_table()
    job = Job(input_ids = torch.tensor(request["token_ids"], dtype = torch.long).unsqueeze(0),
              max_new_tokens = new_tokens, sampler = ArgmaxSampler(), stop_conditions = [])
    generator.enqueue(job)
    generated, empty_steps = [], 0
    while generator.num_remaining_jobs():
        results = generator.iterate()
        empty_steps = 0 if results else empty_steps + 1
        if empty_steps == GENERATOR_EMPTY_STEPS_LIMIT:
            raise RuntimeError(f"request {request['name']}: the Generator returned nothing {empty_steps} times in a "
                               f"row, the job is stuck after {len(generated)} of {new_tokens} tokens")
        for result in results:
            if result.get("stage") == "error":
                error = result.get("error")
                raise RuntimeError(f"the generator reported {error!r}") from (error if isinstance(error, BaseException) else None)
            token_ids = result.get("token_ids")
            if token_ids is not None:
                generated += token_ids.flatten().tolist()
    if len(generated) != new_tokens:
        raise RuntimeError(f"request {request['name']}: {len(generated)} tokens were generated, --new-tokens asked "
                           f"for {new_tokens}")
    return generated, job


def first_difference(runs):
    """Index of the first generated token the runs disagree on, None when they are identical."""
    shortest = min(len(run) for run in runs)
    for index in range(shortest):
        if len({run[index] for run in runs}) > 1:
            return index
    return None if all(len(run) == shortest for run in runs) else shortest


def capture_tokens(arguments, model, cache, tokenizer, writer):
    from exllamav3 import Generator
    for request in arguments.request_list:
        if max(request["token_ids"]) >= model.config.vocab_size:
            raise RuntimeError(f"request {request['name']}: token id {max(request['token_ids'])} is outside the "
                               f"vocabulary of {model.config.vocab_size}")
        writer.add(f"request.{request['name']}.prompt_token_ids", torch.tensor(request["token_ids"], dtype = torch.int64),
                   None, f"prompt of request {request['name']}")
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer, max_batch_size = GENERATOR_SEQUENCES,
                          max_chunk_size = arguments.chunk_tokens)
    runs_by_request = {request["name"]: [] for request in arguments.request_list}
    for run in range(1, GENERATION_RUNS + 1):
        for request in arguments.request_list:
            generated, job = generate_greedy(generator, request, arguments.new_tokens)
            writer.add(f"request.{request['name']}.run_{run}.generated_token_ids",
                       torch.tensor(generated, dtype = torch.int64), None,
                       f"greedy continuation of request {request['name']}, run {run}",
                       prompt_tokens = len(request["token_ids"]),
                       prompt_tokens_found_in_the_cache = job.cached_pages * CACHE_PAGE_TOKENS + job.cached_tokens)
            runs_by_request[request["name"]].append(generated)
            print(f"run {run}, request {request['name']}: {len(generated)} tokens", flush = True)
    comparisons = []
    for name, runs in runs_by_request.items():
        differs_at = first_difference(runs)
        comparisons.append({"name": name, "identical": differs_at is None, "first_difference": differs_at})
        print(f"request {name}: " + ("identical in every run" if differs_at is None
                                     else f"the runs differ from generated token {differs_at} on"))
    writer.results.update(runs = GENERATION_RUNS, requests = comparisons,
                          all_identical = all(comparison["identical"] for comparison in comparisons))


CAPTURES = {"layers": capture_layers, "state": capture_state, "tokens": capture_tokens,
            "likelihood": capture_likelihood, "logits": capture_logits}


def read_requests(parser, arguments, tokens_beyond_the_prompt):
    try:
        with open(arguments.requests, encoding = "utf8") as handle:
            requests = json.load(handle)
    except (OSError, ValueError) as failure:
        parser.error(f"--requests {arguments.requests}: {failure}")
    if not isinstance(requests, list) or not requests:
        parser.error(f"--requests {arguments.requests}: a non-empty list of objects is expected")
    names = set()
    for request in requests:
        name, token_ids = (request.get("name"), request.get("token_ids")) if isinstance(request, dict) else (None, None)
        if not isinstance(name, str) or not name or name in names:
            parser.error(f"--requests {arguments.requests}: every request needs a \"name\" of its own, found {name!r}")
        if not isinstance(token_ids, list) or not token_ids \
                or not all(type(token_id) is int and token_id >= 0 for token_id in token_ids):
            parser.error(f"--requests {arguments.requests}: request {name} needs \"token_ids\", a list of token ids")
        if cache_rows_for(len(token_ids) + tokens_beyond_the_prompt) > arguments.cache_tokens:
            parser.error(f"request {name}: {len(token_ids)} prompt tokens and {tokens_beyond_the_prompt} beyond them "
                         f"do not fit --cache-tokens {arguments.cache_tokens}")
        names.add(name)
    return requests


def check_arguments(parser, arguments):
    """Refuses, before the model is loaded, what can be refused then."""
    if sys.byteorder != REQUIRED_BYTE_ORDER:
        parser.error(f"this host is {sys.byteorder}-endian: tensors.bin must be {REQUIRED_BYTE_ORDER}-endian")
    for file_name in (TENSORS_FILE_NAME, MANIFEST_FILE_NAME):
        if os.path.exists(os.path.join(arguments.output_directory, file_name)):
            parser.error(f"--output-directory {arguments.output_directory} holds a {file_name} already")
    if not 1 <= arguments.chunk_tokens <= PREFILL_CHUNK_TOKENS:
        parser.error(f"--chunk-tokens must be 1 to {PREFILL_CHUNK_TOKENS}, the chunk the model is loaded for")
    if arguments.cache_tokens < CACHE_PAGE_TOKENS or arguments.cache_tokens % CACHE_PAGE_TOKENS:
        parser.error(f"--cache-tokens must be a positive multiple of {CACHE_PAGE_TOKENS}")
    if arguments.capture == "tokens":
        if arguments.requests is None or arguments.new_tokens is None or arguments.new_tokens < 1:
            parser.error("--capture tokens needs --requests and --new-tokens of at least 1")
        if arguments.chunk_tokens < CACHE_PAGE_TOKENS or arguments.chunk_tokens % CACHE_PAGE_TOKENS:
            parser.error(f"--capture tokens needs --chunk-tokens to be a multiple of {CACHE_PAGE_TOKENS}: the "
                         "Generator prefills whole cache pages, and never advances on less than one")
        arguments.request_list = read_requests(parser, arguments, arguments.new_tokens + GENERATOR_RESERVED_TOKENS)
        return
    if arguments.capture == "logits":
        if arguments.requests is None or arguments.request_index is None:
            parser.error("--capture logits needs --requests and --request-index")
        arguments.request_list = read_requests(parser, arguments, 0)
        if not 0 <= arguments.request_index < len(arguments.request_list):
            parser.error(f"--request-index {arguments.request_index}: the file holds "
                         f"{len(arguments.request_list)} requests")
        return
    if arguments.tokens is None or arguments.tokens < 1:
        parser.error(f"--capture {arguments.capture} needs --tokens of at least 1")
    if cache_rows_for(arguments.tokens) > arguments.cache_tokens:
        parser.error(f"--cache-tokens {arguments.cache_tokens} cannot hold {arguments.tokens} tokens: "
                     f"{cache_rows_for(arguments.tokens)} are needed")


def run_capture(arguments):
    writer = CaptureWriter(arguments.output_directory)       # before the load: a folder that cannot be used fails now
    complete = False
    try:
        printed = PrintedTextCopy(sys.stdout)
        with contextlib.redirect_stdout(printed):
            _, model, cache, tokenizer = load_model(arguments)
        placement = next((line for line in "".join(printed.pieces).splitlines()
                          if line.startswith(PLACEMENT_LINE_START)), None)
        if placement is None:
            raise RuntimeError("load_model printed no placement line")
        writer.manifest["conditions"] = describe_run(arguments, model, placement)
        CAPTURES[arguments.capture](arguments, model, cache, tokenizer, writer)
        complete = True
    finally:
        writer.close(complete)
    print(f"{arguments.capture}: {len(writer.records)} tensors, {writer.byte_offset} bytes in {arguments.output_directory}")


def absolute_path(text):
    """The type of the path options: a relative path is taken from the working directory."""
    if not text:
        raise argparse.ArgumentTypeError("the path is empty")
    return os.path.abspath(text)


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", choices = tuple(CAPTURES), required = True)
    parser.add_argument("--output-directory", required = True, type = absolute_path,
                        help = "receives tensors.bin and manifest.json")
    parser.add_argument("--model", default = DEFAULT_MODEL_DIRECTORY)
    parser.add_argument("--corpus", default = DEFAULT_CORPUS_DIRECTORY)
    parser.add_argument("--ram-layers", default = "0", help = "layers whose experts stay in RAM, as the port's -mcl takes them")
    parser.add_argument("--ram-split", default = "0", help = "experts per layer kept in RAM, as the port's -mcs takes them")
    parser.add_argument("--layers-per-device", default = None)
    parser.add_argument("--gigabytes-per-device", default = None)
    parser.add_argument("--cache-tokens", type = int, default = DEFAULT_CACHE_TOKENS)
    parser.add_argument("--tokens", type = int, default = None, help = "corpus tokens of the prompt: layers, state, likelihood")
    parser.add_argument("--chunk-tokens", type = int, default = PREFILL_CHUNK_TOKENS,
                        help = "rows per model.forward call; for tokens, the Generator's chunk, in whole cache pages")
    parser.add_argument("--requests", default = None, type = absolute_path,
                        help = "tokens, logits: JSON list of objects with \"name\" and \"token_ids\"")
    parser.add_argument("--request-index", type = int, default = None, help = "logits: which request of the file")
    parser.add_argument("--new-tokens", type = int, default = None, help = "tokens: tokens generated per request")
    arguments = parser.parse_args()
    check_arguments(parser, arguments)
    try:
        with torch.inference_mode():
            run_capture(arguments)
    except Exception as failure:
        traceback.print_exc()
        print(f"port_oracle_capture FAILED: {type(failure).__name__}: {failure}", file = sys.stderr)
        return EXIT_CODE_FAILURE
    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
