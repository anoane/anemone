#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What stepwise_capture.py records of DeepSeek's reference while it runs, and how (design/first_token_plan.md 3.1
item 2, unit U13a): 13b's switch recording each kind of state before and after its u16 rounding, the gate's scores
formed again from its own linear output, and the step recorder's hooks and wrappers, each calling what it wraps.
stepwise_capture.py's module text lists the records; compare_engine_capture.py names them. Nothing of the
reference's arithmetic changes. Measurement tool: not part of the engine.
"""
import collections
import re

import torch

import compare_engine_capture as gates
import reference_oracle as oracle
import state_precision_true as unit_13b

if set(gates.KIND_RECORDS) != set(unit_13b.KINDS):
    raise ImportError("compare_engine_capture.py names other kinds of state than state_precision.py")
CAPTURE_LAYERS = (0, 1, 2, 3, 8, 14, 15, 16, 20, 21, 24, 31, 32, 39)    # 3.1 item 2
CAPTURE_POSITIONS = (0, 1, 2, 3, 127, 128, 129, 255, 256, 511, 512, 513, 1023, 1024, 1025, 1026, 2047)
CAPTURE_GENERATED_STEPS = (1, 32, 64)
SNAPSHOT_POSITIONS = (129, 1026, 2047)
ROUTE_WEIGHT_EPSILON = 1e-20                 # model.py:825
EXPERT_NAME = re.compile(r"layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)")
BLOCK_STREAMS_ARGUMENT, BLOCK_PRE_MIX_ARGUMENT = 0, 2    # of a block's forward (model.py:968, called at :1267)


class CaptureSwitch(unit_13b.FormSwitch):
    """13b's switch; at a capture point the value of a kind is recorded before and after the u16 rounding, the
    attention output before and after the reference's rounding (46cc8310:787), and the last queries are kept on the
    card for the scores formed again."""

    def __init__(self, reference_model, break_name):
        super().__init__(reference_model, break_name)
        self.recorder, self.query = None, None

    def in_place(self, kind, values, quantize):
        recorder = self.recorder
        before = values.detach().clone() if recorder is not None and recorder.active() else None
        result = super().in_place(kind, values, quantize)
        if kind == unit_13b.QUERIES:
            self.query = result
        if before is not None:
            recorder.point(gates.KIND_RECORDS[kind] + "_unrounded", before[0, 0])
            recorder.point(gates.KIND_RECORDS[kind], result[0, 0])
        return result

    def output_rounded(self, values, quantize):                            # values: a view of attention_output
        recorder = self.recorder
        if recorder is not None:
            recorder.point("attn.output", recorder.attention_output[0, 0])
        result = super().output_rounded(values, quantize)
        if recorder is not None:
            recorder.point("attn.output_rounded", recorder.attention_output[0, 0])
        return result


class RouterWatch:
    """The gate's own linear output, caught by a wrapper around model.py's linear() while a gate runs, and the
    scores the gate forms from it (model.py:811-827) formed again, checked against the gate's ids and weights."""

    def __init__(self, session, on_gate, broken):
        self.reference_model, self.on_gate, self.broken = session.reference.model, on_gate, broken
        self.gates = {block.ffn.gate: block.layer_id for block in session.model.layers}
        self.open, self.logits, self.calls, self.differing, self.handles, self.inner = None, None, 0, 0, [], None

    def linear(self, inputs, weight, bias = None):                         # as model.py:181 is called
        output = self.inner(inputs, weight, bias)
        if self.open is not None and weight.dtype == torch.float32 and weight.shape == self.open.weight.shape:
            self.logits = output
        return output

    def begins(self, gate, inputs):
        self.open, self.logits = gate, None

    def ends(self, gate, inputs, output):
        weights, indices = output
        self.open = None
        if self.logits is None or gate.score_func != "sqrtsoftplus":
            raise RuntimeError("a gate ran no linear this tool saw, or scores by another function than sqrtsoftplus")
        scores = torch.nn.functional.softplus(self.logits / gate.gate_temp).sqrt()
        biased = scores + gate.bias
        ids = biased.topk(gate.topk, dim = -1)[1]
        again = scores.gather(1, ids)
        if gate.norm_topk_prob and gate.topk > 1:
            again /= again.sum(dim = -1, keepdim = True) + ROUTE_WEIGHT_EPSILON
        again *= gate.route_scale
        held = oracle.same_bits(ids, indices.roll(1, dims = -1) if self.broken else indices) \
            and oracle.same_bits(again, weights)
        self.calls, self.differing = self.calls + 1, self.differing + (not held)
        self.on_gate(self.gates[gate], self.logits, scores, biased, indices, weights)

    def install(self):
        self.inner, self.reference_model.linear = self.reference_model.linear, self.linear
        for gate in self.gates:
            self.handles += [gate.register_forward_pre_hook(self.begins), gate.register_forward_hook(self.ends)]

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles, self.reference_model.linear = [], self.inner


class StepRecorder:
    """What a stepwise run records (stepwise_capture.py's module text): hooks on the reference's own modules and
    wrappers around its sparse_attn and hc_split_sinkhorn, each calling what it wraps."""

    def __init__(self, session, writer, switch, steps, capture_positions, snapshot_positions, broken):
        model, sizes = session.model, session.model_arguments
        self.session, self.model, self.writer, self.switch = session, model, writer, switch
        self.reference_model, self.shared = session.reference.model, session.reference.model.shared_attn
        self.capture_positions, self.snapshot_positions = set(capture_positions), set(snapshot_positions)
        self.position, self.layer, self.capturing, self.handles, self.wrapped = None, None, False, [], {}
        self.mixes, self.attention_output, self.weights_projection, self.pairs, self.shared_output = (None,) * 5
        layers, hc, dim, experts = len(model.layers), sizes.hc_mult, sizes.dim, sizes.n_routed_experts
        self.streams = torch.empty((layers, steps, hc, dim), dtype = torch.bfloat16)
        self.carried = torch.empty((layers, steps, hc), dtype = torch.float32)
        self.embeddings = torch.empty((steps, dim), dtype = torch.bfloat16)
        self.ids = torch.empty((layers, steps, sizes.n_activated_experts), dtype = torch.int64)
        self.weights = torch.empty((layers, steps, sizes.n_activated_experts), dtype = torch.float32)
        self.biased = torch.empty((layers, steps, experts), dtype = torch.float32)
        layout = None if model.engram_hash is None else model.engram_hash.layout      # a model without n-grams
        self.engram_layers = () if layout is None else tuple(layout.layer_ids)
        columns = 0 if layout is None else (layout.max_ngram_size - 1) * layout.n_heads
        head_dim = 0 if layout is None else layout.head_dim
        self.row_ids = torch.empty((steps, len(self.engram_layers), columns), dtype = torch.int64)
        self.engram_rows = torch.empty((len(self.engram_layers), steps, columns, head_dim), dtype = torch.bfloat16)
        self.engram_output = torch.empty((len(self.engram_layers), steps, hc, dim), dtype = torch.bfloat16)
        self.ran = torch.zeros((layers, steps), dtype = torch.int64)
        self.points = collections.defaultdict(set)
        self.counts = collections.Counter()
        self.router = RouterWatch(session, self.gate_ran, broken)

    def active(self):
        return self.capturing and self.layer in CAPTURE_LAYERS

    def point(self, record, tensor, **facts):
        if self.active():
            self.writer.add(f"layer_{self.layer}.position_{self.position}.{record}", tensor, self.layer,
                            gates.POINT_RECORDS[record], position = self.position, **facts)
            self.points[(self.layer, self.position)].add(record)

    # ---- the hooks ----
    def block_begins(self, block, inputs):
        self.layer = block.layer_id
        self.point("attn_site.carried_pre", inputs[BLOCK_PRE_MIX_ARGUMENT][0, 0])

    def block_ends(self, block, inputs, output):
        layer, position = block.layer_id, self.position
        self.streams[layer, position].copy_(output[0][0, 0])
        self.carried[layer, position].copy_(output[1][0, 0])
        self.ran[layer, position] += 1
        self.layer = None

    def hc_mixes(self, block, x, hc_fn, hc_scale, hc_base):
        pre, post, comb = type(block).hc_mixes(block, x, hc_fn, hc_scale, hc_base)     # model.py:948-955
        site = "attn_site" if hc_fn is block.hc_attn_fn else "ffn_site"
        for part, tensor in (("streams", x), ("mixes", self.mixes), ("pre", pre), ("post", post), ("comb", comb)):
            self.point(f"{site}.{part}", tensor[0, 0])
        return pre, post, comb

    def sinkhorn(self, mixes, *rest, **named):
        self.mixes = mixes
        return self.wrapped["hc_split_sinkhorn"](mixes, *rest, **named)

    def attention(self, q, kv, attn_sink, topk_idxs, softmax_scale):
        output = self.wrapped["sparse_attn"](q, kv, attn_sink, topk_idxs, softmax_scale)
        self.point("attn.q", q[0, 0])
        self.point("attn.sink", attn_sink)
        self.point("attn.topk_idxs", topk_idxs[0, 0],                   # the prompt branch's window at 0 is 1 row
                   window_rows = self.model.layers[self.layer].attn.window_size if self.position else 1)
        self.point("attn.output_rotated", output[0, 0])
        self.attention_output = output                                     # rotated back in place after the call
        return output

    def wo_b_begins(self, module, inputs):
        self.point("attn.wo_a", inputs[0][0, 0])
        if "attn.output" not in self.points.get((self.layer, self.position), ()):     # else at its rounding
            self.point("attn.output", self.attention_output[0, 0])

    def selection(self, indexer, inputs, output):
        """The scores formed again as model.py:554-580 forms them, from the queries, the slot and the weights this
        step left, and checked: their picks must be the indexer's own."""
        if not self.active():
            return
        x, _, _, start_pos, offset = inputs
        rows, ratio, end_pos = x.size(1), indexer.compress_ratio, start_pos + x.size(1)
        weights = self.weights_projection * (indexer.softmax_scale * indexer.n_heads ** -0.5)
        scores = torch.einsum("bshd,btd->bsht", self.switch.query, self.shared.index_k[:x.size(0), :end_pos // ratio])
        scores = (scores.relu_() * weights.unsqueeze(-1)).sum(dim = 2)
        visible = end_pos // ratio
        if start_pos == 0:
            visible = (torch.arange(1, rows + 1, device = x.device) // ratio).unsqueeze(-1)
            scores.masked_fill_(torch.arange(rows // ratio, device = x.device) >= visible, -torch.inf)
        if indexer.uses_candidates:
            scores = scores.masked_fill(~self.shared.candidates, -torch.inf)
        picks = scores.topk(min(indexer.index_topk, end_pos // ratio), dim = -1, sorted = False).indices
        picks = picks.sort(dim = -1).values
        again = torch.where(picks < visible, picks + offset, -1).int()
        self.counts["selection"] += 1
        self.counts["selection_differing"] += not oracle.same_bits(again, output)
        self.point("indexer.weights", weights[0, 0])
        self.point("indexer.scores", scores[0, 0])
        self.point("indexer.selection", torch.where(output >= 0, output - offset, -1)[0, 0], window_rows = offset)

    def gate_ran(self, layer, logits, scores, biased, indices, weights):
        if self.position is None:                                          # not a step of this recorder's
            return
        self.ids[layer, self.position].copy_(indices[0])
        self.weights[layer, self.position].copy_(weights[0])
        self.biased[layer, self.position].copy_(biased[0])
        for part, tensor in (("logits", logits), ("scores", scores), ("biased_scores", biased),
                             ("expert_ids", indices), ("route_weights", weights)):
            self.point(f"router.{part}", tensor[0])

    def expert_ran(self, stage_expert, inputs, output):
        if self.active():
            name = EXPERT_NAME.fullmatch(self.stage_of[stage_expert].occupant_name)
            if name is None or int(name["layer"]) != self.layer:
                raise RuntimeError("a routed expert ran that is not of the layer running")
            self.pairs.append((int(name["expert"]), output))

    def experts_end(self, moe, inputs, output):
        if not self.active():
            return
        experts = [expert for expert, _ in self.pairs]
        row_sum = torch.zeros_like(output[0, 0], dtype = torch.float32)
        for _, pair_output in self.pairs:                                  # model.py:895-900: ascending, from +0
            row_sum += pair_output[0]
        combined = (row_sum + self.shared_output[0]).to(output.dtype)      # model.py:903-904
        self.counts["moe"] += 1
        self.counts["moe_differing"] += not (experts == sorted(experts) and oracle.same_bits(combined, output[0, 0]))
        self.point("experts.pair_outputs", torch.stack([pair_output[0] for _, pair_output in self.pairs]),
                   experts = experts)
        self.point("experts.row_sum", row_sum)
        self.point("shared_expert.output", self.shared_output[0])
        self.point("ffn.output", output[0, 0])

    def recorded(self, record, shape = None, inputs = False):
        """A hook that points a module's output (its first input) at `record` and returns nothing."""
        def record_it(_, arguments, output = None):
            tensor = (arguments[0] if inputs else output)[0, 0]
            self.point(record, tensor if shape is None else tensor.view(shape))
        return record_it

    def keeper(self, buffer, table = None):
        """A forward hook that copies the step's row of an output into buffer[position] (buffer[table, position])
        and returns nothing: a hook that returns a value replaces the module's output."""
        def keep(_, inputs, output):
            (buffer[self.position] if table is None else buffer[table, self.position]).copy_(output[0, 0])
        return keep

    def compressor_ends(self, compressor, inputs, output):
        self.point("compressor.kv_state", compressor.kv_state[0])
        self.point("compressor.score_state", compressor.score_state[0])

    def install(self):
        """Inside the run's conditions, so that sparse_attn wraps what they put in place."""
        model, reference_model, hook = self.model, self.reference_model, self.handles.append
        for name, wrapper in (("sparse_attn", self.attention), ("hc_split_sinkhorn", self.sinkhorn)):
            self.wrapped[name] = getattr(reference_model, name)
            setattr(reference_model, name, wrapper)
        self.router.install()
        self.stage_of = {stage.expert: stage for stage in self.session.stages}
        for stage_expert in self.stage_of:
            hook(stage_expert.register_forward_hook(self.expert_ran))
        hook(model.embed.register_forward_hook(self.keeper(self.embeddings)))
        if model.engram_hash is not None:
            hook(model.engram_hash.register_forward_hook(self.keeper(self.row_ids)))
        hook(model.norm.register_forward_pre_hook(lambda _, inputs: self.final("collapsed", inputs[0])))
        hook(model.norm.register_forward_hook(lambda _, inputs, output: self.final("normed", output)))
        for block in model.layers:
            self.install_block(block, hook)

    def install_block(self, block, hook):
        attention = block.attn
        block.hc_mixes = lambda *arguments, block = block: self.hc_mixes(block, *arguments)
        hook(block.register_forward_pre_hook(self.block_begins))
        hook(block.register_forward_hook(self.block_ends))
        for site, norm in (("attn_site", block.attn_norm), ("ffn_site", block.ffn_norm)):
            hook(norm.register_forward_pre_hook(self.recorded(f"{site}.collapsed", inputs = True)))
            hook(norm.register_forward_hook(self.recorded(f"{site}.normed")))
        heads = (attention.n_local_heads, attention.head_dim)
        outputs = [(attention.wq_a, "attn.wq_a", None), (attention.q_norm, "attn.qr", None),
                   (attention.wq_b, "attn.wq_b", heads), (attention.wkv, "attn.wkv", None),
                   (attention.kv_norm, "attn.kv_normed", None), (attention.wo_b, "attn.wo_b", None)]
        hook(attention.wo_b.register_forward_pre_hook(self.wo_b_begins))
        hook(block.ffn.shared_experts.register_forward_hook(lambda _, inputs, output: setattr(  # kept for the sum
            self, "shared_output", output)))
        if attention.compressor is not None:
            compressor = attention.compressor
            outputs += [(compressor.wkv, "compressor.wkv", None), (compressor.norm, "compressor.latent", None)]
            if compressor.compress_ratio > 1:
                outputs.append((compressor.wgate, "compressor.wgate", None))
                hook(compressor.register_forward_hook(self.compressor_ends))
        if attention.indexer is not None:
            indexer = attention.indexer
            outputs += [(indexer.wq_b, "indexer.wq_b", (indexer.n_local_heads, indexer.index_head_dim)),
                        (indexer.weights_proj, "indexer.weights_projection", None)]
            hook(indexer.weights_proj.register_forward_hook(lambda _, inputs, output: setattr(
                self, "weights_projection", output)))
            hook(indexer.register_forward_hook(self.selection))
            if indexer.owns_k:
                outputs += [(indexer.wk, "indexer.wk", None), (indexer.k_norm, "indexer.k_normed", None)]
                hook(indexer.wk.register_forward_pre_hook(self.recorded("indexer.latent_bfloat16", inputs = True)))
        for module, record, shape in outputs:
            hook(module.register_forward_hook(self.recorded(record, shape)))
        hook(block.ffn.register_forward_pre_hook(lambda _, inputs: setattr(self, "pairs", [])))
        hook(block.ffn.register_forward_hook(self.experts_end))
        if block.engram is not None:
            self.install_engram(block.engram, self.engram_layers.index(block.layer_id), hook)

    def install_engram(self, engram, table, hook):
        def begins(_, inputs):
            self.layer = engram.layer_id                                   # it runs before its block (model.py:1263)
        hook(engram.register_forward_pre_hook(begins))
        hook(engram.wkv.register_forward_hook(self.recorded("ngram.projection")))
        hook(engram.embed.register_forward_hook(self.keeper(self.engram_rows, table)))
        hook(engram.register_forward_hook(self.keeper(self.engram_output, table)))

    def final(self, part, tensor):
        if self.capturing:
            self.writer.add(f"position_{self.position}.final.{part}", tensor[0, 0], None,
                            f"the final collapse{'' if part == 'collapsed' else ', normed'}: the head's input",
                            position = self.position)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.router.remove()
        for name, wrapped in self.wrapped.items():
            setattr(self.reference_model, name, wrapped)
        for block in self.model.layers:
            if "hc_mixes" in vars(block):
                delattr(block, "hc_mixes")
        self.handles, self.wrapped = [], {}

    # ---- steps ----
    def begin_step(self, position):
        self.position, self.capturing = position, position in self.capture_positions

    def snapshot(self, position):
        """The state a step leaves, as stepwise_capture.py's module text lists it."""
        self.capturing = False
        for layer in CAPTURE_LAYERS:
            attention = self.model.layers[layer].attn
            records = [("ring", attention.window_kv_cache[0])]
            if attention.is_kv_source:
                entries, compressor = (position + 1) // attention.compress_ratio, attention.compressor
                records += [("pool", attention.compress_kv_cache[0, :entries]),
                            ("keys", attention.indexer.k_cache[0, :entries])]
                if compressor.compress_ratio > 1:
                    records += [("kv_state", compressor.kv_state[0]), ("score_state", compressor.score_state[0])]
            for part, tensor in records:
                self.writer.add(f"layer_{layer}.position_{position}.snapshot.{part}", tensor, layer,
                                gates.POINT_RECORDS[f"snapshot.{part}"], position = position)
