#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The weights of a GGUF quantization of DeepSeek-V4.1-Flash, put into DeepSeek's own reference: what stands behind
reference_oracle.py --weights-gguf.

The question it serves: on the same token sequence, how far are the logits of the model with a re-quantized file's
weights from the logits with the checkpoint's own? The file this was written for (GSQ-RCO) holds the routed experts
as Q2_K or Q3_K, the other matrices, the embeddings, the output head and the two n-gram tables as Q8_0 and small
tensors as F32. reference_oracle.py runs the reference on the checkpoint; with
    reference_oracle.py ... --mode forward --all-logits --weights-gguf FILE --gguf-linear-inputs unquantized
    reference_oracle.py ... --mode forward --all-logits --weights-gguf FILE --gguf-linear-inputs as-the-reference
    reference_oracle.py ... --mode forward --all-logits --weights-gguf FILE --gguf-linear-inputs int8-per-256
the same reference code runs with every weight taken from FILE, and compare_logit_captures.py compares the captures.

How a pass takes its weights. install() runs once, after the oracle has loaded and checked the checkpoint:
  * a float parameter the reference keeps on the card is overwritten with the file's values, cast to the
    parameter's dtype. The cast loses nothing where the parameter is float32 (the head, the compressors that pool,
    the hyper-connection, sink and routing-bias tensors: a float16 scale times a signed byte has at most 18
    significant bits) and nothing where an F32 tensor holds the checkpoint's bfloat16 values widened (the norms,
    the gate, the n-gram key and query weights). It rounds to 8 significant bits, at most 2 ** -8 of a value,
    where a Q8_0 tensor fills a bfloat16 parameter: the embeddings, the attention output projection wo_a, the
    indexer's weights_proj and wk, the compressor that does not pool. The change is measured and reported by kind;
  * every QUANTIZED linear of the reference (model.py:181-205: the 8-bit ones of the trunk and the 4-bit routed
    experts) is answered by torch.nn.functional.linear with the file's matrix in float32, cast to the default
    dtype as the reference's products return theirs (kernel.py:299, :583). With "unquantized" the input is taken
    as it is, in float32: float32 products without any rounding of activations, which is what the file's author
    evaluated. With "as-the-reference" the input is first replaced by what the reference's product multiplies:
    act_quant's 8-bit floats times their power-of-two scales (reference_quantization.e4m3_with_scales). With
    "int8-per-256" and "int8-per-32" it is replaced by 8-bit integers times one float32 scale per 256 or per 32
    inputs of a row, by the rule llama.cpp quantizes activations with (int8_with_scales below): the values a
    kernel with 8-bit integer inputs would multiply, in this same float32 product. The checkpoint's bytes stay
    in those parameters and are never read again;
  * the 8-bit matrices of the trunk are decoded once and kept on the card in float32 (5.5 billion weights, about
    20.5 GiB). A routed expert's three matrices are decoded one at a time when the reference's Expert.forward
    reaches each linear, and dropped after the product: 11.6 MB (Q2_K) or 15.2 MB (Q3_K) read per expert, nothing
    cached. The oracle's move of the checkpoint's 4-bit bytes onto its stage is replaced by a note of which
    expert is being called;
  * an n-gram lookup gathers its rows from the file's table, decodes them on the card and returns them in
    bfloat16, as the reference's lookup does (model.py:320): the same rounding to 8 significant bits.
Not everything has to come from the file. install() takes the parts wanted, among the matrices of the trunk's
quantized linears, the float parameters, the tables and the routed experts, and for the experts the layers wanted;
whatever is not taken keeps the checkpoint's own values, multiplied through the SAME stand-in (an expert's 4-bit
codes times their scales, decoded to float32 when it is called). With nothing taken the capture is the control:
it differs from a capture of the checkpoint only by the stand-in, and a capture that takes the file differs from
it only by the values. With a part taken, the capture shows what that part of the file alone costs.
Unchanged in every variant: the reference's in-place activation quantizations (the attention's keys by act_quant,
the indexer's and the compressed keys by fp4_act_quant), its sparse attention, hyper-connections and n-gram
hashing, and the bfloat16 of the residual streams. A capture made this way differs from a capture of the
checkpoint by the weights, by the order in which a quantized linear's float32 product sums its terms (cuBLAS here,
the reference's kernel there: the stand-in's check below measures what it is worth) and, with "unquantized", by the
rounding of the inputs of the quantized linears that is left out, or with the 8-bit integer inputs by the other
rounding that takes its place; by nothing else.

The map (ROOT_MAP, LAYER_MAP, EXPERT_MAP and TABLE_ROW below). Each row names a parameter of model.py, its tensor
and the lines that pair the two in the llama.cpp tree that converts this model: conversion/deepseek.py (class
DeepseekV41Model and its base) and gguf-py/gguf/constants.py. No tensor is transposed, reshaped or split: the
converter hands each one on as it is (deepseek.py:892, base.py:1012) and the writer stores the dimensions
innermost first (gguf_writer.py:268), so a tensor read with its dimensions reversed has the parameter's shape,
[outputs, inputs]. The block scales of the 8-bit weights are multiplied in by the converter (deepseek.py:1253-1258,
blocks of 32 by 32; for wo_a the reference's own convert.py:157-173 does the same) and the row scales of the tables
likewise (deepseek.py:1438-1441): the .scale of a quantized weight has no tensor. A layer's routed experts are one
stacked tensor per matrix, expert outermost (deepseek.py:736-737), named gate for w1, down for w2 and up for w3
(deepseek.py:783-785). The file was not written by that tree as it stands (there the experts stay 4-bit and the
token map is metadata), so those lines are the reason for a row and the checks below are its proof.

Checked at start, and the tool stops with the name:
  * every tensor's bytes lie inside the file and overlap no other tensor's; every type is one decoded here;
  * the decoders on the card against the numpy decoders (Q2_K and Q3_K: gguf_expert_weight_error.py's, confirmed
    on the real file; Q8_0: decode_q8_0 below), bit for bit, on blocks from the start, the middle and the end of
    the first and of the last tensor of each type;
  * every parameter of the reference model has a row or is one a text forward leaves unused
    (PARAMETERS_A_TEXT_FORWARD_DOES_NOT_USE: which, and why); every tensor of the file is taken by a row or is in
    TENSORS_A_TEXT_FORWARD_DOES_NOT_USE; the shape of every row; a row's parameter is a quantized one exactly
    where the map says so; a layer's stacked tensors hold exactly the experts the model routes among;
  * against the checkpoint's own values (a float parameter as the oracle loaded it; an 8-bit weight times its
    block scales; an expert's 4-bit codes times their scales; table rows as the oracle's lookup returns them): the
    cosine and the ratio of the norms of EVERY trunk tensor, of the three matrices of EXPERTS_CHECKED_PER_LAYER
    experts of every layer and of TABLE_ROWS_CHECKED rows of each table, against AGREEMENT_LIMITS;
  * the stand-in's arithmetic against the reference's kernels, both on the CHECKPOINT's weights and before
    anything is replaced: one 8-bit linear and one whole routed expert on random rows, inputs quantized as the
    reference. This is what shows that "as-the-reference" is the reference's product, and that the checkpoint's
    8-bit and 4-bit values, which the cosines are taken against, are decoded here as the reference reads them;
  * with 8-bit integer inputs: the quantizer on the card against the loop of its definition on the host, codes
    and scales bit for bit, on seeded rows of normal values and four blocks made for it: zeros; products that
    fall half way between two whole numbers under a largest value that is positive, and one that is negative;
    the largest magnitude under both signs. The two are one reading of the rule written twice, so the card's
    codes and scales are also held to what any reading must give: whole numbers up to 127 in magnitude, 127
    reached in every block that is not zeros, and a code times its scale within half a step of its value.
At every call with 8-bit integer inputs: a row that is not whole blocks, or a block without finite codes (a value
that is not finite, or a largest magnitude so small that 127 over it is no float32), stops the tool.
A draft-head or vision module that runs stops the tool: its weights are not in the file.

Why these limits. A tensor paired with another, read transposed or from a shifted offset gives a cosine near 0
(two unrelated matrices of millions of weights); a scale left out or applied twice is a power of two, so it moves
the norm by a factor of 2 at least. F32 tensors are copies, and two different norm weights are both close to a
vector of ones, which 0.9999 would not tell apart: cosine 0.999999, norms within 0.1%. Q8_0 rounds a weight to a
step of 1/127 of the largest of its block of 32, an error of under 1% of a typical block's norm and a cosine of
0.9999 or more: the limit is 0.999 with norms within 2%. Q2_K and Q3_K showed 0.96 to 0.99 on the experts
gguf_expert_weight_error.py sampled from the real file. Four levels to a weight reach about 0.94 on bell-shaped
weights and eight about 0.98 (the checkpoint's 4-bit codes are easier), and a fit to layer outputs, which is what
made the file, may give some of that away on an expert it saw little of: the limits are 0.80 for Q2_K and 0.90
for Q3_K, with norms within 0.6 to 1.6, which no power of two is. The stand-in against the kernels: the two sum
the same float32 terms in another order and round to bfloat16, which differs in the last bit of a value here and
there. Measured on the checkpoint (batch_f05 to batch_f07, the same seeded rows every run): 3.2e-9 of one linear's
output and 4.5e-7 of an expert's. The limits are 1e-6 and 1e-4, some hundred times that; a wrong block size,
half-byte order or scale is a difference of the order of 1.

What it does not show.
  * Not the file under its author's runtime (a float32 graph with float32 key caches): here the streams between
    sublayers are bfloat16 and the keys are quantized as the reference quantizes them, in every variant.
  * Not the model's quality: a weight cosine is a check of the map, and the logits are for the comparison tool.
  * Not the arithmetic of a kernel with 8-bit integer inputs: its sums of integer products are exact in any
    order, the float32 sums of code times scale here are not. Such a capture prices the rounding of the inputs
    and no order of additions.
  * Nothing about the draft head or the vision tower (other files), nor about a prompt with images.
  * The hash multipliers, primes and offsets in the file's metadata are not read: a pass hashes with the
    reference's own. The file's token map IS compared with the reference's and reported; it does not stop the tool.
  * The experts not sampled are covered by the shape and extent checks and by the stacking shown on the samples.
Measurement tool: not part of the engine.
"""
import collections
import contextlib
import functools
import json
import math
import os
import re
import sys
import time

import numpy
import torch

import reference_quantization

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "measure"))
from gguf_expert_weight_error import (BLOCK_WEIGHTS, E2M1_MAGNITUDES, GGUF_TYPE_Q2_K, GGUF_TYPE_Q3_K, GROUP_WEIGHTS,
                                      HALF_CODE_BYTES, HALF_WEIGHTS, Q2_K_BLOCK_BYTES, Q3_K_BLOCK_BYTES,
                                      WEIGHTS_PER_SCALE, decode_q2_k, decode_q3_k, read_gguf_tensors)

# ---- The file: types, and the layout of a block of each --------------------------------------------------------------
GGUF_BYTE_ORDER = "little"
GGUF_TYPE_F32, GGUF_TYPE_Q8_0, GGUF_TYPE_I32 = 0, 8, 26
PLAIN_VALUE_BYTES = 4                        # F32 and I32: one little-endian value of four bytes
Q8_0_BLOCK_WEIGHTS, Q8_0_BLOCK_BYTES = 32, 34
Q8_0_D, Q8_0_VALUES = slice(0, 2), slice(2, 34)                # a float16 scale, then 32 signed bytes
Q2_K_SCALES, Q2_K_CODES, Q2_K_D, Q2_K_DMIN = slice(0, 16), slice(16, 80), slice(80, 82), slice(82, 84)
Q3_K_HIGH_BITS, Q3_K_CODES, Q3_K_SCALES, Q3_K_D = slice(0, 32), slice(32, 96), slice(96, 108), slice(108, 110)
BLOCK_GROUPS = BLOCK_WEIGHTS // GROUP_WEIGHTS                  # a Q2_K or Q3_K block: 16 groups of 16 weights ...
BLOCK_HALVES = BLOCK_WEIGHTS // HALF_WEIGHTS                   # ... in two halves, each with 32 bytes of codes:
HALF_RUNS = 2                                                  # 16 bytes for a half's even groups, 16 for its odd
RUN_BYTES = HALF_CODE_BYTES // HALF_RUNS
RUN_GROUPS = HALF_WEIGHTS // (HALF_RUNS * GROUP_WEIGHTS)       # four groups to a run: two bits of every byte each
PAIR_BITS, PAIR_MASK = 2, 3                  # a field of two bits: a code, or the two high bits of a Q3_K scale
NIBBLE_BITS, NIBBLE_MASK = 4, 15
BITS_PER_BYTE = 8
Q3_K_HIGH_BIT_AXES = (0, 3, 4, 1, 2)         # [block, run, weight, half, group] to [block, half, group, run, weight]
Q3_K_CODE_OFFSET = 4                         # a code whose high bit is clear lies 4 lower
Q3_K_SCALE_OFFSET = 32                       # a 6-bit scale is stored 32 higher
Q3_K_WORD_BYTES = 4                          # the 12 scale bytes are three words of four bytes:
Q3_K_LOW_WORDS = 2                           # two words with the low four bits of the 16 scales, a half byte each,
Q3_K_HIGH_PAIRS = slice(8, 12)               # and one with their two high bits
Q3_K_LOW_HALF_SCALES = 8                     # scales 0..7 lie in the low halves of those bytes, 8..15 in the high
TensorType = collections.namedtuple("TensorType", ("name", "block_weights", "block_bytes"))
TENSOR_TYPES = {GGUF_TYPE_F32: TensorType("F32", 1, PLAIN_VALUE_BYTES),
                GGUF_TYPE_I32: TensorType("I32", 1, PLAIN_VALUE_BYTES),
                GGUF_TYPE_Q8_0: TensorType("Q8_0", Q8_0_BLOCK_WEIGHTS, Q8_0_BLOCK_BYTES),
                GGUF_TYPE_Q2_K: TensorType("Q2_K", BLOCK_WEIGHTS, Q2_K_BLOCK_BYTES),
                GGUF_TYPE_Q3_K: TensorType("Q3_K", BLOCK_WEIGHTS, Q3_K_BLOCK_BYTES)}
GgufTensor = collections.namedtuple("GgufTensor", ("name", "shape", "type_code", "byte_offset", "byte_count"))

# ---- The reference ---------------------------------------------------------------------------------------------------
QUANTIZED_WEIGHT_DTYPES = (torch.float4_e2m1fn_x2, torch.float8_e4m3fn)    # the two branches of model.py:186-205
FP4_VALUES_PER_BYTE = 2                      # model.py:221: [outputs, inputs / 2] bytes; the low half first
MATRIX_DIMENSIONS = 2                        # what a quantized linear multiplies by: [outputs, inputs]
TABLE_VALUE_DTYPE = torch.bfloat16           # model.py:320: what the reference's n-gram lookup returns
INPUTS_UNQUANTIZED, INPUTS_AS_THE_REFERENCE = "unquantized", "as-the-reference"
INT8_INPUT_BLOCKS = {"int8-per-256": 256, "int8-per-32": 32}   # 8-bit integer inputs: the inputs of a row per scale
LINEAR_INPUTS = (INPUTS_UNQUANTIZED, INPUTS_AS_THE_REFERENCE, *INT8_INPUT_BLOCKS)
LARGEST_INT8_CODE = 127                      # ggml-quants.c:2791: the value of largest magnitude becomes -127
SCALE_SUFFIX, WEIGHT_SUFFIX = ".scale", ".weight"
LAYER_NAME = re.compile(r"layers\.(?P<layer>\d+)\.(?P<rest>.+)")
ROUTED_EXPERT_NAME = re.compile(r"layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)")
TABLE_NAME = re.compile(r"layers\.(?P<layer>\d+)\.engram\.embed")
TOKEN_MAP_TENSOR = "engram_token_map.weight"
PARTS_NOT_IN_THE_FILE = ("mtp", "vision", "aligner")           # attributes of the reference's Transformer

# ---- The map: reference parameter -> GGUF tensor, how a pass uses it, the lines that pair the two --------------------
USE_PARAMETER = "a float parameter on the card, overwritten"
USE_LINEAR = "the matrix of a quantized linear, kept in float32"
USE_EXPERT = "one routed expert's matrix, decoded when it is called"
USE_TABLE = "rows of an n-gram table, decoded when they are looked up"
MapRow = collections.namedtuple("MapRow", ("parameter", "tensor", "use", "evidence"))
ROOT_MAP = (
    MapRow("embed.weight", "token_embd.weight", USE_PARAMETER, "deepseek.py:811 constants.py:1420"),
    MapRow("norm.weight", "output_norm.weight", USE_PARAMETER, "deepseek.py:812 constants.py:1426"),
    MapRow("head.weight", "output.weight", USE_PARAMETER, "deepseek.py:813 constants.py:1427"),
)
LAYER_MAP = (                                # "layers.N." + parameter -> "blk.N." + tensor
    MapRow("hc_attn_fn", "hc_attn_fn.weight", USE_PARAMETER, "deepseek.py:830 constants.py:1575"),
    MapRow("hc_attn_base", "hc_attn_base.weight", USE_PARAMETER, "deepseek.py:831 constants.py:1576"),
    MapRow("hc_attn_scale", "hc_attn_scale.weight", USE_PARAMETER, "deepseek.py:832 constants.py:1577"),
    MapRow("hc_ffn_fn", "hc_ffn_fn.weight", USE_PARAMETER, "deepseek.py:833 constants.py:1578"),
    MapRow("hc_ffn_base", "hc_ffn_base.weight", USE_PARAMETER, "deepseek.py:834 constants.py:1579"),
    MapRow("hc_ffn_scale", "hc_ffn_scale.weight", USE_PARAMETER, "deepseek.py:835 constants.py:1580"),
    MapRow("attn_norm.weight", "attn_norm.weight", USE_PARAMETER, "deepseek.py:854 constants.py:1439"),
    MapRow("ffn_norm.weight", "ffn_norm.weight", USE_PARAMETER, "deepseek.py:855 constants.py:1455"),
    MapRow("attn.attn_sink", "attn_sinks.weight", USE_PARAMETER, "deepseek.py:836 constants.py:1447"),
    MapRow("attn.wq_a.weight", "attn_q_a.weight", USE_LINEAR, "deepseek.py:837 constants.py:1563"),
    MapRow("attn.q_norm.weight", "attn_q_a_norm.weight", USE_PARAMETER, "deepseek.py:839 constants.py:1569"),
    MapRow("attn.wq_b.weight", "attn_q_b.weight", USE_LINEAR, "deepseek.py:838 constants.py:1564"),
    MapRow("attn.wkv.weight", "attn_kv.weight", USE_LINEAR, "deepseek.py:840 constants.py:1571"),
    MapRow("attn.kv_norm.weight", "attn_kv_a_norm.weight", USE_PARAMETER, "deepseek.py:841 constants.py:1572"),
    MapRow("attn.wo_a.weight", "attn_output_a.weight", USE_PARAMETER,          # bfloat16 in the reference: convert.py
           "deepseek.py:842 constants.py:1573 convert.py:157-173"),           # multiplies its block scales in as well
    MapRow("attn.wo_b.weight", "attn_output_b.weight", USE_LINEAR, "deepseek.py:843 constants.py:1574"),
    MapRow("attn.compressor.wkv.weight", "attn_compressor_kv.weight", USE_PARAMETER,
           "deepseek.py:845 constants.py:1595"),
    MapRow("attn.compressor.wgate.weight", "attn_compressor_gate.weight", USE_PARAMETER,
           "deepseek.py:846 constants.py:1596"),
    MapRow("attn.compressor.norm.weight", "attn_compressor_norm.weight", USE_PARAMETER,
           "deepseek.py:847 constants.py:1598"),
    MapRow("attn.indexer.wq_b.weight", "indexer.attn_q_b.weight", USE_LINEAR, "deepseek.py:848 constants.py:1663"),
    MapRow("attn.indexer.weights_proj.weight", "indexer.proj.weight", USE_PARAMETER,
           "deepseek.py:849 constants.py:1661"),
    MapRow("attn.indexer.wk.weight", "indexer.attn_k.weight", USE_PARAMETER, "deepseek.py:1479 constants.py:1662"),
    MapRow("attn.indexer.k_norm.weight", "indexer.k_norm.weight", USE_PARAMETER, "deepseek.py:1478 constants.py:1656"),
    MapRow("ffn.gate.weight", "ffn_gate_inp.weight", USE_PARAMETER, "deepseek.py:856 constants.py:1453"),
    MapRow("ffn.gate.bias", "exp_probs_b.bias", USE_PARAMETER, "deepseek.py:857 constants.py:1476"),
    MapRow("ffn.gate.bias_vl", "exp_probs_b_vl.bias", USE_PARAMETER, "deepseek.py:858 constants.py:1477"),
    MapRow("ffn.shared_experts.w1.weight", "ffn_gate_shexp.weight", USE_LINEAR, "deepseek.py:860 constants.py:1464"),
    MapRow("ffn.shared_experts.w2.weight", "ffn_down_shexp.weight", USE_LINEAR, "deepseek.py:861 constants.py:1465"),
    MapRow("ffn.shared_experts.w3.weight", "ffn_up_shexp.weight", USE_LINEAR, "deepseek.py:862 constants.py:1466"),
    MapRow("engram.wkv.weight", "engram_wkv.weight", USE_LINEAR, "deepseek.py:1477 constants.py:1660"),
    MapRow("engram.q_weight", "engram_q.weight", USE_PARAMETER, "deepseek.py:1476 constants.py:1659"),
    MapRow("engram.k_weight", "engram_k.weight", USE_PARAMETER, "deepseek.py:1475 constants.py:1658"),
)
EXPERT_MAP = (                               # "layers.N.ffn.experts.E." + parameter -> entry E of "blk.N." + tensor
    MapRow("w1.weight", "ffn_gate_exps.weight", USE_EXPERT, "deepseek.py:783 and :736-737 constants.py:1472"),
    MapRow("w2.weight", "ffn_down_exps.weight", USE_EXPERT, "deepseek.py:784 and :736-737 constants.py:1473"),
    MapRow("w3.weight", "ffn_up_exps.weight", USE_EXPERT, "deepseek.py:785 and :736-737 constants.py:1474"),
)
TABLE_ROW = MapRow("engram.embed.weight", "engram_embd.weight", USE_TABLE,     # "layers.N." and "blk.N." as above
                   "deepseek.py:1474 and :1438-1441 constants.py:1657")
ROOT_ROWS = {row.parameter: row for row in ROOT_MAP}
LAYER_ROWS = {row.parameter: row for row in LAYER_MAP}
PARAMETERS_A_TEXT_FORWARD_DOES_NOT_USE = (   # a name's pattern, and why no tensor of the file is needed for it
    (re.compile(r"mtp\..+"), "the draft head: Transformer.forward (model.py:1242-1272) never calls it, and its "
                             "weights are a file of their own"),
    (re.compile(r"(vision|aligner)\..+|image_(start|end|newline)"),
     "the vision tower, its aligner and the image delimiters: reached only with images (model.py:1254-1256), and "
     "a file of their own"),
)
SCALES_ARE_MULTIPLIED_IN = "the block scales of a quantized weight: the converter multiplied them into the values " \
                           "(deepseek.py:1253-1258)"
TENSORS_A_TEXT_FORWARD_DOES_NOT_USE = (
    (re.compile(re.escape(TOKEN_MAP_TENSOR)), "token id to n-gram id: the reference builds its own from the "
                                              "tokenizer (model.py:1198-1200); compared with it, not used"),
    (re.compile(r"blk\.\d+\.exp_probs_b_vl\.bias"), "the routing bias of image positions, where the reference was "
                                                    "built without vision and declares none (model.py:807)"),
)

# ---- The checks ------------------------------------------------------------------------------------------------------
AGREEMENT_LIMITS = {                         # by GGUF type: the lowest cosine with the checkpoint's values, and the
    GGUF_TYPE_F32: (0.999999, 0.999, 1.001), # range of the norm ratio |GGUF| / |checkpoint|; the module text says
    GGUF_TYPE_Q8_0: (0.999, 0.98, 1.02),     # why these
    GGUF_TYPE_Q2_K: (0.80, 0.6, 1.6),
    GGUF_TYPE_Q3_K: (0.90, 0.6, 1.6),
}
DECODER_CHECK_BLOCKS = 256                   # blocks decoded both ways at each place
DECODER_CHECK_PLACES = (0.0, 0.5, 1.0)       # start, middle and end of a tensor
EXPERTS_CHECKED_PER_LAYER = 3                # evenly spread over a layer's experts: the first, a middle one, the last
TABLE_ROWS_CHECKED = 4096                    # drawn at random, with the first and the last row of the table
TABLE_ROW_COSINE_FLOOR = 0.99                # a single row below this counts as a low row;
TABLE_LOW_ROW_SHARE_LIMIT = 0.01             # more than this share of low rows stops the tool
SELF_CHECK_ROWS = 33                         # one more than the 32 rows the reference's kernels take at a time
SELF_CHECK_SEED = 271828
SELF_CHECK_LIMIT_ONE_LINEAR = 1e-6           # stand-in against kernel, difference over the output: one 8-bit linear
SELF_CHECK_LIMIT_EXPERT = 1e-4               # (3.2e-9 measured), a routed expert's three 4-bit linears (4.5e-7)
INT8_CHECK_BLOCKS_PER_ROW = 8                # the quantizer's check: SELF_CHECK_ROWS rows of this many blocks
HALF_WAY = 0.5                               # a product that far from a whole number is rounded to the even one
INT8_STEP_SHARE_LIMIT = 0.501                # of a step (a scale) between a value and its code times scale: rounding
                                             # leaves half a step, float32's rounding of the two products 1e-4 at most
COMPARISON_CHUNK_VALUES = 2 ** 26            # values compared at a time, in float64: 0.5 GiB an operand
BYTES_PER_GIB = 2 ** 30
FLOAT32_BYTES = 4
Agreement = collections.namedtuple("Agreement", ("cosine", "norm_ratio", "relative_difference"))
PARTS_OF_THE_FILE = ("trunk", "parameters", "tables", "experts")   # what a pass takes from the file, each or not
Taken = collections.namedtuple("Taken", ("trunk", "parameters", "tables", "expert_layers"))   # three yes-or-no and
                                             # the layers whose routed experts are the file's; else the checkpoint's
PlanRow = collections.namedtuple("PlanRow", ("name", "kind", "use", "tensor", "parameter"))
PlanExpert = collections.namedtuple("PlanExpert", ("module", "layer", "expert"))
PlanTable = collections.namedtuple("PlanTable", ("module", "tensor"))
Plan = collections.namedtuple("Plan", ("rows", "experts", "tables", "parameters_without_a_tensor",
                                       "tensors_without_a_parameter"))


# ---- Decoders --------------------------------------------------------------------------------------------------------
def decode_q8_0(raw):
    """numpy: Q8_0 bytes -> float32 weights. A block is a float16 scale d and 32 signed bytes; a weight is d * byte."""
    blocks = raw.reshape(-1, Q8_0_BLOCK_BYTES)
    d = blocks[:, Q8_0_D].copy().view("<f2").astype(numpy.float32)
    return (d * blocks[:, Q8_0_VALUES].copy().view(numpy.int8).astype(numpy.float32)).reshape(-1)


def half_floats(two_bytes):
    """[blocks, 2] bytes on the card, one little-endian float16 each -> [blocks, 1] float32."""
    return two_bytes.contiguous().view(torch.float16).to(torch.float32)


def card_decode_q8_0(raw):
    blocks = raw.reshape(-1, Q8_0_BLOCK_BYTES)
    values = blocks[:, Q8_0_VALUES].contiguous().view(torch.int8).to(torch.float32)
    return (half_floats(blocks[:, Q8_0_D]) * values).reshape(-1)


def card_code_groups(code_bytes):
    """[blocks, 64] bytes of 2-bit codes -> [blocks, 16 groups, 16 weights]: group_codes of the numpy decoders. Group
    2 j + run of a half takes bits 2 j and 2 j + 1 of the 16 bytes of that run."""
    halves = code_bytes.reshape(-1, BLOCK_HALVES, 1, HALF_RUNS, RUN_BYTES)
    shifts = torch.arange(RUN_GROUPS, device = code_bytes.device, dtype = torch.uint8) * PAIR_BITS
    return ((halves >> shifts.reshape(1, 1, -1, 1, 1)) & PAIR_MASK).reshape(-1, BLOCK_GROUPS, GROUP_WEIGHTS)


def card_decode_q2_k(raw):
    """decode_q2_k on the card, the same float32 operations in the same order: d * scale * code - dmin * minimum."""
    blocks = raw.reshape(-1, Q2_K_BLOCK_BYTES)
    scale_bytes = blocks[:, Q2_K_SCALES]
    codes = card_code_groups(blocks[:, Q2_K_CODES]).to(torch.float32)
    scale = (half_floats(blocks[:, Q2_K_D]) * (scale_bytes & NIBBLE_MASK).to(torch.float32)).unsqueeze(-1)
    minimum = (half_floats(blocks[:, Q2_K_DMIN]) * (scale_bytes >> NIBBLE_BITS).to(torch.float32)).unsqueeze(-1)
    return (scale * codes - minimum).reshape(-1)


def card_decode_q3_k(raw):
    """decode_q3_k on the card, the same float32 operations in the same order: d * (scale - 32) * code, a code being
    its two low bits, 4 lower when its high bit is clear."""
    blocks, device = raw.reshape(-1, Q3_K_BLOCK_BYTES), raw.device
    bit_of_byte = torch.arange(BITS_PER_BYTE, device = device, dtype = torch.uint8)
    bits = (blocks[:, Q3_K_HIGH_BITS].unsqueeze(-1) >> bit_of_byte) & 1    # [block, byte, bit], the low bit first
    high = bits.reshape(-1, HALF_RUNS, GROUP_WEIGHTS, BLOCK_HALVES, RUN_GROUPS).permute(Q3_K_HIGH_BIT_AXES) \
        .reshape(-1, BLOCK_GROUPS, GROUP_WEIGHTS)
    codes = card_code_groups(blocks[:, Q3_K_CODES]).to(torch.int32) - (high == 0).to(torch.int32) * Q3_K_CODE_OFFSET
    scale_bytes, scale_index = blocks[:, Q3_K_SCALES], torch.arange(BLOCK_GROUPS, device = device)
    byte_of_word, word = scale_index % Q3_K_WORD_BYTES, scale_index // Q3_K_WORD_BYTES
    source = scale_bytes[:, byte_of_word + Q3_K_WORD_BYTES * (word % Q3_K_LOW_WORDS)]
    low = torch.where(scale_index < Q3_K_LOW_HALF_SCALES, source & NIBBLE_MASK, source >> NIBBLE_BITS)
    high_pair = (scale_bytes[:, Q3_K_HIGH_PAIRS][:, byte_of_word] >> (PAIR_BITS * word).to(torch.uint8)) & PAIR_MASK
    scales = (low | (high_pair << NIBBLE_BITS)).to(torch.int32) - Q3_K_SCALE_OFFSET
    d = half_floats(blocks[:, Q3_K_D])
    return ((d * scales.to(torch.float32)).unsqueeze(-1) * codes.to(torch.float32)).reshape(-1)


NUMPY_DECODERS = {GGUF_TYPE_Q8_0: decode_q8_0, GGUF_TYPE_Q2_K: decode_q2_k, GGUF_TYPE_Q3_K: decode_q3_k}
CARD_DECODERS = {GGUF_TYPE_Q8_0: card_decode_q8_0, GGUF_TYPE_Q2_K: card_decode_q2_k, GGUF_TYPE_Q3_K: card_decode_q3_k,
                 GGUF_TYPE_F32: lambda raw: raw.view(torch.float32), GGUF_TYPE_I32: lambda raw: raw.view(torch.int32)}


class GgufFile:
    """The tensor table of a GGUF and a read-only map of the file. A tensor's shape is the reference's: the stored
    dimensions reversed. What is read is copied into fresh host memory, moved to the card and decoded there."""

    def __init__(self, path):
        assert sys.byteorder == GGUF_BYTE_ORDER, "the oracle refuses another byte order before it loads anything"
        self.name, self.byte_count = os.path.basename(path), os.path.getsize(path)
        self.mapped = numpy.memmap(path, dtype = numpy.uint8, mode = "r")
        self.tensors = {}
        for name, (dimensions, type_code, byte_offset) in read_gguf_tensors(path).items():
            if type_code not in TENSOR_TYPES or not dimensions:
                raise RuntimeError(f"GGUF tensor {name}: type {type_code} is none this tool decodes, or it has no "
                                   "dimension")
            shape, block = tuple(reversed(dimensions)), TENSOR_TYPES[type_code]
            if shape[-1] % block.block_weights:
                raise RuntimeError(f"GGUF tensor {name}: rows of {shape[-1]} weights in {block.name} blocks of "
                                   f"{block.block_weights}")
            self.tensors[name] = GgufTensor(name, shape, type_code, byte_offset,
                                            math.prod(shape) // block.block_weights * block.block_bytes)
        end_of_previous = 0
        for tensor in sorted(self.tensors.values(), key = lambda tensor: tensor.byte_offset):
            if tensor.byte_offset < end_of_previous or tensor.byte_offset + tensor.byte_count > self.byte_count:
                raise RuntimeError(f"GGUF tensor {tensor.name}: bytes {tensor.byte_offset} to "
                                   f"{tensor.byte_offset + tensor.byte_count} overlap the tensor before it or pass "
                                   f"the end of the file ({self.byte_count} bytes)")
            end_of_previous = tensor.byte_offset + tensor.byte_count

    def raw(self, tensor, first_byte, byte_count):
        assert 0 <= first_byte and 0 <= byte_count and first_byte + byte_count <= tensor.byte_count, \
            f"GGUF tensor {tensor.name}: bytes {first_byte} to {first_byte + byte_count} of {tensor.byte_count}"
        start = tensor.byte_offset + first_byte
        return numpy.array(self.mapped[start:start + byte_count])          # a copy: fresh, writable host memory

    def values(self, tensor, device):
        """The whole tensor, decoded on the card, in the reference's shape."""
        raw = torch.from_numpy(self.raw(tensor, 0, tensor.byte_count)).to(device)
        return CARD_DECODERS[tensor.type_code](raw).reshape(tensor.shape)

    def entry(self, tensor, index, device):
        """One entry along the first dimension (one expert of a stacked tensor), decoded on the card."""
        entry_bytes = tensor.byte_count // tensor.shape[0]                 # a row is a whole number of blocks
        raw = torch.from_numpy(self.raw(tensor, index * entry_bytes, entry_bytes)).to(device)
        return CARD_DECODERS[tensor.type_code](raw).reshape(tensor.shape[1:])

    def rows(self, tensor, row_ids, device):
        """The entries of a sorted numpy list of row ids along the first dimension, decoded on the card."""
        row_count, row_bytes = tensor.shape[0], tensor.byte_count // tensor.shape[0]
        assert len(row_ids) and 0 <= int(row_ids.min()) and int(row_ids.max()) < row_count, \
            f"GGUF tensor {tensor.name}: no row, or a row outside its {row_count} rows"
        extent = slice(tensor.byte_offset, tensor.byte_offset + tensor.byte_count)
        gathered = numpy.asarray(self.mapped[extent]).reshape(row_count, row_bytes)[row_ids]   # indexing copies
        decoded = CARD_DECODERS[tensor.type_code](torch.from_numpy(gathered).to(device))
        return decoded.reshape(len(row_ids), *tensor.shape[1:])


def check_card_decoders(gguf, device):
    """Every decoder on the card against its numpy definition, bit for bit, on real blocks of the file."""
    blocks_compared = {}
    for type_code, numpy_decoder in NUMPY_DECODERS.items():
        of_type = sorted((tensor for tensor in gguf.tensors.values() if tensor.type_code == type_code),
                         key = lambda tensor: tensor.byte_offset)
        block, compared, nonzero_values = TENSOR_TYPES[type_code], 0, 0
        for tensor in of_type[:1] + of_type[1:][-1:]:                      # the first and the last in the file
            blocks = tensor.byte_count // block.block_bytes
            count = min(blocks, DECODER_CHECK_BLOCKS)
            for first_block in sorted({int((blocks - count) * place) for place in DECODER_CHECK_PLACES}):
                raw = gguf.raw(tensor, first_block * block.block_bytes, count * block.block_bytes)
                expected = torch.from_numpy(numpy_decoder(raw))
                found = CARD_DECODERS[type_code](torch.from_numpy(raw).to(device)).cpu()
                if not reference_quantization.same_bits(expected, found):
                    raise RuntimeError(f"{block.name}, blocks {first_block} to {first_block + count} of {tensor.name}: "
                                       "the decoder on the card does not give the bits of the numpy decoder")
                compared, nonzero_values = compared + count, nonzero_values + int(torch.count_nonzero(found))
        if of_type and not nonzero_values:
            raise RuntimeError(f"{block.name}: every block compared decoded to zeros: the comparison showed nothing")
        if of_type:
            blocks_compared[block.name] = compared
    return blocks_compared


# ---- The checkpoint's own values, and comparisons --------------------------------------------------------------------
def e2m1_matrix(packed, scales):
    """[outputs, inputs / 2] bytes of two 4-bit floats each, the low half first (convert.py:31-34), and float32
    scales [outputs, inputs / 32] -> float32 [outputs, inputs]: code 8 + m is minus code m (convert.py:13-15)."""
    codes = torch.stack((packed & NIBBLE_MASK, packed >> NIBBLE_BITS), dim = -1).flatten(1).long()
    magnitudes = torch.tensor(E2M1_MAGNITUDES, dtype = torch.float32, device = packed.device)
    values = torch.cat((magnitudes, -magnitudes))[codes]
    return (values.unflatten(1, (-1, WEIGHTS_PER_SCALE)) * scales.unsqueeze(-1)).flatten(1)


def checkpoint_matrix(weight, block_size):
    """What the reference's product multiplies by, in float32 [outputs, inputs], from a quantized parameter and
    the scales the reference attached to it (model.py:222, :227): an 8-bit weight times the scale of its block of
    block_size by block_size (as deepseek.py:1253-1258 expands it), or 4-bit codes times a scale per 32 inputs."""
    if weight.dtype == torch.float8_e4m3fn:
        outputs, inputs = weight.shape
        scales = weight.scale.float().repeat_interleave(block_size, 0)[:outputs].repeat_interleave(block_size, 1)
        return weight.float() * scales[:, :inputs]
    return e2m1_matrix(weight.detach().contiguous().view(torch.uint8), weight.scale.float())


def compare(candidate, original):
    """Cosine of two tensors of one shape, the ratio of their norms (candidate over original) and the norm of
    their difference over the original's, summed in float64 a chunk at a time. Two all-zero tensors agree."""
    assert candidate.shape == original.shape, f"shapes {tuple(candidate.shape)} and {tuple(original.shape)}"
    first, second, sums = candidate.reshape(-1), original.reshape(-1), [0.0, 0.0, 0.0, 0.0]
    for start in range(0, first.numel(), COMPARISON_CHUNK_VALUES):
        left = first[start:start + COMPARISON_CHUNK_VALUES].to(torch.float64)
        right = second[start:start + COMPARISON_CHUNK_VALUES].to(torch.float64)
        for index, term in enumerate((left * right, left * left, right * right, (left - right).square())):
            sums[index] += float(term.sum())
    product, candidate_square, original_square, difference_square = sums
    if candidate_square == 0.0 or original_square == 0.0:
        same = candidate_square == original_square
        return Agreement(float(same), 1.0 if same else math.inf, 0.0 if same else math.inf)
    return Agreement(product / math.sqrt(candidate_square * original_square),
                     math.sqrt(candidate_square / original_square), math.sqrt(difference_square / original_square))


def dequantized_as_the_reference(inputs, block_size):
    """The input of a quantized linear as the reference's product multiplies it: act_quant's 8-bit floats times
    their power-of-two scales (kernel.py:74-90, by its definition in reference_quantization.py), in float32."""
    quantized, scales = reference_quantization.e4m3_with_scales(inputs, block_size)
    return (quantized.to(torch.float32) * scales.unsqueeze(-1)).reshape(inputs.shape)


def int8_with_scales(inputs, block_size):
    """llama.cpp's quantization of activations to 8-bit integers (quantize_row_q8_K_ref, ggml-quants.c:2769-2805,
    where a block is 256 values), written from its definition, in float32, per row and block of block_size values.
    With m the value of largest magnitude (of equal ones the first): the inverse scale is -127 / m; a code is the
    inverse scale times a value, rounded to the nearest whole number (half way to the even one, as nearest_int,
    :621-626, rounds) and at most 127; the scale is 1 / inverse scale. So m becomes -127 whatever its sign, and a
    block of zeros has codes and scale 0. Returns the codes [rows, blocks of a row, block_size], whole numbers in
    float32, and the scales [rows, blocks of a row]."""
    if inputs.shape[-1] % block_size:
        raise RuntimeError(f"an input of shape {tuple(inputs.shape)} to a quantized linear: its rows are not whole "
                           f"blocks of {block_size} values, which 8-bit integer inputs per {block_size} need")
    blocks = reference_quantization.in_blocks(inputs, block_size)
    largest = blocks.gather(-1, blocks.abs().argmax(dim = -1, keepdim = True))
    # tensor over tensor: a number over a tensor is computed as its reciprocal times the number, rounded twice
    inverse_scale = (largest.new_tensor(-LARGEST_INT8_CODE) / largest).masked_fill(largest == 0, 0.0)
    codes = torch.round(inverse_scale * blocks).clamp(max = LARGEST_INT8_CODE)
    if not bool(torch.isfinite(codes).all()):
        raise RuntimeError(f"an input of shape {tuple(inputs.shape)} to a quantized linear: a block of {block_size} "
                           "has no 8-bit integer codes. It holds a value that is not finite, or its largest "
                           f"magnitude is so small that {LARGEST_INT8_CODE} over it is no float32")
    return codes, (largest.new_ones(()) / inverse_scale).masked_fill(largest == 0, 0.0).squeeze(-1)


def host_int8_with_scales(blocks):
    """int8_with_scales as the loop of its definition over float32 blocks [blocks, block size] in host memory."""
    codes, scales = numpy.zeros_like(blocks), numpy.zeros(len(blocks), dtype = numpy.float32)
    for index, block in enumerate(blocks):
        largest = block[numpy.abs(block).argmax()]                         # of equal magnitudes the first
        if largest != 0:
            inverse_scale = numpy.float32(-LARGEST_INT8_CODE) / largest
            codes[index] = numpy.minimum(numpy.rint(inverse_scale * block), numpy.float32(LARGEST_INT8_CODE))
            scales[index] = numpy.float32(1) / inverse_scale
    return codes, scales


def check_int8_inputs(block_size, device):
    """int8_with_scales on the card against the loop of its definition on the host, codes and scales bit for bit,
    on seeded rows of normal values whose first row begins with four blocks made for it (the module text names
    them); then the card's codes and scales against what any reading of the rule must give: 8-bit integers that
    reach 127 in every block that is not zeros, and a code times its scale within half a step of its value.
    Returns the blocks compared, how many of them are zeros, the farthest such product in steps, and how far the
    products lie from the rows of normal values alone."""
    generator = numpy.random.default_rng(SELF_CHECK_SEED)
    rows = generator.standard_normal((SELF_CHECK_ROWS, INT8_CHECK_BLOCKS_PER_ROW * block_size), dtype = numpy.float32)
    half_way = numpy.arange(block_size, dtype = numpy.float32) % LARGEST_INT8_CODE + numpy.float32(HALF_WAY)
    half_way[0] = LARGEST_INT8_CODE          # the inverse scale is -1: every other product lies half way
    both_signs = half_way.copy()
    both_signs[1] = -LARGEST_INT8_CODE       # the same magnitude again with the other sign: it becomes code 127
    made = numpy.concatenate((numpy.zeros_like(half_way), half_way, -half_way, both_signs))
    rows[0, :len(made)] = made
    blocks, values = rows.reshape(-1, block_size), torch.from_numpy(rows).to(device)
    expected_codes, expected_scales = host_int8_with_scales(blocks)
    codes, scales = int8_with_scales(values, block_size)
    quantizer = f"8-bit integer inputs per {block_size}: the quantizer on {device}"
    if not (reference_quantization.same_bits(codes.reshape(blocks.shape).cpu(), torch.from_numpy(expected_codes))
            and reference_quantization.same_bits(scales.reshape(-1).cpu(), torch.from_numpy(expected_scales))):
        raise RuntimeError(f"{quantizer} does not give the codes and scales of the loop of its definition on the host")
    is_zeros = scales == 0                   # from here on the card's own codes and scales, whatever the loop gave
    if not bool(((codes == codes.round()).all(dim = -1)
                 & (codes.abs().amax(dim = -1) == ~is_zeros * LARGEST_INT8_CODE)).all()):
        raise RuntimeError(f"{quantizer} gives codes that are not whole numbers up to {LARGEST_INT8_CODE} in "
                           f"magnitude, or a block that is not zeros without a code of magnitude {LARGEST_INT8_CODE}")
    restored = (codes * scales.unsqueeze(-1)).reshape(values.shape)        # as the stand-in multiplies them
    steps = (restored - values).reshape(codes.shape).abs() / scales.abs().masked_fill(is_zeros, 1.0).unsqueeze(-1)
    farthest = float(steps.max())
    if not farthest <= INT8_STEP_SHARE_LIMIT:                              # a value that is not a number fails too
        raise RuntimeError(f"{quantizer} gives a code times its scale {farthest:.3f} of a step (the scale) from its "
                           f"value: rounding to the nearest whole number leaves {HALF_WAY} at most, so the scale is "
                           "not the step of the codes")
    difference = compare(restored[1:], values[1:]).relative_difference    # without the made blocks
    return {"blocks_compared_bit_for_bit": len(blocks), "blocks_of_zeros_among_them": int(is_zeros.sum()),
            "farthest_product_from_its_value_in_steps": farthest, "relative_difference_on_normal_values": difference}


class LinearStandIn:
    """Stands where model.py looks its linear() up (model.py:181-207). A float weight goes to the reference's own
    function; a quantized one is answered by a float32 product with the matrix that matrix_of gives for it."""

    def __init__(self, reference_model, matrix_of, linear_inputs):
        self.reference_model, self.matrix_of, self.linear_inputs = reference_model, matrix_of, linear_inputs
        self.reference_linear = reference_model.linear

    def linear(self, inputs, weight, bias = None):                         # as model.py:181 is called
        if weight.dtype not in QUANTIZED_WEIGHT_DTYPES:
            return self.reference_linear(inputs, weight, bias)             # model.py:207: F.linear on a float weight
        assert bias is None, "the reference's quantized linear takes no bias (model.py:184)"
        matrix = self.matrix_of(weight)
        assert matrix.dim() == MATRIX_DIMENSIONS and matrix.shape[1] == inputs.shape[-1] \
            and matrix.dtype == torch.float32, f"a matrix of shape {tuple(matrix.shape)} and dtype {matrix.dtype}"
        if self.linear_inputs in INT8_INPUT_BLOCKS:
            codes, scales = int8_with_scales(inputs, INT8_INPUT_BLOCKS[self.linear_inputs])
            values = (codes * scales.unsqueeze(-1)).reshape(inputs.shape)  # ggml-quants.c:2814: a scale times a code
        elif self.linear_inputs == INPUTS_AS_THE_REFERENCE:
            values = dequantized_as_the_reference(inputs, self.reference_model.fp8_block_size)
        else:
            values = inputs.to(torch.float32)
        return torch.nn.functional.linear(values, matrix).to(torch.get_default_dtype())   # kernel.py:299 and :583

    @contextlib.contextmanager
    def installed(self):
        self.reference_model.linear = self.linear
        try:
            yield
        finally:
            self.reference_model.linear = self.reference_linear


def check_products(reference_model, plan, device):
    """The stand-in against the reference's own kernels, both multiplying by the CHECKPOINT's weights, before any
    weight is replaced: one 8-bit linear of the trunk and one whole routed expert (three 4-bit linears) on random
    rows, the inputs quantized as the reference. Returns the relative difference of the outputs of each."""
    block_size = reference_model.fp8_block_size
    stand_in = LinearStandIn(reference_model, lambda weight: checkpoint_matrix(weight, block_size),
                             INPUTS_AS_THE_REFERENCE)
    generator = torch.Generator(device = device).manual_seed(SELF_CHECK_SEED)

    def random_rows(width):
        return torch.randn(SELF_CHECK_ROWS, width, generator = generator, device = device,
                           dtype = torch.float32).to(torch.get_default_dtype())

    linear_row = next(row for row in plan.rows if row.use == USE_LINEAR)
    rows = random_rows(linear_row.parameter.shape[1])
    outputs = {linear_row.name: (stand_in.linear(rows, linear_row.parameter),
                                 reference_model.linear(rows, linear_row.parameter), SELF_CHECK_LIMIT_ONE_LINEAR)}
    host_expert = plan.experts[0].module
    rows = random_rows(host_expert.stage.expert.w1.in_features)
    route_weights = torch.rand(SELF_CHECK_ROWS, 1, generator = generator, device = device, dtype = torch.float32)
    by_the_reference = host_expert(rows, route_weights)  # the oracle moves the expert in; the reference's kernels run
    with stand_in.installed():                           # the same bytes on the stage, through the stand-in
        outputs[host_expert.checkpoint_name] = (host_expert.stage.expert(rows, route_weights), by_the_reference,
                                                SELF_CHECK_LIMIT_EXPERT)
    differences = {}
    for name, (found, expected, limit) in outputs.items():
        if found.shape != expected.shape or found.dtype != expected.dtype:
            raise RuntimeError(f"{name}: the stand-in returned {tuple(found.shape)} {found.dtype}, the reference "
                               f"{tuple(expected.shape)} {expected.dtype}")
        differences[name] = compare(found, expected).relative_difference
        if not differences[name] <= limit:                                 # a value that is not a number fails too
            raise RuntimeError(f"{name}: with the checkpoint's own weights the stand-in's output lies "
                               f"{differences[name]:.3e} of the reference's output away from it (rounding is "
                               f"expected to stay below {limit}): the stand-in, or the decoding of the checkpoint's "
                               "weights here, is wrong")
    return differences


# ---- The map, resolved against a model and a file --------------------------------------------------------------------
def map_row_of(name):
    """The map's row for a parameter name and the name of its GGUF tensor, or (None, None)."""
    if name in ROOT_ROWS:
        return ROOT_ROWS[name], ROOT_ROWS[name].tensor
    match = LAYER_NAME.fullmatch(name)
    if match is None or match["rest"] not in LAYER_ROWS:
        return None, None
    return LAYER_ROWS[match["rest"]], f"blk.{match['layer']}.{LAYER_ROWS[match['rest']].tensor}"


def kind_of(name):
    """A parameter's name without its layer's number: what the report groups by."""
    match = LAYER_NAME.fullmatch(name)
    return name if match is None else f"layers.*.{match['rest']}"


def reason_no_tensor_is_needed(name, parameters):
    """Why a parameter (or a stand-in) without a row needs no tensor. One that has no reason stops the tool."""
    for pattern, reason in PARAMETERS_A_TEXT_FORWARD_DOES_NOT_USE:
        if pattern.fullmatch(name):
            return reason
    weight_name = name.removesuffix(SCALE_SUFFIX) + WEIGHT_SUFFIX
    if name.endswith(SCALE_SUFFIX) and weight_name in parameters and map_row_of(weight_name)[0] is not None \
            and parameters[weight_name].dtype in QUANTIZED_WEIGHT_DTYPES:
        return SCALES_ARE_MULTIPLIED_IN
    raise RuntimeError(f"{name} of the reference model has no row in the map of gguf_weights.py and is not among "
                       "those a text forward leaves unused")


def build_plan(model, expert_class, table_class, gguf):
    """Every parameter and stand-in of the model paired with its tensor, shapes checked; every tensor accounted for."""
    parameters = dict(model.named_parameters())
    rows, experts, tables, without_tensor, claimed = [], [], [], collections.Counter(), set()
    for name, parameter in parameters.items():
        row, tensor_name = map_row_of(name)
        if row is None:
            without_tensor[reason_no_tensor_is_needed(name, parameters)] += 1
            continue
        tensor = gguf.tensors.get(tensor_name)
        if tensor is None:
            raise RuntimeError(f"{name} ({row.evidence}): the GGUF holds no tensor {tensor_name}")
        if (parameter.dtype in QUANTIZED_WEIGHT_DTYPES) != (row.use == USE_LINEAR):
            raise RuntimeError(f"{name} is of {parameter.dtype} in the reference; the map takes it for "
                               f"{row.use} ({row.evidence})")
        if tensor.shape != tuple(parameter.shape) or tensor.type_code == GGUF_TYPE_I32:
            raise RuntimeError(f"{name}: the reference declares {tuple(parameter.shape)}, {tensor_name} holds "
                               f"{TENSOR_TYPES[tensor.type_code].name} {tensor.shape} ({row.evidence})")
        claimed.add(tensor_name)
        rows.append(PlanRow(name, kind_of(name), row.use, tensor, parameter))
    experts_of_layer = collections.defaultdict(set)
    for module_name, module in model.named_modules():
        if isinstance(module, expert_class):
            match = ROUTED_EXPERT_NAME.fullmatch(module_name)
            if match is None:
                without_tensor[reason_no_tensor_is_needed(module_name, parameters)] += 1
                continue
            layer, expert = int(match["layer"]), int(match["expert"])
            for row in EXPERT_MAP:
                tensor, slot = gguf.tensors.get(f"blk.{layer}.{row.tensor}"), module.stage.slots[row.parameter]
                matrix_shape = (slot.shape[0], slot.shape[1] * FP4_VALUES_PER_BYTE)
                if tensor is None or len(tensor.shape) != len(matrix_shape) + 1 or tensor.shape[1:] != matrix_shape \
                        or tensor.type_code == GGUF_TYPE_I32:
                    raise RuntimeError(f"{module_name}.{row.parameter}: the reference declares {matrix_shape}; "
                                       f"blk.{layer}.{row.tensor} is missing or does not stack that ({row.evidence})")
                claimed.add(tensor.name)
            experts_of_layer[layer].add(expert)
            experts.append(PlanExpert(module, layer, expert))
        elif isinstance(module, table_class):
            match = TABLE_NAME.fullmatch(module_name)
            tensor = None if match is None else gguf.tensors.get(f"blk.{match['layer']}.{TABLE_ROW.tensor}")
            if tensor is None or tensor.shape != (module.row_count, module.row_width) \
                    or tensor.type_code == GGUF_TYPE_I32:
                raise RuntimeError(f"{module_name}: a table of {(module.row_count, module.row_width)} has no such "
                                   f"tensor {TABLE_ROW.tensor} in its block ({TABLE_ROW.evidence})")
            claimed.add(tensor.name)
            tables.append(PlanTable(module, tensor))
    for layer, found in sorted(experts_of_layer.items()):
        for row in EXPERT_MAP:
            stacked = gguf.tensors[f"blk.{layer}.{row.tensor}"].shape[0]
            if found != set(range(stacked)):
                raise RuntimeError(f"blk.{layer}.{row.tensor} stacks {stacked} experts, the reference routes among "
                                   f"{len(found)} in layer {layer}")
    if not experts or not any(row.use == USE_LINEAR for row in rows):
        raise RuntimeError("the reference model shows no quantized linear with a row, or no routed expert")
    unused = {}
    for name in sorted(set(gguf.tensors) - claimed):
        reasons = [reason for pattern, reason in TENSORS_A_TEXT_FORWARD_DOES_NOT_USE if pattern.fullmatch(name)]
        if not reasons:
            raise RuntimeError(f"GGUF tensor {name} is taken by no parameter of the reference model and is not "
                               "among those a text forward leaves unused")
        unused[name] = reasons[0]
    return Plan(rows, experts, tables, dict(without_tensor), unused)


# ---- Putting the file's weights in -----------------------------------------------------------------------------------
class GgufWeights:
    """The file's weights in a built reference model: take_trunk, check_experts and check_tables compare each kind
    with the checkpoint's values; stand_in then makes the reference compute with the file's where a part is taken,
    and with the checkpoint's own through the same stand-in where it is not."""

    def __init__(self, reference_model, model, stages, gguf, plan, linear_inputs, taken):
        self.reference_model, self.model, self.stages, self.gguf, self.plan = reference_model, model, stages, gguf, plan
        # every comparison with the file is made whatever is taken; what is not taken keeps the CHECKPOINT's own
        # values, multiplied through the same stand-in: with nothing taken, what the stand-in alone changes
        self.taken = taken
        self.expert_modules = {expert.module.checkpoint_name: expert.module for expert in plan.experts}
        self.layer_of_expert = {expert.module.checkpoint_name: expert.layer for expert in plan.experts}
        self.device = next(model.parameters()).device
        self.linear_stand_in = LinearStandIn(reference_model, self.matrix_of, linear_inputs)
        self.matrices = {}                   # id of an 8-bit parameter of the trunk -> its float32 matrix on the card
        self.staged = {id(parameter): (stage, slot_name) for stage in stages
                       for slot_name, parameter in stage.slots.items()}
        self.expert_tensors = {(expert.module.checkpoint_name, row.parameter):
                               (gguf.tensors[f"blk.{expert.layer}.{row.tensor}"], expert.expert)
                               for expert in plan.experts for row in EXPERT_MAP}
        self.report = {}                     # by kind: what the comparisons with the checkpoint showed

    def judge(self, name, kind, tensor, agreement, reference_dtype):
        """Holds one tensor's agreement with the checkpoint to its type's limits, and records it under its kind."""
        lowest_cosine, lowest_ratio, highest_ratio = AGREEMENT_LIMITS[tensor.type_code]
        type_name = TENSOR_TYPES[tensor.type_code].name
        if not (agreement.cosine >= lowest_cosine and lowest_ratio <= agreement.norm_ratio <= highest_ratio):
            raise RuntimeError(f"{name} against GGUF tensor {tensor.name} ({type_name}): cosine {agreement.cosine:.6f} "
                               f"with the checkpoint's values (at least {lowest_cosine} is expected) and a norm "
                               f"{agreement.norm_ratio:.6f} of the checkpoint's ({lowest_ratio} to {highest_ratio}): "
                               "the map, a decoder or the file is wrong for this tensor")
        entry = self.report.setdefault(kind, {
            "gguf_types": [], "reference_dtypes": [], "compared": 0, "lowest_cosine": agreement.cosine,
            "lowest_norm_ratio": agreement.norm_ratio, "highest_norm_ratio": agreement.norm_ratio,
            "largest_relative_difference": agreement.relative_difference,
            "largest_relative_change_by_the_cast": None})                 # None: nothing of this kind was cast
        entry.update(
            gguf_types = sorted({*entry["gguf_types"], type_name}),
            reference_dtypes = sorted({*entry["reference_dtypes"], str(reference_dtype).removeprefix("torch.")}),
            compared = entry["compared"] + 1, lowest_cosine = min(entry["lowest_cosine"], agreement.cosine),
            lowest_norm_ratio = min(entry["lowest_norm_ratio"], agreement.norm_ratio),
            highest_norm_ratio = max(entry["highest_norm_ratio"], agreement.norm_ratio),
            largest_relative_difference = max(entry["largest_relative_difference"], agreement.relative_difference))
        return entry

    @staticmethod
    def note_cast(entry, change):
        """Records what the cast to the reference's dtype changed in values the pass will compute with."""
        known = entry["largest_relative_change_by_the_cast"]
        entry["largest_relative_change_by_the_cast"] = change if known is None else max(known, change)

    def take_trunk(self):
        """Every tensor of the trunk: decoded, compared with the checkpoint's, then (where its part is taken from
        the file; else the checkpoint's values stay) written over its float parameter
        or kept as the matrix of its quantized linear."""
        block_size = self.reference_model.fp8_block_size
        for row in self.plan.rows:
            values = self.gguf.values(row.tensor, self.device)
            original = checkpoint_matrix(row.parameter, block_size) if row.use == USE_LINEAR else row.parameter
            entry = self.judge(row.name, row.kind, row.tensor, compare(values, original), row.parameter.dtype)
            if row.use == USE_LINEAR:
                self.matrices[id(row.parameter)] = values if self.taken.trunk else original
                continue
            if not self.taken.parameters:
                continue
            row.parameter.copy_(values)                                    # the cast to the parameter's dtype
            self.note_cast(entry, compare(row.parameter, values).relative_difference)

    def check_experts(self):
        """Three matrices of a few experts of every layer against the checkpoint's 4-bit codes times their scales."""
        modules = collections.defaultdict(dict)
        for expert in self.plan.experts:
            modules[expert.layer][expert.expert] = expert.module
        for layer, of_layer in sorted(modules.items()):
            spread = numpy.linspace(0, len(of_layer) - 1, EXPERTS_CHECKED_PER_LAYER).round().astype(int).tolist()
            for expert in sorted(set(spread)):
                module = of_layer[expert]
                for row in EXPERT_MAP:
                    tensor, _ = self.expert_tensors[(module.checkpoint_name, row.parameter)]
                    scale_name = row.parameter.removesuffix(WEIGHT_SUFFIX) + SCALE_SUFFIX
                    weight_slot, scale_slot = module.stage.slots[row.parameter], module.stage.slots[scale_name]
                    packed = module.host_bytes[row.parameter].to(self.device).reshape(weight_slot.shape)
                    scales = module.host_bytes[scale_name].to(self.device).reshape(scale_slot.shape)
                    original = e2m1_matrix(packed, scales.view(scale_slot.dtype).float())  # as model.py:319 reads them
                    self.judge(f"{module.checkpoint_name}.{row.parameter}", f"layers.*.ffn.experts.*.{row.parameter}",
                               tensor, compare(self.gguf.entry(tensor, expert, self.device), original),
                               weight_slot.dtype)

    def check_tables(self):
        """Random rows of each table against the same rows as the oracle's lookup returns them from the checkpoint."""
        generator = numpy.random.default_rng(SELF_CHECK_SEED)
        for table in self.plan.tables:
            module, name = table.module, f"{table.module.checkpoint_name}{WEIGHT_SUFFIX}"
            row_ids = numpy.unique(numpy.concatenate((
                numpy.array([0, module.row_count - 1]), generator.integers(0, module.row_count, TABLE_ROWS_CHECKED))))
            rows_moved = module.rows_moved
            original = module(torch.from_numpy(row_ids).to(self.device)).to(torch.float64)   # ids sorted, each once:
            module.rows_moved = rows_moved                                                   # rows in this order
            decoded = self.gguf.rows(table.tensor, row_ids, self.device)
            entry = self.judge(name, kind_of(name), table.tensor, compare(decoded, original), TABLE_VALUE_DTYPE)
            if self.taken.tables:
                self.note_cast(entry, compare(decoded.to(TABLE_VALUE_DTYPE), decoded).relative_difference)
            found = decoded.to(torch.float64)
            norms = found.norm(dim = -1) * original.norm(dim = -1)
            both_zero = (found.abs().sum(dim = -1) == 0) & (original.abs().sum(dim = -1) == 0)
            row_cosine = torch.where(norms > 0, (found * original).sum(dim = -1) / norms, both_zero.to(torch.float64))
            low_rows = int((row_cosine < TABLE_ROW_COSINE_FLOOR).sum())
            entry.update(rows_compared = entry.get("rows_compared", 0) + len(row_ids),
                         rows_below_the_row_floor = entry.get("rows_below_the_row_floor", 0) + low_rows)
            if low_rows > TABLE_LOW_ROW_SHARE_LIMIT * len(row_ids):
                raise RuntimeError(f"{name} against GGUF tensor {table.tensor.name}: {low_rows} of {len(row_ids)} rows "
                                   f"have a cosine below {TABLE_ROW_COSINE_FLOOR} with the checkpoint's row of the "
                                   "same number: rows are read from the wrong place")

    def matrix_of(self, weight):
        """The float32 matrix for a quantized weight the reference is about to multiply by: the file's, or the
        checkpoint's own where that part is not taken from the file."""
        matrix = self.matrices.get(id(weight))
        if matrix is not None:
            return matrix
        assert id(weight) in self.staged, "a quantized weight without a row: the oracle runs no part that has one"
        stage, slot_name = self.staged[id(weight)]
        assert (stage.occupant_name, slot_name) in self.expert_tensors, f"{stage.occupant_name}.{slot_name}"
        started = time.perf_counter()
        tensor, expert = self.expert_tensors[(stage.occupant_name, slot_name)]
        if self.layer_of_expert[stage.occupant_name] in self.taken.expert_layers:
            matrix = self.gguf.entry(tensor, expert, self.device)
            stage.moved_bytes += tensor.byte_count // tensor.shape[0]
        else:                                                              # as check_experts reads the checkpoint's
            module = self.expert_modules[stage.occupant_name]
            scale_name = slot_name.removesuffix(WEIGHT_SUFFIX) + SCALE_SUFFIX
            weight_slot, scale_slot = stage.slots[slot_name], stage.slots[scale_name]
            packed = module.host_bytes[slot_name].to(self.device).reshape(weight_slot.shape)
            scales = module.host_bytes[scale_name].to(self.device).reshape(scale_slot.shape)
            matrix = e2m1_matrix(packed, scales.view(scale_slot.dtype).float())
            stage.moved_bytes += packed.numel() * packed.element_size() + scales.numel() * scales.element_size()
        stage.move_seconds += time.perf_counter() - started
        return matrix

    def staged_run(self, stage, host_expert, rows, route_weights):
        """Stands where the oracle's ExpertStage.run moved the checkpoint's bytes in: the stage only learns which
        expert is called. Its matrices come from the file (or, for a layer not taken, from the checkpoint's bytes
        the module holds) when the reference's Expert.forward (model.py:841-851)
        reaches each of its three linears; the 4-bit bytes left on the stage are never multiplied by."""
        stage.occupant_name = host_expert.checkpoint_name
        stage.moves += 1
        return stage.expert(rows, route_weights)

    def table_lookup(self, module, tensor, indices):
        """Stands where the oracle's table looked rows up in the checkpoint (model.py:312-325 for one rank): the
        same rows of the file's table, decoded, in the dtype the reference's lookup returns."""
        assert 0 <= int(indices.min()) and int(indices.max()) < module.row_count, module.checkpoint_name
        row_ids, renumbered = torch.unique(indices, return_inverse = True)
        rows = self.gguf.rows(tensor, row_ids.cpu().numpy(), self.device)
        module.rows_moved += len(row_ids)
        return rows.to(TABLE_VALUE_DTYPE)[renumbered]

    def token_map_is_the_reference_s(self):
        """Whether the file's token map is the one the reference hashes n-grams with; None where either is absent."""
        tensor = self.gguf.tensors.get(TOKEN_MAP_TENSOR)
        own = getattr(getattr(self.model, "engram_hash", None), "token_map", None)
        if tensor is None or not torch.is_tensor(own):
            return None
        stored = self.gguf.values(tensor, self.device).reshape(-1).long()
        own = own.to(self.device).reshape(-1).long()
        return stored.shape == own.shape and bool((stored == own).all())

    def stand_in(self):
        """From here on the reference computes through the stand-in, with the file's weights where a part is
        taken, for the rest of the process."""
        def refusing(part_name):
            def refuse(module, arguments):
                raise AssertionError(f"a module of the reference's {part_name} ran: the oracle calls none of them")
            return refuse

        self.reference_model.linear = self.linear_stand_in.linear
        for stage in self.stages:
            stage.run = functools.partial(self.staged_run, stage)
        for table in self.plan.tables if self.taken.tables else ():
            table.module.forward = functools.partial(self.table_lookup, table.module, table.tensor)
        for part_name in PARTS_NOT_IN_THE_FILE:
            part = getattr(self.model, part_name, None)
            modules = list(part) if isinstance(part, torch.nn.ModuleList) else [] if part is None else [part]
            for module in modules:
                module.register_forward_pre_hook(refusing(part_name))


def install(reference_model, model, stages, expert_class, table_class, gguf_path, linear_inputs,
            parts = PARTS_OF_THE_FILE, expert_layers = None):
    """Checks a GGUF against a reference model that holds its checkpoint, and makes the model compute with the
    file's weights (the module text says how). Returns what it found, for the capture's manifest. parts names
    what is taken from the file, and expert_layers the layers whose routed experts are (all of them when None);
    everything else keeps the checkpoint's own values, multiplied through the same stand-in. With no part taken
    that is the control: what the stand-in's float32 products alone change in the logits."""
    assert linear_inputs in LINEAR_INPUTS, linear_inputs
    if reference_model.scale_fmt is None or reference_model.scale_dtype != torch.float8_e8m0fnu:
        raise RuntimeError("the reference does not round its activation scales to a power of two (model.py:29-30): "
                           "reference_quantization.py defines no other act_quant")
    gguf = GgufFile(gguf_path)
    device = next(model.parameters()).device
    decoder_blocks = check_card_decoders(gguf, device)
    print(f"{gguf.name}: {len(gguf.tensors)} tensors; the decoders on the card give the numpy decoders' bits on "
          + ", ".join(f"{blocks} {name} blocks" for name, blocks in decoder_blocks.items()), flush = True)
    plan = build_plan(model, expert_class, table_class, gguf)
    layers = {expert.layer for expert in plan.experts}
    chosen = (layers if "experts" in parts else set()) if expert_layers is None else set(expert_layers)
    assert set(parts) <= set(PARTS_OF_THE_FILE), f"parts {tuple(parts)}: the oracle's options name no other"
    assert expert_layers is None or ("experts" in parts and chosen), "layers go with experts, at least one"
    if not chosen <= layers:                 # only here is the model's number of layers known
        raise RuntimeError(f"layers {sorted(chosen - layers)} are not among the model's {len(layers)} layers: the "
                           "routed experts of a layer it has not cannot be taken from the file")
    taken = Taken("trunk" in parts, "parameters" in parts, "tables" in parts, frozenset(chosen))
    with torch.inference_mode():             # as the reference's forward runs its kernels (model.py:1241), and as
        products = check_products(reference_model, plan, device)          # the oracle's own staged-expert check does
    print("the stand-in of linear() against the reference's kernels, on the checkpoint's weights: "
          + "; ".join(f"{name} differs by {difference:.2e} of the output" for name, difference in products.items()),
          flush = True)
    int8_check = None                        # with 8-bit integer inputs only: the quantizer against its definition
    if linear_inputs in INT8_INPUT_BLOCKS:
        int8_check = check_int8_inputs(INT8_INPUT_BLOCKS[linear_inputs], device)
        print(f"linear inputs {linear_inputs}: 8-bit integers with one scale per {INT8_INPUT_BLOCKS[linear_inputs]} "
              f"inputs of a row. On {int8_check['blocks_compared_bit_for_bit']} seeded blocks the card gives the "
              "codes and scales of the loop of the definition on the host, and every block but the "
              f"{int8_check['blocks_of_zeros_among_them']} of zeros reaches a code of magnitude {LARGEST_INT8_CODE}; "
              f"code times scale lies at most {int8_check['farthest_product_from_its_value_in_steps']:.3f} of a "
              f"step from its value and {int8_check['relative_difference_on_normal_values']:.2e} of their norm "
              "from normal values", flush = True)
    weights = GgufWeights(reference_model, model, stages, gguf, plan, linear_inputs, taken)
    weights.take_trunk()
    if any(matrix.requires_grad for matrix in weights.matrices.values()):
        raise RuntimeError("the trunk's float32 matrices were formed with autograd on: each keeps its two factors, "
                           "two more float32 copies of the trunk on the card; install() runs under torch.no_grad()")
    weights.check_experts()
    weights.check_tables()
    token_map_agrees = weights.token_map_is_the_reference_s()
    weights.stand_in()
    for kind, entry in weights.report.items():
        cast = entry["largest_relative_change_by_the_cast"]
        print(f"  {kind:42s} {'/'.join(entry['gguf_types']):9s} {entry['compared']:4d} compared: cosine at least "
              f"{entry['lowest_cosine']:.6f}, norm {entry['lowest_norm_ratio']:.4f} to "
              f"{entry['highest_norm_ratio']:.4f} of the checkpoint's, difference at most "
              f"{entry['largest_relative_difference']:.4f}" + ("" if cast is None else
              f"; as {'/'.join(entry['reference_dtypes'])} changed by at most {cast:.1e}"), flush = True)
    overwritten = sum(row.use == USE_PARAMETER for row in plan.rows) if taken.parameters else 0
    tables_from_the_file = len(plan.tables) if taken.tables else 0
    experts_from_the_file = sum(expert.layer in taken.expert_layers for expert in plan.experts)
    matrix_bytes = sum(matrix.numel() for matrix in weights.matrices.values()) * FLOAT32_BYTES
    token_map_text = "was not compared with the reference's" if token_map_agrees is None \
        else "is the reference's" if token_map_agrees \
        else "DIFFERS from the reference's: the file's own runtime would read other table rows than this pass"
    from_the_file = {
        "trunk_matrices": taken.trunk, "float_parameters": taken.parameters, "tables": taken.tables,
        "routed_experts_of_layers": sorted(taken.expert_layers)}
    if not (taken.trunk and taken.parameters and taken.tables and taken.expert_layers == layers):
        print(("CONTROL: nothing is taken from the file. " if not any((*taken[:-1], taken.expert_layers)) else
               "PART of the file is taken. ") + f"From the file: {json.dumps(from_the_file)}; everything else keeps "
              "the CHECKPOINT's own values, multiplied through the same stand-in", flush = True)
    print(f"{overwritten} float parameters overwritten; {len(weights.matrices)} matrices of quantized linears on the "
          f"card in float32 ({matrix_bytes / BYTES_PER_GIB:.2f} GiB), the "
          f"{'file' if taken.trunk else 'checkpoint'}'s; {experts_from_the_file} of {len(plan.experts)} routed "
          f"experts and {tables_from_the_file} of {len(plan.tables)} tables read from the file when called; linear "
          f"inputs {linear_inputs}; the file's token map {token_map_text}", flush = True)
    return {"file": gguf.name, "file_bytes": gguf.byte_count, "tensors": len(gguf.tensors),
            "taken_from_the_file": from_the_file,
            "linear_inputs": linear_inputs, "int8_inputs_against_their_definition": int8_check,
            "decoder_blocks_compared_bit_for_bit": decoder_blocks,
            "float_parameters_overwritten": overwritten,
            "matrices_of_quantized_linears_kept_in_float32": len(weights.matrices),
            "bytes_of_those_matrices_on_the_card": matrix_bytes,
            "routed_experts_read_from_the_file_when_called": experts_from_the_file,
            "tables_read_from_the_file_when_looked_up": tables_from_the_file,
            "parameters_without_a_tensor": plan.parameters_without_a_tensor,
            "tensors_without_a_parameter": plan.tensors_without_a_parameter,
            "agreement_limits_by_gguf_type": {TENSOR_TYPES[type_code].name: list(limits)
                                              for type_code, limits in AGREEMENT_LIMITS.items()},
            "agreement_with_the_checkpoint_by_kind": weights.report,
            "stand_in_against_the_reference_relative_difference": products,
            "token_map_is_the_reference_s": token_map_agrees}
