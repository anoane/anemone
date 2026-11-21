# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/cli/: options, model source, load, fit, start, generate, serve, stats, capture, axes, policy, knobs
# (every module but a model's).
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/cli_REQUIRES := common engine/formats/weights engine/formats/digest engine/backend engine/backend/cuda \
    engine/topology engine/program engine/storage engine/formats/pack engine/formats/exl3 engine/library \
    engine/library/block_scaled_linear engine/library/bf16_linear engine/library/exl3 engine/library/moe \
    engine/library/collectives engine/library/rings engine/library/wire_codecs engine/library/tensor_parallel \
    engine/library/sampling engine/storage/io engine/storage/load engine/storage/replicas engine/storage/profile \
    engine/storage/autotune engine/engram engine/input/tokenizer engine/input/image models engine/runtime/step \
    engine/runtime/placement engine/runtime/speculation engine/runtime/scheduler engine/server/api engine/server
