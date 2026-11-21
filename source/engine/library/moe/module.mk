# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/library/moe/: MoE plumbing.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/library/moe_REQUIRES := common engine/backend engine/program engine/formats/weights engine/library \
    engine/library/block_scaled_linear
