# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/library/rings/: BF16 rings, cut-through.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/library/rings_REQUIRES := common engine/backend engine/program engine/library engine/library/wire_codecs \
    engine/library/collectives
REQUIRED_TRAPS_kernel_bf16_compressed_ring := bf16_compressed_ring
REQUIRED_TRAPS_kernel_bf16_round_once_ring := bf16_round_once_ring
REQUIRED_TRAPS_kernel_exact_ring := exact_ring exact_ring_unequal
