# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/library/bf16_linear/: BF16/F32 products.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/library/bf16_linear_REQUIRES := common engine/backend engine/program engine/formats/weights engine/library \
    engine/library/block_scaled_linear
SM_80_ONLY_KERNEL_SOURCES += source/engine/library/bf16_linear/kernel_head_mma.cu
SM_80_ONLY_KERNEL_SOURCES += source/engine/library/bf16_linear/kernel_trunk_mma.cu
