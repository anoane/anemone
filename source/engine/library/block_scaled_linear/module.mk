# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/library/block_scaled_linear/: E4M3/E2M1 block-scaled products, the MoE routing arrays' records.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/library/block_scaled_linear_REQUIRES := common engine/backend engine/program engine/formats/weights \
    engine/library
NATIVE_KERNEL_SOURCES += source/engine/library/block_scaled_linear/kernel_expert_linear.cu
REGISTER_CHECKED_INSTANCES += expert_(gate_up(_quantize)?|down)
WIDE_SLICE_INSTANCES += gate_up
SM_80_ONLY_KERNEL_SOURCES += source/engine/library/block_scaled_linear/kernel_expert_relaxed.cu
NATIVE_KERNEL_SOURCES += source/engine/library/block_scaled_linear/kernel_trunk_linear.cu
REGISTER_CHECKED_INSTANCES += expert_shared_join
REGISTER_CHECKED_INSTANCES += trunk_linear_(quantized|bfloat16)(_wide|_interleaved)?
SM_80_ONLY_KERNEL_SOURCES += source/engine/library/block_scaled_linear/kernel_trunk_relaxed.cu
