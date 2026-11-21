# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/library/exl3/: EXL3 products.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/library/exl3_REQUIRES := common engine/backend engine/program engine/formats/weights engine/library \
    engine/library/block_scaled_linear engine/library/moe engine/formats/pack engine/formats/exl3
NATIVE_KERNEL_SOURCES += source/engine/library/exl3/kernel_exl3.cu
SM_80_ONLY_KERNEL_SOURCES += source/engine/library/exl3/kernel_exl3_product.cu
SM_80_ONLY_KERNEL_SOURCES += source/engine/library/exl3/kernel_exl3_tc.cu
SM_80_ONLY_KERNEL_SOURCES += source/engine/library/exl3/kernel_exl3_tcp.cu
REQUIRED_TRAPS_kernel_exl3_tcp := exl3_tcp_route
NATIVE_KERNEL_SOURCES += source/engine/library/exl3/kernel_exl3_wide.cu
NATIVE_ONLY_KERNEL_SOURCES += source/engine/library/exl3/kernel_exl3_wide.cu
