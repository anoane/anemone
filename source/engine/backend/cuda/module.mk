# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/backend/cuda/: the CUDA driver backend, the kernel image table, the fill kernel.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/backend/cuda_REQUIRES := common engine/backend engine/formats/weights engine/formats/digest
