# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/formats/exl3/: the EXL3 map's sidecar descriptor.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/formats/exl3_REQUIRES := common engine/formats/pack engine/formats/weights
