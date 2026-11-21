# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/engram/: Engram n-gram tables.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/engram_REQUIRES := common engine/backend engine/program engine/formats/weights engine/formats/digest \
    engine/formats/pack engine/formats/exl3 engine/storage engine/storage/io engine/storage/load \
    engine/storage/replicas engine/storage/profile engine/storage/autotune
