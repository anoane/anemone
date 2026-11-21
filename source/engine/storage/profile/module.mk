# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/storage/profile/: the storage profile document.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/storage/profile_REQUIRES := common engine/storage engine/formats/weights engine/formats/digest \
    engine/formats/pack engine/formats/exl3 engine/backend engine/storage/io engine/program engine/storage/load \
    engine/storage/replicas
