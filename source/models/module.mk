# SPDX-License-Identifier: AGPL-3.0-or-later
# source/models/: the model contract and the registry.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
models_REQUIRES := common engine/backend engine/topology engine/program engine/formats/weights engine/formats/digest \
    engine/formats/pack engine/formats/exl3 engine/storage engine/storage/load engine/input/tokenizer \
    engine/input/image
