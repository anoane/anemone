# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/formats/pack/: the pack container and reader, the tokenizer-tables binding.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/formats/pack_REQUIRES := common engine/formats/weights engine/formats/digest engine/storage engine/program
