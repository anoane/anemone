# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/input/tokenizer/: byte-level BPE, the pre-tokenizer contract.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/input/tokenizer_REQUIRES := common engine/formats/pack
