# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/library/wire_codecs/: the lossless BF16 codec and the quantised-activation wire.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/library/wire_codecs_REQUIRES := common engine/backend engine/program engine/library \
    engine/library/block_scaled_linear
