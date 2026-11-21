# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/library/sampling/: sampler, vocabulary-sharded sampler.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/library/sampling_REQUIRES := common engine/backend engine/program engine/library \
    engine/library/block_scaled_linear
REQUIRED_TRAPS_kernel_sampler_shard := sampler_shard
