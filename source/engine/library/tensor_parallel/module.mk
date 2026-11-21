# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/library/tensor_parallel/: TP exchange body, TP join, peer row copy.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/library/tensor_parallel_REQUIRES := common engine/backend engine/program engine/library \
    engine/library/collectives engine/library/rings engine/library/wire_codecs engine/library/block_scaled_linear
REQUIRED_TRAPS_kernel_tp_exchange := tp_exchange tp_exchange_ce_owner_ring_sum tp_exchange_ce_owner_sum \
    tp_exchange_done tp_exchange_owner_publish tp_exchange_owner_ring_publish tp_exchange_owner_ring_sum \
    tp_exchange_owner_sum tp_exchange_publish tp_exchange_ready tp_exchange_sum
