# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/library/collectives/: ordered collectives, pushes, routes, mailboxes.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/library/collectives_REQUIRES := common engine/backend engine/program engine/library
REQUIRED_TRAPS_kernel_ordered_collective := ordered_all_reduce ordered_gather ordered_reduce_scatter ordered_two_shot
REQUIRED_TRAPS_kernel_ordered_push := ordered_push
REQUIRED_TRAPS_kernel_ordered_push_parallel := ordered_push_parallel_publish ordered_push_parallel_reduce
REQUIRED_TRAPS_kernel_ordered_push_two := ordered_push_two
