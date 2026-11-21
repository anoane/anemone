# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/runtime/placement/: placement, sessions that fit.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/runtime/placement_REQUIRES := common engine/topology engine/program engine/formats/pack engine/storage/load \
    models engine/backend
