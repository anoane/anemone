# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/runtime/scheduler/: the scheduler behind engine_*.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/runtime/scheduler_REQUIRES := engine/runtime/step engine/runtime/speculation engine/input/tokenizer \
    engine/input/image common engine/program
