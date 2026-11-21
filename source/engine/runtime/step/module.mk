# SPDX-License-Identifier: AGPL-3.0-or-later
# source/engine/runtime/step/: the step driver.
# What the layout cannot say: the modules its files may include, and its kernels' build classes, required traps
# and register-checked instances, which the root Makefile gathers.
engine/runtime/step_REQUIRES := common engine/backend engine/program engine/formats/pack engine/engram \
    engine/library/sampling engine/input/image models
